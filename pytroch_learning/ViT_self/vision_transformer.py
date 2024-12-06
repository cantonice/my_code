from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from patch_embed import PatchEmbed
from block import Block


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,  # depth of transformer，就是我们上面的Block堆叠多少次
                 num_heads=12,
                 mlp_ratio=4.0,  # mlp的hidden dim比embedding dim
                 qkv_bias=True,
                 qkv_scale=None,
                 representation_size=None,  # enable and set representation layer (pre-logits) to this value if set。对应的是最后的MLP中的pre-logits中的全连接层的节点个数。默认是none，也就是不会去构建MLP中的pre-logits，mlp中只有一个全连接层。
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None, ):
        super().__init__()
        self.num_classes=num_classes
        self.num_features=self.embed_dim=embed_dim
        # 不用管distilled，所有self.num_tokens=1
        self.num_tokens=2 if distilled else 1
        # norm_layer默认为none，所有norm_layer=nn.LayerNorm，用partial方法给一个默认参数。partial 函数的功能就是：把一个函数的某些参数给固定住，返回一个新的函数。
        norm_layer=norm_layer or partial(nn.LayerNorm,esp=1e-6)
        # act_layer默认等于GELU函数
        act_layer=act_layer or nn.GELU
        # 通过embed_layer构建PatchEmbed
        self.patch_embed=embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        # 获得num_patches的总个数196
        num_patches=self.patch_embed.num_patches
        # 创建一个cls_token，形状为(1,768),直接通过0矩阵进行初始化，后面在训练学习。下面要和num_patches进行拼接，加上一个类别向量，从而变成(197,768)
        self.cls_token=nn.Parameter(torch.zores(1,1,embed_dim))
        # 不用管，用不到，因为distilled默认为none
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # 创建一个位置编码，形状为（197，768）
        self.pos_embed=nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,self.embed_dim))
        # 此处的dropout为加上位置编码后的dropout层
        self.pos_drop = nn.Dropout(p=drop_ratio)
        # 根据传入的drop_path_ratio构建一个等差序列，总共depth个元素，即在每个Encoder Block中的失活比例都不一样。默认为0，可以传入参数改变
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        # 构建blocks，首先通过列表创建depth次，也就是12次。然后通过nn.Sequential方法把列表中的所有元素打包成整体赋值给self.blocks。
        # *[...]：这是 Python 中的可变参数列表（unpacking operator），它允许将列表或元组中的元素作为独立的参数传递给函数。
        self.block=nn.Sequential(*[
            Block(embed_dim,num_heads, mlp_ratio, qkv_bias, qkv_scale, drop_ratio, attn_drop_ratio,
                  drop_path_ratio=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)
        ])
        # 通过norm_layer层
        self.norm = norm_layer(embed_dim)

        # distilled不用管，只用看representation_size即可，如果有传入representation_size，在MLP中就会构建pre-logits。否者直接 self.has_logits = False，然后执行self.pre_logits = nn.Identity()，相当于没有pre-logits。
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # 整个网络的最后一层全连接层，输出就是分类类别个数，前提要num_classes > 0
        self.head=nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist=None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim,self.num_classes) if num_classes > 0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        # 先进行patch_embeding处理
        x = self.patch_embed(x)
        # 对cls_token在batch维度进行复制batch_size份
        cls_token=self.cls_token.expand(x.shape[0],-1,-1)  # 在维度1进行操作连接[1, 1, 768] -> [B, 1, 768]
        # self.dist_token默认为none。
        if self.dist_token is None:
            # 在dim=1的维度上进行拼接，输出shape：[B, 197, 768]
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # 位置编码后有个dropout层
        x=self.pos_drop(x+self.pos_embed)
        x=self.block(x)
        x=self.norm(x)
        # 提取clc_token对应的输出，也就是提取出类别向量。
        if self.dist_token is None:
            # 返回所有的batch维度和第二个维度上面索引为0的数据
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        # 首先将x传给forward_features()函数，输出shape为(1,768)
        x = self.forward_features(x)
        # self.head_dist默认为none,自动执行else后面的语句
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            # 输出特征大小为(1,1000)，对应1000分类
            x = self.head(x)
        return x

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
