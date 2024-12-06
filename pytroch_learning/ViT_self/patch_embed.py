import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
        Image --> Patch Embedding --> Linear Proj --> Pos Embedding
        Image size -> [224,224,3]
        Patch size -> 16*16
        Patch num -> (224^2)/(16^2)=196
        Patch dim -> 16*16*3 =768
        Patch Embedding: [224,224,3] -> [196,768]
        Linear Proj: [196,768] -> [196,768]
     	Positional Embedding: [196,768] -> [197,768]
    """
    """
        2D Image to Patch Embedding
    """
    def __init__(self,img_size=224,in_c=3,embed_dim=768,patch_size=16,norm_layer=None):
        """
        构造函数
        :param img_size:默认输入的图片是224*224
        :param in_c: 默认输入的图片是3个通道
        :param embed_dim: 规定嵌入层维度是768维=16*16*3
        :param patch_size: 默认切片是16*16
        :param norm_layer:默认嵌入层不需要层归一化
        :return:返回网络传播后的输入x
        """
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size  # 实例变量，可以在其他方法(forward)中也可以访问
        self.patch_size = patch_size
        self.grid_size = (img_size[0]//patch_size[0],img_size[1]//patch_size[1])
        self.num_patches = self.grid_size[0]*self.grid_size[1]
        # [224,224,3] >> [14,14,768]
        self.proj=nn.Conv2d(in_channels=in_c,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        # 计算每个维度大小
        B,C,H,W = x.shape
        # 断言输入是否符合规范
        assert H == self.img_size[0] and W ==self.img_size[1],\
        f"输入的图片是{H}*{W},图片不符合要求：({self.img_size[0]} * {self.img_size[1]})"
        """
            flatten: [B, C, H, W] -> [B, C, HW]
            transpose: [B, C, HW] -> [B, HW, C]
            进过卷积之后从第2维度开始展平，之后再交换1,2维度(为了计算方便)
        """
        x = self.proj(x).flatten(2).transpose(1,2)  # [1*3*224*224]>>[1*768*14*14]>>[1*768*196]>>[1*196*768]
        x = self.norm(x)
        # print(x.shape)
        # """
        #     在 PyTorch 中，nn.Parameter 是一个特殊的类，用于将一个张量（Tensor）标记为模型的参数。
        #     这意味着这个张量将被视为模型的一部分，会在神经网络的训练过程中被优化（即进行梯度更新）。
        # """
        # cls_token = nn.Parameter(torch.zeros(1, 1, 768))  # 引入一个类别的token用于表征类别
        # """
        #     在 PyTorch 中，nn.init.trunc_normal_ 是一个初始化函数，用于对张量进行截断正态分布的初始化。
        #     这种初始化方法特别适用于深度学习模型中的参数初始化，因为它可以控制权重的分布范围，从而避免在训练
        #     初期出现过大或过小的梯度，这有助于模型的稳定训练。
        # """
        # nn.init.trunc_normal_(self.cls_token, std=0.02)  # 初始化
        # cls_token = cls_token.expand(x.shape[0], -1, -1)  # [1,1,768]>>[128,1,768]
        # x = torch.cat((cls_token,x),dim=1)
        return x
