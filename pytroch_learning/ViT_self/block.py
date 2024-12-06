import torch.nn as nn
from multi_head_attention import Attention
from  mlp import Mlp
from  drop_path import DropPath

class Block(nn.Module):
    """
    这就是一个 Encoder Block
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,  # 对应multi-head attention最后全连接层的失活比例
                 attn_drop_ratio=0.,  # q,k矩阵相乘之后通过softmax之后的全连接层的失活比例
                 drop_path_ratio=0.,  # 框图中Droppath失活比例。也可以使用dropout,没啥影响
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # layernorm层
        self.norm1=norm_layer(dim)
        # 实例化上面讲的Attention类
        self.attn=Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                            attn_drop_ratio=attn_drop_ratio,proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 失活比例大于0就实例化DropPath，否者不做任何操作
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        # 第二个layernorm层
        self.norm2=norm_layer(dim)
        # 第一个全连接之后输出维度翻四倍
        mlp_hidden_dim = int(dim * mlp_ratio)
        # 实例化mlp
        self.mlp=Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_ratio)

    def forward(self,x):
        x=x+self.drop_path(self.attn(self.norm1(x)))
        x=x+self.drop_path(self.mlp(self.norm2(x)))
        return x
