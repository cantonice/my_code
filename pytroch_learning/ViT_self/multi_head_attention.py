import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,
                 dim, # 输入token的dim，768
                 num_heads=8,  # Transformer是8个头，ViT是12个头，所以等下实例化12头
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        # 计算每个head的dim,直接均分操作。
        head_dim=dim//num_heads
        # 计算分母,q, k相乘之后要除以一个根号下dk。
        self.scale=qk_scale or head_dim** -0.5
        # 直接使用一个全连接实现q,k,v
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias)  # 因为它需要同时输出查询、键和值3个向量，每个都有与输入相同的维度。
        self.attn_drop=nn.Dropout(attn_drop_ratio)
        # 多头拼接之后通过W进行映射，跟上面的qkv一样，也是通过全连接实现
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop_ratio)

    def forward(self,x):
        # [batch_size, num_patches + 1, total_embed_dim]，即(B,197,768)
        B,N,C=x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head],即(B,197,3,12,64)
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]，即(3,B,12,197,64)
        # C // self.num_heads:每个head的q,k,v对应的维度
        # Linear函数可以接收多维的矩阵输入但是只对最后一维起效果，其他维度不变。permute()函数用于调整维度。
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # 通过切片获取q,k,v
        q,k,v=qkv[0],qkv[1],qkv[2]

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]。@为矩阵乘法，q,k是多维矩阵，只有最后两个维度进行矩阵乘法。
        # 每个head的q,k进行相乘。输出维度大小为(1,12,197,197)
        attn=(q@k.transpose(-2,-1))*self.scale
        # 在最后一个维度，即每一行进行softmax处理
        attn=attn.softmax(dim=-1)
        # softmax处理后要经过一个dropout层
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        # q,k矩阵相乘的结果要和v相乘得到一个加权结果，输出维度为(1,12,197,64)，然后交换1,2维度，再进行reshape操作
        # 其实这个reshape操作就是对多头的拼接，得到最后的输出shape为(1,197,768)
        x=(attn@v).transpose(1,2).reshape(B,N,C)
        # 经过Woy映射,也就是一个全连接层
        x=self.proj(x)
        # 经过一个dropout层。一般全连接后面都跟一个dropout层
        x=self.proj_drop(x)
        return x










