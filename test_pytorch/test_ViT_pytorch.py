# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class PatchEmbed(nn.Module):
#     """
#     2D Image to Patch Embedding
#     """
#
#     def __init__(self, image_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         """
#         Map input tensor to patch.
#         Args:
#             image_size: input image size
#             patch_size: patch size
#             in_c: number of input channels
#             embed_dim: embedding dimension. dimension = patch_size * patch_size * in_c
#             norm_layer: The function of normalization
#         """
#         super().__init__()
#         image_size = (image_size, image_size)
#         patch_size = (patch_size, patch_size)
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         # The input tensor is divided into patches using 16x16 convolution
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.image_size[0] and W == self.image_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
#
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#
#         return x
#
# class Block(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_ratio=0.,
#                  attn_drop_ratio=0.,
#                  drop_path_ratio=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super(Block, self).__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
# class VisionTransformer(nn.Module):
#     def __init__(self, image_size=224, patch_size=16, in_c=3, num_classes=1000,
#                  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
#                  qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
#                  attn_drop_ratio=0., drop_path_ratio=0.5, embed_layer=PatchEmbed, norm_layer=None,
#                  act_layer=None):
#         """
#         Args:
#             image_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_c (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension, dim = patch_size * patch_size * in_c
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             distilled (bool): model includes a distillation token and head as in DeiT models
#             drop_ratio (float): dropout rate
#             attn_drop_ratio (float): attention dropout rate
#             drop_path_ratio (float): stochastic depth rate
#             embed_layer (nn.Module): patch embedding layer
#             norm_layer: (nn.Module): normalization layer
#         """
#         super(VisionTransformer, self).__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_tokens = 2 if distilled else 1
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # partial类似闭包函数，第一个参数是函数对象，后面的参数是第一个函数对象的实参
#         act_layer = act_layer or nn.GELU
#
#         self.patch_embed = embed_layer(image_size=image_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))     # nn.Parameter转换函数，把参数转换为可训练变量
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_ratio)
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
#         self.blocks = nn.Sequential(*[
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
#                   norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)
#         ])
#         self.norm = norm_layer(embed_dim)
#
#         # Representation layer
#         if representation_size and not distilled:
#             self.has_logits = True
#             self.num_features = representation_size
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ("fc", nn.Linear(embed_dim, representation_size)),
#                 ("act", nn.Tanh())
#             ]))
#         else:
#             self.has_logits = False
#             self.pre_logits = nn.Identity()     # placeholder
#
#         # Classifier head(s)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#         self.head_dist = None
#         if distilled:
#             self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
#
#         # Weight init
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         if self.dist_token is not None:
#             nn.init.trunc_normal_(self.dist_token, std=0.02)
#
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         self.apply(_init_vit_weights)
#
#     def forward_features(self, x):
#         # [B, C, H, W] -> [B, num_patches, embed_dim]
#         x = self.patch_embed(x)  # [B, 196, 768]
#         # [1, 1, 768] -> [B, 1, 768]
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         if self.dist_token is None:
#             x = torch.cat((cls_token, x), dim=1)  # 在维度1进行操作连接[B, 197, 768]
#         else:
#             x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         x = self.norm(x)
#         if self.dist_token is None:
#             return self.pre_logits(x[:, 0])
#         else:
#             return x[:, 0], x[:, 1]
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         if self.head_dist is not None:
#             x, x_dist = self.head(x[0]), self.head_dist(x[1])
#             if self.training and not torch.jit.is_scripting():
#                 # during inference, return the average of both classifier predictions
#                 return x, x_dist
#             else:
#                 return (x + x_dist) / 2
#         else:
#             x = self.head(x)
#
#         return x
