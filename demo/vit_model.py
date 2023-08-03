""""
参考rwightman 
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
进行的vit demo书写
"""
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

# droppath减少过拟合，学习自bilibili.com/video/BV1Jh411Y7WQ
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    该函数用于在残差块的主路径中对样本进行随机深度丢弃（Stochastic Depth）。

    参数：
    x：输入张量
    drop_prob：丢弃概率，默认为0.（即不进行丢弃）
    training：是否处于训练模式，默认为False
    返回：
    如果丢弃概率为0或者不处于训练模式，则返回输入张量x
    否则，返回通过随机深度丢弃后的张量output
    """
    if drop_prob != 0. and training:
        keep_prob = 1 - drop_prob
        random_tensor = torch.rand(x.size(), dtype=x.dtype, device=x.device) + keep_prob
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    else:
        return x


# 通过实例化方式使用droppath
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

#进行嵌入向量的转化
class PatchEmbed(nn.Module):
    """
    该类用于进行2D图像到Patch的嵌入。

    参数：
    img_size：输入图像的尺寸，默认为224
    patch_size：Patch的尺寸，默认为16
    in_c：输入图像的通道数，默认为3（默认图片为RGB和YUV图像，均为3通道）
    embed_dim：图像词嵌入维度，默认为768（即16*16*3）
    norm_layer：规范化层，用于对嵌入后的特征进行规范化，默认为None

    成员变量：
    img_size：输入图像的尺寸
    patch_size：Patch的尺寸
    grid_size：Patch的网格大小，即图像被分成了多少个Patch
    num_patches：总的Patch数量
    proj：用于将输入图像进行卷积嵌入的卷积层
    norm：用于对嵌入特征进行规范化的规范化层
    前向传播函数（forward）：

    获取输入张量x的维度信息，包括批量大小B、通道数C、高度H、宽度W。
    判断输入图像的尺寸是否与模型的期望尺寸相匹配，如果不匹配则抛出异常。
    使用卷积层proj对输入张量x进行卷积嵌入，得到维度为[B, embed_dim, grid_size[0], grid_size[1]]的输出张量。
    将输出张量进行flatten操作，将维度变为[B, embed_dim, num_patches]。
    对flatten后的输出张量进行转置操作，将维度变为[B, num_patches, embed_dim]。
    使用规范化层norm对转置后的输出张量进行规范化。
    返回规范化后的张量x。
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# 注意力机制的实现
class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # 获取输入的维度信息
        B, N, C = x.shape
        
        # 将输入通过全连接层 qkv，得到 q、k、v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 对注意力矩阵进行dropout操作
        attn = self.attn_drop(attn)

        # 将注意力矩阵与v相乘，并进行维度转换和重塑
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 通过全连接层proj进行特征映射，并进行dropout操作
        x = self.proj(x)
        x = self.proj_drop(x)

        return x




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()

        # Layer normalization for the input
        self.norm1 = norm_layer(dim)

        # Self-attention layer
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        # Dropout layer for the attention
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        # Layer normalization for the output of attention layer
        self.norm2 = norm_layer(dim)

        # Multi-Layer Perceptron (MLP) layer
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # Apply layer normalization to the input
        x = self.norm1(x)
        
        # Pass the input through the self-attention layer and add it to the input (residual connection)
        x = x + self.drop_path(self.attn(x))
        
        # Apply layer normalization to the output of the attention layer
        x = self.norm2(x)
        
        # Pass the output of the attention layer through the MLP layer and add it to the output (residual connection)
        x = x + self.drop_path(self.mlp(x))
        
        return x



class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Patch Embedding layer
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class and Distilled Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        # Stochastic Depth Decay Rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        # Transformer Blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        # Layer normalization
        self.norm = norm_layer(embed_dim)

        # Representation layer
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

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # Patch Embedding
        x = self.patch_embed(x)

        # Class and Distilled Tokens
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)  # Positional Embedding and Dropout
        x = self.blocks(x)  # Transformer Blocks
        x = self.norm(x)  # Layer normalization

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)

        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # During inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)

        return x



def _init_vit_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):  # 合并Linear和Conv2d的初始化操作
    # 使用trunc_normal_初始化权重，均值为0，标准差为0.01
        nn.init.trunc_normal(m.weight, std=.01)
    if m.bias is not None:
    # 初始化偏置为0
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
    # 初始化LayerNorm的偏置为0，权重为1
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224(num_classes: int = 1000):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
