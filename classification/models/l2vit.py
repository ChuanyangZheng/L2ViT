import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from torch.nn.modules import module

def build_norm(cfg, dim):
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type == 'SyncBN':
        return nn.SyncBatchNorm(dim)
    elif layer_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_type == 'LN_custom':
        return LayerNorm(dim, eps=1e-6, data_format="channels_first")
    else:
        return nn.BatchNorm2d(dim)


class LayerNorm(nn.Module):
    """
    a more flexible implementation from ConvNeXt in paper:A ConvNet for the 2020s,
    thanks.
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def build_block(block_type, index=1, **kwargs):
    if block_type == 'local':
        return LWABlock(**kwargs)
    elif block_type == 'enhanced_vit':
        return LGABlock(enhanced=True, **kwargs)
    elif block_type == 'mix_local_enhanced_vit':
        if index % 2 == 0:
            return LWABlock(**kwargs)
        else:
            return LGABlock(enhanced=True, **kwargs)


class Mul(nn.Module):
    # add Mul Module and hook for calculating FLOPs.
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        return q @ k


def mul_flops_counter_hook(module, input, output):
    q, k = input
    # multiply batch because batch is always 1,
    # and swin reshape number of windows into batch.
    mul_flops = q.numel() * k.shape[-1]
    module.__flops__ += int(mul_flops)

# better deal with tuple input than nn.Sequential
class CustomSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CPE(nn.Module):
    """
    condition position embedding.
    """
    def __init__(self, dim, k=3, act=False):
        super(CPE, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).contiguous().view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    local window attention in paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qk_mul = Mul()
        self.kv_mul = Mul()

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = self.qk_mul(q, k.transpose(-2, -1))
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = self.kv_mul(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class LWABlock(nn.Module):
    """
    local windows attention block.
    """
    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([CPE(dim=dim, k=3, act=cpe_act),
                                  CPE(dim=dim, k=3, act=cpe_act)])

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


class LCM(nn.Module):
    """
    local concentration module.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 drop=0., kernel_size=7):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x, size):
        H, W = size
        B, N, C = x.shape
        x = x.transpose(-1, -2).contiguous().view(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.flatten(2).transpose(-1, -2)
        return x


class LinearAttention(nn.Module):
    """
    the linear attention, kv first is equal to qk first,
    we use the equivalent qk first function for attention map visualization.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * self.scale)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qk_mul = Mul()
        self.kv_mul = Mul()
        self.extra_mul = Mul()
        self.act = nn.ReLU()

    def forward(self, x):  # kv first, then q*attn
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.act(q)
        k = self.act(k)
        denom = torch.clamp(self.qk_mul(q, k.transpose(-2, -1).sum(dim=-1, keepdim=True)), 1e2)
        # clamp makes training stable and boost performance as shown in our paper.
        attn = self.kv_mul(k.transpose(-2, -1), v) * self.temperature
        # add scale on attn out is better than add scale on k,
        # which doesn't change the attn output, as the former is more like softmax attention.
        attn = self.extra_mul(q, attn)
        attn = attn / denom

        x = attn.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # def forward(self, x):  # qk first, then attn*v
    #     B, N, C = x.shape
    #     qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    #     q, k, v = qkv[0], qkv[1], qkv[2]

    #     q = self.act(q)
    #     k = self.act(k)
    #     attn = self.qk_mul(q, k.transpose(-2, -1))
    #     denom = torch.clamp(attn.sum(dim=-1, keepdim=True), 1.0)
    #     attn = attn / denom * self.scale
    #     attn = self.attn_drop(attn)

    #     x = self.kv_mul(attn, v).transpose(1, 2).reshape(B, N, C)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
    #     return x


class LGABlock(nn.Module):
    """
    linear global attention block.
    """
    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False, enhanced=False, **kwargs):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([CPE(dim=dim, k=3, act=cpe_act)])
        self.norm1 = norm_layer(dim)
        self.attn = LinearAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

        self.local_mp = None
        if enhanced:
            self.norm3 = norm_layer(dim)
            self.local_mp = LCM(in_features=dim, act_layer=act_layer)

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)
        if self.local_mp is not None:
            x = x + self.drop_path(self.local_mp(self.norm3(x), size))
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


class ConvStem(nn.Module):
    """
    convolutional stem.
    """
    def __init__(self, in_chans, embed_dim, patch_size=4, act_layer=nn.ReLU, norm=dict(type='BN')):
        super(ConvStem, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            build_norm(norm, embed_dim // 2),
            act_layer(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            build_norm(norm, embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class ConvStem_4L(nn.Module):
    """
    scaled relu: 3Conv+3BN+3ReLU+1Proj form paper
    <Scaled ReLU Matters for Training Vision Transformers>.
    We ablate this stem in our experiments.
    """
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, patch_norm=None):
        super(ConvStem_4L, self).__init__()

        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        hid_dim = embed_dim // 2
        stem_stride = 2
        self.proj = nn.Sequential(
                nn.Conv2d(in_chans,hid_dim,kernel_size=7,stride=stem_stride,padding=3,bias=False), #112x112
                nn.BatchNorm2d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_dim,hid_dim,kernel_size=3,stride=1,padding=1,bias=False), #112x112
                nn.BatchNorm2d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_dim,hid_dim,kernel_size=3,stride=1,padding=1,bias=False), #112x112
                nn.BatchNorm2d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_dim,embed_dim,kernel_size=patch_size//stem_stride,stride=patch_size//stem_stride),
            )

    def forward(self, x):
        x = self.proj(x)
        return x

def build_conv_stem(conv_stem_type, **kwargs):
    if conv_stem_type == '2L':
        return ConvStem(**kwargs)
    elif conv_stem_type == '4L':
        return ConvStem_4L(**kwargs)
    else:
        raise NotImplementedError('not supported conv stem')

class PatchEmbed(nn.Module):
    def __init__(self,
                patch_size=16,
                in_chans=3,
                embed_dim=96,
                conv_stem=False,
                conv_stem_type='2L',
                stem_overlapped=False,
                overlapped=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        stem_kernel = 7 if stem_overlapped else 4
        padding = stem_kernel // 2 if stem_overlapped else 0
        if patch_size[0] == 4:
            if conv_stem:
                self.proj = build_conv_stem(conv_stem_type, 
                                            in_chans=in_chans, 
                                            embed_dim=embed_dim,
                                            patch_size=patch_size)
            else:
                self.proj = nn.Conv2d(in_chans, embed_dim,
                                kernel_size=stem_kernel,
                                stride=patch_size,
                                padding=padding)
            self.norm = nn.LayerNorm(embed_dim)
        if patch_size[0] == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(in_chans, embed_dim,
                                kernel_size=to_2tuple(kernel),
                                stride=patch_size,
                                padding=to_2tuple(pad))
            self.norm = nn.LayerNorm(in_chans)

    def forward(self, x, size):
        H, W = size
        dim = len(x.shape)
        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        if dim == 4:
            x = self.norm(x)
        return x, newsize

class L2ViT(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, mlp_ratio=[4., 4., 4., 4., ],
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 cpe=False, sub_cpe=False, cpe_kernel=3, cpe_act=False,
                 conv_stem=False, conv_stem_type='2L',
                 stem_overlapped=False, overlapped=False,
                 ffn_conv=False, rffn_conv=False,
                 window_size=7,
                 token_mixer='csmix',
                 channel_mixer='mlp',
                 block_type='convnext',
                 norm=dict(type='LN'), last_proj=False,
                 ):
        super().__init__()
        self.num_stages = len(dims)
        self.downsample_layers = nn.ModuleList([
            PatchEmbed(patch_size=4 if i == 0 else 2,
                       in_chans=in_chans if i == 0 else dims[i - 1],
                       embed_dim=dims[i],
                       conv_stem=conv_stem,
                       conv_stem_type=conv_stem_type,
                       stem_overlapped=stem_overlapped,
                       overlapped=overlapped)
            for i in range(self.num_stages)])

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = CustomSequential(
                *[build_block(block_type, index=j, dim=dims[i], drop_path=dp_rates[cur + j], mlp_ratio=mlp_ratio[i],
                  num_heads=num_heads[i], layer_scale_init_value=layer_scale_init_value,
                  cpe=cpe, sub_cpe=sub_cpe, cpe_kernel=cpe_kernel, cpe_act=cpe_act,
                  ffn_conv=ffn_conv, rffn_conv=rffn_conv, window_size=window_size,
                  token_mixer=token_mixer, channel_mixer=channel_mixer,
                  norm=norm) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = build_norm(norm, dims[-1]) # final norm layer
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.last_proj = None
        if last_proj:
            self.last_proj = nn.Linear(dims[-1], 1280, 1)
            self.last_act = nn.ReLU()
        self.head = nn.Linear(dims[-1] if not last_proj else 1280, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.custom_modules_hooks = {Mul: mul_flops_counter_hook}

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        size = [x.size(2), x.size(3)]
        for i in range(4):
            x, size = self.downsample_layers[i](x, size)
            x, size = self.stages[i](x, size)
        x = self.norm(x)   # B, L, C
        if self.last_proj is not None:
            x = self.last_proj(x)
            x = self.last_act(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def L2ViT_Tiny(pretrained=False, **kwargs):
    model = L2ViT(depths=[2, 2, 6, 2], mlp_ratio=[4., 4., 4., 4.], dims=[96, 192, 384, 768],
                       block_type='mix_local_enhanced_vit', conv_stem=True,
                       **kwargs)
    return model


@register_model
def L2ViT_Small(pretrained=False, **kwargs):
    model = L2ViT(depths=[2, 2, 18, 2], mlp_ratio=[4., 4., 4., 4.], dims=[96, 192, 384, 768],
                       block_type='mix_local_enhanced_vit', conv_stem=True,
                       **kwargs)
    return model


@register_model
def L2ViT_Base(pretrained=False, **kwargs):
    model = L2ViT(depths=[2, 2, 18, 2], mlp_ratio=[4., 4., 4., 4.], dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32],
                       block_type='mix_local_enhanced_vit', conv_stem=True,
                       **kwargs)
    return model


@register_model
def L2ViT_Base_384(pretrained=False, **kwargs):
    model = L2ViT(depths=[2, 2, 18, 2], mlp_ratio=[4., 4., 4., 4.], dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32],
                       block_type='mix_local_enhanced_vit', conv_stem=True, window_size=12,
                       **kwargs)
    return model