import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import trunc_normal_
import numpy as np


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
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

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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


class Attention(nn.Module):
    def __init__(self, dim, head_dim = 64, grid_size=4, ds_ratio=4, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout2d(proj_drop, inplace=True)

        if grid_size > 1:
            self.grid_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
            self.avg_pool = nn.AvgPool2d(ds_ratio, stride=ds_ratio)
            self.q = nn.Conv2d(dim, dim, 1)
            self.kv = nn.Conv2d(dim, dim * 2, 1)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        # stage2,3,4:[16, 401, 384] -> [16, 400, 384] -> [16, 384, 400] -> [16, 384, 20, 20]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        qkv = self.qkv(x)

        if self.grid_size > 1:
            grid_h, grid_w = H // self.grid_size, W // self.grid_size
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, grid_h,
                              self.grid_size, grid_w, self.grid_size)  # B QKV Heads Dim H GSize W GSize
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3)
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            grid_x = (attn @ v).reshape(B, self.num_heads, grid_h, grid_w,
                                        self.grid_size, self.grid_size, self.head_dim)
            grid_x = grid_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, C, H, W)
            grid_x = self.grid_norm(x + grid_x)

            q = self.q(grid_x).reshape(B, self.num_heads, self.head_dim, -1)
            q = q.transpose(-2, -1)
            kv = self.kv(self.avg_pool(grid_x))
            kv = kv.reshape(B, 2, self.num_heads, self.head_dim, -1)
            kv = kv.permute(1, 0, 2, 4, 3)
            k, v = kv[0], kv[1]
        else:
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        global_x = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        if self.grid_size > 1:
            global_x = global_x + grid_x
        x = self.drop(self.proj(global_x))
        x = x.flatten(2).transpose(1, 2)

        return x

'''block on the transformer branch: Multihead-6 self-attention + MLP'''
class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

'''ConvNeXt Block'''
class ConvBlock(nn.Module):

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x, x_t=None, return_x_2=True):
        shortcut = x # stage2,3,4:[16,96,80,80]
        # stage2,3,4:[16,96,80,80]
        x = self.dwconv(x) if x_t is None else self.dwconv(x + x_t)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x2 = self.norm(x) # stage2,3,4:[16,96,80,80]
        x = self.pwconv1(x2)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x.permute(0, 2, 3, 1)
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = shortcut + self.drop_path(x)
        if return_x_2:
            return x, x2
        else:
            return x

class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """
    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        x = self.conv_project(x)  # [N, C, H, W] -> [N, 384, H, W] adjust channel -> 384
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        return x

class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.GELU):
        super(FCUUp, self).__init__()
        # Upsample + 1x1conv + batch norm
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = LayerNorm(outplanes, eps=1e-6, data_format='channels_first')
        self.act = act_layer()


    def forward(self, x, H, W):
        B, _, C = x.shape # stage2,3,4: [16, 401, 384]
        x_r = x.transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        x_r = F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))
        return x_r

class Med_ConvBlock(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)

        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)

        return x

class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """
    def __init__(self, dw_stride, dim, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 downsample=False, num_med_block=0):

        super(ConvTransBlock, self).__init__()
        self.norm = LayerNorm(dim // 2, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv2d(dim // 2, dim, kernel_size=2, stride=2)
        self.cnn_block = ConvBlock(dim=dim, drop_rate=drop_path_rate)
        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(dim=dim, drop_rate=drop_path_rate))
            self.med_block = nn.ModuleList(self.med_block)
        self.squeeze_block = FCUDown(inplanes=dim, outplanes=embed_dim, dw_stride=dw_stride)
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=dim, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        self.fusion_block = ConvBlock(dim=dim, drop_rate=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.downsample = downsample

    def forward(self, x, x_t):
        if self.downsample:
            # stage 5,6,7,8:[16,96,80,80] -> [16,192,40,40]
            # stage 9,10,11:[16,192,40,40] -> [16,384,20,20]
            x = self.norm(x)
            x = self.conv(x)
        x, x2 = self.cnn_block(x)
        _, _, H, W = x2.shape
        # The feature map on the conv branch is converted to the transformer branch
        x_st = self.squeeze_block(x2) # stage2,3,4: [16,401,384]
        # Multi-head attention after feature fusion
        x_t = self.trans_block(x_st + x_t) # stage2,3,4: [16,401,384]
        # num_med_block=0
        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)
        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t

'''HWMSA_convnext_conformer'''
class HWMSA_convnext_conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, num_med_block=0,
                 dims=None, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., head_init_scale=1.):
        # Transformer
        super().__init__()
        if dims is None:
            dims = [96, 192, 384, 768]
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Classifier head
        self.trans_norm = nn.LayerNorm(embed_dim)  # (384, )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # (384, class)
        self.conv_norm = nn.LayerNorm(dims[2], eps=1e-6)  # final norm layer
        self.conv_cls_head = nn.Linear(dims[2], num_classes) # (384,class)

        self.conv1 = nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(dims[0])
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 1 stage
        stage1_channel =dims[0]  # 96
        trans_dw_stride = patch_size // 4  # 16 / 4
        self.conv_1 = ConvBlock(dim=stage1_channel, drop_rate=self.trans_dpr[0])
        self.trans_patch_conv = nn.Conv2d(stage1_channel, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim,  mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                             )

        init_stage = 2
        fin_stage = depth // 3 + 1  # 12/3=4,5
        # stage 2,3,4, trans_dw_stride=4, stage1_channel=96
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                dw_stride=trans_dw_stride, dim=stage1_channel, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                num_med_block=num_med_block
                            )
                            )

        stage2_channel = dims[1]
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 5+4=9
        for i in range(init_stage, fin_stage):
            downsample = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                dw_stride=trans_dw_stride // 2, dim=stage2_channel, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                downsample = downsample, num_med_block=num_med_block
                            )
                            )

        stage3_channel = dims[2]
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 9+4=13
        for i in range(init_stage, fin_stage):
            downsample = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                dw_stride=trans_dw_stride // 4, dim=stage3_channel, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                drop_path_rate=self.trans_dpr[i - 1],
                                downsample = downsample, num_med_block=num_med_block
                            )
                            )

        self.fin_stage = fin_stage

        self.apply(self._init_weights)
        self.conv_cls_head.weight.data.mul_(head_init_scale)
        self.conv_cls_head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x)))) # [16,96,80,80]

        x = self.conv_1(x_base, return_x_2=False) # [16,96,80,80]
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)# [16,400,384]
        x_t = self.trans_1(x_t)  # transformer encoder [16,401,384]

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)

        # conv classification [N,1024,7,7] -> [N,1,1,1024] -> [N,1024] -> [N,num_classes]
        x_p = self.conv_norm(x.mean([-2, -1]))
        conv_cls = self.conv_cls_head(x_p)

        # trans classification [16,400,384] -> layer norm -> [16,384] -> [N,num_classes]
        x_t = self.trans_norm(x_t)
        x_t = self.avgpool(x_t.transpose(1, 2))
        x_t = torch.flatten(x_t, 1)
        tran_cls = self.trans_cls_head(x_t)

        return [conv_cls, tran_cls]

    def forward_features(self, x):

        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = self.trans_1(x_t)

        for i in range(2, self.fin_stage):
            x, x_t = getattr(self, f'conv_trans_{i}')(x, x_t)

        feat_c = self.conv_norm(x.mean([-2, -1]))  # (B, C)

        x_t = self.trans_norm(x_t)  # (B, 400, C)
        x_t = self.avgpool(x_t.transpose(1, 2))  # → (B, C, 1)
        feat_s = torch.flatten(x_t, 1)  # → (B, C)

        return feat_c, feat_s