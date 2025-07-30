import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg
from pathlib import Path
from model.vision_transformer import VisionTransformer, _cfg
from model.conformer import Conformer
from model.convnext_conformer import ConvNeXt_conformer #no
from timm.models.registry import register_model
from model.HWMSA_convnext_conformer import HWMSA_convnext_conformer
from model.new_HWMSA_convnext_conformer import new_HWMSA_convnext_conformer #no
from model.IRMLP_convnext_conformer import IRMLP_convnext_conformer #no
from model.pvt import PyramidVisionTransformer
from model.ceit import Image2Tokens, CeIT
from model.vim import VisionMamba
from model.medmamba import VSSM
from model.swiftformer import SwiftFormer, SwiftFormer_depth, SwiftFormer_width
from model.mambavision import MambaVision, resolve_pretrained_cfg, update_args

@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=320,patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_med_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def Conformer_tiny_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch16(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch32(pretrained=False, **kwargs):
    model = Conformer(patch_size=32, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_base_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def ConvNeXt_conformer_small_patch16(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model = ConvNeXt_conformer(patch_size=16, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def HWMSA_conformer_small_patch16(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model = HWMSA_convnext_conformer(patch_size=16, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def newHWMSA_convnext_conformer(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model = new_HWMSA_convnext_conformer(dims=[96, 192, 384, 768], head_dim=32, grid_sizes=[8, 8, 4, 1],
                                         ds_ratios=[8, 4, 2, 1], depth=12, mlp_ratio=4, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def IRMLPconvnext_conformer(pretrained=False, **kwargs):
    model = IRMLP_convnext_conformer(patch_size=16, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def pvt_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None,  **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model

@register_model
def ceit_small_patch16_224(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None,  leff_local_size=3, leff_with_bn=True, **kwargs):
    """
    convolutional + pooling stem
    local enhanced feedforward
    attention over cls_tokens
    """
    i2t = Image2Tokens()
    model = CeIT(
        hybrid_backbone=i2t,
        patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_divide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def medmamba(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model = VSSM(depths=[2, 2, 4, 2], dims=[96,192,384,768], num_classes=2, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


@register_model
def SwiftFormer_S(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model = SwiftFormer(
        layers=SwiftFormer_depth['S'],
        embed_dims=SwiftFormer_width['S'],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model


@register_model
def mamba_vision_T(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,  cache_dir=None, **kwargs):
    model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")
    depths = kwargs.pop("depths", [1, 3, 8, 4])
    num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
    window_size = kwargs.pop("window_size", [8, 8, 14, 7])
    dim = kwargs.pop("dim", 80)
    in_dim = kwargs.pop("in_dim", 32)
    mlp_ratio = kwargs.pop("mlp_ratio", 4)
    resolution = kwargs.pop("resolution", 224)
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T').to_dict()
    update_args(pretrained_cfg, kwargs, kwargs_filter=None)
    model = MambaVision(depths=depths,
                        num_heads=num_heads,
                        window_size=window_size,
                        dim=dim,
                        in_dim=in_dim,
                        mlp_ratio=mlp_ratio,
                        resolution=resolution,
                        drop_path_rate=drop_path_rate,
                        **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg
    if pretrained:
        if not Path(model_path).is_file():
            url = model.default_cfg['url']
            torch.hub.download_url_to_file(url=url, dst=model_path)
        model._load_state_dict(model_path)
    return model

@register_model
def vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, embed_dim=768, d_state=16, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model