"""Load the two generators, density decoders, and boundary encoders."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from direct3d_s2.models.autoencoders.decoder import SparseSDFDecoder
from direct3d_s2.models.autoencoders.dense_vae import SparseStructureDecoder
from direct3d_s2.models.autoencoders.encoder import SparseSDFEncoder
from direct3d_s2.modules import sparse as sp
from direct3d_s2.modules.sparse.attention import modules as attention
from .bc_vae import SparseBCSDFEncoder
from .dit import DenseDensityDiT, SparseDensityDiT


_window_attention = attention.sparse_windowed_scaled_dot_product_self_attention


def _window_attention_fp32(qkv, window_size, shift_window=(0, 0, 0)):
    if qkv.feats.dtype != torch.float32:
        return _window_attention(qkv, window_size, shift_window)
    result = _window_attention(qkv.replace(qkv.feats.bfloat16()), window_size, shift_window)
    return result.replace(result.feats.float())


attention.sparse_windowed_scaled_dot_product_self_attention = _window_attention_fp32


def load_models(weights, device):
    root = Path(weights)
    config = json.loads((root / "architecture.json").read_text())
    stats = json.loads((root / "normalization.json").read_text())

    def load(model, name, dtype=torch.float32):
        model.load_state_dict(load_file(str(root / f"{name}.safetensors")), strict=True)
        return model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    sparse = dict(resolution=64, model_channels=512, num_blocks=4, num_heads=8, num_head_channels=64)
    decoder = SparseSDFDecoder(latent_channels=16, **sparse)
    decoder.out_active = sp.SparseSigmoid()
    return SimpleNamespace(
        dense_cfg=config["dense"], sparse_cfg=config["sparse"],
        dense_dit=load(DenseDensityDiT(config["dense"]), "dense_dit", torch.bfloat16),
        sparse_dit=load(SparseDensityDiT(config["sparse"]), "sparse_dit", torch.bfloat16),
        dense_decoder=load(SparseStructureDecoder(1, 8, 4, [512, 128, 64], num_res_blocks_middle=4), "dense_decoder"),
        sparse_decoder=load(decoder, "sparse_decoder"),
        fixed_vae=load(SparseBCSDFEncoder(in_channels=1), "support_encoder"),
        load_vae=load(SparseBCSDFEncoder(in_channels=1), "load_encoder"),
        bng_vae=load(SparseSDFEncoder(in_channels=512, latent_channels=12, **sparse), "bng_encoder"),
        stats={name: {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in group.items()} for name, group in stats.items()},
    )
