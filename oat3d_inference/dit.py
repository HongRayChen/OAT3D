"""Dense and sparse flow-matching density DiTs (36 AdaLN blocks, packed cross-attention over BC/BNG tokens)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

from direct3d_s2.modules import sparse as sp
from direct3d_s2.modules.norm import LayerNorm32
from direct3d_s2.modules.spatial import patchify, unpatchify
from direct3d_s2.modules.sparse.transformer.modulated import ModulatedSparseTransformerCrossBlock
from direct3d_s2.modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock


def geometry_width(bands):
    return 6 * (1 + 2 * int(bands))


class BF16LayerNorm32(nn.LayerNorm):
    def forward(self, x):
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps).to(x.dtype)


def replace_layer_norm32(module):
    for name, child in list(module.named_children()):
        if isinstance(child, LayerNorm32):
            norm = BF16LayerNorm32(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
            if child.elementwise_affine:
                norm.weight.data.copy_(child.weight.data)
                norm.bias.data.copy_(child.bias.data)
            setattr(module, name, norm)
        else:
            replace_layer_norm32(child)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, head_dim, num_heads):
        super().__init__()
        self.scale = head_dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x):
        return (F.normalize(x.float(), dim=-1) * self.gamma.float() * self.scale).to(x.dtype)


class PackedCrossAttention(nn.Module):
    def __init__(self, channels, context_channels, num_heads, qk_rms_norm=False):
        super().__init__()
        self.channels, self.num_heads, self.head_dim = channels, num_heads, channels // num_heads
        self.to_q = nn.Linear(channels, channels)
        self.to_kv = nn.Linear(context_channels, channels * 2)
        self.to_out = nn.Linear(channels, channels)
        self.qk_rms_norm = qk_rms_norm
        if qk_rms_norm:
            self.q_rms_norm, self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads), MultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(self, x, context):
        sparse = hasattr(x, "feats") and hasattr(x, "layout")
        if sparse:
            feats, lengths = x.feats, [item.stop - item.start for item in x.layout]
            batch_size = len(lengths)
            cu_q = torch.tensor([0] + list(np.cumsum(lengths)), device=x.device, dtype=torch.int32)
            max_q = max(lengths)
        else:
            batch_size, length, _ = x.shape
            feats = x.reshape(batch_size * length, self.channels)
            cu_q = torch.arange(0, (batch_size + 1) * length, length, device=x.device, dtype=torch.int32)
            max_q = length
        if int(context["batch_size"]) != batch_size:
            raise RuntimeError(f"Conditioning batch mismatch: {context['batch_size']} != {batch_size}")
        tokens = context["tokens"].to(device=feats.device, dtype=feats.dtype)
        q = self.to_q(feats).reshape(-1, self.num_heads, self.head_dim)
        k, v = self.to_kv(tokens).reshape(-1, 2, self.num_heads, self.head_dim).unbind(dim=1)
        if self.qk_rms_norm:
            q, k = self.q_rms_norm(q), self.k_rms_norm(k)
        h = flash_attn_varlen_func(q.contiguous(), k.contiguous(), v.contiguous(), cu_q, context["cu_seqlens"], max_q, int(context["max_seqlen"]), causal=False)
        h = self.to_out(h.reshape(-1, self.channels))
        return x.replace(h) if sparse else h.reshape(batch_size, length, self.channels)


class GeometryPackedCrossAttention(nn.Module):
    def __init__(self, channels, context_channels, num_heads, geometry_bands):
        super().__init__()
        width = geometry_width(geometry_bands)
        self.channels, self.num_heads, self.head_dim = channels, num_heads, channels // num_heads
        self.to_q, self.to_k, self.to_v, self.to_out = nn.Linear(channels, channels), nn.Linear(context_channels, channels), nn.Linear(context_channels, channels), nn.Linear(channels, channels)
        self.q_pos, self.k_pos = nn.Linear(width, channels, bias=False), nn.Linear(width, channels, bias=False)
        self.q_content_norm, self.q_position_norm = BF16LayerNorm32(channels), BF16LayerNorm32(channels)
        self.q_mix = nn.Sequential(nn.Linear(2 * channels, channels), nn.GELU(), nn.Linear(channels, channels))
        self.q_rms_norm, self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads), MultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(self, x, context):
        sparse = hasattr(x, "feats") and hasattr(x, "layout")
        if sparse:
            feats, lengths = x.feats, [item.stop - item.start for item in x.layout]
            batch_size = len(lengths)
            cu_q = torch.tensor([0] + list(np.cumsum(lengths)), device=x.device, dtype=torch.int32)
            max_q = max(lengths)
            query_geometry = context["query_geometry"]
        else:
            batch_size, length, _ = x.shape
            feats = x.reshape(batch_size * length, self.channels)
            cu_q = torch.arange(0, (batch_size + 1) * length, length, device=x.device, dtype=torch.int32)
            max_q = length
            query_geometry = context["query_geometry"].expand(batch_size, -1, -1).reshape(batch_size * length, -1)
        tokens = context["tokens"].to(device=feats.device, dtype=feats.dtype)
        q = self.q_mix(torch.cat([self.q_content_norm(self.to_q(feats)), self.q_position_norm(self.q_pos(query_geometry))], dim=-1))
        k = self.to_k(tokens) + self.k_pos(context["geometry"])
        v = self.to_v(tokens)
        q = self.q_rms_norm(q.reshape(-1, self.num_heads, self.head_dim))
        k = self.k_rms_norm(k.reshape(-1, self.num_heads, self.head_dim))
        h = flash_attn_varlen_func(q.contiguous(), k.contiguous(), v.reshape(-1, self.num_heads, self.head_dim).contiguous(), cu_q, context["cu_seqlens"], max_q, int(context["max_seqlen"]), causal=False)
        h = self.to_out(h.reshape(-1, self.channels))
        return x.replace(h) if sparse else h.reshape(batch_size, length, self.channels)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        embedding = torch.cat([torch.cos(t.float()[:, None] * frequencies[None]), torch.sin(t.float()[:, None] * frequencies[None])], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


def cube_geometry(centers, extents, bands):
    values = torch.cat([centers.float(), extents.float()], dim=-1)
    frequencies = (2.0 ** torch.arange(bands, device=values.device, dtype=torch.float32)) * math.pi
    phase = values[..., None] * frequencies
    return torch.cat([values, phase.sin().flatten(-2), phase.cos().flatten(-2)], dim=-1).to(torch.bfloat16)


class DenseDensityDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels, cond_channels, heads = int(cfg["model_channels"]), int(cfg["cond_channels"]), int(cfg["num_heads_dense"])
        resolution, in_channels, blocks = int(cfg["dense_resolution"]), int(cfg["dense_channels"]), int(cfg["num_blocks"])
        self.resolution, self.in_channels, self.geometry_cross_attention = resolution, in_channels, bool(cfg["geometry_cross_attention"])
        self.t_embedder = TimestepEmbedder(channels)
        self.problem_embedder = nn.Sequential(nn.Linear(4 * cond_channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        position_embedder = AbsolutePositionEmbedder(channels, 3)
        coords = torch.stack(torch.meshgrid(*[torch.arange(resolution) for _ in range(3)], indexing="ij"), dim=-1).reshape(-1, 3)
        self.register_buffer("pos_emb", position_embedder(coords), persistent=False)
        centers = (coords.float() + 0.5) / resolution
        self.register_buffer("query_geometry", cube_geometry(centers, torch.full_like(centers, 1.0 / resolution), int(cfg["geometry_bands"])), persistent=False)
        self.input_layer = nn.Linear(in_channels, channels)
        self.blocks = nn.ModuleList([ModulatedTransformerCrossBlock(channels, cond_channels, num_heads=heads, mlp_ratio=float(cfg["mlp_ratio"]), attn_mode="full", use_checkpoint=False, use_rope=False, qk_rms_norm=bool(cfg["qk_rms_norm_self"]), qk_rms_norm_cross=bool(cfg["qk_rms_norm_cross_dense"]), share_mod=False) for _ in range(blocks)])
        for block in self.blocks:
            block.cross_attn = GeometryPackedCrossAttention(channels, cond_channels, heads, int(cfg["geometry_bands"])) if self.geometry_cross_attention else PackedCrossAttention(channels, cond_channels, heads, bool(cfg["qk_rms_norm_cross_dense"]))
        self.out_layer = nn.Linear(channels, in_channels)
        replace_layer_norm32(self)
        self.bfloat16()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, t, cond):
        batch_size = x.shape[0]
        h = patchify(x.to(device=self.device, dtype=torch.bfloat16), 1).view(batch_size, self.in_channels, -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h) + self.pos_emb[None].to(device=self.device, dtype=torch.bfloat16)
        problem = self.problem_embedder(cond["global_tokens"].reshape(batch_size, -1))
        modulation = self.t_embedder(t.to(self.device)) + problem
        if self.geometry_cross_attention:
            cond = {**cond, "query_geometry": self.query_geometry}
        for block in self.blocks:
            h = block(h, modulation, cond)
        h = self.out_layer(F.layer_norm(h.float(), h.shape[-1:]).to(torch.bfloat16))
        return unpatchify(h.permute(0, 2, 1).contiguous().view(batch_size, self.in_channels, self.resolution, self.resolution, self.resolution), 1).contiguous().to(torch.bfloat16)


class SparseDensityDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels, cond_channels, heads = int(cfg["model_channels"]), int(cfg["cond_channels"]), int(cfg["num_heads_sparse"])
        in_channels, blocks, resolution = int(cfg["sparse_channels"]), int(cfg["num_blocks"]), int(cfg["sparse_resolution"])
        self.geometry_cross_attention, self.resolution, self.geometry_bands = bool(cfg["geometry_cross_attention"]), resolution, int(cfg["geometry_bands"])
        self.t_embedder = TimestepEmbedder(channels)
        self.problem_embedder = nn.Sequential(nn.Linear(4 * cond_channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.pos_embedder = AbsolutePositionEmbedder(channels)
        self.input_layer = sp.SparseLinear(in_channels, channels)
        self.blocks = nn.ModuleList([ModulatedSparseTransformerCrossBlock(channels, cond_channels, num_heads=heads, num_kv_heads=int(cfg["num_kv_heads_sparse"]), mlp_ratio=float(cfg["mlp_ratio"]), attn_mode="full", compression_block_size=int(cfg["sparse_compression_block_size"]), selection_block_size=int(cfg["sparse_selection_block_size"]), topk=int(cfg["sparse_topk"]), compression_version=cfg["sparse_compression_version"], use_checkpoint=False, use_rope=False, qk_rms_norm=bool(cfg["qk_rms_norm_self"]), qk_rms_norm_cross=bool(cfg["qk_rms_norm_cross_sparse"]), share_mod=False, use_ssa=True, resolution=resolution, window_size=int(cfg["sparse_window_size"]), shift_window=int(cfg["sparse_window_size"]) // 2 * (index % 2)) for index in range(blocks)])
        for block in self.blocks:
            block.cross_attn = GeometryPackedCrossAttention(channels, cond_channels, heads, self.geometry_bands) if self.geometry_cross_attention else PackedCrossAttention(channels, cond_channels, heads, bool(cfg["qk_rms_norm_cross_sparse"]))
        self.out_layer = sp.SparseLinear(channels, in_channels)
        replace_layer_norm32(self)
        self.bfloat16()

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, t, cond):
        x = x.type(torch.bfloat16)
        h = self.input_layer(x).type(torch.bfloat16)
        h = (h + self.pos_embedder(h.coords[:, 1:], factor=1.0).to(device=h.device, dtype=torch.bfloat16)).type(torch.bfloat16)
        problem = self.problem_embedder(cond["global_tokens"].reshape(x.shape[0], -1))
        modulation = self.t_embedder(t.to(h.device)) + problem
        if self.geometry_cross_attention:
            centers = (h.coords[:, 1:].float() + 0.5) / self.resolution
            cond = {**cond, "query_geometry": cube_geometry(centers, torch.full_like(centers, 1.0 / self.resolution), self.geometry_bands)}
        for block in self.blocks:
            h = block(h, modulation, cond).type(torch.bfloat16)
        return self.out_layer(h.replace(F.layer_norm(h.feats.float(), h.feats.shape[-1:]).to(torch.bfloat16))).type(torch.bfloat16)
