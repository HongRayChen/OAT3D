"""Sparse boundary-condition field VAE (TorchSparse + FlashAttention), 513^3 fields -> 65^3 latents."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import flash_attn
import torchsparse
import torchsparse.nn as spnn
from torchsparse import SparseTensor


VERSION = "v1_5"


def bxyz_to_xyzb(coords_bxyz: torch.Tensor) -> torch.Tensor:
    return coords_bxyz.contiguous()


def xyzb_to_bxyz(coords_bxyz: torch.Tensor) -> torch.Tensor:
    return coords_bxyz.contiguous()


def make_spatial_range(batch_size: int, spatial_size: int) -> Tuple[int, int, int, int]:
    return (int(batch_size), int(spatial_size), int(spatial_size), int(spatial_size))


def _stride_tuple(stride) -> Tuple[int, int, int]:
    if isinstance(stride, torch.Tensor):
        stride = stride.detach().cpu().tolist()
    if isinstance(stride, (tuple, list)):
        if len(stride) == 0:
            return (1, 1, 1)
        if len(stride) == 1:
            s0 = int(stride[0])
            return (s0, s0, s0)
        return (int(stride[0]), int(stride[1]), int(stride[2]))
    s0 = int(stride)
    return (s0, s0, s0)


def _stride_scalar(stride) -> int:
    return int(_stride_tuple(stride)[0])


def _normalize_spatial_range(spatial_range):
    if spatial_range is None:
        return None
    if isinstance(spatial_range, torch.Tensor):
        spatial_range = spatial_range.detach().cpu().tolist()
    return tuple(int(v) for v in spatial_range)


def make_sparse_tensor(
    feats: torch.Tensor,
    coords_bxyz: torch.Tensor,
    stride=1,
    spatial_range: Optional[Tuple[int, int, int, int]] = None,
) -> SparseTensor:
    coords_bxyz = coords_bxyz.contiguous().int()
    feats = feats.contiguous()

    kwargs = {}
    spatial_range = _normalize_spatial_range(spatial_range)
    if spatial_range is not None:
        kwargs["spatial_range"] = spatial_range

    return SparseTensor(feats=feats, coords=coords_bxyz, stride=_stride_tuple(stride), **kwargs)


def sparse_feats(x: SparseTensor) -> torch.Tensor:
    return x.F


def sparse_coords(x: SparseTensor) -> torch.Tensor:
    return x.C


def sparse_stride_tuple(x: SparseTensor) -> Tuple[int, int, int]:
    if hasattr(x, "s"):
        return _stride_tuple(x.s)
    return _stride_tuple(getattr(x, "stride", 1))


def sparse_stride(x: SparseTensor) -> int:
    return _stride_scalar(sparse_stride_tuple(x))


def sparse_spatial_range(x: SparseTensor):
    return _normalize_spatial_range(getattr(x, "spatial_range", None))


def sparse_batch_size(x: SparseTensor) -> int:
    sr = sparse_spatial_range(x)
    if sr is not None and len(sr) > 0:
        return int(sr[0])
    coords = sparse_coords(x)
    if coords.numel() == 0:
        return 0
    return int(coords[:, 0].max().detach().item()) + 1


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _downsampled_spatial_range(x: SparseTensor, factor: int):
    sr = sparse_spatial_range(x)
    if sr is None:
        return None
    return (int(sr[0]), _ceil_div(sr[1], factor), _ceil_div(sr[2], factor), _ceil_div(sr[3], factor))


def _target_spatial_range(x: SparseTensor, target_spatial_size: Optional[int]):
    if target_spatial_size is None:
        sr = sparse_spatial_range(x)
        if sr is None:
            return None
        return (int(sr[0]), int(sr[1]) * 2, int(sr[2]) * 2, int(sr[3]) * 2)
    return make_spatial_range(sparse_batch_size(x), int(target_spatial_size))


def sparse_replace(x: SparseTensor, feats: torch.Tensor, coords: Optional[torch.Tensor] = None) -> SparseTensor:
    coords_out = sparse_coords(x) if coords is None else coords
    y = make_sparse_tensor(feats, coords_out, sparse_stride_tuple(x), sparse_spatial_range(x))
    if coords is None and hasattr(x, "_caches"):
        y._caches = x._caches
    return y


def coord_key(coords_bxyz: torch.Tensor, base: int = 1024) -> torch.Tensor:
    c = coords_bxyz.long()
    b, x, y, z = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    return (((b * base + x) * base + y) * base + z)


def sparse_add(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    if sparse_coords(a).shape == sparse_coords(b).shape and torch.equal(sparse_coords(a), sparse_coords(b)):
        return sparse_replace(a, sparse_feats(a) + sparse_feats(b))

    coords = torch.cat([sparse_coords(a), sparse_coords(b)], dim=0)
    feats = torch.cat([sparse_feats(a), sparse_feats(b)], dim=0)
    key = coord_key(coords)
    key_sorted, order = torch.sort(key)
    coords_sorted = coords[order]
    feats_sorted = feats[order]
    unique_key, inverse = torch.unique_consecutive(key_sorted, return_inverse=True)
    out_feats = torch.zeros((unique_key.shape[0], feats.shape[1]), device=feats.device, dtype=feats.dtype)
    out_feats.index_add_(0, inverse, feats_sorted)
    is_start = torch.ones_like(key_sorted, dtype=torch.bool)
    is_start[1:] = key_sorted[1:] != key_sorted[:-1]
    out_coords = coords_sorted[is_start]
    return make_sparse_tensor(out_feats, out_coords, sparse_stride_tuple(a), sparse_spatial_range(a))


class SparseLinear(nn.Linear):
    def forward(self, input: SparseTensor) -> SparseTensor:
        return sparse_replace(input, super().forward(sparse_feats(input)))


class SparseGroupNorm(nn.GroupNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        feats = sparse_feats(input)
        coords = sparse_coords(input)
        dtype = feats.dtype
        nfeats = torch.zeros_like(feats)
        batch_ids = torch.unique(coords[:, 0], sorted=True)
        for batch_id in batch_ids:
            mask = coords[:, 0] == batch_id
            bfeats = feats[mask].permute(1, 0).reshape(1, feats.shape[1], -1)
            bfeats = F.group_norm(bfeats.float(), self.num_groups, self.weight.float(), self.bias.float(), self.eps)
            nfeats[mask] = bfeats.reshape(feats.shape[1], -1).permute(1, 0).to(dtype)
        return sparse_replace(input, nfeats)


def _sparse_type(x: SparseTensor, dtype: torch.dtype) -> SparseTensor:
    return sparse_replace(x, sparse_feats(x).to(dtype))


class SparseGroupNorm32(SparseGroupNorm):
    def forward(self, x: SparseTensor) -> SparseTensor:
        dtype = sparse_feats(x).dtype
        y = super().forward(_sparse_type(x, torch.float32))
        return _sparse_type(y, dtype)


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class SparseSiLU(nn.Module):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_replace(x, F.silu(sparse_feats(x)))


class SparseGELU(nn.Module):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_replace(x, F.gelu(sparse_feats(x), approximate="tanh"))


class SparseSigmoid(nn.Module):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return sparse_replace(x, torch.sigmoid(sparse_feats(x)))


class SparseConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.conv = spnn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.conv(x)


class SparseSequential(nn.Sequential):
    pass


class SparseDownsample(nn.Module):
    """Downsample by floor-dividing spatial coords [x,y,z] and mean-pooling features.

    v1_5 uses TorchSparse-native [batch, x, y, z] coordinates; only columns 1:4
    are spatial coordinates.
    """

    def __init__(self, factor: int = 2, mode: str = "mean"):
        super().__init__()
        self.factor = int(factor)
        self.mode = mode

    def forward(self, input: SparseTensor) -> SparseTensor:
        coords = sparse_coords(input)
        feats = sparse_feats(input)
        new_coords = coords.clone()
        new_coords[:, 1:] = torch.div(new_coords[:, 1:], self.factor, rounding_mode="floor")

        key = coord_key(new_coords)
        key_sorted, order = torch.sort(key)
        coords_sorted = new_coords[order]
        feats_sorted = feats[order]
        unique_key, inverse = torch.unique_consecutive(key_sorted, return_inverse=True)

        out_feats = torch.zeros((unique_key.shape[0], feats.shape[1]), device=feats.device, dtype=torch.float64)
        out_feats.index_add_(0, inverse, feats_sorted.double())
        if self.mode == "mean":
            counts = torch.bincount(inverse, minlength=unique_key.shape[0]).to(out_feats.dtype).unsqueeze(1).clamp_min_(1.0)
            out_feats = out_feats / counts
        else:
            raise ValueError(f"Unsupported SparseDownsample mode: {self.mode}")
        out_feats = out_feats.to(feats.dtype)

        is_start = torch.ones_like(key_sorted, dtype=torch.bool)
        is_start[1:] = key_sorted[1:] != key_sorted[:-1]
        out_coords = coords_sorted[is_start]
        return make_sparse_tensor(out_feats, out_coords, sparse_stride(input) * self.factor, _downsampled_spatial_range(input, self.factor))


class SparseDownBlock3d(nn.Module):
    def __init__(self, channels: int, out_channels: Optional[int] = None, num_groups: int = 32, use_checkpoint: bool = False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.act_layers = SparseSequential(SparseGroupNorm32(min(num_groups, channels), channels), SparseSiLU())
        self.down = SparseDownsample(2)
        self.out_layers = SparseSequential(
            SparseConv3d(channels, self.out_channels, 3, padding=1),
            SparseGroupNorm32(min(num_groups, self.out_channels), self.out_channels),
            SparseSiLU(),
            SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip_connection = nn.Identity() if self.out_channels == channels else SparseConv3d(channels, self.out_channels, 1)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x: SparseTensor) -> SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        x = self.down(x)
        h = self.out_layers(h)
        h = sparse_add(h, self.skip_connection(x))
        return h

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self._forward(x)


class AbsolutePositionEmbedder(nn.Module):
    """Direct3D-S2 sinusoidal absolute position embedder."""

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        freqs = 1.0 / (10000 ** freqs)
        self.register_buffer("freqs", freqs, persistent=False)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.outer(x, self.freqs.to(x.device))
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor, factor: float = None) -> torch.Tensor:
        if factor is not None:
            x = x * factor
        x = x.float()
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim=-1) * self.gamma * self.scale).to(x.dtype)


def calc_window_partition_bxyz(coords_bxyz: torch.Tensor, window_size: int, shift_window: int | Tuple[int, int, int] = 0):
    """Partition [batch, x, y, z] sparse coords into local xyz windows."""
    shift_window = (shift_window,) * 3 if isinstance(shift_window, int) else shift_window
    shifted_coords = coords_bxyz.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=coords_bxyz.device, dtype=coords_bxyz.dtype).unsqueeze(0)
    max_coords = shifted_coords[:, 1:].max(dim=0).values.tolist()
    num_windows = [math.ceil((mc + 1) / window_size) for mc in max_coords]
    offset = torch.cumprod(torch.tensor([1] + num_windows[::-1]), dim=0).tolist()[::-1]
    shifted_coords[:, 1:] //= torch.tensor(window_size, device=coords_bxyz.device, dtype=coords_bxyz.dtype).unsqueeze(0)
    shifted_indices = (shifted_coords.long() * torch.tensor(offset, device=coords_bxyz.device, dtype=torch.long).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=coords_bxyz.device)
    seq_lens = torch.bincount(shifted_indices)
    mask = seq_lens != 0
    seq_lens = seq_lens[mask]
    return fwd_indices, bwd_indices, seq_lens


def _flash_attn_dtype(feats: torch.Tensor) -> torch.dtype:
    if feats.is_cuda:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return feats.dtype


class SparseWindowFlashSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, window_size: int = 8, shift_window: int | Tuple[int, int, int] = 0, qk_rms_norm: bool = False):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = int(window_size)
        self.shift_window = shift_window
        self.to_qkv = nn.Linear(channels, channels * 3, bias=True)
        self.to_out = nn.Linear(channels, channels, bias=True)
        self.qk_rms_norm = qk_rms_norm
        if qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(self, x: SparseTensor) -> SparseTensor:
        feats = sparse_feats(x)
        if feats.shape[0] == 0:
            return x

        model_dtype = feats.dtype
        qkv = self.to_qkv(feats.float()).reshape(feats.shape[0], 3, self.num_heads, self.head_dim)

        if self.qk_rms_norm:
            q, k, v = qkv.unbind(dim=1)
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
            qkv = torch.stack([q, k, v], dim=1)

        fwd_indices, bwd_indices, seq_lens = calc_window_partition_bxyz(sparse_coords(x), self.window_size, self.shift_window)
        qkv_feats = qkv[fwd_indices].contiguous().to(_flash_attn_dtype(feats))
        cu_seqlens = torch.cat([
            torch.zeros(1, device=feats.device, dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0).to(torch.int32),
        ], dim=0)
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, cu_seqlens, int(seq_lens.max().item()))
        out = out.to(model_dtype)[bwd_indices].reshape(feats.shape[0], self.channels)
        out = self.to_out(out.float()).to(model_dtype)
        return sparse_replace(x, out)


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = SparseSequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(),
            SparseLinear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)


class SparseTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0, window_size: int = 8, shift_window: int | Tuple[int, int, int] = 0, use_checkpoint: bool = False, qk_rms_norm: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseWindowFlashSelfAttention(channels, num_heads=num_heads, window_size=window_size, shift_window=shift_window, qk_rms_norm=qk_rms_norm)
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def _forward(self, x: SparseTensor) -> SparseTensor:
        h = sparse_replace(x, self.norm1(sparse_feats(x)))
        h = self.attn(h)
        x = sparse_add(x, h)
        h = sparse_replace(x, self.norm2(sparse_feats(x)))
        h = self.mlp(h)
        x = sparse_add(x, h)
        return x

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self._forward(x)


class SparseTransformerBase(nn.Module):
    def __init__(self, in_channels: int, model_channels: int = 512, num_blocks: int = 4, num_heads: int = 8, mlp_ratio: float = 4.0, window_size: int = 8, use_checkpoint: bool = False, qk_rms_norm: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_blocks = num_blocks
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.qk_rms_norm = qk_rms_norm
        self.pos_embedder = AbsolutePositionEmbedder(model_channels)
        self.input_layer = SparseLinear(in_channels, model_channels)
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                model_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size,
                shift_window=window_size // 2 * (i % 2),
                use_checkpoint=use_checkpoint,
                qk_rms_norm=qk_rms_norm,
            )
            for i in range(num_blocks)
        ])
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        self.apply(_basic_init)

    def forward(self, x: SparseTensor, factor: float = None) -> SparseTensor:
        h = self.input_layer(x)
        pe = self.pos_embedder(sparse_coords(x)[:, 1:], factor)
        h = sparse_replace(h, sparse_feats(h) + pe.to(sparse_feats(h).dtype))
        for block in self.blocks:
            h = block(h)
        return h


class SparseBCSDFEncoder(nn.Module):
    def __init__(self, in_channels: int, model_channels: int = 512, latent_channels: int = 16, num_blocks: int = 4, num_heads: int = 8, window_size: int = 8, use_checkpoint: bool = False, qk_rms_norm: bool = False):
        super().__init__()
        self.input_layer1 = SparseLinear(in_channels, model_channels // 16)
        self.downsample = nn.ModuleList([
            SparseDownBlock3d(model_channels // 16, model_channels // 8, use_checkpoint=use_checkpoint),
            SparseDownBlock3d(model_channels // 8, model_channels // 4, use_checkpoint=use_checkpoint),
            SparseDownBlock3d(model_channels // 4, model_channels, use_checkpoint=use_checkpoint),
        ])
        self.transformer = SparseTransformerBase(
            in_channels=model_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=4.0,
            window_size=window_size,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.out_layer = SparseLinear(model_channels, 2 * latent_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.constant_(self.out_layer.weight, 0.0)
        nn.init.constant_(self.out_layer.bias, 0.0)

    def forward(self, x: SparseTensor, factor: float = None) -> SparseTensor:
        h = self.input_layer1(x)
        for block in self.downsample:
            h = block(h)
        h = self.transformer(h, factor)
        h = sparse_replace(h, F.layer_norm(sparse_feats(h), sparse_feats(h).shape[-1:]))
        h = self.out_layer(h)
        return h
