from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from ...modules import sparse as sp
from .base import SparseTransformerBase


class SparseSubdivideBlock3d(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.act_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, padding=1),
            sp.SparseSiLU()
        )

        self.sub = sp.SparseSubdivide()

        self.out_layers = nn.Sequential(
            sp.SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
            sp.SparseSiLU(),
        )

    def _forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.sub(h)
        h = self.out_layers(h)
        return h

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


class SparseSDFDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
        out_channels: int = 1,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.out_channels = out_channels
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                out_channels=model_channels // 4,
                use_checkpoint=use_checkpoint,
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                out_channels=model_channels // 8,
                use_checkpoint=use_checkpoint,
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 8,
                out_channels=model_channels // 16,
                use_checkpoint=use_checkpoint,
            )
        ])

        self.out_layer = sp.SparseLinear(model_channels // 16, self.out_channels)
        self.out_active = sp.SparseTanh()

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)

    def forward(self, x: sp.SparseTensor, factor: float = None):
        h = super().forward(x, factor)
        for block in self.upsample:
            h = block(h)
        return self.out_active(self.out_layer(h.type(x.dtype)))
