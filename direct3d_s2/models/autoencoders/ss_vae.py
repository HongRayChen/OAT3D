"""Encode a sparse native field at its posterior mode."""
import torch

from ...modules import sparse as sp
from .distributions import posterior_mode


def encode_field(encoder, xyz, values):
    batch = torch.zeros_like(xyz[:, :1])
    moments = encoder(sp.SparseTensor(values.reshape(-1, 1), torch.cat([batch, xyz], dim=1).int()))
    return moments.replace(posterior_mode(moments.feats))
