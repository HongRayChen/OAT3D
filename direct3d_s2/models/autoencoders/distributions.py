"""Deterministic latent used by the conditioning encoders."""


def posterior_mode(moments):
    return moments.chunk(2, dim=1)[0]
