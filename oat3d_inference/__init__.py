"""OAT3D inference-only package: boundary conditions -> dense support -> sparse density."""
import os
import sys
from pathlib import Path

for key, value in (("ATTN_BACKEND", "flash_attn"), ("SPARSE_BACKEND", "torchsparse"), ("SPARSE_ATTN_BACKEND", "flash_attn")):
    os.environ.setdefault(key, value)
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
