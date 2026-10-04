"""P59 control-useful functional geometry adaptation.

This namespace is intentionally independent of the frozen P50 implementation.
P50 is loaded read-only and receives the same 13 inputs it was trained on.
"""

from .contracts import P59Contract, PublicTransition
from .checkpoint import load_frozen_checkpoint
from .core import (
    CanonicalGeometryCorrection,
    FrozenP50Action,
    P59AdaptationCore,
    P59Runtime,
)

__all__ = [
    "CanonicalGeometryCorrection",
    "FrozenP50Action",
    "P59AdaptationCore",
    "P59Contract",
    "P59Runtime",
    "PublicTransition",
    "load_frozen_checkpoint",
]
