"""Fast Adapter prototype for PROVE toy meta-learning experiments.

This package is intentionally scoped to P0/P1 development. It implements the
z-only MAML-style inner update and a numerical 2D toy environment; it does not
load DINOv3, RGB observations, LIBERO, or robot services.
"""

from .core import (
    AdaptationLossConfig,
    AdapterObservation,
    BaseSkillPolicy,
    FastAdapter,
    FastAdapterSystem,
    FrozenVisualEncoder,
    InnerUpdateResult,
    InteractionEncoder,
    ScriptedNominalPolicy2D,
    DynamicsPredictor,
    adaptation_loss,
    expand_latent,
    inner_update_z,
)
from .baselines import HistoryConditionedAdapter
from .toy2d import (
    Toy2DConfig,
    Toy2DDomain,
    Toy2DObservation,
    Toy2DVectorEnv,
    sample_toy2d_domains,
)

__all__ = [
    "AdaptationLossConfig",
    "AdapterObservation",
    "BaseSkillPolicy",
    "FastAdapter",
    "FastAdapterSystem",
    "FrozenVisualEncoder",
    "InnerUpdateResult",
    "HistoryConditionedAdapter",
    "InteractionEncoder",
    "ScriptedNominalPolicy2D",
    "DynamicsPredictor",
    "Toy2DConfig",
    "Toy2DDomain",
    "Toy2DObservation",
    "Toy2DVectorEnv",
    "adaptation_loss",
    "expand_latent",
    "inner_update_z",
    "sample_toy2d_domains",
]
