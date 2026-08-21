"""Optional sampled Deep CFR training support for canonical Stockpile Lite.

Importing :mod:`stockpile` does not import PyTorch.  Training symbols are
loaded lazily from this subpackage so rules, complexity, and play clients keep
the lightweight core dependency set.
"""

from .config import (
    DEFAULT_CURRICULUM,
    CurriculumConfig,
    DeepCFRConfig,
    NetworkConfig,
    parse_curriculum,
)


def __getattr__(name: str):
    if name in {"DeepCFRPolicy", "DeepCFRTrainer", "TrainingResult"}:
        from .policy import DeepCFRPolicy
        from .trainer import DeepCFRTrainer, TrainingResult

        return {
            "DeepCFRPolicy": DeepCFRPolicy,
            "DeepCFRTrainer": DeepCFRTrainer,
            "TrainingResult": TrainingResult,
        }[name]
    raise AttributeError(name)


__all__ = [
    "DEFAULT_CURRICULUM",
    "CurriculumConfig",
    "DeepCFRConfig",
    "DeepCFRPolicy",
    "DeepCFRTrainer",
    "NetworkConfig",
    "TrainingResult",
    "parse_curriculum",
]
