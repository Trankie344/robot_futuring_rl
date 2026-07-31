import dataclasses

from typing_extensions import override

import openpi.models.pi0.config as pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class ArmPi0Config(pi0_config.Pi0Config):
    """Pi0 config that instantiates the ARM VLA model class."""

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.arm_awbc.model import ArmPi0

        import flax.nnx as nnx

        return ArmPi0(self, rngs=nnx.Rngs(rng))
