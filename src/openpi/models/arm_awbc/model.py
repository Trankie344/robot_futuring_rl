import flax.nnx as nnx

import openpi.models.pi0.config as pi0_config
import openpi.models.pi0.model as pi0


class ArmPi0(pi0.Pi0):
    """Pi0 architecture entry point for ARM weighted VLA training."""

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config, rngs)
