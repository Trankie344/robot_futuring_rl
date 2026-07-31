"""Train the standalone RL Token Stage 1 autoencoder."""

from openpi.training.rl_token import config as rl_token_config
from openpi.training.rl_token.stage1 import trainer


def main() -> None:
    trainer.main(rl_token_config.stage1_cli())


if __name__ == "__main__":
    main()
