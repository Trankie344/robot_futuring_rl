"""Independent configuration registry for both RL Token training stages."""

from __future__ import annotations

import dataclasses
import difflib
from pathlib import Path
from typing import TypeAlias

import flax.nnx as nnx
from typing_extensions import override
import tyro

from openpi import transforms as _transforms
from openpi.models import tokenizer as _tokenizer
from openpi.models.rl_token import actor_critic as _actor_critic
from openpi.models.rl_token.config import RLTokenPi0Config
from openpi.shared import nnx_utils
from openpi.training import config as _openpi_config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from openpi.training.rl_token.stage2 import trainer as _stage2_trainer


@dataclasses.dataclass(frozen=True)
class RLTokenModelTransformFactory(_openpi_config.GroupFactory):
    """PI0.5 transforms without coupling to the original Pi0Config class."""

    default_prompt: str | None = None

    @override
    def __call__(self, model_config: RLTokenPi0Config) -> _transforms.Group:
        if not isinstance(model_config, RLTokenPi0Config) or not model_config.pi05:
            raise ValueError("RL Token Stage 1 requires an RLTokenPi0Config in PI0.5 mode")
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ]
        )


@dataclasses.dataclass(frozen=True)
class RLTokenTrainConfig(_openpi_config.TrainConfig):
    """Stage 1 config extensions for partial EMA and deployment export."""

    model: RLTokenPi0Config = dataclasses.field(default_factory=RLTokenPi0Config)
    ema_filter: _openpi_config.Filter = dataclasses.field(default_factory=nnx.Everything)
    deployment_base_params_path: str | None = None

    @property
    def ema_params_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, self.ema_filter)

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.model.rl_token_only:
            if self.ema_decay is None:
                raise ValueError("RL Token Stage 1 requires partial EMA")
            if self.deployment_base_params_path is None:
                raise ValueError("RL Token Stage 1 requires deployment_base_params_path")


@dataclasses.dataclass(frozen=True)
class RLTokenStage2Config:
    """Named Stage 2 runtime template selected before explicit round inputs."""

    name: str
    runtime: _stage2_trainer.Stage2TrainerConfig = dataclasses.field(
        default_factory=_stage2_trainer.Stage2TrainerConfig
    )
    training_root: str = "/mnt/workspace/ys/futuring/openpi_runtime/rl_token_stage2"
    default_stage1_checkpoint: str = "./checkpoints/rl_token/pi05_lite0030_rltoken_only/54999"


Config: TypeAlias = RLTokenTrainConfig | RLTokenStage2Config
TrainConfig = RLTokenTrainConfig
DataConfig = _openpi_config.DataConfig
AssetsConfig = _openpi_config.AssetsConfig
SimpleDataConfig = _openpi_config.SimpleDataConfig
Lite0030Inputs = _openpi_config.Lite0030Inputs
Filter = _openpi_config.Filter

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_BASE_STEP = _REPOSITORY_ROOT / "checkpoints/rl_token/pi05_lite0030_base/29999"
_STAGE1_CHECKPOINT = _REPOSITORY_ROOT / "checkpoints/rl_token/pi05_lite0030_rltoken_only/54999"
_STAGE1_DATASET = _REPOSITORY_ROOT / "data/rl_token/lite0030_stage1"
_ASSET_ID = "lite0030_joints_fps20_openpi_drop_last4s_min20s"

_STAGE1_MODEL = RLTokenPi0Config(
    pi05=True,
    action_horizon=50,
    rl_token_enabled=True,
    rl_token_only=True,
    rl_token_reconstruction_weight=1.0,
    rl_token_encoder_depth=2,
    rl_token_decoder_depth=2,
    rl_token_width=2048,
    rl_token_num_heads=16,
    rl_token_mlp_dim=8192,
    rl_token_max_prefix_len=968,
    rl_token_dropout=0.1,
    rl_token_compute_dtype="bfloat16",
)


def _stage1_data(repo_id: str) -> SimpleDataConfig:
    return SimpleDataConfig(
        repo_id=repo_id,
        assets=AssetsConfig(
            assets_dir=str(_BASE_STEP / "assets"),
            asset_id=_ASSET_ID,
        ),
        base_config=DataConfig(
            prompt_from_task=True,
            video_tolerance_s=0.05,
            action_sequence_keys=("action",),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "top": "observation.images.top",
                                "left_wrist": "observation.images.left_wrist",
                                "right_wrist": "observation.images.right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
        ),
        data_transforms=lambda model: _transforms.Group(
            inputs=[
                _openpi_config.Lite0030Inputs(),
                _transforms.DeltaActions(_transforms.make_bool_mask(16)),
            ],
            outputs=[
                _transforms.AbsoluteActions(_transforms.make_bool_mask(16)),
                _openpi_config.Lite0030Outputs(action_dim=16),
            ],
        ),
        model_transforms=RLTokenModelTransformFactory(),
    )


_STAGE1_CONFIGS = (
    RLTokenTrainConfig(
        name="rl_token_stage1",
        model=_STAGE1_MODEL,
        data=_stage1_data(str(_STAGE1_DATASET)),
        project_name="openpi-rl-token",
        checkpoint_base_dir="./checkpoints/rl_token/stage1",
        batch_size=256,
        num_workers=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        ema_filter=nnx_utils.PathRegex(r"rl_token(?:/.*)?"),
        freeze_filter=_STAGE1_MODEL.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_BASE_STEP / "params")),
        deployment_base_params_path=str(_BASE_STEP / "params"),
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
        fsdp_devices=1,
    ),
    RLTokenTrainConfig(
        name="rl_token_stage1_debug",
        model=dataclasses.replace(
            _STAGE1_MODEL,
            rl_token_encoder_depth=1,
            rl_token_decoder_depth=1,
            rl_token_num_heads=4,
            rl_token_mlp_dim=2048,
            rl_token_dropout=0.0,
            rl_token_compute_dtype="float32",
        ),
        data=_stage1_data("fake"),
        exp_name="debug",
        project_name="openpi-rl-token",
        checkpoint_base_dir="./checkpoints/rl_token/stage1",
        batch_size=1,
        num_workers=0,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=0,
            peak_lr=5e-5,
            decay_steps=1,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        ema_filter=nnx_utils.PathRegex(r"rl_token(?:/.*)?"),
        freeze_filter=_STAGE1_MODEL.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(_BASE_STEP / "params")),
        deployment_base_params_path=str(_BASE_STEP / "params"),
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        keep_period=1,
        wandb_enabled=False,
        fsdp_devices=1,
    ),
)

_STAGE2_CONFIGS = (
    RLTokenStage2Config(
        name="rl_token_stage2",
        default_stage1_checkpoint=str(_STAGE1_CHECKPOINT),
    ),
    RLTokenStage2Config(
        name="rl_token_stage2_debug",
        runtime=_stage2_trainer.Stage2TrainerConfig(
            network=_actor_critic.RLTActorCriticConfig(
                actor_hidden_dims=(64, 64, 64),
                critic_hidden_dims=(64, 64, 64),
            ),
            batch_size=2,
            log_interval=1,
            temp_checkpoint_interval=1,
            temp_max_to_keep=1,
            replay_max_open_shards=2,
        ),
        default_stage1_checkpoint=str(_STAGE1_CHECKPOINT),
    ),
)

_CONFIGS = (*_STAGE1_CONFIGS, *_STAGE2_CONFIGS)
if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("RL Token config names must be unique")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_config(config_name: str) -> Config:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT, n=1, cutoff=0.0)
        suggestion = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{suggestion}")
    return _CONFIGS_DICT[config_name]


def get_stage1_config(config_name: str) -> RLTokenTrainConfig:
    config = get_config(config_name)
    if not isinstance(config, RLTokenTrainConfig):
        raise ValueError(f"Config '{config_name}' is not a Stage 1 config")
    return config


def get_stage2_config(config_name: str) -> RLTokenStage2Config:
    config = get_config(config_name)
    if not isinstance(config, RLTokenStage2Config):
        raise ValueError(f"Config '{config_name}' is not a Stage 2 config")
    return config


def stage1_cli() -> RLTokenTrainConfig:
    configs = {config.name: (config.name, config) for config in _STAGE1_CONFIGS}
    return tyro.extras.overridable_config_cli(configs)
