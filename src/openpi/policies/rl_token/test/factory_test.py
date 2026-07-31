from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models.rl_token import actor_critic as rlt_actor_critic
from openpi.policies.rl_token import factory as policy_config
from openpi.policies.rl_token import actor_policy as rlt_actor_policy
from openpi.training.rl_token import config as _config
from openpi.training.rl_token.stage2 import checkpoints as rlt_stage2_checkpoints
from openpi.training.rl_token.stage2 import td3 as rlt_td3
from openpi.training.rl_token.stage2 import train_state as rlt_td3_state

pytestmark = pytest.mark.filterwarnings(
    r"ignore:shape requires ndarray or scalar arguments.*:DeprecationWarning:flax\.core\.scope"
)

_FEATURE_IDENTITY = "f" * 64
_FROZEN_PARAMS_SHA256 = "1" * 64
_NORM_STATS_SHA256 = "2" * 64
_SAMPLER_NUM_STEPS = 10


@dataclasses.dataclass(frozen=True)
class _Marker:
    name: str

    def __call__(self, data):
        return data


class _PromptEqualityImpostor(str):
    def __ne__(self, other):
        return False


@dataclasses.dataclass
class _CapturedPolicy:
    model: object
    kwargs: dict[str, object]

    @property
    def metadata(self):
        return self.kwargs["metadata"]


def _network_config() -> rlt_actor_critic.RLTActorCriticConfig:
    return rlt_actor_critic.RLTActorCriticConfig(
        z_dim=2048,
        state_dim=16,
        action_horizon=20,
        action_dim=16,
        actor_state_proj_dim=4,
        actor_reference_proj_dim=5,
        critic_state_proj_dim=4,
        critic_action_proj_dim=5,
        actor_hidden_dims=(8, 7, 6),
        critic_hidden_dims=(8, 7, 6),
        compute_dtype="bfloat16",
    )


def _save_actor_step(root: Path, *, complete: bool) -> Path:
    checkpoint_root = root / "checkpoints"
    network = _network_config()
    algorithm = rlt_td3.TD3Config()
    state, _, _ = rlt_td3_state.initialize_train_state(
        rlt_actor_critic.RLTActor(network),
        rlt_actor_critic.RLTCritic(network),
        algorithm,
        jax.random.key(13),
    )
    step = 1 if complete else 0
    state = state.replace(
        critic_step=jnp.asarray(step, dtype=jnp.int32),
        round_critic_step=jnp.asarray(step, dtype=jnp.int32),
    )
    metadata = rlt_stage2_checkpoints.RLTCheckpointMetadata(
        schema_version=3,
        stage1_config=rlt_stage2_checkpoints.RLT_STAGE1_CONFIG_NAME,
        stage2_config=rlt_stage2_checkpoints.RLT_STAGE2_CONFIG_NAME,
        asset_id=rlt_stage2_checkpoints.RLT_ASSET_ID,
        base_checkpoint_step=rlt_stage2_checkpoints.RLT_BASE_CHECKPOINT_STEP,
        reward_source="tristate",
        reward_label_values=(-1, 0, 1, 2),
        completion_label=2,
        reward_aggregation="sum_20_frames",
        reward_schema_version=1,
        feature_identity=_FEATURE_IDENTITY,
        frozen_params_sha256=_FROZEN_PARAMS_SHA256,
        norm_stats_sha256=_NORM_STATS_SHA256,
        sampler_num_steps=_SAMPLER_NUM_STEPS,
        round_id="round_000001",
        admission_sha256="a" * 64,
        replay_snapshot_sha256="b" * 64,
        network_config=network,
        algorithm_config=algorithm,
        batch_size=256,
        round_start_step=0,
        round_critic_updates=1,
        critic_step=step,
        round_critic_step=step,
        replay_rng_state=np.random.Generator(np.random.PCG64(123)).bit_generator.state,
        jax_rng_impl=jax.random.key_impl(state.rng),
        round_complete=complete,
    )
    manager, resuming = rlt_stage2_checkpoints.initialize_checkpoint_dir(
        checkpoint_root,
        keep_period=None,
        overwrite=False,
        resume=False,
        max_to_keep=3,
    )
    assert not resuming
    rlt_stage2_checkpoints.save_rlt_checkpoint(manager, state=state, metadata=metadata)
    manager.close()
    return checkpoint_root / str(step)


@pytest.fixture(scope="module")
def actor_step(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _save_actor_step(tmp_path_factory.mktemp("actor-complete"), complete=True)


@pytest.fixture(scope="module")
def incomplete_actor_step(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _save_actor_step(tmp_path_factory.mktemp("actor-incomplete"), complete=False)


@pytest.fixture
def base_checkpoint(tmp_path: Path) -> Path:
    base = tmp_path / "54999"
    (base / "assets" / rlt_stage2_checkpoints.RLT_ASSET_ID).mkdir(parents=True)
    (base / "params").mkdir()
    return base


def _train_config() -> _config.TrainConfig:
    return _config.get_stage1_config(rlt_stage2_checkpoints.RLT_STAGE1_CONFIG_NAME)


def _patch_success_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    actor_step: Path,
    base_checkpoint: Path,
):
    records: dict[str, object] = {
        "events": [],
        "hash_calls": [],
        "restore_calls": [],
        "data_create_calls": [],
        "norm_calls": [],
        "model_load_calls": [],
        "policy_calls": [],
    }
    original_restore = policy_config._model.restore_params  # noqa: SLF001
    base_params = {"base": jnp.ones((2,), dtype=jnp.bfloat16)}
    fake_model = object()
    stats = transforms.NormStats(
        mean=np.zeros((32,), dtype=np.float32),
        std=np.ones((32,), dtype=np.float32),
        q01=-np.ones((32,), dtype=np.float32),
        q99=np.ones((32,), dtype=np.float32),
    )
    norm_stats = {"state": stats, "actions": stats}
    data_input = _Marker("data-input")
    data_output = _Marker("data-output")
    model_input = _Marker("model-input")
    model_output = _Marker("model-output")
    dataset_repack = _Marker("dataset-repack-must-not-run-online")

    def restore_params(path, *, dtype=None, **kwargs):
        path = Path(path)
        records["events"].append(("restore", path))
        records["restore_calls"].append((path, dtype, kwargs))
        if path == actor_step / "params":
            return original_restore(path, dtype=dtype, **kwargs)
        if path == base_checkpoint / "params":
            return base_params
        raise AssertionError(f"unexpected params path: {path}")

    def data_create(self, assets_dirs, model_config):
        records["data_create_calls"].append((self, Path(assets_dirs), model_config))
        return _config.DataConfig(
            asset_id=rlt_stage2_checkpoints.RLT_ASSET_ID,
            norm_stats={"must": "not be selected"},
            repack_transforms=transforms.Group(inputs=(dataset_repack,), outputs=(dataset_repack,)),
            data_transforms=transforms.Group(inputs=(data_input,), outputs=(data_output,)),
            model_transforms=transforms.Group(inputs=(model_input,), outputs=(model_output,)),
            use_quantile_norm=True,
        )

    def load_norm_stats(assets_dir, asset_id):
        records["norm_calls"].append((Path(assets_dir), asset_id))
        return norm_stats

    def load_model(self, params):
        records["model_load_calls"].append((self, params))
        return fake_model

    def make_policy(model, **kwargs):
        records["policy_calls"].append((model, kwargs))
        return _CapturedPolicy(model, kwargs)

    def checkpoint_tree_sha256(path):
        path = Path(path)
        records["events"].append(("hash", path))
        records["hash_calls"].append(path)
        if path == base_checkpoint / "params":
            return _FROZEN_PARAMS_SHA256
        if path == base_checkpoint / "assets" / rlt_stage2_checkpoints.RLT_ASSET_ID:
            return _NORM_STATS_SHA256
        raise AssertionError(f"unexpected identity tree: {path}")

    monkeypatch.setattr(policy_config._model, "restore_params", restore_params)  # noqa: SLF001
    monkeypatch.setattr(
        policy_config._feature_identity,  # noqa: SLF001
        "checkpoint_tree_sha256",
        checkpoint_tree_sha256,
    )
    monkeypatch.setattr(_config.SimpleDataConfig, "create", data_create)
    monkeypatch.setattr(policy_config._checkpoints, "load_norm_stats", load_norm_stats)  # noqa: SLF001
    monkeypatch.setattr(type(_train_config().model), "load", load_model)
    monkeypatch.setattr(policy_config._rlt_actor_policy, "RLTActorPolicy", make_policy)  # noqa: SLF001

    return records, {
        "base_params": base_params,
        "fake_model": fake_model,
        "norm_stats": norm_stats,
        "data_input": data_input,
        "data_output": data_output,
        "model_input": model_input,
        "model_output": model_output,
        "dataset_repack": dataset_repack,
    }


def test_factory_restores_actor_first_and_reuses_exact_openpi_transform_order(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    records, expected = _patch_success_dependencies(
        monkeypatch,
        actor_step=actor_step,
        base_checkpoint=base_checkpoint,
    )

    policy = policy_config.create_rlt_actor_policy(
        _train_config(),
        base_checkpoint,
        actor_step,
        mode=rlt_actor_policy.RLTActorMode.MEAN,
        sampler_num_steps=_SAMPLER_NUM_STEPS,
        seed=19,
    )

    restore_calls = records["restore_calls"]
    assert [(path, dtype) for path, dtype, _ in restore_calls] == [
        (actor_step / "params", jnp.float32),
        (base_checkpoint / "params", jnp.bfloat16),
    ]
    assert records["hash_calls"] == [
        base_checkpoint / "params",
        base_checkpoint / "assets" / rlt_stage2_checkpoints.RLT_ASSET_ID,
    ]
    assert records["events"].index(("hash", base_checkpoint / "params")) < records["events"].index(
        ("restore", base_checkpoint / "params")
    )
    assert records["events"].index(
        ("hash", base_checkpoint / "assets" / rlt_stage2_checkpoints.RLT_ASSET_ID)
    ) < records["events"].index(("restore", base_checkpoint / "params"))
    assert records["model_load_calls"] == [(_train_config().model, expected["base_params"])]
    assert len(records["policy_calls"]) == 1
    model, kwargs = records["policy_calls"][0]
    assert policy.model is expected["fake_model"]
    assert model is expected["fake_model"]
    assert kwargs["mode"] is rlt_actor_policy.RLTActorMode.MEAN
    assert kwargs["sample_kwargs"] == {"num_steps": _SAMPLER_NUM_STEPS}
    np.testing.assert_array_equal(jax.random.key_data(kwargs["rng"]), jax.random.key_data(jax.random.key(19)))

    data_factory, assets_dirs, model_config = records["data_create_calls"][0]
    assert Path(data_factory.assets.assets_dir) == base_checkpoint / "assets"
    assert assets_dirs == base_checkpoint / "assets"
    assert model_config is _train_config().model
    assert records["norm_calls"] == [(base_checkpoint / "assets", rlt_stage2_checkpoints.RLT_ASSET_ID)]

    input_transforms = kwargs["transforms"]
    assert isinstance(input_transforms[0], transforms.InjectDefaultPrompt)
    assert input_transforms[0].prompt == "fold clothes"
    assert input_transforms[1] is expected["data_input"]
    assert isinstance(input_transforms[2], transforms.Normalize)
    assert input_transforms[2].norm_stats is expected["norm_stats"]
    assert input_transforms[3] is expected["model_input"]
    assert expected["dataset_repack"] not in input_transforms

    output_transforms = kwargs["output_transforms"]
    assert output_transforms[0] is expected["model_output"]
    assert isinstance(output_transforms[1], transforms.Unnormalize)
    assert output_transforms[1].norm_stats is expected["norm_stats"]
    assert output_transforms[2] is expected["data_output"]
    assert expected["dataset_repack"] not in output_transforms

    missing = input_transforms[0]({})
    explicit = input_transforms[0]({"prompt": "explicit task"})
    assert missing["prompt"].item() == "fold clothes"
    assert explicit["prompt"] == "explicit task"

    deployment_metadata = kwargs["metadata"]["rlt_stage2"]
    assert deployment_metadata["round_complete"] is True
    assert deployment_metadata["mode"] == "mean"
    assert deployment_metadata["sampler_num_steps"] == _SAMPLER_NUM_STEPS
    assert deployment_metadata["feature_identity"] == _FEATURE_IDENTITY
    assert deployment_metadata["frozen_params_sha256"] == _FROZEN_PARAMS_SHA256
    assert deployment_metadata["norm_stats_sha256"] == _NORM_STATS_SHA256
    assert deployment_metadata["stage1_config"] == rlt_stage2_checkpoints.RLT_STAGE1_CONFIG_NAME
    assert deployment_metadata["stage2_config"] == rlt_stage2_checkpoints.RLT_STAGE2_CONFIG_NAME
    assert deployment_metadata["asset_id"] == rlt_stage2_checkpoints.RLT_ASSET_ID
    assert deployment_metadata["base_checkpoint_step"] == 54999
    assert deployment_metadata["network_config"]["actor_hidden_dims"] == [8, 7, 6]
    assert "replay_rng_state" not in deployment_metadata
    json.dumps(kwargs["metadata"])


@pytest.mark.parametrize(
    "mode",
    [rlt_actor_policy.RLTActorMode.MEAN, rlt_actor_policy.RLTActorMode.COLLECTION],
)
def test_factory_rejects_incomplete_actor_step_before_any_parameter_load(
    incomplete_actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: rlt_actor_policy.RLTActorMode,
):
    calls = []

    def unexpected_restore(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("incomplete metadata must fail before loading params")

    monkeypatch.setattr(policy_config._model, "restore_params", unexpected_restore)  # noqa: SLF001

    with pytest.raises(ValueError, match="complete"):
        policy_config.create_rlt_actor_policy(
            _train_config(),
            base_checkpoint,
            incomplete_actor_step,
            mode=mode,
        )

    assert calls == []


@pytest.mark.parametrize("mismatch", ["stage1_config", "asset_id", "base_checkpoint_step"])
def test_factory_rejects_identity_mismatch_without_loading_base(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
):
    train_config = _train_config()
    selected_base = base_checkpoint
    if mismatch == "stage1_config":
        train_config = dataclasses.replace(train_config, name="wrong_config")
    elif mismatch == "asset_id":
        train_config = dataclasses.replace(
            train_config,
            data=dataclasses.replace(
                train_config.data,
                assets=dataclasses.replace(train_config.data.assets, asset_id="wrong_asset"),
            ),
        )
    else:
        selected_base = base_checkpoint.parent / "55000"
        selected_base.mkdir()

    original_restore = policy_config._model.restore_params  # noqa: SLF001
    calls = []

    def tracked_restore(path, *, dtype=None, **kwargs):
        path = Path(path)
        calls.append(path)
        if path == actor_step / "params":
            return original_restore(path, dtype=dtype, **kwargs)
        raise AssertionError("identity mismatch must not load the base checkpoint")

    monkeypatch.setattr(policy_config._model, "restore_params", tracked_restore)  # noqa: SLF001

    with pytest.raises(ValueError, match=mismatch):
        policy_config.create_rlt_actor_policy(
            train_config,
            selected_base,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
        )

    assert base_checkpoint / "params" not in calls
    assert selected_base / "params" not in calls


@pytest.mark.parametrize("tree", ["params", "norm_stats"])
def test_factory_rejects_physical_base_identity_mismatch_before_base_restore(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    tree: str,
):
    original_restore = policy_config._model.restore_params  # noqa: SLF001
    restore_calls: list[Path] = []

    def restore_params(path, *, dtype=None, **kwargs):
        path = Path(path)
        restore_calls.append(path)
        if path == actor_step / "params":
            return original_restore(path, dtype=dtype, **kwargs)
        raise AssertionError("mismatched frozen identity reached base restore")

    def checkpoint_tree_sha256(path):
        path = Path(path)
        if path == base_checkpoint / "params":
            return "9" * 64 if tree == "params" else _FROZEN_PARAMS_SHA256
        if path == base_checkpoint / "assets" / rlt_stage2_checkpoints.RLT_ASSET_ID:
            return "9" * 64 if tree == "norm_stats" else _NORM_STATS_SHA256
        raise AssertionError(f"unexpected identity tree: {path}")

    monkeypatch.setattr(policy_config._model, "restore_params", restore_params)  # noqa: SLF001
    monkeypatch.setattr(
        policy_config._feature_identity,  # noqa: SLF001
        "checkpoint_tree_sha256",
        checkpoint_tree_sha256,
    )

    with pytest.raises(ValueError, match="physical.*identity"):
        policy_config.create_rlt_actor_policy(
            _train_config(),
            base_checkpoint,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
        )

    assert base_checkpoint / "params" not in restore_calls


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("sampler_num_steps", 5, "sampler_num_steps"),
        ("default_prompt", "other task", "default_prompt"),
    ],
)
def test_factory_rejects_online_feature_identity_override_before_parameter_load(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    value: object,
    message: str,
):
    calls = []
    monkeypatch.setattr(
        policy_config._model,  # noqa: SLF001
        "restore_params",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match=message):
        policy_config.create_rlt_actor_policy(
            _train_config(),
            base_checkpoint,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
            **{keyword: value},
        )

    assert calls == []


def test_factory_rejects_prompt_string_subclass_before_base_identity_or_parameter_load(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    hash_calls = []
    restore_calls = []
    monkeypatch.setattr(
        policy_config._feature_identity,  # noqa: SLF001
        "checkpoint_tree_sha256",
        lambda *args, **kwargs: hash_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        policy_config._model,  # noqa: SLF001
        "restore_params",
        lambda *args, **kwargs: restore_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="default_prompt"):
        policy_config.create_rlt_actor_policy(
            _train_config(),
            base_checkpoint,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
            default_prompt=_PromptEqualityImpostor("other task"),
        )

    assert hash_calls == []
    assert restore_calls == []


@pytest.mark.parametrize("drift", ["model", "data_transforms"])
def test_factory_rejects_non_registry_train_config_before_base_identity_or_parameter_load(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
):
    registered = _train_config()
    if drift == "model":
        train_config = dataclasses.replace(
            registered,
            model=dataclasses.replace(registered.model, max_token_len=registered.model.max_token_len + 1),
        )
    else:
        train_config = dataclasses.replace(
            registered,
            data=dataclasses.replace(
                registered.data,
                data_transforms=lambda model: transforms.Group(),
            ),
        )

    hash_calls = []
    restore_calls = []
    monkeypatch.setattr(
        policy_config._feature_identity,  # noqa: SLF001
        "checkpoint_tree_sha256",
        lambda *args, **kwargs: hash_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        policy_config._model,  # noqa: SLF001
        "restore_params",
        lambda *args, **kwargs: restore_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="registered TrainConfig"):
        policy_config.create_rlt_actor_policy(
            train_config,
            base_checkpoint,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
        )

    assert hash_calls == []
    assert restore_calls == []


def _corrupt_actor_params(params, corruption: str):
    flat = traverse_util.flatten_dict(params)
    first_path = next(iter(flat))
    first = np.asarray(flat[first_path])
    if corruption == "tree":
        flat[("unexpected",)] = jnp.zeros((1,), dtype=jnp.float32)
    elif corruption == "shape":
        flat[first_path] = jnp.asarray(first.reshape(-1)[:-1], dtype=jnp.float32)
    elif corruption == "dtype":
        flat[first_path] = jnp.asarray(first, dtype=jnp.float16)
    else:
        bad = first.copy()
        bad.reshape(-1)[0] = np.nan
        flat[first_path] = jnp.asarray(bad, dtype=jnp.float32)
    return traverse_util.unflatten_dict(flat)


@pytest.mark.parametrize("corruption", ["tree", "shape", "dtype", "finite"])
def test_factory_validates_exact_actor_params_before_base_load(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
):
    original_restore = policy_config._model.restore_params  # noqa: SLF001
    valid = original_restore(actor_step / "params", dtype=jnp.float32)
    corrupted = _corrupt_actor_params(valid, corruption)
    calls = []

    def restore_params(path, *, dtype=None, **kwargs):
        path = Path(path)
        calls.append((path, dtype))
        if path == actor_step / "params":
            return corrupted
        raise AssertionError("invalid actor params must fail before base load")

    monkeypatch.setattr(policy_config._model, "restore_params", restore_params)  # noqa: SLF001

    with pytest.raises(ValueError, match="actor params"):
        policy_config.create_rlt_actor_policy(
            _train_config(),
            base_checkpoint,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
        )

    assert calls == [(actor_step / "params", jnp.float32)]


@pytest.mark.parametrize(
    "model_changes",
    [
        {"action_horizon": 49},
        {"action_dim": 31},
        {"pi05": False},
        {"rl_token_enabled": False, "rl_token_only": False},
        {"rl_token_only": False},
    ],
)
def test_factory_accepts_only_jax_pi05_rlt_50x32_model_config(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_changes: dict[str, object],
):
    train_config = _train_config()
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, **model_changes),
    )
    calls = []
    monkeypatch.setattr(
        policy_config._model,  # noqa: SLF001
        "restore_params",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="JAX PI0.5 RLT.*50x32"):
        policy_config.create_rlt_actor_policy(
            train_config,
            base_checkpoint,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
        )

    assert calls == []


def test_factory_rejects_seed_outside_jax_uint32_range_before_parameter_load(
    actor_step: Path,
    base_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(
        policy_config._model,  # noqa: SLF001
        "restore_params",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="seed.*uint32"):
        policy_config.create_rlt_actor_policy(
            _train_config(),
            base_checkpoint,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
            seed=2**32,
        )

    assert calls == []


def test_factory_rejects_pytorch_base_before_loading_base_params(
    actor_step: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    base = tmp_path / "54999"
    base.mkdir()
    (base / "model.safetensors").touch()
    original_restore = policy_config._model.restore_params  # noqa: SLF001
    calls = []

    def restore_params(path, *, dtype=None, **kwargs):
        path = Path(path)
        calls.append(path)
        if path == actor_step / "params":
            return original_restore(path, dtype=dtype, **kwargs)
        raise AssertionError("PyTorch base must be rejected before base params load")

    monkeypatch.setattr(policy_config._model, "restore_params", restore_params)  # noqa: SLF001

    with pytest.raises(ValueError, match="JAX"):
        policy_config.create_rlt_actor_policy(
            _train_config(),
            base,
            actor_step,
            mode=rlt_actor_policy.RLTActorMode.MEAN,
        )

    assert calls == [actor_step / "params"]
