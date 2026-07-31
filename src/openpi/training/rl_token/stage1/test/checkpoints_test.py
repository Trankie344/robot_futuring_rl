import dataclasses
import os
import pathlib
import shutil

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize
from openpi.training.rl_token.stage1 import checkpoints
from openpi.training.rl_token import config as _config
from openpi.training import utils as training_utils


class _Branch(nnx.Module):
    def __init__(self, value: float, dtype: jnp.dtype):
        self.kernel = nnx.Param(jnp.full((2, 2), value, dtype=dtype))


class _TinyModel(_model.BaseModel):
    def __init__(self, rngs: nnx.Rngs):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.vla = _Branch(1.0, jnp.bfloat16)
        self.action_head = _Branch(2.0, jnp.bfloat16)
        self.rl_token = _Branch(3.0, jnp.float32)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, observation, actions, train
        return jnp.zeros((1, 1), dtype=jnp.float32)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        return jnp.zeros((1, 1, 1), dtype=jnp.float32)


class _Loader:
    def __init__(self):
        self._data_config = _config.DataConfig(
            repo_id="fake",
            asset_id="tiny_asset",
            norm_stats={
                "state": _normalize.NormStats(
                    mean=np.array([1.0], dtype=np.float32),
                    std=np.array([2.0], dtype=np.float32),
                )
            },
        )

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        return iter(())


def _is_exact_rlt(path, _value) -> bool:
    return bool(path) and path[0] == "rl_token"


def _make_state(
    *,
    vla_value: float = 1.0,
    action_value: float = 2.0,
    raw_rlt_value: float = 3.0,
    ema_rlt_value: float = 9.0,
    dropout_seed: int = 17,
    dropout_advance_count: int = 4,
    optimizer_update_count: int = 0,
    step: int = 123,
) -> training_utils.TrainState:
    model = _TinyModel(nnx.Rngs(dropout=dropout_seed))
    model.vla.kernel.value = jnp.full((2, 2), vla_value, dtype=jnp.bfloat16)
    model.action_head.kernel.value = jnp.full((2, 2), action_value, dtype=jnp.bfloat16)
    model.rl_token.kernel.value = jnp.full((2, 2), raw_rlt_value, dtype=jnp.float32)
    for _ in range(dropout_advance_count):
        model.dropout.rngs.dropout()

    graphdef, state = nnx.split(model)
    rlt_params = state.filter(nnx.All(nnx.Param, _is_exact_rlt))
    tx = optax.adam(1e-3)
    opt_state = tx.init(rlt_params)
    unit_grads = rlt_params.map(
        lambda _path, variable: variable.replace(jnp.ones_like(variable.value, dtype=jnp.float32))
    )
    for _ in range(optimizer_update_count):
        _, opt_state = tx.update(unit_grads, opt_state, rlt_params)
    ema_params = rlt_params.map(
        lambda _path, variable: variable.replace(jnp.full(variable.value.shape, ema_rlt_value, dtype=jnp.float32))
    )
    with at.disable_typechecking():
        return training_utils.TrainState(
            step=jnp.asarray(step, dtype=jnp.int32),
            params=state,
            model_def=graphdef,
            tx=tx,
            opt_state=opt_state,
            ema_decay=0.999,
            ema_params=ema_params,
        )


def _pure_base_params(
    *,
    vla_value: float = 11.0,
    action_value: float = 22.0,
    rlt_value: float = 33.0,
) -> dict:
    return {
        "vla": {"kernel": np.full((2, 2), vla_value, dtype=np.float32)},
        "action_head": {"kernel": np.full((2, 2), action_value, dtype=np.float32)},
        "rl_token": {"kernel": np.full((2, 2), rlt_value, dtype=np.float32)},
    }


def _save_base(path: pathlib.Path, params: dict) -> None:
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": params})


def _flat_pure(params: nnx.State | dict) -> dict[tuple[str, ...], object]:
    if isinstance(params, nnx.State):
        return {tuple(map(str, path)): variable.value for path, variable in params.flat_state().items()}
    return {tuple(map(str, path)): value for path, value in traverse_util.flatten_dict(params).items()}


def _rng_state(state: training_utils.TrainState) -> dict[tuple[str, ...], np.ndarray]:
    def as_numpy(value) -> np.ndarray:
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            value = jax.random.key_data(value)
        return np.asarray(value)

    return {
        tuple(map(str, path)): as_numpy(variable.value)
        for path, variable in state.params.filter(nnx.RngState).flat_state().items()
    }


def _assert_tree_arrays_equal(expected, actual) -> None:
    expected_with_paths, expected_structure = jax.tree_util.tree_flatten_with_path(expected)
    actual_with_paths, actual_structure = jax.tree_util.tree_flatten_with_path(actual)
    assert actual_structure == expected_structure
    assert [jax.tree_util.keystr(path) for path, _ in actual_with_paths] == [
        jax.tree_util.keystr(path) for path, _ in expected_with_paths
    ]
    for (_, expected_leaf), (_, actual_leaf) in zip(expected_with_paths, actual_with_paths, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))


def _assert_tree_arrays_differ(expected, actual) -> None:
    expected_with_paths, expected_structure = jax.tree_util.tree_flatten_with_path(expected)
    actual_with_paths, actual_structure = jax.tree_util.tree_flatten_with_path(actual)
    assert actual_structure == expected_structure
    assert [jax.tree_util.keystr(path) for path, _ in actual_with_paths] == [
        jax.tree_util.keystr(path) for path, _ in expected_with_paths
    ]
    assert any(
        not np.array_equal(np.asarray(expected_leaf), np.asarray(actual_leaf))
        for (_, expected_leaf), (_, actual_leaf) in zip(expected_with_paths, actual_with_paths, strict=True)
    )


@pytest.mark.parametrize(("max_to_keep", "expected"), [(2, 2), (None, 1)])
def test_initialize_checkpoint_dir_configures_max_to_keep(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    max_to_keep: int | None,
    expected: int,
):
    captured = {}

    def make_manager(directory, *, item_handlers, options):
        captured.update(directory=directory, item_handlers=item_handlers, options=options)
        return object()

    monkeypatch.setattr(ocp, "CheckpointManager", make_manager)
    kwargs = {} if max_to_keep is None else {"max_to_keep": max_to_keep}

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / f"checkpoints-{expected}",
        keep_period=None,
        overwrite=False,
        resume=False,
        **kwargs,
    )

    assert manager is not None
    assert not resuming
    assert captured["options"].max_to_keep == expected


@pytest.mark.parametrize("max_to_keep", [True, False, 0, -1, 1.5, "2", None])
def test_initialize_checkpoint_dir_rejects_invalid_max_to_keep(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    max_to_keep,
):
    def unexpected_manager(*args, **kwargs):
        raise AssertionError("manager construction must follow validation")

    monkeypatch.setattr(ocp, "CheckpointManager", unexpected_manager)

    with pytest.raises(ValueError, match="max_to_keep.*positive integer"):
        checkpoints.initialize_checkpoint_dir(
            tmp_path / "checkpoints",
            keep_period=None,
            overwrite=False,
            resume=False,
            max_to_keep=max_to_keep,
        )


def test_build_deployment_params_is_complete_fp32_and_ema_wins(tmp_path: pathlib.Path):
    state = _make_state()
    base_path = tmp_path / "base_params"
    _save_base(base_path, _pure_base_params(rlt_value=123.0))

    deployment = checkpoints.build_deployment_params(state, base_path)
    flat = _flat_pure(deployment)

    assert set(flat) == {
        ("action_head", "kernel"),
        ("rl_token", "kernel"),
        ("vla", "kernel"),
    }
    assert not deployment.filter(nnx.RngState).flat_state()
    assert all(np.asarray(value).dtype == np.float32 for value in flat.values())
    np.testing.assert_array_equal(flat[("vla", "kernel")], np.full((2, 2), 11.0, dtype=np.float32))
    np.testing.assert_array_equal(flat[("action_head", "kernel")], np.full((2, 2), 22.0, dtype=np.float32))
    np.testing.assert_array_equal(flat[("rl_token", "kernel")], np.full((2, 2), 9.0, dtype=np.float32))


def test_saved_deployment_survives_base_deletion(tmp_path: pathlib.Path):
    state = _make_state()
    base_path = tmp_path / "base_params"
    deployment_path = tmp_path / "deployment_params"
    _save_base(base_path, _pure_base_params())
    deployment = checkpoints.build_deployment_params(state, base_path)

    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(deployment_path, {"params": deployment})
    shutil.rmtree(base_path)

    restored = _model.restore_params(deployment_path, restore_type=np.ndarray)
    flat = _flat_pure(restored)
    assert set(flat) == {
        ("action_head", "kernel"),
        ("rl_token", "kernel"),
        ("vla", "kernel"),
    }
    assert all(np.asarray(value).dtype == np.float32 for value in flat.values())
    np.testing.assert_array_equal(flat[("vla", "kernel")], np.full((2, 2), 11.0, dtype=np.float32))
    np.testing.assert_array_equal(flat[("rl_token", "kernel")], np.full((2, 2), 9.0, dtype=np.float32))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda params: params.pop("vla"), r"Missing deployment base parameter.*vla/kernel"),
        (
            lambda params: params["action_head"].__setitem__("kernel", np.zeros((3, 2), dtype=np.float32)),
            r"Deployment base shape mismatch.*action_head/kernel.*expected \(2, 2\).*got \(3, 2\)",
        ),
        (
            lambda params: params["vla"].__setitem__("kernel", np.zeros((2, 2), dtype=np.float16)),
            r"Deployment base dtype mismatch.*vla/kernel.*float32.*float16",
        ),
    ],
)
def test_build_deployment_params_rejects_invalid_base(tmp_path: pathlib.Path, mutation, message):
    params = _pure_base_params()
    mutation(params)
    base_path = tmp_path / "invalid_base"
    _save_base(base_path, params)

    with pytest.raises(ValueError, match=message):
        checkpoints.build_deployment_params(_make_state(), base_path)


def test_build_deployment_params_requires_base_for_partial_ema():
    with pytest.raises(ValueError, match="deployment_base_params_path.*partial EMA"):
        checkpoints.build_deployment_params(_make_state(), None)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (np.zeros((1, 2), dtype=np.float32), r"EMA shape mismatch.*rl_token/kernel"),
        (np.zeros((2, 2), dtype=np.float16), r"EMA dtype mismatch.*rl_token/kernel.*float32"),
    ],
)
def test_build_deployment_params_validates_partial_ema(replacement, message):
    state = _make_state()
    ema = nnx.State.from_flat_path(
        {path: variable.replace(replacement) for path, variable in state.ema_params.flat_state().items()}
    )
    state = dataclasses.replace(state, ema_params=ema)

    with pytest.raises(ValueError, match=message):
        checkpoints.build_deployment_params(state, "/not/needed/because/ema/is-validated-first")


def test_build_deployment_params_rejects_extra_ema_path(tmp_path: pathlib.Path):
    state = _make_state()
    extra_ema = nnx.State.merge(
        state.ema_params,
        nnx.State(
            {
                "not_a_model_param": nnx.VariableState(
                    nnx.Param,
                    jnp.ones((2, 2), dtype=jnp.float32),
                )
            }
        ),
    )
    state = dataclasses.replace(state, ema_params=extra_ema)
    base_path = tmp_path / "base"
    _save_base(base_path, _pure_base_params())

    with pytest.raises(ValueError, match=r"EMA contains unexpected parameter.*not_a_model_param"):
        checkpoints.build_deployment_params(state, base_path)


def test_save_and_restore_partial_ema_checkpoint_is_self_contained(tmp_path: pathlib.Path):
    state = _make_state(step=123, optimizer_update_count=2)
    original_params = _flat_pure(state.params.filter(nnx.Param))
    original_ema = _flat_pure(state.ema_params)
    original_rng = _rng_state(state)
    original_opt_state = state.opt_state
    base_path = tmp_path / "base_params"
    checkpoint_dir = tmp_path / "checkpoints"
    _save_base(base_path, _pure_base_params())
    loader = _Loader()
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=5000,
        overwrite=False,
        resume=False,
    )
    assert not resuming

    checkpoints.save_state(
        manager,
        state,
        loader,
        5000,
        deployment_base_params_path=base_path,
    )
    manager.wait_until_finished()
    assert (checkpoint_dir / "5000" / "assets" / "tiny_asset" / "norm_stats.json").is_file()
    shutil.rmtree(base_path)

    deployment = _model.restore_params(
        checkpoint_dir / "5000" / "params",
        restore_type=np.ndarray,
    )
    deployment_flat = _flat_pure(deployment)
    assert set(deployment_flat) == {
        ("action_head", "kernel"),
        ("rl_token", "kernel"),
        ("vla", "kernel"),
    }
    assert all(np.asarray(value).dtype == np.float32 for value in deployment_flat.values())

    restore_target = _make_state(
        vla_value=-11.0,
        action_value=-22.0,
        raw_rlt_value=-3.0,
        ema_rlt_value=-9.0,
        dropout_seed=99,
        dropout_advance_count=1,
        optimizer_update_count=0,
        step=0,
    )
    target_params = _flat_pure(restore_target.params.filter(nnx.Param))
    target_ema = _flat_pure(restore_target.ema_params)
    target_rng = _rng_state(restore_target)
    assert int(restore_target.step) != int(state.step)
    assert target_params.keys() == original_params.keys()
    assert all(not np.array_equal(target_params[path], value) for path, value in original_params.items())
    assert target_ema.keys() == original_ema.keys()
    assert all(not np.array_equal(target_ema[path], value) for path, value in original_ema.items())
    assert target_rng.keys() == original_rng.keys()
    assert all(not np.array_equal(target_rng[path], value) for path, value in original_rng.items())
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree_util.tree_leaves(original_opt_state))
    _assert_tree_arrays_differ(original_opt_state, restore_target.opt_state)

    restored = checkpoints.restore_state(manager, restore_target, loader, step=5000)
    restored_params = _flat_pure(restored.params.filter(nnx.Param))
    assert restored_params.keys() == original_params.keys()
    for path, value in original_params.items():
        np.testing.assert_array_equal(restored_params[path], value)
    restored_ema = _flat_pure(restored.ema_params)
    assert restored_ema.keys() == original_ema.keys()
    for path, value in original_ema.items():
        np.testing.assert_array_equal(restored_ema[path], value)
    assert int(restored.step) == int(state.step)
    assert _rng_state(restored).keys() == original_rng.keys()
    for path, value in original_rng.items():
        np.testing.assert_array_equal(_rng_state(restored)[path], value)
    _assert_tree_arrays_equal(original_opt_state, restored.opt_state)


def test_split_merge_preserves_legacy_full_ema_semantics_despite_rng_state():
    state = _make_state()
    full_ema = state.params.filter(nnx.Param).map(
        lambda _path, variable: variable.replace(variable.value.astype(jnp.float32))
    )
    state = dataclasses.replace(state, ema_params=full_ema)

    assert not checkpoints._is_partial_ema(state)  # noqa: SLF001
    train_state, params = checkpoints._split_params(state)  # noqa: SLF001
    assert train_state.ema_params is None
    assert set(_flat_pure(params)) == set(_flat_pure(full_ema))

    restored = checkpoints._merge_params(train_state, {"params": params})  # noqa: SLF001
    assert set(_flat_pure(restored.params)) == set(_flat_pure(state.params))
    assert set(_flat_pure(restored.ema_params)) == set(_flat_pure(full_ema))


def test_split_merge_preserves_legacy_no_ema_runtime_state():
    state = dataclasses.replace(_make_state(), ema_decay=None, ema_params=None)
    original_rng = _rng_state(state)

    train_state, params = checkpoints._split_params(state)  # noqa: SLF001
    assert train_state.params == nnx.State({})
    split_rng = _rng_state(dataclasses.replace(state, params=params))
    assert split_rng.keys() == original_rng.keys()
    for path, value in original_rng.items():
        np.testing.assert_array_equal(split_rng[path], value)

    restored = checkpoints._merge_params(train_state, {"params": params})  # noqa: SLF001
    assert set(_flat_pure(restored.params)) == set(_flat_pure(state.params))
    assert _rng_state(restored).keys() == original_rng.keys()
    for path, value in original_rng.items():
        np.testing.assert_array_equal(_rng_state(restored)[path], value)


def test_partial_detection_compares_parameter_paths_not_runtime_state():
    state = _make_state()
    full_param_ema = state.params.filter(nnx.Param)
    assert state.params.filter(nnx.RngState).flat_state()

    assert not checkpoints._is_partial_ema(dataclasses.replace(state, ema_params=full_param_ema))  # noqa: SLF001
    assert checkpoints._is_partial_ema(state)  # noqa: SLF001
