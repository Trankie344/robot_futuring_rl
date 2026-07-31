import dataclasses
from types import MappingProxyType

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rl_token import actor_critic as rlt

# Flax 0.10.2 probes ShapeDtypeStruct values with jnp.shape during parameter initialization.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:shape requires ndarray or scalar arguments.*:DeprecationWarning:flax\.core\.scope"
)


def _small_config(*, compute_dtype: str = "bfloat16") -> rlt.RLTActorCriticConfig:
    return rlt.RLTActorCriticConfig(
        z_dim=8,
        state_dim=3,
        action_horizon=2,
        action_dim=2,
        actor_state_proj_dim=5,
        actor_reference_proj_dim=6,
        critic_state_proj_dim=5,
        critic_action_proj_dim=6,
        actor_hidden_dims=(9, 8, 7),
        critic_hidden_dims=(9, 8, 7),
        compute_dtype=compute_dtype,
    )


def _inputs(config: rlt.RLTActorCriticConfig, *, batch_size: int = 2):
    z_key, state_key, action_key = jax.random.split(jax.random.key(0), 3)
    z_rl = jax.random.normal(z_key, (batch_size, config.z_dim), dtype=jnp.float32)
    state = jax.random.normal(state_key, (batch_size, config.state_dim), dtype=jnp.float32)
    action = jax.random.normal(
        action_key,
        (batch_size, config.action_horizon, config.action_dim),
        dtype=jnp.float32,
    )
    return z_rl, state, action


def _parameter_count(variables) -> int:
    return sum(parameter.size for parameter in jax.tree_util.tree_leaves(variables["params"]))


def _assert_finite_tree(tree) -> None:
    leaves = jax.tree_util.tree_leaves(tree)
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf.astype(jnp.float32))) for leaf in leaves)


def _dot_general_dtype_signatures(module, variables, inputs):
    closed_jaxpr = jax.make_jaxpr(lambda *args: module.apply(variables, *args))(*inputs)
    return tuple(
        tuple(variable.aval.dtype for variable in (*equation.invars, *equation.outvars))
        for equation in closed_jaxpr.jaxpr.eqns
        if equation.primitive.name == "dot_general"
    )


def _network_config_json_payload() -> dict[str, object]:
    return {
        "z_dim": 8,
        "state_dim": 3,
        "action_horizon": 2,
        "action_dim": 2,
        "actor_state_proj_dim": 5,
        "actor_reference_proj_dim": 6,
        "critic_state_proj_dim": 5,
        "critic_action_proj_dim": 6,
        "actor_hidden_dims": [9, 8, 7],
        "critic_hidden_dims": [9, 8, 7],
        "compute_dtype": "float32",
    }


def test_config_defaults_and_derived_properties_are_the_public_contract():
    config = rlt.RLTActorCriticConfig()

    assert dataclasses.is_dataclass(config)
    assert config.z_dim == 2048
    assert config.state_dim == 16
    assert config.action_horizon == 20
    assert config.action_dim == 16
    assert config.actor_state_proj_dim == 2048
    assert config.actor_reference_proj_dim == 2048
    assert config.critic_state_proj_dim == 2048
    assert config.critic_action_proj_dim == 2048
    assert config.actor_hidden_dims == (1024, 1024, 1024)
    assert config.critic_hidden_dims == (1024, 1024, 1024)
    assert config.compute_dtype == "bfloat16"
    assert config.flat_action_dim == 320
    assert config.jnp_compute_dtype == jnp.dtype(jnp.bfloat16)
    assert config.validate() is None

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.z_dim = 1


def test_decode_network_config_accepts_only_the_canonical_json_shape():
    payload = _network_config_json_payload()

    config = rlt.decode_network_config(payload, label="policy.network_config")

    assert config == rlt.RLTActorCriticConfig(
        z_dim=8,
        state_dim=3,
        action_horizon=2,
        action_dim=2,
        actor_state_proj_dim=5,
        actor_reference_proj_dim=6,
        critic_state_proj_dim=5,
        critic_action_proj_dim=6,
        actor_hidden_dims=(9, 8, 7),
        critic_hidden_dims=(9, 8, 7),
        compute_dtype="float32",
    )
    round_trip = dataclasses.asdict(config)
    round_trip["actor_hidden_dims"] = list(round_trip["actor_hidden_dims"])
    round_trip["critic_hidden_dims"] = list(round_trip["critic_hidden_dims"])
    assert round_trip == payload


def test_decode_network_config_accepts_a_read_only_mapping():
    payload = _network_config_json_payload()

    config = rlt.decode_network_config(MappingProxyType(payload))

    assert config == rlt.decode_network_config(payload)


class _FormatBomb:
    def __init__(self):
        self.format_calls = 0

    def __format__(self, _format_spec: str) -> str:
        self.format_calls += 1
        raise AssertionError("untrusted label was formatted")


class _FormatBombStr(str):
    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance.format_calls = 0
        return instance

    def __format__(self, _format_spec: str) -> str:
        self.format_calls += 1
        raise AssertionError("untrusted string subclass was formatted")


@pytest.mark.parametrize("label", [_FormatBomb(), _FormatBombStr("policy.network_config")])
def test_decode_network_config_rejects_nonexact_labels_without_formatting_them(label):
    with pytest.raises(ValueError, match="label must be a nonempty exact string"):
        rlt.decode_network_config(_network_config_json_payload(), label=label)

    assert label.format_calls == 0


def test_decode_network_config_rejects_an_empty_label():
    with pytest.raises(ValueError, match="label must be a nonempty exact string"):
        rlt.decode_network_config(_network_config_json_payload(), label="")


class _StringSubclass(str):
    pass


@pytest.mark.parametrize("invalid_key", [1, _StringSubclass("z_dim")])
def test_decode_network_config_rejects_nonexact_json_string_keys(invalid_key):
    payload = _network_config_json_payload()
    payload[invalid_key] = payload.pop("z_dim")

    with pytest.raises(ValueError, match="policy\\.network_config"):
        rlt.decode_network_config(payload, label="policy.network_config")


@pytest.mark.parametrize("payload", [None, [], (("z_dim", 8),)])
def test_decode_network_config_rejects_non_mappings(payload):
    with pytest.raises(ValueError, match="policy\\.network_config"):
        rlt.decode_network_config(payload, label="policy.network_config")


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_decode_network_config_requires_the_exact_key_set(mutation):
    payload = _network_config_json_payload()
    if mutation == "missing":
        del payload["state_dim"]
    else:
        payload["unexpected"] = 1

    with pytest.raises(ValueError, match="policy\\.network_config"):
        rlt.decode_network_config(payload, label="policy.network_config")


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "missing_only"),
        ("missing_and_extra", "key_set"),
        ("extra", "key_set"),
    ],
)
def test_decode_network_config_reports_a_typed_key_set_reason(mutation, reason):
    payload = _network_config_json_payload()
    if mutation != "extra":
        del payload["state_dim"]
    if mutation != "missing":
        payload["unexpected"] = 1

    with pytest.raises(rlt.NetworkConfigDecodeError) as exc_info:
        rlt.decode_network_config(payload)

    assert exc_info.value.reason == reason


@pytest.mark.parametrize(
    "field",
    [
        "z_dim",
        "state_dim",
        "action_horizon",
        "action_dim",
        "actor_state_proj_dim",
        "actor_reference_proj_dim",
        "critic_state_proj_dim",
        "critic_action_proj_dim",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, 8.0, "8", np.int64(8)])
def test_decode_network_config_requires_exact_json_integer_scalars(field, invalid_value):
    payload = _network_config_json_payload()
    payload[field] = invalid_value

    with pytest.raises(ValueError, match="policy\\.network_config"):
        rlt.decode_network_config(payload, label="policy.network_config")


class _ListSubclass(list):
    pass


@pytest.mark.parametrize("field", ["actor_hidden_dims", "critic_hidden_dims"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        (9, 8, 7),
        _ListSubclass([9, 8, 7]),
        [9, 8],
        [9, 8, 7, 6],
        [9, True, 7],
        [9, 8.0, 7],
        [9, 0, 7],
    ],
)
def test_decode_network_config_requires_exact_json_hidden_dimension_lists(field, invalid_value):
    payload = _network_config_json_payload()
    payload[field] = invalid_value

    with pytest.raises(ValueError, match="policy\\.network_config"):
        rlt.decode_network_config(payload, label="policy.network_config")


@pytest.mark.parametrize("invalid_value", [True, 1, "float16"])
def test_decode_network_config_requires_a_supported_exact_json_string_dtype(invalid_value):
    payload = _network_config_json_payload()
    payload["compute_dtype"] = invalid_value

    with pytest.raises(ValueError, match="policy\\.network_config"):
        rlt.decode_network_config(payload, label="policy.network_config")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"z_dim": 0}, "z_dim"),
        ({"state_dim": -1}, "state_dim"),
        ({"action_horizon": 0}, "action_horizon"),
        ({"action_dim": -1}, "action_dim"),
        ({"actor_state_proj_dim": 0}, "actor_state_proj_dim"),
        ({"actor_reference_proj_dim": -1}, "actor_reference_proj_dim"),
        ({"critic_state_proj_dim": 0}, "critic_state_proj_dim"),
        ({"critic_action_proj_dim": -1}, "critic_action_proj_dim"),
        ({"actor_hidden_dims": (1, 2)}, "actor_hidden_dims"),
        ({"actor_hidden_dims": (1, 0, 3)}, "actor_hidden_dims"),
        ({"critic_hidden_dims": (1, 2, 3, 4)}, "critic_hidden_dims"),
        ({"critic_hidden_dims": (1, -2, 3)}, "critic_hidden_dims"),
        ({"compute_dtype": "float16"}, "compute_dtype"),
    ],
)
def test_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rlt.RLTActorCriticConfig(**kwargs)


@pytest.mark.parametrize(
    "dimension_name",
    [
        "z_dim",
        "state_dim",
        "action_horizon",
        "action_dim",
        "actor_state_proj_dim",
        "actor_reference_proj_dim",
        "critic_state_proj_dim",
        "critic_action_proj_dim",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, 1.0, "1"])
def test_config_rejects_non_integer_scalar_dimensions(dimension_name, invalid_value):
    with pytest.raises(ValueError, match=dimension_name):
        rlt.RLTActorCriticConfig(**{dimension_name: invalid_value})


@pytest.mark.parametrize(
    ("dimension_name", "invalid_value"),
    [
        ("actor_hidden_dims", [1, 2, 3]),
        ("critic_hidden_dims", [1, 2, 3]),
        ("actor_hidden_dims", (1, True, 3)),
        ("critic_hidden_dims", (1, 2.0, 3)),
    ],
)
def test_config_rejects_mutable_or_non_integer_hidden_dimensions(dimension_name, invalid_value):
    with pytest.raises(ValueError, match=dimension_name):
        rlt.RLTActorCriticConfig(**{dimension_name: invalid_value})


@pytest.mark.parametrize(
    "boolean_value",
    [np.bool_(0), np.bool_(1), jnp.bool_(0), jnp.bool_(1)],
)
def test_config_rejects_boolean_scalar_variants(boolean_value):
    with pytest.raises(ValueError, match="z_dim"):
        rlt.RLTActorCriticConfig(z_dim=boolean_value)
    with pytest.raises(ValueError, match="actor_hidden_dims"):
        rlt.RLTActorCriticConfig(actor_hidden_dims=(1, boolean_value, 3))


def test_config_accepts_float32_compute():
    config = _small_config(compute_dtype="float32")

    assert config.jnp_compute_dtype == jnp.dtype(jnp.float32)


def test_config_normalizes_integer_like_dimensions_and_remains_hashable():
    config = rlt.RLTActorCriticConfig(
        z_dim=np.int64(8),
        state_dim=jnp.int32(3),
        action_horizon=np.int16(2),
        action_dim=jnp.int16(2),
        actor_state_proj_dim=5,
        actor_reference_proj_dim=np.int32(6),
        critic_state_proj_dim=jnp.int32(5),
        critic_action_proj_dim=6,
        actor_hidden_dims=(np.int64(9), jnp.int32(8), 7),
        critic_hidden_dims=(jnp.int16(9), 8, np.int32(7)),
    )

    scalar_dimensions = (
        config.z_dim,
        config.state_dim,
        config.action_horizon,
        config.action_dim,
        config.actor_state_proj_dim,
        config.actor_reference_proj_dim,
        config.critic_state_proj_dim,
        config.critic_action_proj_dim,
    )
    assert {type(dimension) for dimension in scalar_dimensions} == {int}
    assert {type(config.actor_hidden_dims), type(config.critic_hidden_dims)} == {tuple}
    assert {type(dimension) for dimension in (*config.actor_hidden_dims, *config.critic_hidden_dims)} == {int}
    assert isinstance(hash(config), int)


@pytest.mark.parametrize("compute_dtype", ["bfloat16", "float32"])
def test_actor_and_critic_dense_compute_dtypes_are_locked(compute_dtype):
    config = _small_config(compute_dtype=compute_dtype)
    inputs = _inputs(config)
    trunk_dtype = jnp.dtype(compute_dtype)
    fp32_dtype = jnp.dtype(jnp.float32)
    dense_dtypes = (trunk_dtype,) * 5 + (fp32_dtype,)
    expected_signatures = tuple((dense_dtype,) * 3 for dense_dtype in dense_dtypes)

    for module, key in (
        (rlt.RLTActor(config), jax.random.key(14)),
        (rlt.RLTCritic(config), jax.random.key(15)),
    ):
        variables = module.init(key, *inputs)
        assert _dot_general_dtype_signatures(module, variables, inputs) == expected_signatures


def test_kaiming_uniform_initializer_matches_fixed_key_reference():
    key = jax.random.key(16)
    shape = (17, 11)
    reference_initializer = nn.initializers.variance_scaling(2.0, "fan_in", "uniform")

    expected = reference_initializer(key, shape, jnp.float32)
    actual = rlt._KAIMING_UNIFORM(key, shape, jnp.float32)  # noqa: SLF001

    assert jnp.array_equal(actual, expected)


def test_actor_and_critic_projection_and_hidden_kernels_use_kaiming(monkeypatch):
    sentinel_value = jnp.asarray(0.125, dtype=jnp.float32)

    def sentinel_initializer(_key, shape, dtype=jnp.float32):
        return jnp.full(shape, sentinel_value, dtype=dtype)

    monkeypatch.setattr(rlt, "_KAIMING_UNIFORM", sentinel_initializer)
    config = _small_config()
    inputs = _inputs(config)
    cases = (
        (
            rlt.RLTActor(config),
            jax.random.key(17),
            ("state_projection", "reference_projection", "hidden_0", "hidden_1", "hidden_2"),
        ),
        (
            rlt.RLTCritic(config),
            jax.random.key(18),
            ("state_projection", "action_projection", "hidden_0", "hidden_1", "hidden_2"),
        ),
    )

    for module, key, layer_names in cases:
        params = module.init(key, *inputs)["params"]
        for layer_name in layer_names:
            kernel = params[layer_name]["kernel"]
            assert jnp.array_equal(kernel, jnp.full_like(kernel, sentinel_value))


def test_actor_and_critic_small_config_contract_and_parameter_topology():
    config = _small_config()
    z_rl, state, action = _inputs(config)
    actor = rlt.RLTActor(config)
    critic = rlt.RLTCritic(config)
    actor_variables = actor.init(jax.random.key(1), z_rl, state, action)
    critic_variables = critic.init(jax.random.key(2), z_rl, state, action)

    actor_output = actor.apply(actor_variables, z_rl, state, action)
    critic_output = critic.apply(critic_variables, z_rl, state, action)

    assert actor_output.shape == (2, config.action_horizon, config.action_dim)
    assert actor_output.dtype == jnp.float32
    assert jnp.all(actor_output >= -1.0)
    assert jnp.all(actor_output <= 1.0)
    assert critic_output.shape == (2, 1)
    assert critic_output.dtype == jnp.float32

    assert set(actor_variables["params"]) == {
        "state_projection",
        "reference_projection",
        "hidden_0",
        "hidden_1",
        "hidden_2",
        "output",
    }
    assert set(critic_variables["params"]) == {
        "state_projection",
        "action_projection",
        "hidden_0",
        "hidden_1",
        "hidden_2",
        "output",
    }
    parameter_leaves = jax.tree_util.tree_leaves((actor_variables["params"], critic_variables["params"]))
    assert parameter_leaves
    assert all(parameter.dtype == jnp.float32 for parameter in parameter_leaves)

    for variables in (actor_variables, critic_variables):
        output_parameters = variables["params"]["output"]
        assert jnp.all(jnp.abs(output_parameters["kernel"]) <= 3e-3)
        assert jnp.all(jnp.abs(output_parameters["bias"]) <= 3e-3)


def test_output_initializer_and_real_heads_are_bounded_and_bilateral():
    bound = jnp.asarray(3e-3, dtype=jnp.float32)
    initializer_samples = rlt._OUTPUT_UNIFORM(  # noqa: SLF001
        jax.random.key(11),
        (16_384,),
        jnp.float32,
    )

    assert jnp.all(initializer_samples >= -bound)
    assert jnp.all(initializer_samples <= bound)
    assert jnp.any(initializer_samples < 0)
    assert jnp.any(initializer_samples > 0)

    config = _small_config()
    z_rl, state, action = _inputs(config)
    actor_variables = rlt.RLTActor(config).init(jax.random.key(12), z_rl, state, action)
    critic_variables = rlt.RLTCritic(config).init(jax.random.key(13), z_rl, state, action)
    real_head_values = []
    for variables in (actor_variables, critic_variables):
        for parameter in variables["params"]["output"].values():
            assert jnp.all(parameter >= -bound)
            assert jnp.all(parameter <= bound)
            real_head_values.append(parameter.reshape(-1))

    concatenated_head_values = jnp.concatenate(real_head_values)
    assert jnp.any(concatenated_head_values < 0)
    assert jnp.any(concatenated_head_values > 0)


def test_default_parameter_counts_use_abstract_initialization():
    config = rlt.RLTActorCriticConfig()
    z_rl = jnp.zeros((2, config.z_dim), dtype=jnp.float32)
    state = jnp.zeros((2, config.state_dim), dtype=jnp.float32)
    action = jnp.zeros(
        (2, config.action_horizon, config.action_dim),
        dtype=jnp.float32,
    )

    actor_variables = jax.eval_shape(
        rlt.RLTActor(config).init,
        jax.random.key(3),
        z_rl,
        state,
        action,
    )
    critic_variables = jax.eval_shape(
        rlt.RLTCritic(config).init,
        jax.random.key(4),
        z_rl,
        state,
        action,
    )

    assert _parameter_count(actor_variables) == 9_411_904
    assert _parameter_count(critic_variables) == 9_084_929


@pytest.mark.parametrize(
    ("z_shape", "state_shape", "reference_shape", "input_name"),
    [
        ((2, 7), (2, 3), (2, 2, 2), "z_rl"),
        ((2, 8), (2, 4), (2, 2, 2), "state"),
        ((2, 8), (2, 3), (2, 3, 2), "reference"),
        ((2, 8), (2, 3), (2, 2, 3), "reference"),
        ((2, 8), (1, 3), (2, 2, 2), "state"),
    ],
)
def test_actor_shape_errors_report_expected_and_got(
    z_shape,
    state_shape,
    reference_shape,
    input_name,
):
    config = _small_config()
    z_rl = jnp.zeros(z_shape, dtype=jnp.float32)
    state = jnp.zeros(state_shape, dtype=jnp.float32)
    reference = jnp.zeros(reference_shape, dtype=jnp.float32)

    with pytest.raises(ValueError, match=rf"{input_name}.*expected.*got"):
        rlt.RLTActor(config).init(jax.random.key(5), z_rl, state, reference)


def test_critic_action_shape_error_reports_expected_and_got():
    config = _small_config()
    z_rl, state, _ = _inputs(config)
    action = jnp.zeros((2, config.action_horizon + 1, config.action_dim), dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"action.*expected.*got"):
        rlt.RLTCritic(config).init(jax.random.key(6), z_rl, state, action)


def test_batch_one_actor_and_critic_are_jittable():
    config = _small_config()
    z_rl, state, action = _inputs(config, batch_size=1)
    actor = rlt.RLTActor(config)
    critic = rlt.RLTCritic(config)
    actor_variables = actor.init(jax.random.key(7), z_rl, state, action)
    critic_variables = critic.init(jax.random.key(8), z_rl, state, action)

    actor_output = jax.jit(actor.apply)(actor_variables, z_rl, state, action)
    critic_output = jax.jit(critic.apply)(critic_variables, z_rl, state, action)

    assert actor_output.shape == (1, config.action_horizon, config.action_dim)
    assert actor_output.dtype == jnp.float32
    assert critic_output.shape == (1, 1)
    assert critic_output.dtype == jnp.float32
    _assert_finite_tree((actor_output, critic_output))


def test_actor_and_critic_input_gradients_are_finite():
    config = _small_config()
    z_rl, state, action = _inputs(config)
    actor = rlt.RLTActor(config)
    critic = rlt.RLTCritic(config)
    actor_variables = actor.init(jax.random.key(9), z_rl, state, action)
    critic_variables = critic.init(jax.random.key(10), z_rl, state, action)

    def actor_loss(z_value, state_value, reference_value):
        return jnp.sum(actor.apply(actor_variables, z_value, state_value, reference_value))

    def critic_loss(z_value, state_value, action_value):
        return jnp.sum(critic.apply(critic_variables, z_value, state_value, action_value))

    actor_gradients = jax.grad(actor_loss, argnums=(0, 1, 2))(z_rl, state, action)
    critic_gradients = jax.grad(critic_loss, argnums=(0, 1, 2))(z_rl, state, action)

    _assert_finite_tree(actor_gradients)
    _assert_finite_tree(critic_gradients)
