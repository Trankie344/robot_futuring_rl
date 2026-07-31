# Positional train arguments exercise Flax remat's static_argnums contract.
# ruff: noqa: FBT003

from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.rl_token import autoencoder as rl_token
import openpi.training.sharding as sharding


def _walk_jaxpr_equations(value):
    if hasattr(value, "jaxpr"):
        yield from _walk_jaxpr_equations(value.jaxpr)
        return
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            yield from _walk_jaxpr_equations(equation.params)
        return
    if isinstance(value, dict):
        nested_values = value.values()
    elif isinstance(value, list | tuple):
        nested_values = value
    else:
        return
    for nested_value in nested_values:
        yield from _walk_jaxpr_equations(nested_value)


def _small_autoencoder_case(*, dropout: float = 0.0):
    inputs = jax.random.normal(jax.random.key(0), (2, 5, 16), dtype=jnp.bfloat16)
    mask = jnp.array(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )
    module = rl_token.PrefixRLTokenAutoencoder(
        width=16,
        max_prefix_len=8,
        encoder_depth=2,
        decoder_depth=2,
        num_heads=4,
        mlp_dim=32,
        dropout=dropout,
        compute_dtype="bfloat16",
    )
    return module, inputs, mask


def _initialize_decode_variables(module, params, *, batch_size: int, max_length: int):
    _, updates = module.apply(
        {"params": params},
        batch_size,
        max_length,
        method=module.initialize_decode_cache,
        mutable=["cache"],
    )
    return {"params": params, "cache": updates["cache"]}


def _cached_decode_with_teacher_inputs(module, variables, z_rl, target):
    predictions = []
    for position in range(target.shape[1]):
        decoder_input = z_rl if position == 0 else target[:, position - 1, :]
        prediction, updates = module.apply(
            variables,
            decoder_input,
            jnp.asarray(position, dtype=jnp.int32),
            method=module.decode_step,
            mutable=["cache"],
        )
        variables = {"params": variables["params"], "cache": updates["cache"]}
        predictions.append(prediction)
    return jnp.stack(predictions, axis=1), variables


def test_prefix_rl_token_autoencoder_uses_bfloat16_compute_fp32_params_loss_and_metrics():
    module, inputs, mask = _small_autoencoder_case()
    variables = module.init(jax.random.key(1), inputs, mask, False)

    z_rl = module.apply(variables, inputs, mask, False, method=module.encode)
    assert z_rl.shape == (2, 16)
    assert z_rl.dtype == jnp.bfloat16

    loss, metrics = module.apply(variables, inputs, mask, False, method=module.reconstruction_loss)
    assert loss.shape == ()
    assert loss.dtype == jnp.float32
    assert set(metrics) == {
        "rl_token/recon_loss",
        "rl_token/valid_tokens",
        "rl_token/z_rms",
        "rl_token/pred_rms",
        "rl_token/target_rms",
    }
    assert all(metric.shape == () for metric in metrics.values())
    assert all(metric.dtype == jnp.float32 for metric in metrics.values())
    assert all(jnp.isfinite(metric) for metric in metrics.values())
    assert metrics["rl_token/valid_tokens"] == jnp.asarray(8.0, dtype=jnp.float32)
    assert jnp.isfinite(loss)

    parameter_leaves = jax.tree_util.tree_leaves(variables["params"])
    assert parameter_leaves
    assert all(parameter.dtype == jnp.float32 for parameter in parameter_leaves)


def test_prefix_rl_token_all_padding_reports_raw_zero_count_and_finite_zero_losses():
    module, inputs, _ = _small_autoencoder_case()
    mask = jnp.zeros(inputs.shape[:2], dtype=bool)
    variables = module.init(jax.random.key(7), inputs, mask, False)

    loss, metrics = module.apply(variables, inputs, mask, False, method=module.reconstruction_loss)

    assert metrics["rl_token/valid_tokens"] == jnp.asarray(0.0, dtype=jnp.float32)
    assert loss == jnp.asarray(0.0, dtype=jnp.float32)
    assert metrics["rl_token/recon_loss"] == jnp.asarray(0.0, dtype=jnp.float32)
    assert metrics["rl_token/pred_rms"] == jnp.asarray(0.0, dtype=jnp.float32)
    assert metrics["rl_token/target_rms"] == jnp.asarray(0.0, dtype=jnp.float32)
    assert all(jnp.isfinite(metric) for metric in metrics.values())


def test_prefix_rl_token_decoder_mask_is_causal_and_excludes_padded_teacher_forcing_keys():
    module, _, _ = _small_autoencoder_case()
    prefix_mask = jnp.array([[True, True, False, False]])

    decoder_mask = module._decoder_mask(prefix_mask)  # noqa: SLF001

    expected = jnp.array(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [True, True, True, False],
                    [True, True, True, False],
                ]
            ]
        ]
    )
    assert jnp.array_equal(decoder_mask, expected)


def test_prefix_rl_token_decoder_is_parallel_causal_teacher_forcing():
    module, target, mask = _small_autoencoder_case()
    variables = module.init(jax.random.key(2), target, mask, False)
    z_rl = module.apply(variables, target, mask, False, method=module.encode)

    changed_target = target.at[:, 3:, :].set((target[:, 3:, :].astype(jnp.float32) + 100.0).astype(jnp.bfloat16))
    pred = module.apply(variables, z_rl, target, mask, False, method=module.decode)
    changed_pred = module.apply(variables, z_rl, changed_target, mask, False, method=module.decode)

    assert pred.shape == target.shape
    assert pred.dtype == jnp.bfloat16
    assert jnp.array_equal(pred[:, :4, :], changed_pred[:, :4, :])


def test_prefix_rl_token_cached_teacher_inputs_match_parallel_decoder():
    module, target, mask = _small_autoencoder_case()
    variables = module.init(jax.random.key(20), target, mask, False)
    z_rl = module.apply(variables, target, mask, False, method=module.encode)
    parallel_prediction = module.apply(variables, z_rl, target, mask, False, method=module.decode)
    decode_variables = _initialize_decode_variables(
        module,
        variables["params"],
        batch_size=target.shape[0],
        max_length=target.shape[1],
    )

    cached_prediction, decode_variables = _cached_decode_with_teacher_inputs(
        module,
        decode_variables,
        z_rl,
        target,
    )

    valid = np.asarray(mask)
    np.testing.assert_allclose(
        np.asarray(cached_prediction, dtype=np.float32)[valid],
        np.asarray(parallel_prediction, dtype=np.float32)[valid],
        rtol=3e-2,
        atol=3e-2,
    )
    flattened_cache = traverse_util.flatten_dict(decode_variables["cache"])
    cache_indices = [value for path, value in flattened_cache.items() if path[-1] == "cache_index"]
    assert len(cache_indices) == module.decoder_depth
    assert all(np.asarray(cache_index) == target.shape[1] for cache_index in cache_indices)


def test_prefix_rl_token_decode_cache_does_not_change_parameter_topology():
    module, target, mask = _small_autoencoder_case()
    variables = module.init(jax.random.key(21), target, mask, False)
    params_before = traverse_util.flatten_dict(variables["params"])

    decode_variables = _initialize_decode_variables(
        module,
        variables["params"],
        batch_size=target.shape[0],
        max_length=target.shape[1],
    )
    params_after = traverse_util.flatten_dict(decode_variables["params"])

    assert params_after.keys() == params_before.keys()
    for path, parameter_before in params_before.items():
        parameter_after = params_after[path]
        assert parameter_after.shape == parameter_before.shape
        assert parameter_after.dtype == parameter_before.dtype
        assert np.array_equal(np.asarray(parameter_after), np.asarray(parameter_before))


@pytest.mark.parametrize(("batch_size", "max_length"), [(0, 5), (2, 0), (2, 9)])
def test_prefix_rl_token_decode_cache_rejects_invalid_shape(batch_size, max_length):
    module, target, mask = _small_autoencoder_case()
    variables = module.init(jax.random.key(22), target, mask, False)

    with pytest.raises(ValueError, match=r"batch_size|max_length"):
        _initialize_decode_variables(
            module,
            variables["params"],
            batch_size=batch_size,
            max_length=max_length,
        )


def test_prefix_rl_token_reconstruction_stops_gradient_at_prefix_boundary():
    module, inputs, mask = _small_autoencoder_case()
    variables = module.init(jax.random.key(3), inputs, mask, False)

    def loss_fn(prefix):
        loss, _ = module.apply(variables, prefix, mask, False, method=module.reconstruction_loss)
        return loss

    prefix_gradient = jax.grad(loss_fn)(inputs)
    assert jnp.array_equal(prefix_gradient, jnp.zeros_like(prefix_gradient))


def test_prefix_rl_token_autoencoder_remats_each_block_and_constrains_activations():
    module, inputs, mask = _small_autoencoder_case(dropout=0.1)
    variables = module.init(
        {"params": jax.random.key(4), "dropout": jax.random.key(5)},
        inputs,
        mask,
        True,
    )
    mesh = jax.make_mesh((1, 1), (sharding.BATCH_AXIS, sharding.FSDP_AXIS))

    def loss_fn(prefix, prefix_mask, dropout_key):
        loss, _ = module.apply(
            variables,
            prefix,
            prefix_mask,
            True,
            method=module.reconstruction_loss,
            rngs={"dropout": dropout_key},
        )
        return loss

    with sharding.set_mesh(mesh):
        apply_jaxpr = jax.make_jaxpr(loss_fn)(inputs, mask, jax.random.key(6))

    equations = list(_walk_jaxpr_equations(apply_jaxpr))
    remat_equations = [equation for equation in equations if equation.primitive.name in {"remat", "remat2"}]
    sharding_constraints = [equation for equation in equations if equation.primitive.name == "sharding_constraint"]
    assert len(remat_equations) == 4, [equation.primitive.name for equation in remat_equations]
    assert all(equation.params["prevent_cse"] is True for equation in remat_equations)
    expected_policy = jax.checkpoint_policies.nothing_saveable
    assert all(
        equation.params["policy"] is expected_policy
        or getattr(equation.params["policy"], "__name__", None) == getattr(expected_policy, "__name__", None)
        for equation in remat_equations
    )
    assert len(sharding_constraints) >= 24

    def parameter_loss(params):
        loss, _ = module.apply(
            {"params": params},
            inputs,
            mask,
            False,
            method=module.reconstruction_loss,
        )
        return loss

    with sharding.set_mesh(mesh):
        grad_jaxpr = jax.make_jaxpr(jax.grad(parameter_loss))(variables["params"])

    grad_remat_equations = [
        equation
        for equation in _walk_jaxpr_equations(grad_jaxpr)
        if equation.primitive.name in {"remat", "remat2"} and equation.params.get("differentiated") is True
    ]
    assert len(grad_remat_equations) == 4


def test_fp32_logits_attention_uses_fp32_logits_and_bfloat16_values():
    query_key, key_key, value_key = jax.random.split(jax.random.key(1), 3)
    query = jax.random.normal(query_key, (2, 3, 4, 4), dtype=jnp.bfloat16)
    key = jax.random.normal(key_key, (2, 3, 4, 4), dtype=jnp.bfloat16)
    value = jax.random.normal(value_key, (2, 3, 4, 4), dtype=jnp.bfloat16)
    mask = jnp.array(
        [
            [[[True, True, False], [True, True, True], [False, True, True]]],
            [[[True, False, False], [True, True, False], [True, True, True]]],
        ]
    )

    def attention_fn(q, k, v):
        return rl_token._fp32_logits_attention(  # noqa: SLF001
            q,
            k,
            v,
            mask=mask,
            force_fp32_for_softmax=True,
        )

    with pytest.raises(ValueError, match="force_fp32_for_softmax"):
        rl_token._fp32_logits_attention(query, key, value)  # noqa: SLF001

    output = attention_fn(query, key, value)
    assert output.dtype == jnp.bfloat16

    jaxpr = jax.make_jaxpr(attention_fn)(query, key, value).jaxpr
    dot_generals = [equation for equation in jaxpr.eqns if equation.primitive.name == "dot_general"]
    assert len(dot_generals) == 2
    assert [variable.aval.dtype for variable in dot_generals[0].invars[:2]] == [jnp.float32, jnp.float32]
    assert [variable.aval.dtype for variable in dot_generals[1].invars[:2]] == [jnp.bfloat16, jnp.bfloat16]

    softmax_exps = [equation for equation in jaxpr.eqns if equation.primitive.name == "exp"]
    assert softmax_exps
    assert all(variable.aval.dtype == jnp.float32 for equation in softmax_exps for variable in equation.outvars)


def test_fp32_logits_attention_mask_excludes_changed_high_score_value():
    query = jnp.ones((1, 1, 1, 1), dtype=jnp.bfloat16)
    mask = jnp.array([[[[True, False]]]])
    key_a = jnp.array([[[[0.0]], [[100.0]]]], dtype=jnp.bfloat16)
    value_a = jnp.array([[[[3.0]], [[1000.0]]]], dtype=jnp.bfloat16)
    key_b = jnp.array([[[[0.0]], [[80.0]]]], dtype=jnp.bfloat16)
    value_b = jnp.array([[[[3.0]], [[-800.0]]]], dtype=jnp.bfloat16)

    output_a = rl_token._fp32_logits_attention(  # noqa: SLF001
        query,
        key_a,
        value_a,
        mask=mask,
        force_fp32_for_softmax=True,
    )
    output_b = rl_token._fp32_logits_attention(  # noqa: SLF001
        query,
        key_b,
        value_b,
        mask=mask,
        force_fp32_for_softmax=True,
    )

    expected = jnp.array([[[[3.0]]]], dtype=jnp.bfloat16)
    assert jnp.array_equal(output_a, expected)
    assert jnp.array_equal(output_b, expected)


def test_transformer_block_apply_wires_fp32_qk_and_bfloat16_value_attention():
    input_key, init_key = jax.random.split(jax.random.key(6))
    inputs = jax.random.normal(input_key, (1, 3, 8), dtype=jnp.bfloat16)
    mask = jnp.ones((1, 1, 3, 3), dtype=bool)
    block = rl_token.TransformerBlock(
        width=8,
        num_heads=2,
        mlp_dim=16,
        compute_dtype="bfloat16",
    )
    variables = block.init(init_key, inputs, mask, train=False)

    apply_jaxpr = jax.make_jaxpr(lambda x: block.apply(variables, x, mask, train=False))(inputs)
    dot_generals = [
        equation for equation in _walk_jaxpr_equations(apply_jaxpr) if equation.primitive.name == "dot_general"
    ]
    signatures = [
        (
            tuple(tuple(variable.aval.shape) for variable in equation.invars[:2]),
            tuple(variable.aval.dtype for variable in equation.invars[:2]),
        )
        for equation in dot_generals
    ]

    qk_shape = (1, 3, 2, 4)
    probability_shape = (1, 2, 3, 3)
    value_shape = (1, 3, 2, 4)
    assert any(
        shapes == (qk_shape, qk_shape) and dtypes == (jnp.float32, jnp.float32) for shapes, dtypes in signatures
    ), signatures
    assert any(
        set(shapes) == {probability_shape, value_shape} and dtypes == (jnp.bfloat16, jnp.bfloat16)
        for shapes, dtypes in signatures
    ), signatures


def test_transformer_block_uses_fp32_parameters_and_bfloat16_compute():
    input_key, init_key = jax.random.split(jax.random.key(2))
    inputs = jax.random.normal(input_key, (2, 5, 16), dtype=jnp.bfloat16)
    mask = jnp.ones((2, 1, 5, 5), dtype=bool)
    block = rl_token.TransformerBlock(
        width=16,
        num_heads=4,
        mlp_dim=32,
        dropout=0.1,
        compute_dtype="bfloat16",
    )

    variables = block.init(init_key, inputs, mask, train=False)
    parameter_leaves = jax.tree_util.tree_leaves(variables["params"])
    assert parameter_leaves
    assert all(parameter.dtype == jnp.float32 for parameter in parameter_leaves)

    output = block.apply(variables, inputs, mask, train=False)
    assert output.dtype == jnp.bfloat16


def test_transformer_block_dropout_is_train_only():
    input_key, init_key = jax.random.split(jax.random.key(3))
    inputs = jax.random.normal(input_key, (2, 5, 16), dtype=jnp.bfloat16)
    mask = jnp.ones((2, 1, 5, 5), dtype=bool)
    block = rl_token.TransformerBlock(
        width=16,
        num_heads=4,
        mlp_dim=32,
        dropout=0.1,
        compute_dtype="bfloat16",
    )
    variables = block.init(init_key, inputs, mask, train=False)

    eval_output_a = block.apply(variables, inputs, mask, train=False)
    eval_output_b = block.apply(variables, inputs, mask, train=False)
    assert jax.device_get(eval_output_a).tobytes() == jax.device_get(eval_output_b).tobytes()

    train_output_a = block.apply(
        variables,
        inputs,
        mask,
        train=True,
        rngs={"dropout": jax.random.key(4)},
    )
    train_output_b = block.apply(
        variables,
        inputs,
        mask,
        train=True,
        rngs={"dropout": jax.random.key(5)},
    )
    assert not jnp.array_equal(train_output_a, train_output_b)
