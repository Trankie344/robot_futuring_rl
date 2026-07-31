"""RL-token encoder-decoder readout for PI0/PI0.5 prefix embeddings."""

from flax import linen as nn
import jax
import jax.numpy as jnp

import openpi.training.sharding as sharding


def _fp32_logits_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    bias: jax.Array | None = None,
    mask: jax.Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: jax.Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    dtype=None,
    precision=None,
    module: nn.Module | None = None,
    force_fp32_for_softmax: bool = False,
) -> jax.Array:
    """Compute QK logits and softmax in FP32, then apply values in their compute dtype."""
    del broadcast_dropout, dropout_rng, deterministic, dtype
    if not force_fp32_for_softmax:
        raise ValueError("_fp32_logits_attention requires force_fp32_for_softmax=True.")
    if bias is not None:
        raise ValueError("_fp32_logits_attention does not support additive attention bias.")
    if dropout_rate != 0.0:
        raise ValueError("_fp32_logits_attention does not support attention-probability dropout.")

    depth = jnp.asarray(query.shape[-1], dtype=jnp.float32)
    scaled_query = query.astype(jnp.float32) / jnp.sqrt(depth)
    key_fp32 = key.astype(jnp.float32)
    logits = jnp.einsum("...qhd,...khd->...hqk", scaled_query, key_fp32, precision=precision)
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(jnp.float32).min)

    attention_weights_fp32 = jax.nn.softmax(logits, axis=-1)
    if module is not None:
        module.sow("intermediates", "attention_weights", attention_weights_fp32)
    attention_weights = attention_weights_fp32.astype(value.dtype)
    output = jnp.einsum("...hqk,...khd->...qhd", attention_weights, value, precision=precision)
    return output.astype(value.dtype)


class TransformerBlock(nn.Module):
    """Small transformer block used by the RL-token readout."""

    width: int
    num_heads: int
    mlp_dim: int | None = None
    dropout: float = 0.0
    compute_dtype: str = "float32"
    decode: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        train: bool = False,  # noqa: FBT001, FBT002 -- preserve the established positional API.
    ) -> jax.Array:
        compute_dtype = jnp.dtype(self.compute_dtype)
        x = sharding.activation_sharding_constraint(x.astype(compute_dtype))

        y = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="pre_attention_norm",
        )(x.astype(jnp.float32))
        y = sharding.activation_sharding_constraint(y.astype(compute_dtype))
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            dropout_rate=0.0,
            deterministic=True,
            attention_fn=_fp32_logits_attention,
            force_fp32_for_softmax=True,
            decode=self.decode,
            name="attention",
        )(y, y, mask=mask)
        y = sharding.activation_sharding_constraint(y.astype(compute_dtype))
        y = nn.Dropout(rate=self.dropout, name="post_attention_dropout")(y, deterministic=not train)
        x = sharding.activation_sharding_constraint((x + y).astype(compute_dtype))

        y = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="pre_mlp_norm",
        )(x.astype(jnp.float32))
        y = y.astype(compute_dtype)
        y = nn.Dense(
            self.mlp_dim or 4 * self.width,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            name="mlp_in",
        )(y)
        y = nn.gelu(y)
        y = nn.Dense(
            self.width,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            name="mlp_out",
        )(y)
        y = sharding.activation_sharding_constraint(y.astype(compute_dtype))
        y = nn.Dropout(rate=self.dropout, name="post_mlp_dropout")(y, deterministic=not train)
        return sharding.activation_sharding_constraint((x + y).astype(compute_dtype))


_RematTransformerBlock = nn.remat(
    TransformerBlock,
    prevent_cse=True,
    static_argnums=(3,),
    policy=jax.checkpoint_policies.nothing_saveable,
)


class PrefixRLTokenAutoencoder(nn.Module):
    """Compresses final-layer VLA prefix tokens into one RL token and reconstructs them."""

    width: int = 2048
    max_prefix_len: int = 1024
    encoder_depth: int = 2
    decoder_depth: int = 2
    num_heads: int = 16
    mlp_dim: int | None = None
    dropout: float = 0.0
    compute_dtype: str = "float32"

    def _check_prefix_shape(self, prefix_out: jax.Array, prefix_mask: jax.Array | None = None) -> None:
        if prefix_out.ndim != 3:
            raise ValueError(f"Expected rank-3 prefix embeddings, got shape {prefix_out.shape}.")
        if prefix_out.shape[-1] != self.width:
            raise ValueError(f"Expected prefix width {self.width}, got {prefix_out.shape[-1]}.")
        if prefix_out.shape[1] > self.max_prefix_len:
            raise ValueError(f"Prefix length {prefix_out.shape[1]} exceeds max_prefix_len={self.max_prefix_len}.")
        if prefix_mask is not None and prefix_mask.shape != prefix_out.shape[:2]:
            raise ValueError(f"Expected prefix mask shape {prefix_out.shape[:2]}, got {prefix_mask.shape}.")

    def _encoder_mask(self, prefix_mask: jax.Array) -> jax.Array:
        batch_size = prefix_mask.shape[0]
        rl_mask = jnp.ones((batch_size, 1), dtype=prefix_mask.dtype)
        valid_keys = jnp.concatenate([prefix_mask, rl_mask], axis=1)
        return valid_keys[:, None, None, :]

    def _decoder_mask(self, prefix_mask: jax.Array) -> jax.Array:
        batch_size, prefix_len = prefix_mask.shape
        key_mask = jnp.concatenate(
            [jnp.ones((batch_size, 1), dtype=prefix_mask.dtype), prefix_mask[:, :-1]],
            axis=1,
        )
        causal = jnp.tril(jnp.ones((prefix_len, prefix_len), dtype=bool))
        return jnp.logical_and(causal[None, None, :, :], key_mask[:, None, None, :])

    @nn.compact
    def encode(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
        train: bool = False,  # noqa: FBT001, FBT002 -- train is positional for remat.
    ) -> jax.Array:
        self._check_prefix_shape(prefix_out, prefix_mask)
        batch_size, prefix_len, _ = prefix_out.shape
        compute_dtype = jnp.dtype(self.compute_dtype)

        rl_token = self.param(
            "rl_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            jnp.float32,
        ).astype(compute_dtype)
        rl_token = jnp.broadcast_to(rl_token, (batch_size, 1, self.width))
        x = jnp.concatenate([prefix_out.astype(compute_dtype), rl_token], axis=1)

        pos_embedding = self.param(
            "encoder_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_prefix_len + 1, self.width),
            jnp.float32,
        ).astype(compute_dtype)
        x = x + pos_embedding[:, : prefix_len + 1, :]
        mask = self._encoder_mask(prefix_mask)

        for i in range(self.encoder_depth):
            x = _RematTransformerBlock(
                width=self.width,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                compute_dtype=self.compute_dtype,
                name=f"encoder_block_{i}",
            )(x, mask, train)
        x = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="encoder_norm",
        )(x.astype(jnp.float32))
        return x.astype(compute_dtype)[:, -1, :]

    @nn.compact
    def decode(
        self,
        z_rl: jax.Array,
        target: jax.Array,
        prefix_mask: jax.Array,
        train: bool = False,  # noqa: FBT001, FBT002 -- train is positional for remat.
    ) -> jax.Array:
        self._check_prefix_shape(target, prefix_mask)
        if z_rl.shape != (target.shape[0], self.width):
            raise ValueError(f"Expected RL token shape {(target.shape[0], self.width)}, got {z_rl.shape}.")
        compute_dtype = jnp.dtype(self.compute_dtype)
        target = target.astype(compute_dtype)
        decoder_input = jnp.concatenate([z_rl.astype(compute_dtype)[:, None, :], target[:, :-1, :]], axis=1)
        pos_embedding = self.param(
            "decoder_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_prefix_len, self.width),
            jnp.float32,
        ).astype(compute_dtype)
        x = decoder_input + pos_embedding[:, : target.shape[1], :]
        mask = self._decoder_mask(prefix_mask)

        for i in range(self.decoder_depth):
            x = _RematTransformerBlock(
                width=self.width,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                compute_dtype=self.compute_dtype,
                name=f"decoder_block_{i}",
            )(x, mask, train)
        x = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="decoder_norm",
        )(x.astype(jnp.float32)).astype(compute_dtype)
        return nn.Dense(
            self.width,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            name="decoder_out_proj",
        )(x).astype(compute_dtype)

    @nn.compact
    def initialize_decode_cache(self, batch_size: int, max_length: int) -> jax.Array:
        if batch_size <= 0:
            raise ValueError(f"Decode cache batch_size must be positive, got {batch_size}.")
        if max_length <= 0 or max_length > self.max_prefix_len:
            raise ValueError(
                f"Decode cache max_length must be in [1, {self.max_prefix_len}], got {max_length}."
            )
        compute_dtype = jnp.dtype(self.compute_dtype)
        dummy = jnp.zeros((batch_size, max_length, self.width), dtype=compute_dtype)
        for i in range(self.decoder_depth):
            TransformerBlock(
                width=self.width,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                compute_dtype=self.compute_dtype,
                decode=True,
                name=f"decoder_block_{i}",
            )(dummy, None, False)  # noqa: FBT003 -- inference uses established positional train API.
        return jnp.asarray(0, dtype=jnp.int32)

    @nn.compact
    def decode_step(self, decoder_input: jax.Array, position: jax.Array) -> jax.Array:
        if decoder_input.ndim != 2 or decoder_input.shape[-1] != self.width:
            raise ValueError(f"Expected decoder input shape [batch, {self.width}], got {decoder_input.shape}.")
        compute_dtype = jnp.dtype(self.compute_dtype)
        position = jnp.asarray(position, dtype=jnp.int32)
        pos_embedding = self.param(
            "decoder_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.max_prefix_len, self.width),
            jnp.float32,
        )
        position_embedding = jax.lax.dynamic_slice(
            pos_embedding,
            (0, position, 0),
            (1, 1, self.width),
        ).astype(compute_dtype)
        x = decoder_input.astype(compute_dtype)[:, None, :] + position_embedding

        for i in range(self.decoder_depth):
            x = TransformerBlock(
                width=self.width,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                compute_dtype=self.compute_dtype,
                decode=True,
                name=f"decoder_block_{i}",
            )(x, None, False)  # noqa: FBT003 -- inference uses established positional train API.
        x = nn.LayerNorm(
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            name="decoder_norm",
        )(x.astype(jnp.float32)).astype(compute_dtype)
        x = nn.Dense(
            self.width,
            dtype=compute_dtype,
            param_dtype=jnp.float32,
            name="decoder_out_proj",
        )(x).astype(compute_dtype)
        return x[:, 0, :]

    @nn.compact
    def reconstruction_loss(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
        train: bool = False,  # noqa: FBT001, FBT002 -- train is positional for remat.
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        self._check_prefix_shape(prefix_out, prefix_mask)
        compute_dtype = jnp.dtype(self.compute_dtype)
        target = jax.lax.stop_gradient(prefix_out).astype(compute_dtype)
        z_rl = self.encode(target, prefix_mask, train)
        pred = self.decode(z_rl, target, prefix_mask, train)

        pred_fp32 = pred.astype(jnp.float32)
        target_fp32 = target.astype(jnp.float32)
        valid = prefix_mask.astype(jnp.float32)
        raw_valid_count = jnp.sum(valid)
        valid_count = jnp.maximum(raw_valid_count, jnp.asarray(1.0, dtype=jnp.float32))
        per_token_loss = jnp.mean(jnp.square(pred_fp32 - target_fp32), axis=-1)
        recon_loss = jnp.sum(per_token_loss * valid) / valid_count

        valid_elements = valid_count * jnp.asarray(self.width, dtype=jnp.float32)
        pred_rms = jnp.sqrt(jnp.sum(jnp.square(pred_fp32) * valid[..., None]) / valid_elements)
        target_rms = jnp.sqrt(jnp.sum(jnp.square(target_fp32) * valid[..., None]) / valid_elements)
        z_rms = jnp.sqrt(jnp.mean(jnp.square(z_rl.astype(jnp.float32))))
        metrics = {
            "rl_token/recon_loss": recon_loss,
            "rl_token/valid_tokens": raw_valid_count,
            "rl_token/z_rms": z_rms,
            "rl_token/pred_rms": pred_rms,
            "rl_token/target_rms": target_rms,
        }
        return recon_loss, metrics

    def __call__(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
        train: bool = False,  # noqa: FBT001, FBT002 -- preserve the positional API.
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        return self.reconstruction_loss(prefix_out, prefix_mask, train)
