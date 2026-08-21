# Copyright 2026 zkasuran. SPDX-License-Identifier: Apache-2.0
"""Geometry Tesseract (JAX, analytic autodiff).

Maps a few symmetric design coefficients to a smooth, positive, symmetric
thickness profile: the marimba-bar undercut. It is differentiated analytically
by JAX (jvp/vjp), then composed across a real boundary with the
finite-difference beam eigensolver, so end-to-end gradients reach the design
coefficients even though the physics solver on the other side is a black box.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

N_NODES = 25      # thickness samples along the bar (matches the eigensolver)
H0 = 0.020        # base thickness (m)
H_MIN = 0.004     # minimum thickness at the deepest undercut (m)


class InputSchema(BaseModel):
    coeffs: Differentiable[Array[(None,), Float64]]  # symmetric undercut coefficients


class OutputSchema(BaseModel):
    profile: Differentiable[Array[(N_NODES,), Float64]]  # thickness at nodes (m)


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    c = inputs["coeffs"]
    x = jnp.linspace(0.0, 1.0, N_NODES)
    ks = jnp.arange(1, c.shape[0] + 1)
    # cosine bases that vanish at the bar ends, so the struck tips keep full thickness
    basis = 1.0 - jnp.cos(2.0 * jnp.pi * ks[:, None] * x[None, :])  # (K, N_NODES)
    depth = jnp.sum(jax.nn.softplus(c)[:, None] * basis, axis=0)    # >= 0
    profile = H_MIN + (H0 - H_MIN) * jnp.exp(-depth)               # in (H_MIN, H0]
    return {"profile": profile}


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
