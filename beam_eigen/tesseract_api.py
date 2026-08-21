# Copyright 2026 zkasuran. SPDX-License-Identifier: Apache-2.0
"""Beam eigensolver Tesseract (black box, finite-difference jacobian).

Free-free Euler-Bernoulli beam of rectangular cross-section with variable
thickness h(x). It assembles Hermite finite-element stiffness and mass matrices
and solves the generalized eigenproblem for the first flexural frequencies.

The eigensolve is a black box: JAX cannot differentiate through
`scipy.linalg.eigh`, and eigenvector gradients are ill-defined near modal
degeneracies. So this component reports a finite-difference jacobian. Composing
that with the analytic-autodiff geometry Tesseract, across this differentiation
boundary, is the whole point of the project.
"""

import numpy as np
from pydantic import BaseModel
from scipy.linalg import eigh

from tesseract_core.runtime import Array, Differentiable, Float64

L = 0.35        # bar length (m)
WIDTH = 0.045   # bar width (m)
E = 1.4e10      # Young's modulus (Pa)
RHO = 800.0     # density (kg/m^3); absolute scale cancels in frequency ratios
N_MODES = 3     # flexural frequencies returned


class InputSchema(BaseModel):
    profile: Differentiable[Array[(None,), Float64]]  # thickness at nodes (m)


class OutputSchema(BaseModel):
    freqs: Differentiable[Array[(N_MODES,), Float64]]  # first flexural freqs (Hz)


def _frequencies(profile: np.ndarray) -> np.ndarray:
    h = np.asarray(profile, dtype=float)
    n_nodes = h.shape[0]
    n_elem = n_nodes - 1
    le = L / n_elem
    ndof = 2 * n_nodes
    kt = np.array(
        [[12.0, 6 * le, -12.0, 6 * le],
         [6 * le, 4 * le**2, -6 * le, 2 * le**2],
         [-12.0, -6 * le, 12.0, -6 * le],
         [6 * le, 2 * le**2, -6 * le, 4 * le**2]]
    )
    mt = np.array(
        [[156.0, 22 * le, 54.0, -13 * le],
         [22 * le, 4 * le**2, 13 * le, -3 * le**2],
         [54.0, 13 * le, 156.0, -22 * le],
         [-13 * le, -3 * le**2, -22 * le, 4 * le**2]]
    )
    K = np.zeros((ndof, ndof))
    M = np.zeros((ndof, ndof))
    for e in range(n_elem):
        he = 0.5 * (h[e] + h[e + 1])
        second_moment = WIDTH * he**3 / 12.0
        area = WIDTH * he
        ke = (E * second_moment / le**3) * kt
        me = (RHO * area * le / 420.0) * mt
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        K[np.ix_(idx, idx)] += ke
        M[np.ix_(idx, idx)] += me
    w2 = np.sort(eigh(K, M, eigvals_only=True))
    # free-free: the first two modes are rigid-body (near zero); drop them
    flex = w2[2 : 2 + N_MODES]
    return np.sqrt(np.maximum(flex, 0.0)) / (2.0 * np.pi)


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(freqs=_frequencies(np.asarray(inputs.profile)))


def _fd_jacobian(h: np.ndarray) -> np.ndarray:
    """Central finite-difference jacobian d(freqs)/d(profile), shape (N_MODES, n).
    This is the numerical derivative of the black-box eigensolve."""
    n = h.shape[0]
    eps = 1e-6 * max(float(np.mean(np.abs(h))), 1e-3)
    jac = np.zeros((N_MODES, n))
    for i in range(n):
        hp = h.copy()
        hp[i] += eps
        hm = h.copy()
        hm[i] -= eps
        jac[:, i] = (_frequencies(hp) - _frequencies(hm)) / (2.0 * eps)
    return jac


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return {"freqs": {"profile": _fd_jacobian(np.asarray(inputs.profile, dtype=float))}}


def jacobian_vector_product(inputs: InputSchema, jvp_inputs, jvp_outputs, tangent_vector):
    jac = _fd_jacobian(np.asarray(inputs.profile, dtype=float))
    return {"freqs": jac @ np.asarray(tangent_vector["profile"], dtype=float)}


def vector_jacobian_product(inputs: InputSchema, vjp_inputs, vjp_outputs, cotangent_vector):
    jac = _fd_jacobian(np.asarray(inputs.profile, dtype=float))
    return {"profile": np.asarray(cotangent_vector["freqs"], dtype=float) @ jac}


def abstract_eval(abstract_inputs):
    # Output is always the first N_MODES flexural frequencies; tesseract-jax uses
    # this for shape inference under jax.grad.
    return {"freqs": {"shape": (N_MODES,), "dtype": "float64"}}
