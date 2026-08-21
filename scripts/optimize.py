# Copyright 2026 zkasuran. SPDX-License-Identifier: Apache-2.0
"""Compose the geometry and beam-eigensolver Tesseracts into one differentiable
pipeline, then inverse-design a marimba-bar undercut so its first three flexural
partials land on the target harmonic ratios (1 : 4 : 10).

Gradients flow end to end across a real boundary: analytic JAX autodiff in the
geometry Tesseract, finite differences through the black-box eigensolve in the
beam Tesseract, stitched by tesseract-jax. Neither side could do this alone.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)  # schemas use Float64

TARGET = jnp.array([4.0, 10.0])  # target f2/f1, f3/f1 (marimba tuning)
STEPS = 400
ART = Path(__file__).resolve().parent.parent / "artifacts"
ART.mkdir(exist_ok=True)


def main() -> None:
    with Tesseract.from_image("geometry") as geo, Tesseract.from_image("beam_eigen") as beam:

        def pipeline(coeffs):
            profile = apply_tesseract(geo, {"coeffs": coeffs})["profile"]
            freqs = apply_tesseract(beam, {"profile": profile})["freqs"]
            return freqs, profile

        def loss(coeffs):
            freqs, _ = pipeline(coeffs)
            ratios = freqs[1:] / freqs[0]
            return jnp.sum((ratios - TARGET) ** 2)

        coeffs0 = jnp.full(6, -2.0)  # start near a straight bar (minimal undercut)

        # Gradient across the boundary: jax.grad (via Tesseract endpoints) vs a
        # global finite-difference check over the whole composed pipeline.
        g = np.asarray(jax.grad(loss)(coeffs0))
        eps = 1e-4
        gfd = np.zeros_like(np.asarray(coeffs0))
        for i in range(coeffs0.shape[0]):
            gfd[i] = float(
                (loss(coeffs0.at[i].add(eps)) - loss(coeffs0.at[i].add(-eps))) / (2 * eps)
            )
        print("grad (tesseract-jax):", np.round(g, 4))
        print("grad (global FD)    :", np.round(gfd, 4))
        rel = np.max(np.abs(g - gfd)) / (np.max(np.abs(gfd)) + 1e-12)
        print(f"max abs diff {np.max(np.abs(g - gfd)):.3e}   max rel diff {rel:.3e}")

        # Inverse design.
        opt = optax.adam(5e-2)
        state = opt.init(coeffs0)
        coeffs = coeffs0
        vg = jax.value_and_grad(loss)
        hist = []
        for step in range(STEPS):
            val, grad = vg(coeffs)
            updates, state = opt.update(grad, state)
            coeffs = optax.apply_updates(coeffs, updates)
            if step % 5 == 0 or step == STEPS - 1:
                freqs, profile = pipeline(coeffs)
                ratios = np.asarray(freqs[1:] / freqs[0])
                hist.append({
                    "step": step,
                    "loss": float(val),
                    "ratios": ratios.tolist(),
                    "freqs": np.asarray(freqs).tolist(),
                    "profile": np.asarray(profile).tolist(),
                    "coeffs": np.asarray(coeffs).tolist(),
                })
                print(f"step {step:3d}  loss {float(val):.5f}  ratios {np.round(ratios, 3)}")

        (ART / "history.json").write_text(json.dumps(hist, indent=2))
        print(f"\nsaved {ART/'history.json'}  ({len(hist)} snapshots)")
        print("final ratios:", np.round(hist[-1]["ratios"], 4), " target [4, 10]")


if __name__ == "__main__":
    main()
