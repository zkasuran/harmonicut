# Copyright 2026 zkasuran. SPDX-License-Identifier: Apache-2.0
"""Visuals for the tuned-bar inverse design: the undercut morphing, the first
three mode shapes, the partials sliding onto the target harmonic ratios, and the
loss curve. Reads artifacts/history.json produced by optimize.py."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from scipy.linalg import eigh

L, WIDTH, E, RHO, N_MODES = 0.35, 0.045, 1.4e10, 800.0, 3
ACCENT, MUTED, TARGETC = "#0d6e4e", "#93a4b3", "#d9534f"
ART = Path(__file__).resolve().parent.parent / "artifacts"


def assemble(h):
    n = h.shape[0]
    ne = n - 1
    le = L / ne
    kt = np.array([[12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
                   [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]], float)
    mt = np.array([[156, 22*le, 54, -13*le], [22*le, 4*le**2, 13*le, -3*le**2],
                   [54, 13*le, 156, -22*le], [-13*le, -3*le**2, -22*le, 4*le**2]], float)
    K = np.zeros((2*n, 2*n))
    M = np.zeros((2*n, 2*n))
    for e in range(ne):
        he = 0.5 * (h[e] + h[e+1])
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(idx, idx)] += (E * WIDTH * he**3 / 12.0 / le**3) * kt
        M[np.ix_(idx, idx)] += (RHO * WIDTH * he * le / 420.0) * mt
    return K, M


def mode_shapes(h):
    w2, vecs = eigh(*assemble(h))
    order = np.argsort(w2)
    sel = order[2:2+N_MODES]
    freqs = np.sqrt(np.maximum(w2[sel], 0.0)) / (2*np.pi)
    shapes = vecs[0::2][:, sel]  # transverse DOFs
    return freqs, shapes


def main():
    hist = json.loads((ART / "history.json").read_text())
    first, last = hist[0], hist[-1]
    x = np.linspace(0, L, len(last["profile"])) * 100  # cm
    steps = [h["step"] for h in hist]
    r2 = [h["ratios"][0] for h in hist]
    r3 = [h["ratios"][1] for h in hist]
    loss = [h["loss"] for h in hist]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Tuned marimba bar via composed Tesseracts (autodiff geometry + FD eigensolve)",
                 fontsize=13, fontweight="bold")

    a = ax[0, 0]
    for prof, c, lab in ((first["profile"], MUTED, "start (straight)"), (last["profile"], ACCENT, "tuned undercut")):
        p = np.asarray(prof) * 100
        a.fill_between(x, 0, -p, color=c, alpha=0.5, label=lab)
    a.set_title("Bar side profile: undercut carved by the optimiser")
    a.set_xlabel("position (cm)"); a.set_ylabel("thickness (cm, underside)"); a.legend(loc="lower center")

    a = ax[0, 1]
    _, shapes = mode_shapes(np.asarray(last["profile"]))
    xm = np.linspace(0, L, shapes.shape[0]) * 100
    for k in range(N_MODES):
        s = shapes[:, k]; s = s / np.max(np.abs(s))
        a.plot(xm, s + 0, label=f"mode {k+1} ({last['freqs'][k]:.0f} Hz)")
    a.axhline(0, color=MUTED, lw=0.6)
    a.set_title("First three flexural mode shapes (tuned bar)")
    a.set_xlabel("position (cm)"); a.legend(loc="upper right", fontsize=8)

    a = ax[1, 0]
    a.plot(steps, r2, color=ACCENT, label="f2/f1")
    a.plot(steps, r3, color="#4c72b0", label="f3/f1")
    a.axhline(4, color=TARGETC, ls="--", lw=1); a.axhline(10, color=TARGETC, ls="--", lw=1, label="targets 4, 10")
    a.set_title(f"Partials tuned toward 1:4:10  (end {r2[-1]:.2f} : {r3[-1]:.2f})")
    a.set_xlabel("optimisation step"); a.set_ylabel("frequency ratio"); a.legend(loc="center right")

    a = ax[1, 1]
    a.semilogy(steps, loss, color=ACCENT)
    a.set_title("Objective ||ratios - target||^2 (end-to-end gradient descent)")
    a.set_xlabel("optimisation step"); a.set_ylabel("loss (log)")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ART / "summary.png", dpi=150)
    print("saved", ART / "summary.png")

    # morph animation: profile + a moving dot on the ratio track
    figA, (pax, rax) = plt.subplots(1, 2, figsize=(11, 4))
    def draw(i):
        h = hist[i]
        pax.clear(); rax.clear()
        p = np.asarray(h["profile"]) * 100
        pax.fill_between(x, 0, -p, color=ACCENT, alpha=0.6)
        pax.set_ylim(-2.2, 0.2); pax.set_title(f"undercut  (step {h['step']})")
        pax.set_xlabel("position (cm)"); pax.set_ylabel("thickness (cm)")
        rax.plot(steps, r2, color=ACCENT, lw=1); rax.plot(steps, r3, color="#4c72b0", lw=1)
        rax.axhline(4, color=TARGETC, ls="--", lw=1); rax.axhline(10, color=TARGETC, ls="--", lw=1)
        rax.plot(h["step"], h["ratios"][0], "o", color=ACCENT)
        rax.plot(h["step"], h["ratios"][1], "o", color="#4c72b0")
        rax.set_title(f"partials  {h['ratios'][0]:.2f} : {h['ratios'][1]:.2f}  -> 4 : 10")
        rax.set_xlabel("step"); rax.set_ylabel("ratio")
    writer = PillowWriter(fps=10)
    with writer.saving(figA, str(ART / "morph.gif"), dpi=90):
        for i in range(len(hist)):
            draw(i); writer.grab_frame()
    print("saved", ART / "morph.gif")


if __name__ == "__main__":
    main()
