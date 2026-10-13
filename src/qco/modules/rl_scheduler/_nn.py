"""Module A — tiny dependency-free neural-net pieces for the built-in PPO.

CI installs only ``pyyaml`` + ``pytest`` (no numpy, no torch), so the built-in
PPO must run on the standard library alone. Everything here is plain Python;
when numpy happens to be installed, :class:`ScalarMLP` transparently uses it for
the matrix products (same maths, same results to ~1e-12 — see
``tests/test_rl_ppo.py``), which makes real training runs ~10x faster.

* :class:`ScalarMLP` — ``x -> tanh(W1 x + b1) -> w2 . h + b2`` (scalar output),
  evaluated on a *batch* of rows (the policy scores every legal action with
  the same weights, so any number of actions is supported).
* :class:`Adam` — Adam on a flat parameter list, with global-norm clipping.

Parameters live in one flat ``list[float]`` so checkpoints are plain JSON and
optimizers need no knowledge of layer shapes.
"""

from __future__ import annotations

import math
import random
from operator import mul

try:  # optional accelerator
    import numpy as _np

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - exercised in CI (no numpy)
    _np = None
    HAVE_NUMPY = False

BACKENDS = ("auto", "python", "numpy")


def resolve_backend(backend: str) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "numpy" and not HAVE_NUMPY:
        raise ImportError("backend='numpy' requested but numpy is not installed")
    if backend == "auto":
        return "numpy" if HAVE_NUMPY else "python"
    return backend


class ScalarMLP:
    """One-hidden-layer tanh MLP with a scalar output, on batches of rows."""

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        *,
        rng: random.Random | None = None,
        out_scale: float = 0.01,
        theta: list[float] | None = None,
        backend: str = "auto",
    ):
        self.in_dim = in_dim
        self.hidden = hidden
        self.backend = resolve_backend(backend)
        self.n_params = hidden * in_dim + hidden + hidden + 1
        if theta is None:
            rng = rng or random.Random(0)
            s1 = 1.0 / math.sqrt(in_dim)
            theta = [rng.gauss(0.0, s1) for _ in range(hidden * in_dim)]
            theta += [0.0] * hidden
            theta += [rng.gauss(0.0, out_scale) for _ in range(hidden)]
            theta += [0.0]
        if len(theta) != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {len(theta)}")
        self.set_theta(theta)

    # -- parameters --------------------------------------------------------
    def set_theta(self, theta: list[float]) -> None:
        D, H = self.in_dim, self.hidden
        self.theta = list(theta)
        t = self.theta
        self._W1 = [t[i * D:(i + 1) * D] for i in range(H)]
        o = H * D
        self._b1 = t[o:o + H]
        self._w2 = t[o + H:o + 2 * H]
        self._b2 = t[o + 2 * H]
        if self.backend == "numpy":
            self._nW1 = _np.array(self._W1, dtype=float)
            self._nb1 = _np.array(self._b1, dtype=float)
            self._nw2 = _np.array(self._w2, dtype=float)

    # -- forward ------------------------------------------------------------
    def forward(self, X: list[list[float]]):
        """Return ``(y, cache)`` with ``y[k]`` the scalar output for row ``X[k]``."""
        if self.backend == "numpy":
            Xn = _np.asarray(X, dtype=float)
            Hn = _np.tanh(Xn @ self._nW1.T + self._nb1)
            y = Hn @ self._nw2 + self._b2
            return y.tolist(), (Xn, Hn)
        W1, b1, w2, b2 = self._W1, self._b1, self._w2, self._b2
        tanh = math.tanh
        Hs: list[list[float]] = []
        ys: list[float] = []
        for x in X:
            h = [tanh(b + sum(map(mul, row, x))) for row, b in zip(W1, b1)]
            Hs.append(h)
            ys.append(b2 + sum(map(mul, w2, h)))
        return ys, (X, Hs)

    # -- backward ----------------------------------------------------------
    def backward(self, cache, dy: list[float], grad: list[float]) -> None:
        """Accumulate ``d loss / d theta`` into ``grad`` given ``dy[k] = d loss / d y[k]``."""
        D, H = self.in_dim, self.hidden
        o = H * D
        if self.backend == "numpy":
            Xn, Hn = cache
            dyn = _np.asarray(dy, dtype=float)
            dH = (dyn[:, None] * self._nw2[None, :]) * (1.0 - Hn * Hn)
            dW1 = (dH.T @ Xn).reshape(-1)
            db1 = dH.sum(axis=0)
            dw2 = Hn.T @ dyn
            for i, v in enumerate(dW1.tolist()):
                grad[i] += v
            for i, v in enumerate(db1.tolist()):
                grad[o + i] += v
            for i, v in enumerate(dw2.tolist()):
                grad[o + H + i] += v
            grad[o + 2 * H] += float(dyn.sum())
            return
        X, Hs = cache
        w2 = self._w2
        for x, h, d in zip(X, Hs, dy):
            if d == 0.0:
                continue
            grad[o + 2 * H] += d
            for i in range(H):
                hi = h[i]
                grad[o + H + i] += d * hi
                dh = d * w2[i] * (1.0 - hi * hi)
                if dh == 0.0:
                    continue
                grad[o + i] += dh
                base = i * D
                for j in range(D):
                    grad[base + j] += dh * x[j]


class Adam:
    """Adam on a flat parameter list (with optional global-norm gradient clipping)."""

    def __init__(self, n: int, lr: float = 3e-3, betas=(0.9, 0.999), eps: float = 1e-8, max_grad_norm: float | None = 0.5):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.max_grad_norm = max_grad_norm
        self.m = [0.0] * n
        self.v = [0.0] * n
        self.t = 0

    def step(self, theta: list[float], grad: list[float]) -> float:
        """Update ``theta`` in place; returns the (pre-clip) global gradient norm."""
        norm = math.sqrt(sum(g * g for g in grad))
        scale = 1.0
        if self.max_grad_norm is not None and norm > self.max_grad_norm > 0:
            scale = self.max_grad_norm / (norm + 1e-12)
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for i, g in enumerate(grad):
            g *= scale
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g
            theta[i] -= self.lr * (self.m[i] / c1) / (math.sqrt(self.v[i] / c2) + self.eps)
        return norm

    def state(self) -> dict:
        return {"t": self.t, "m": list(self.m), "v": list(self.v)}

    def load_state(self, s: dict) -> None:
        self.t, self.m, self.v = int(s["t"]), list(s["m"]), list(s["v"])


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]
