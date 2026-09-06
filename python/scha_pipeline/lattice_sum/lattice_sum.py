"""Real-space lattice sum for the short-range dipolar kernel Q_SR_{ab}(R; eta)
and its quadratic (k^2) small-k expansion, for a simple cubic Bravais lattice.

By cubic symmetry, the quadratic-order expansion of
sum_{R!=0} Q_SR_{ab}(R; eta) e^{ik.R} produces exactly three invariants:
k^2 delta_ab, k_a k_b, and delta_ab k_a^2 (no sum on a).
"""

import numpy as np
from scipy.special import erfc


def shell_vectors(a0, n):
    """Lattice vectors R = a0*(i, j, k) on the surface of the cubic shell
    max(|i|, |j|, |k|) == n (n >= 1), for a simple cubic lattice of
    parameter a0."""
    idx = np.arange(-n, n + 1)
    I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
    on_shell = (np.abs(I) == n) | (np.abs(J) == n) | (np.abs(K) == n)
    return np.stack([I[on_shell], J[on_shell], K[on_shell]], axis=-1) * a0


def quadratic_invariants(a0, eta, tol=1e-4, n_max=200):
    """Sum, shell by shell, the rank-4 lattice tensor
    T_{ab,xd} = sum_{R!=0} Q_SR_{ab}(R; eta) R_x R_d
    and return its three cubic invariants (a, b, c) defined by

        T_{ab,xd} k_x k_d = a k^2 delta_ab + 2 b k_a k_b + c delta_ab k_a^2 (no sum).

    Only the (xx, yy, xy) components of Q_SR are needed to read off
    (a, b, c): a = T_yy,xx, b = T_xy,xy, c = T_xx,xx - a - 2b.

    Stops once each of (a, b, c) changes by less than `tol` (relative)
    between two successive shells; the last shell-to-shell increment is
    returned as an error estimate on (a, b, c).
    """
    a = b = txxxx = 0.0
    prev = np.full(3, np.nan)
    for n in range(1, n_max + 1):
        R = shell_vectors(a0, n)
        Rn = np.linalg.norm(R, axis=1)
        xh, yh = R[:, 0] / Rn, R[:, 1] / Rn
        gauss = np.exp(-(eta * Rn) ** 2)
        bracket = erfc(eta * Rn) + 2 * eta * Rn / np.sqrt(np.pi) * gauss

        Qxx = (1 - 3 * xh**2) / Rn**3 * bracket - 4 * eta**3 / np.sqrt(np.pi) * xh**2 * gauss
        Qyy = (1 - 3 * yh**2) / Rn**3 * bracket - 4 * eta**3 / np.sqrt(np.pi) * yh**2 * gauss
        Qxy = -3 * xh * yh / Rn**3 * bracket - 4 * eta**3 / np.sqrt(np.pi) * xh * yh * gauss

        a += np.sum(Qyy * R[:, 0] ** 2)
        b += np.sum(Qxy * R[:, 0] * R[:, 1])
        txxxx += np.sum(Qxx * R[:, 0] ** 2)

        current = np.array([a, b, txxxx])
        if n > 1 and np.all(np.abs(current - prev) < tol * np.maximum(np.abs(current), 1e-30)):
            c = txxxx - a - 2 * b
            err_a, err_b, err_txxxx = np.abs(current - prev)
            err_c = err_txxxx + err_a + 2 * err_b
            return a, b, c, n, (err_a, err_b, err_c)
        prev = current
    raise RuntimeError(f"quadratic_invariants: not converged after n_max={n_max} shells")


def expansion_coefficients(a0, eta, tol=1e-4, n_max=200):
    """Coefficients of the three small-k cubic invariants of the
    short-range kernel -- k^2 delta_ab, k_a k_b, delta_ab k_a^2 (no sum) --
    obtained from the quadratic term -1/2 sum_{R!=0} Q_SR_ab(R) (k.R)^2.
    Each coefficient is returned together with its absolute numerical
    error estimate (last-shell increment that triggered convergence)."""
    a, b, c, n, (err_a, err_b, err_c) = quadratic_invariants(a0, eta, tol=tol, n_max=n_max)
    return {
        "k2_delta": -a / 2,
        "k_a k_b": -b,
        "delta_k_a2": -c / 2,
        "err_k2_delta": err_a / 2,
        "err_k_a k_b": err_b,
        "err_delta_k_a2": err_c / 2,
        "n_shells": n,
    }
