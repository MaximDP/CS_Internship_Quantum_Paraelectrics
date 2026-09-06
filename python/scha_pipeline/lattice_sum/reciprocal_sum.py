"""Reciprocal-space lattice sum for the long-range (LRP) part of the
Ewald-split dipolar kernel, and its quadratic (k^2) small-k expansion, for
a simple cubic Bravais lattice of parameter a0 (reciprocal lattice spacing
2*pi/a0).

The six (denominator order, exponential order, projector order) rows of
the k^2 expansion of (4*pi/Omega_0) sum_{G!=0} (k+G)_a(k+G)_b/|k+G|^2
exp(-|k+G|^2/4eta^2) all reduce, by cubic symmetry of the reciprocal
lattice, to combinations of three scalar/tensor lattice sums:

    S0(p)   = sum_{G!=0} (1/G^2p)              exp(-G^2/4eta^2)
    D(p)    = sum_{G!=0} (G_a G_b/G^2p)         exp(-G^2/4eta^2) = D(p) delta_ab
    (a,c)(p)= sum_{G!=0} (G_a G_b G_x G_d/G^2p) exp(-G^2/4eta^2)
            = a(p)[delta_ab delta_xd + delta_ax delta_bd + delta_ad delta_bx] + c(p) delta_abxd

(the rank-4 sum here is built from the fully index-symmetric product
G_aG_bG_xG_d, so its two off-diagonal cubic invariants coincide, i.e.
b(p) = a(p) -- this is checked implicitly by computing both from the
lattice sum). Contracting each row with k_x k_d and summing over the six
rows gives the LR contribution to the three final invariants.
"""

import numpy as np


def reciprocal_shell_vectors(a0, n):
    """Reciprocal lattice vectors G = (2*pi/a0)*(i, j, k) on the surface of
    the cubic shell max(|i|, |j|, |k|) == n (n >= 1)."""
    b0 = 2 * np.pi / a0
    idx = np.arange(-n, n + 1)
    I, J, K = np.meshgrid(idx, idx, idx, indexing="ij")
    on_shell = (np.abs(I) == n) | (np.abs(J) == n) | (np.abs(K) == n)
    return np.stack([I[on_shell], J[on_shell], K[on_shell]], axis=-1) * b0


def lrp_quadratic_invariants(a0, eta, tol=1e-4, n_max=400):
    """Sum, shell by shell over G != 0, the nine lattice sums needed by the
    six (denom, exp, projector) rows, and combine them into the three
    cubic invariants of the LR (reciprocal-space) contribution to the
    quadratic-order expansion of Q_alpha,beta^latt(k):

        k^2 delta_ab,   k_a k_b,   delta_ab k_a^2 (no sum),

    already multiplied by the 4*pi/Omega_0 prefactor (Omega_0 = a0**3).
    """
    names = ["S0_1", "D_1", "D_2", "a_1", "m_1", "a_2", "m_2", "a_3", "m_3"]
    sums = {name: 0.0 for name in names}
    prev = None
    for n in range(1, n_max + 1):
        G = reciprocal_shell_vectors(a0, n)
        G2 = np.sum(G**2, axis=1)
        w = np.exp(-G2 / (4 * eta**2))
        Gx2, Gy2 = G[:, 0] ** 2, G[:, 1] ** 2

        sums["S0_1"] += np.sum(w / G2)
        sums["D_1"] += np.sum(Gx2 / G2 * w)
        sums["D_2"] += np.sum(Gx2 / G2**2 * w)
        sums["a_1"] += np.sum(Gx2 * Gy2 / G2 * w)
        sums["m_1"] += np.sum(Gx2**2 / G2 * w)
        sums["a_2"] += np.sum(Gx2 * Gy2 / G2**2 * w)
        sums["m_2"] += np.sum(Gx2**2 / G2**2 * w)
        sums["a_3"] += np.sum(Gx2 * Gy2 / G2**3 * w)
        sums["m_3"] += np.sum(Gx2**2 / G2**3 * w)

        current = np.array([sums[name] for name in names])
        if prev is not None and np.all(np.abs(current - prev) < tol * np.maximum(np.abs(current), 1e-30)):
            increment = dict(zip(names, np.abs(current - prev)))
            break
        prev = current
    else:
        raise RuntimeError(f"lrp_quadratic_invariants: not converged after n_max={n_max} shells")

    def invariants_from(s):
        S0_1, D_1, D_2 = s["S0_1"], s["D_1"], s["D_2"]
        a_1, c_1 = s["a_1"], s["m_1"] - 3 * s["a_1"]
        a_2, c_2 = s["a_2"], s["m_2"] - 3 * s["a_2"]
        a_3, c_3 = s["a_3"], s["m_3"] - 3 * s["a_3"]
        # Row (0,2,0): -D_1/(4eta^2) k^2 delta_ab + a_1/(8eta^4) [k^2 delta + 2 k_a k_b + delta k_a^2]
        # Row (1,1,0): +a_2/eta^2 [k^2 delta + 2 k_a k_b + delta k_a^2]
        #   (corrected sign: d_1 = -2k.G/G^4 and e_1 = -k.G/2eta^2 -> d_1*e_1 > 0)
        # Row (0,1,1): -D_1/eta^2 k_a k_b
        # Row (2,0,0): -D_2 k^2 delta_ab
        # Row (1,0,1): -4 D_2 k_a k_b
        # Row (0,0,2): S0_1 k_a k_b + 4 a_3 [k^2 delta + 2 k_a k_b + delta k_a^2]
        k2d = -D_1 / (4 * eta**2) + a_1 / (8 * eta**4) + a_2 / eta**2 - D_2 + 4 * a_3
        kakb = a_1 / (4 * eta**4) + 2 * a_2 / eta**2 - D_1 / eta**2 - 4 * D_2 + S0_1 + 8 * a_3
        dk2 = c_1 / (8 * eta**4) + c_2 / eta**2 + 4 * c_3
        return np.array([k2d, kakb, dk2])

    central = invariants_from(sums)
    # Conservative (triangle-inequality) error bound: sum |coefficient| x
    # (last shell's increment on the corresponding raw sum).
    inc = increment
    err_c1 = inc["m_1"] + 3 * inc["a_1"]
    err_c2 = inc["m_2"] + 3 * inc["a_2"]
    err_c3 = inc["m_3"] + 3 * inc["a_3"]
    err_k2d = inc["D_1"] / (4 * eta**2) + inc["a_1"] / (8 * eta**4) + inc["a_2"] / eta**2 + inc["D_2"] + 4 * inc["a_3"]
    err_kakb = (
        inc["a_1"] / (4 * eta**4) + 2 * inc["a_2"] / eta**2 + inc["D_1"] / eta**2
        + 4 * inc["D_2"] + inc["S0_1"] + 8 * inc["a_3"]
    )
    err_dk2 = err_c1 / (8 * eta**4) + err_c2 / eta**2 + 4 * err_c3
    err_bound = np.array([err_k2d, err_kakb, err_dk2])

    omega0 = a0**3
    prefactor = 4 * np.pi / omega0
    return {
        "k2_delta": prefactor * central[0],
        "k_a k_b": prefactor * central[1],
        "delta_k_a2": prefactor * central[2],
        "err_k2_delta": prefactor * err_bound[0],
        "err_k_a k_b": prefactor * err_bound[1],
        "err_delta_k_a2": prefactor * err_bound[2],
        "n_shells": n,
    }


def g0_quadratic_correction(a0, eta):
    """Quadratic-order (analytic) correction coming from Taylor-expanding
    the Gaussian factor of the non-analytic G=0 reciprocal-space term,
    (4*pi/Omega_0) (k_a k_b/k^2) exp(-k^2/4eta^2) =
    (4*pi/Omega_0) k_a k_b/k^2 - (pi/(Omega_0 eta^2)) k_a k_b + O(k^4).

    The leading 1/k^2 piece is the non-analytic long-range (depolarizing
    field) term and is kept separate; only the second, analytic, k^2-order
    term -pi/(Omega_0 eta^2) k_a k_b is returned here. It contributes
    purely to the k_a k_b invariant.
    """
    omega0 = a0**3
    return {"k2_delta": 0.0, "k_a k_b": -np.pi / (omega0 * eta**2), "delta_k_a2": 0.0}
