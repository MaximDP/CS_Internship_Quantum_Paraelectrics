#!/usr/bin/env python3
"""Paraelectric SCHA solver for the mass equation in rapportCS.tex.

The coefficient sets are documented in the material appendices of
``rapportCS_stage_maxime/rapportCS.tex``.  Nishimatsu BaTiO3 is the default set; the 2016
Nishimatsu SrTiO3 set is available as ``nishimatsu_sto``.  Both include the
sixth- and eighth-order SCHA corrections.
Vanderbilt/ZVR remains available, with zero sixth-order coefficients.

We solve a local-self-energy closure,

    A1(T) = A01
          + 3 kBT beta_eff_abgd G_gd delta_ab / 3
          + 15 (kBT)^2 gamma_abgder G_gd G_er delta_ab / 3
          + 105 (kBT)^3 rho_abgderhl G_gd G_er G_hl delta_ab / 3,

    G_ab = Omega0/(2 pi)^3 int_BZ d^3k sum_n [A(k, omega_n; A1)^(-1)]_ab.

The Matsubara sum is evaluated by diagonalizing the full periodic harmonic
Nishimatsu/FERAM kernel. A2..A5 are retained only for small-k diagnostics and
macroscopic response. The trial kernel must be positive over the integration
grid; negative static eigenvalues are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nishimatsu_harmonic import HarmonicGrid, build_harmonic_grid


KB_HARTREE_PER_K = 3.166811563e-6
AMU_TO_ELECTRON_MASS = 1822.888486217313


@dataclass(frozen=True)
class SchaCoefficients:
    """Coefficients from Appendix A, in Hartree and bohr conventions."""

    name: str
    a0: float = 7.46
    A01: float = -0.035058109
    A02: float = +0.155207480
    A03: float = -5.348957200
    A04: float = +0.905359992
    A05: float = +0.572574326
    b1_eff: float = 1.36449e-1
    b2_eff: float = 2.93874e-1
    k1_6: float = 0.0
    k2_6: float = 0.0
    k3_6: float = 0.0
    k4_8: float = 0.0
    mass_amu: float = 39.05
    zstar: float = 9.956
    eps_inf: float = 5.24
    kappa2: float = 0.0568
    j1: float = -0.02734
    j2: float = +0.04020
    j3: float = +0.00927
    j4: float = -0.00815
    j5: float = +0.00580
    j6: float = +0.00370
    j7: float = +0.00185

    @property
    def omega0(self) -> float:
        return self.a0**3

    @property
    def kmax(self) -> float:
        return math.pi / self.a0

    @property
    def mass_au(self) -> float:
        return self.mass_amu * AMU_TO_ELECTRON_MASS


MATERIALS: dict[str, SchaCoefficients] = {
    "vanderbilt": SchaCoefficients(name="vanderbilt_zvr"),
    "nishimatsu": SchaCoefficients(
        name="nishimatsu_2010_bto",
        a0=7.532372743713335,
        A01=-0.031248507143898818,
        A02=+0.13891652381255355,
        A03=-2.6304665896796883,
        A04=+1.5738147329579442,
        A05=+0.4567855127021116,
        b1_eff=0.1119078117465313,
        b2_eff=0.11589662794120481,
        k1_6=-0.2162513067289395,
        k2_6=+0.15937669276170113,
        k3_6=+0.669944535531705,
        k4_8=+0.14506807378350556,
        mass_amu=38.24,
        zstar=10.33,
        eps_inf=6.8691464565,
        kappa2=0.08782224891906101,
        j1=-0.021446406452141763,
        j2=-0.011618803008767388,
        j3=+0.007095114406006146,
        j4=-0.006291221721453839,
        j5=0.0,
        j6=+0.0028495045061522105,
        j7=0.0,
    ),
    "nishimatsu_sto": SchaCoefficients(
        # Nishimatsu et al., JPSJ 85, 114714 (2016), Table I.  The
        # continuum coefficients are derived from the same FERAM lattice
        # kernel; A01 is twice the tabulated total-energy curvature kappa.
        name="nishimatsu_2016_sto",
        a0=7.371821612165129,
        A01=-0.002593296296554976,
        A02=+0.2689681617221805,
        A03=-2.306510007880982,
        A04=+1.7965640186020102,
        A05=+0.4181661231102177,
        b1_eff=0.034680323402952566,
        b2_eff=0.023595648723710533,
        k1_6=-0.05256587344375355,
        k2_6=+0.09441521634816033,
        k3_6=+0.16274923788971776,
        k4_8=+0.03148947136215039,
        mass_amu=43.61,
        zstar=9.28,
        eps_inf=6.46,
        kappa2=0.10616049442563942,
        j1=-0.020705206939161155,
        j2=-0.018677907850187622,
        j3=+0.006071606408600935,
        j4=-0.005834916667248695,
        j5=0.0,
        j6=+0.0024492242800796993,
        j7=0.0,
    ),
}


def beta_eff_component(a: int, b: int, g: int, d: int, c: SchaCoefficients) -> float:
    """Cubic quartic tensor beta_eff_abgd = b1 S_abgd + b2 delta_abgd."""

    delta_ab = 1.0 if a == b else 0.0
    delta_gd = 1.0 if g == d else 0.0
    delta_ag = 1.0 if a == g else 0.0
    delta_bd = 1.0 if b == d else 0.0
    delta_ad = 1.0 if a == d else 0.0
    delta_bg = 1.0 if b == g else 0.0
    delta_abgd = 1.0 if a == b == g == d else 0.0
    return c.b1_eff * (delta_ab * delta_gd + delta_ag * delta_bd + delta_ad * delta_bg) + c.b2_eff * delta_abgd


def gamma_component(indices: tuple[int, int, int, int, int, int], c: SchaCoefficients) -> float:
    """Symmetric rank-6 tensor for E6 = gamma_abcdef u_a...u_f / 6."""

    counts = sorted([indices.count(axis) for axis in range(3) if indices.count(axis) > 0], reverse=True)
    if counts == [6]:
        return 6.0 * c.k1_6
    if counts == [4, 2]:
        return 2.0 * (3.0 * c.k1_6 + c.k2_6) / 5.0
    if counts == [2, 2, 2]:
        return (6.0 * c.k1_6 + c.k3_6) / 15.0
    return 0.0


def rho_component(indices: tuple[int, int, int, int, int, int, int, int], c: SchaCoefficients) -> float:
    """Symmetric rank-8 tensor for E8 = rho_abcdefgh u_a...u_h / 8."""

    counts = sorted([indices.count(axis) for axis in range(3) if indices.count(axis) > 0], reverse=True)
    if counts == [8]:
        return 8.0 * c.k4_8
    if counts == [6, 2]:
        return 8.0 * c.k4_8 / 7.0
    if counts == [4, 4]:
        return 24.0 * c.k4_8 / 35.0
    if counts == [4, 2, 2]:
        return 8.0 * c.k4_8 / 35.0
    return 0.0


def quartic_self_energy_mass(gmat: np.ndarray, temperature: float, c: SchaCoefficients) -> float:
    """Return the scalar mass correction delta A1 from the quartic SCHA term."""

    kbt = KB_HARTREE_PER_K * temperature
    contraction_trace = 0.0
    for a in range(3):
        for g in range(3):
            for d in range(3):
                contraction_trace += beta_eff_component(a, a, g, d, c) * gmat[g, d]
    return 3.0 * kbt * contraction_trace / 3.0


def sextic_self_energy_mass(gmat: np.ndarray, temperature: float, c: SchaCoefficients) -> float:
    """Return the scalar mass correction delta A1 from the sixth-order SCHA term."""

    kbt = KB_HARTREE_PER_K * temperature
    contraction_trace = 0.0
    for a in range(3):
        for g in range(3):
            for d in range(3):
                for e in range(3):
                    for r in range(3):
                        contraction_trace += gamma_component((a, a, g, d, e, r), c) * gmat[g, d] * gmat[e, r]
    return 15.0 * kbt**2 * contraction_trace / 3.0


def octic_self_energy_mass(gmat: np.ndarray, temperature: float, c: SchaCoefficients) -> float:
    """Return the scalar mass correction delta A1 from the eighth-order SCHA term."""

    kbt = KB_HARTREE_PER_K * temperature
    contraction_trace = 0.0
    for a in range(3):
        for g in range(3):
            for d in range(3):
                for e in range(3):
                    for r in range(3):
                        for h in range(3):
                            for l in range(3):
                                contraction_trace += (
                                    rho_component((a, a, g, d, e, r, h, l), c)
                                    * gmat[g, d]
                                    * gmat[e, r]
                                    * gmat[h, l]
                                )
    return 105.0 * kbt**3 * contraction_trace / 3.0


def static_kernel(kvec: np.ndarray, a1: float, c: SchaCoefficients) -> np.ndarray:
    """Static trial matrix mu(k; A1), without the m omega_n^2 term."""

    k2 = float(np.dot(kvec, kvec))
    mat = (a1 + c.A02 * k2) * np.eye(3)
    mat += c.A03 * np.outer(kvec, kvec)
    mat += c.A04 * np.diag(kvec * kvec)
    if k2 > 0.0:
        mat += c.A05 * np.outer(kvec, kvec) / k2
    return mat


def coth(x: float) -> float:
    if abs(x) < 1.0e-5:
        return 1.0 / x + x / 3.0 - x**3 / 45.0
    if x > 50.0:
        return 1.0
    return 1.0 / math.tanh(x)


def matsubara_mode_sum(mu: float, temperature: float, c: SchaCoefficients) -> float:
    """Return sum_n 1/(m omega_n^2 + mu) in atomic units.

    With omega_n = 2 pi n kBT and hbar=1:

        sum_n 1/(m(omega_n^2 + omega0^2))
        = 1/(2 m kBT omega0) coth(omega0/(2 kBT)),

    where omega0 = sqrt(mu/m).
    """

    kbt = KB_HARTREE_PER_K * temperature
    if mu <= 1.0e-14:
        raise ValueError("non-positive static eigenvalue is not allowed")
    omega = math.sqrt(mu / c.mass_au)
    return coth(omega / (2.0 * kbt)) / (2.0 * c.mass_au * kbt * omega)


def matsubara_summed_propagator(kvec: np.ndarray, a1: float, temperature: float, c: SchaCoefficients) -> np.ndarray:
    """O diag(G_i) O^T from Eqs. (89)--(96)."""

    mu = static_kernel(kvec, a1, c)
    eigvals, eigvecs = np.linalg.eigh(mu)
    diag = np.array([matsubara_mode_sum(float(value), temperature, c) for value in eigvals])
    return (eigvecs * diag) @ eigvecs.T


@lru_cache(maxsize=16)
def cached_harmonic_grid(ngrid: int, cutoff: float, c: SchaCoefficients) -> HarmonicGrid:
    return build_harmonic_grid(c, ngrid=ngrid, cutoff=cutoff)


@lru_cache(maxsize=16)
def cached_harmonic_eigensystem(ngrid: int, cutoff: float, c: SchaCoefficients):
    grid = cached_harmonic_grid(ngrid, cutoff, c)
    eigvals, eigvecs = np.linalg.eigh(grid.offsets)
    return grid, eigvals, eigvecs


def integrate_g(a1: float, temperature: float, ngrid: int, cutoff: float, c: SchaCoefficients) -> np.ndarray:
    """Integrate G using the full FERAM harmonic lattice kernel."""

    if not math.isfinite(a1) or a1 <= 1.0e-14:
        raise ValueError(
            "A1 must be positive: it is the transverse Gamma-limit eigenvalue"
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("the paraelectric Matsubara solver requires temperature > 0")
    if ngrid <= 0 or ngrid % 2 != 0:
        raise ValueError("ngrid must be a positive even integer")
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be positive and finite")
    grid, offset_eigvals, eigvecs = cached_harmonic_eigensystem(ngrid, cutoff, c)
    mu = offset_eigvals + a1
    if np.any(mu <= 1.0e-14):
        raise ValueError("non-positive static eigenvalue is not allowed")
    kbt = KB_HARTREE_PER_K * temperature
    omega = np.sqrt(mu / c.mass_au)
    mode_sums = 1.0 / np.tanh(omega / (2.0 * kbt)) / (2.0 * c.mass_au * kbt * omega)
    return grid.prefactor * np.einsum(
        "n,nai,ni,nbi->ab", grid.weights, eigvecs, mode_sums, eigvecs, optimize=True
    )


def stable_a1_threshold(ngrid: int, cutoff: float, c: SchaCoefficients) -> float:
    """Return the positivity threshold of the full lattice kernel."""

    grid = cached_harmonic_grid(ngrid, cutoff, c)
    min_eigenvalue_at_zero = float(np.linalg.eigvalsh(grid.offsets).min())
    return max(0.0, -min_eigenvalue_at_zero)


def residual(
    a1: float,
    temperature: float,
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
) -> tuple[float, np.ndarray, float, float, float]:
    gmat = integrate_g(a1, temperature=temperature, ngrid=ngrid, cutoff=cutoff, c=c)
    delta_quartic = quartic_self_energy_mass(gmat, temperature=temperature, c=c)
    delta_sextic = sextic_self_energy_mass(gmat, temperature=temperature, c=c)
    delta_octic = octic_self_energy_mass(gmat, temperature=temperature, c=c)
    return (
        a1 - (c.A01 + delta_quartic + delta_sextic + delta_octic),
        gmat,
        delta_quartic,
        delta_sextic,
        delta_octic,
    )


def find_brackets(temperature: float, ngrid: int, cutoff: float, c: SchaCoefficients) -> list[tuple[float, float]]:
    """Find sign-changing brackets for A1 in the stable trial-kernel domain."""

    stable_threshold = stable_a1_threshold(ngrid=ngrid, cutoff=cutoff, c=c)
    samples = np.concatenate(
        [
            stable_threshold + np.logspace(-10.0, 1.0, 180),
            np.linspace(-20.0, -1.0, 160),
            np.linspace(-0.99, 2.0, 220),
            np.linspace(2.05, 200.0, 160),
        ]
    )
    samples = np.array(sorted(set(float(value) for value in samples)))
    previous_a1: float | None = None
    previous_residual: float | None = None
    brackets: list[tuple[float, float]] = []

    for a1 in samples:
        try:
            value, _, _, _, _ = residual(a1, temperature=temperature, ngrid=ngrid, cutoff=cutoff, c=c)
        except (ValueError, np.linalg.LinAlgError):
            previous_a1 = None
            previous_residual = None
            continue

        if previous_residual is not None and previous_residual * value <= 0.0:
            brackets.append((previous_a1, float(a1)))
        previous_a1 = float(a1)
        previous_residual = float(value)

    return brackets


def choose_bracket(
    brackets: list[tuple[float, float]],
    c: SchaCoefficients,
    root_index: int | None = None,
) -> tuple[float, float]:
    """Pick a bracket, defaulting to the local branch nearest A01."""

    if not brackets:
        raise RuntimeError("could not find a stable sign-changing bracket for A1")
    if root_index is not None:
        if root_index < 0 or root_index >= len(brackets):
            raise ValueError(f"--root-index must be between 0 and {len(brackets) - 1}")
        return brackets[root_index]
    return min(brackets, key=lambda item: abs(0.5 * (item[0] + item[1]) - c.A01))


def find_bracket(
    temperature: float,
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
    root_index: int | None = None,
) -> tuple[float, float]:
    """Find one sign-changing bracket for A1."""

    brackets = find_brackets(temperature, ngrid=ngrid, cutoff=cutoff, c=c)
    return choose_bracket(brackets, c=c, root_index=root_index)


def format_brackets(brackets: list[tuple[float, float]]) -> list[list[float]]:
    return [[float(left), float(right)] for left, right in brackets]


def solve_a1(
    temperature: float,
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
    bracket: tuple[float, float] | None = None,
    root_index: int | None = None,
    tolerance: float = 1.0e-12,
    max_iterations: int = 100,
) -> dict:
    """Bisection solve of the restricted Eq. (85)."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if ngrid % 2 != 0:
        raise ValueError("ngrid must be even so no finite quadrature weight is placed at the non-analytic Gamma point")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    brackets = find_brackets(temperature, ngrid=ngrid, cutoff=cutoff, c=c) if bracket is None else [bracket]
    if bracket is not None or root_index is not None:
        indexed_candidates = [(root_index or 0, choose_bracket(brackets, c=c, root_index=root_index))]
    else:
        indexed_candidates = sorted(
            enumerate(brackets),
            key=lambda item: abs(0.5 * (item[1][0] + item[1][1]) - c.A01),
        )

    last_error: Exception | None = None
    for selected_index, selected_bracket in indexed_candidates:
        try:
            return bisect_bracket(
                selected_bracket=selected_bracket,
                selected_index=selected_index,
                detected_brackets=brackets,
                temperature=temperature,
                ngrid=ngrid,
                cutoff=cutoff,
                c=c,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        except RuntimeError as exc:
            last_error = exc
            if bracket is not None or root_index is not None:
                raise

    if last_error is not None:
        raise RuntimeError(f"no detected bracket converged to a regular root; last error: {last_error}")
    raise RuntimeError("could not find a stable sign-changing bracket for A1")


def bisect_bracket(
    selected_bracket: tuple[float, float],
    selected_index: int,
    detected_brackets: list[tuple[float, float]],
    temperature: float,
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
    tolerance: float,
    max_iterations: int,
) -> dict:
    left, right = selected_bracket
    f_left, _, _, _, _ = residual(left, temperature=temperature, ngrid=ngrid, cutoff=cutoff, c=c)
    f_right, _, _, _, _ = residual(right, temperature=temperature, ngrid=ngrid, cutoff=cutoff, c=c)
    if not math.isfinite(f_left) or not math.isfinite(f_right):
        raise ValueError("bracket endpoint residuals must be finite")
    if f_left * f_right > 0.0:
        raise ValueError("bracket does not change sign")

    mid = 0.5 * (left + right)
    f_mid = math.nan
    for iteration in range(1, max_iterations + 1):
        mid = 0.5 * (left + right)
        f_mid, gmat, delta_quartic, delta_sextic, delta_octic = residual(
            mid, temperature=temperature, ngrid=ngrid, cutoff=cutoff, c=c
        )
        if not math.isfinite(f_mid):
            raise RuntimeError("non-finite residual encountered during bisection")
        if abs(f_mid) < tolerance:
            grid, offset_eigvals, _ = cached_harmonic_eigensystem(ngrid, cutoff, c)
            full_lattice_min = float((offset_eigvals + mid).min())
            r_corner = np.array([cutoff, cutoff, cutoff], dtype=float)
            continuum_r_min = float(np.linalg.eigvalsh(static_kernel(r_corner, mid, c))[0])
            return {
                "material": c.name,
                "temperature_K": temperature,
                "A1": mid,
                "A01": c.A01,
                "delta_A1_quartic": delta_quartic,
                "delta_A1_sextic": delta_sextic,
                "delta_A1_octic": delta_octic,
                "delta_A1_total": delta_quartic + delta_sextic + delta_octic,
                "G": gmat.tolist(),
                "G_trace_over_3": float(np.trace(gmat) / 3.0),
                "full_lattice_min_eigenvalue": full_lattice_min,
                "continuum_R_min_eigenvalue": continuum_r_min,
                "harmonic_kernel": "full Nishimatsu/FERAM lattice kernel",
                "residual": f_mid,
                "iterations": iteration,
                "ngrid": ngrid,
                "cutoff_bohr_inv": cutoff,
                "bracket": [left, right],
                "root_index": selected_index,
                "detected_brackets": format_brackets(detected_brackets),
                "coefficients": asdict(c),
            }
        if 0.5 * abs(right - left) <= np.finfo(float).eps * max(1.0, abs(mid)):
            raise RuntimeError(
                f"bisection stagnated before reaching the residual tolerance; residual={f_mid:.6e}"
            )
        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid

    raise RuntimeError(f"bisection did not converge; last residual={f_mid:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve paraelectric Eq. (85) with Matsubara sum.")
    parser.add_argument(
        "--material",
        choices=sorted(MATERIALS),
        default="nishimatsu",
        help="Coefficient set. Default: nishimatsu.",
    )
    parser.add_argument("--temperature", "-T", type=float, default=500.0, help="Temperature in kelvin.")
    parser.add_argument("--ngrid", type=int, default=40, help="Even Gauss-Legendre order per direction. Default: 40.")
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Cubic BZ half-width in bohr^-1. Default: pi/a0.",
    )
    parser.add_argument("--left", type=float, default=None, help="Optional lower A1 bracket.")
    parser.add_argument("--right", type=float, default=None, help="Optional upper A1 bracket.")
    parser.add_argument(
        "--root-index",
        type=int,
        default=None,
        help="Pick a detected sign-changing bracket by index. Default: bracket nearest A01.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coeffs = MATERIALS[args.material]
    cutoff = coeffs.kmax if args.cutoff is None else args.cutoff
    bracket = None
    if args.left is not None or args.right is not None:
        if args.left is None or args.right is None:
            raise SystemExit("--left and --right must be provided together")
        bracket = (args.left, args.right)

    try:
        result = solve_a1(
            temperature=args.temperature,
            ngrid=args.ngrid,
            cutoff=cutoff,
            c=coeffs,
            bracket=bracket,
            root_index=args.root_index,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"material = {result['material']}")
    print(f"T = {result['temperature_K']:.6g} K")
    print(f"A1(T) = {result['A1']:.12g} Hartree/bohr^2")
    print(f"A01 = {result['A01']:.12g} Hartree/bohr^2")
    print(f"quartic delta A1 = {result['delta_A1_quartic']:.12g} Hartree/bohr^2")
    print(f"sextic delta A1 = {result['delta_A1_sextic']:.12g} Hartree/bohr^2")
    print(f"octic delta A1 = {result['delta_A1_octic']:.12g} Hartree/bohr^2")
    print(f"total delta A1 = {result['delta_A1_total']:.12g} Hartree/bohr^2")
    print(f"Tr(G)/3 = {result['G_trace_over_3']:.12g} bohr^2/Hartree")
    print(f"M* = {coeffs.mass_amu:.6g} amu = {coeffs.mass_au:.12g} electron masses")
    print(f"residual = {result['residual']:.3e}")
    print(f"ngrid = {result['ngrid']}, cutoff = {result['cutoff_bohr_inv']:.12g} bohr^-1")
    print(f"root index = {result['root_index']}")
    print(f"bracket = [{result['bracket'][0]:.12g}, {result['bracket'][1]:.12g}]")
    print(f"detected brackets = {len(result['detected_brackets'])}")
    print()
    print("Regard critique sur les valeurs obtenues:")
    print(f"- résidu auto-cohérent = {result['residual']:.3e}")
    print(f"- plus petite valeur propre du noyau de réseau = {result['full_lattice_min_eigenvalue']:.6e} Hartree/bohr^2")
    print(f"- valeur propre minimale du développement k^2 au coin R = {result['continuum_R_min_eigenvalue']:.6e} Hartree/bohr^2")
    print("- A1 résulte d'une compensation importante entre les contributions nue, quartique, sextique et octique;")
    print("  sa convergence en grille doit donc être contrôlée avant d'annoncer plus de trois chiffres significatifs.")
    print()
    print("Le développement continu en k^2, utilisé jusqu’au bord de la zone de Brillouin, rend artificiellement le noyau instable au coin R.")
    print("Les intégrales SCHA utilisent donc le noyau harmonique de réseau Nishimatsu complet; les coefficients A2...A5 sont réservés aux limites petit-k et à la réponse macroscopique.")
    print("Le code officiel FERAM confirme explicitement les deux points structurants : le site R=0 reçoit 2*P_kappa2.")

    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
