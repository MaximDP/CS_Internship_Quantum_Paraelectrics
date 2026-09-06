#!/usr/bin/env python3
"""Static dielectric response from the paraelectric SCHA kernel.

Computes

    chi_ab(k) = Omega0 * [A*(k, omega_0; T)^(-1)]_ab

for k approaching zero along x, y, and z.  A static source has only a zero
Matsubara-frequency component; the Matsubara sum instead belongs to the
equal-time covariance used in the SCHA self-consistency equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SchaKernel:
    """Nishimatsu paraelectric SCHA coefficients in polarization variables."""

    temperature_k: float = 500.0
    omega0: float = 427.3615143
    a1: float = 5.01613131381
    a2: float = +237.76289672944844
    a3: float = -4502.181158493385
    a4: float = +2693.6662360518203
    a5: float = +781.8122977987001

def kernel_from_scha_json(path: Path) -> SchaKernel:
    result = json.loads(path.read_text(encoding="utf-8"))
    coefficients = result["coefficients"]
    omega0 = coefficients["a0"] ** 3
    scale = omega0 / coefficients["zstar"]
    scale2 = scale * scale
    return SchaKernel(
        temperature_k=float(result["temperature_K"]),
        omega0=omega0,
        a1=float(result["A1"]) * scale2,
        a2=coefficients["A02"] * scale2,
        a3=coefficients["A03"] * scale2,
        a4=coefficients["A04"] * scale2,
        a5=coefficients["A05"] * scale2,
    )


DIRECTIONS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def static_kernel(kvec: np.ndarray, c: SchaKernel) -> np.ndarray:
    """Return the static part mu_ab(k) of A*_ab(k, omega_n; T)."""

    k2 = float(np.dot(kvec, kvec))
    mat = (c.a1 + c.a2 * k2) * np.eye(3)
    mat += c.a3 * np.outer(kvec, kvec)
    mat += c.a4 * np.diag(kvec * kvec)
    if k2 > 0.0:
        mat += c.a5 * np.outer(kvec, kvec) / k2
    return mat


def limiting_static_kernel(direction: np.ndarray, c: SchaKernel) -> np.ndarray:
    """Return lim_{q->0+} mu(q direction), keeping the non-analytic projector."""

    unit = direction / np.linalg.norm(direction)
    return c.a1 * np.eye(3) + c.a5 * np.outer(unit, unit)


def chi_from_static_kernel(mu: np.ndarray, c: SchaKernel) -> np.ndarray:
    """Return the dc response Omega0 * A*(k, omega_0)^(-1)."""

    eigvals, eigvecs = np.linalg.eigh(mu)
    if np.any(eigvals <= 0.0):
        raise ValueError(f"static eigenvalues must be positive, got {eigvals}")
    inverse = (eigvecs * eigvals**-1) @ eigvecs.T
    return c.omega0 * inverse


def format_matrix(mat: np.ndarray) -> str:
    rows = []
    for row in mat:
        rows.append("[" + "  ".join(f"{value: .10e}" for value in row) + "]")
    return "\n".join(rows)


def parse_k_values(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the SCHA dielectric response near k=0.")
    parser.add_argument("-T", "--temperature", type=float, default=None, help="Temperature in kelvin.")
    parser.add_argument("--scha-json", type=Path, default=None, help="Read A1 and coefficients from a SCHA JSON result.")
    parser.add_argument(
        "--k-values",
        default="1e-1,1e-2,1e-3,1e-4",
        help="Comma-separated small |k| values in bohr^-1.",
    )
    parser.add_argument(
        "--limit-only",
        action="store_true",
        help="Print only the directional k->0 limits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coeffs = kernel_from_scha_json(args.scha_json) if args.scha_json is not None else SchaKernel()
    if args.temperature is not None and not math.isclose(args.temperature, coeffs.temperature_k, abs_tol=1.0e-12):
        raise SystemExit("temperature does not match the supplied/precomputed SCHA kernel; recompute A1 at that temperature")
    k_values = parse_k_values(args.k_values)

    print("Static SCHA dielectric response")
    print(f"T = {coeffs.temperature_k:g} K")
    print("chi_ab(k) = Omega0 * [A*(k, omega_0; T)^(-1)]_ab")
    print("directions: columns/rows are x, y, z")
    print()

    for name, direction in DIRECTIONS.items():
        print(f"k -> 0 along {name}")
        chi_limit = chi_from_static_kernel(limiting_static_kernel(direction, coeffs), coeffs)
        print(format_matrix(chi_limit))
        print()

        if args.limit_only:
            continue

        for kval in k_values:
            chi = chi_from_static_kernel(static_kernel(kval * direction, coeffs), coeffs)
            print(f"|k| = {kval:.3e} bohr^-1")
            print(format_matrix(chi))
        print()


if __name__ == "__main__":
    main()
