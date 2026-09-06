#!/usr/bin/env python3
"""Quantum-SCHA dielectric profile of bulk SrTiO3 at zero pressure.

The effective-Hamiltonian parameters are the WC-GGA SrTiO3 values in Table I
of Nishimatsu et al., J. Phys. Soc. Jpn. 85, 114714 (2016).  The dynamic
inhomogeneous-strain interaction H^dagger G_w H is evaluated with the same
FERAM lattice vertex and Matsubara summation as for the BaTiO3 profile.  The
antiferrodistortive oxygen-rotation mode responsible for the 105 K
cubic-to-tetragonal transition remains outside this effective Hamiltonian.

For orientation, the plot includes the single-crystal Barrett fit used by
Mueller and Burkard, Phys. Rev. B 19, 3593 (1979): C=8e4 K, T0=35.5 K and
T1=80 K.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_chi_temperature_inhomogeneous as acoustic_scan
import scha_paraelc as scha


BARRETT_C_K = 8.0e4
BARRETT_T0_K = 35.5
BARRETT_T1_K = 80.0


def temperature_grid(start: float, stop: float, step: float) -> np.ndarray:
    if start <= 0.0 or step <= 0.0 or stop < start:
        raise ValueError("require start > 0, step > 0 and stop >= start")
    count = int(round((stop - start) / step))
    grid = start + step * np.arange(count + 1, dtype=float)
    if not np.isclose(grid[-1], stop, rtol=0.0, atol=1.0e-10):
        raise ValueError("the temperature interval must be an integer multiple of the step")
    return grid


def barrett_permittivity(temperature: np.ndarray | float) -> np.ndarray:
    values = np.asarray(temperature, dtype=float)
    denominator = 0.5 * BARRETT_T1_K / np.tanh(
        BARRETT_T1_K / (2.0 * values)
    ) - BARRETT_T0_K
    return BARRETT_C_K / denominator


def solve_polar_root(
    temperature: float,
    ngrid: int,
    coefficients: scha.SchaCoefficients,
) -> float:
    cutoff = coefficients.kmax

    def scalar_residual(a1: float) -> float:
        return float(
            scha.residual(a1, temperature, ngrid, cutoff, coefficients)[0]
        )

    lower = max(
        1.0e-12,
        scha.stable_a1_threshold(ngrid, cutoff, coefficients) + 1.0e-12,
    )
    upper = 1.0
    if scalar_residual(lower) * scalar_residual(upper) <= 0.0:
        return float(
            brentq(
                scalar_residual,
                lower,
                upper,
                xtol=1.0e-13,
                rtol=1.0e-13,
            )
        )

    samples = np.concatenate(([lower], np.logspace(-10.0, 0.0, 160)))
    previous_a1 = float(samples[0])
    previous_value = scalar_residual(previous_a1)
    for current_a1 in samples[1:]:
        current_a1 = float(current_a1)
        current_value = scalar_residual(current_a1)
        if previous_value * current_value <= 0.0:
            return float(
                brentq(
                    scalar_residual,
                    previous_a1,
                    current_a1,
                    xtol=1.0e-13,
                    rtol=1.0e-13,
                )
            )
        previous_a1 = current_a1
        previous_value = current_value
    raise RuntimeError(f"no stable centered SCHA root at T={temperature:g} K")


def scan(temperatures: np.ndarray, ngrid: int) -> list[dict[str, float]]:
    coefficients = scha.MATERIALS["nishimatsu_sto"]
    cutoff = coefficients.kmax
    scha.cached_harmonic_eigensystem(ngrid, cutoff, coefficients)
    acoustic_grid = acoustic_scan.build_inhomogeneous_quartic_grid(
        ngrid,
        coefficients,
        parameters=acoustic_scan.STO_ACOUSTIC_PARAMETERS,
    )
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    rows: list[dict[str, float]] = []

    for temperature_value in temperatures:
        temperature = float(temperature_value)
        a1_u_polar = solve_polar_root(temperature, ngrid, coefficients)
        polar_residual = scha.residual(
            a1_u_polar, temperature, ngrid, cutoff, coefficients
        )
        a1_u_acoustic = acoustic_scan.solve_root(
            temperature,
            0.0,
            coefficients,
            acoustic_grid,
        )
        acoustic_residual = acoustic_scan.residual(
            a1_u_acoustic,
            temperature,
            coefficients,
            acoustic_grid,
        )
        a1_p_polar = scale2 * a1_u_polar
        a1_p_acoustic = scale2 * a1_u_acoustic
        chi_soft_polar = coefficients.omega0 / a1_p_polar
        chi_soft_acoustic = coefficients.omega0 / a1_p_acoustic
        epsilon_r_polar = coefficients.eps_inf + 4.0 * np.pi * chi_soft_polar
        epsilon_r_acoustic = (
            coefficients.eps_inf + 4.0 * np.pi * chi_soft_acoustic
        )
        epsilon_barrett = float(barrett_permittivity(temperature))
        rows.append(
            {
                "temperature_K": temperature,
                "A1_u_acoustic_Ha_per_bohr2": a1_u_acoustic,
                "A1_P_acoustic": a1_p_acoustic,
                "chi_soft_SCHA_acoustic": chi_soft_acoustic,
                "epsilon_r_SCHA_acoustic": epsilon_r_acoustic,
                "A1_u_polar_only_Ha_per_bohr2": a1_u_polar,
                "A1_P_polar_only": a1_p_polar,
                "chi_soft_SCHA_polar_only": chi_soft_polar,
                "epsilon_r_SCHA_polar_only": epsilon_r_polar,
                "epsilon_r_Barrett": epsilon_barrett,
                "delta_A1_quartic_local": float(acoustic_residual[2]),
                "delta_A1_quartic_homogeneous": float(acoustic_residual[3]),
                "delta_A1_quartic_inhomogeneous": float(acoustic_residual[4]),
                "delta_A1_sextic": float(acoustic_residual[5]),
                "delta_A1_octic": float(acoustic_residual[6]),
                "G_trace_over_3_acoustic": float(
                    np.trace(acoustic_residual[1]) / 3.0
                ),
                "residual_acoustic": float(acoustic_residual[0]),
                "residual_polar_only": float(polar_residual[0]),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperature = np.array([row["temperature_K"] for row in rows])
    epsilon_acoustic = np.array(
        [row["epsilon_r_SCHA_acoustic"] for row in rows]
    )
    epsilon_polar = np.array(
        [row["epsilon_r_SCHA_polar_only"] for row in rows]
    )
    dense_temperature = np.linspace(float(temperature[0]), float(temperature[-1]), 800)
    epsilon_barrett = barrett_permittivity(dense_temperature)

    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    axis.plot(
        temperature,
        epsilon_acoustic,
        "o-",
        markersize=3.4,
        linewidth=1.6,
        label=(
            r"SCHA with acoustic kernel "
            r"$\mathcal{H}^{\dagger}G_w\mathcal{H}$"
        ),
    )
    axis.plot(
        temperature,
        epsilon_polar,
        "s-.",
        markerfacecolor="white",
        markersize=3.4,
        linewidth=1.3,
        label="polar-only SCHA",
    )
    axis.plot(
        dense_temperature,
        epsilon_barrett,
        "--",
        linewidth=1.6,
        label=r"Barrett fit: $C=8\times10^4$ K, $T_0=35.5$ K, $T_1=80$ K",
    )
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"Relative permittivity $\varepsilon_r$")
    axis.set_xlim(float(temperature[0]), float(temperature[-1]))
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False, fontsize=8.7)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan the zero-pressure quantum-SCHA permittivity of SrTiO3."
    )
    parser.add_argument("--start", type=float, default=415.0, help="First temperature in K.")
    parser.add_argument("--stop", type=float, default=565.0, help="Last temperature in K.")
    parser.add_argument("--step", type=float, default=5.0, help="Temperature step in K.")
    parser.add_argument(
        "--ngrid",
        type=int,
        default=40,
        help="Even Gauss-Legendre order per reciprocal-space direction.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--plot", type=Path, default=None, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ngrid % 2 != 0:
        raise SystemExit("--ngrid must be even")
    temperatures = temperature_grid(args.start, args.stop, args.step)
    rows = scan(temperatures, args.ngrid)

    stem = (
        f"epsilon_r_vs_temperature_{args.start:g}_{args.stop:g}K_"
        f"n{args.ngrid}_srtio3_acoustic"
    )
    output_dir = (
        Path(__file__).resolve().parents[1] / "outputs" / "paraelectric"
    )
    csv_path = args.csv or output_dir / f"{stem}.csv"
    plot_path = args.plot or output_dir / f"{stem}.png"
    write_csv(rows, csv_path)
    write_plot(rows, plot_path)

    print(f"computed {len(rows)} temperatures from {args.start:g} to {args.stop:g} K")
    print("material = nishimatsu_2016_sto, pressure = 0 GPa")
    print(f"ngrid = {args.ngrid}")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    print(
        f"epsilon_r_acoustic({rows[0]['temperature_K']:g} K) = "
        f"{rows[0]['epsilon_r_SCHA_acoustic']:.10g}"
    )
    print(
        f"epsilon_r_acoustic({rows[-1]['temperature_K']:g} K) = "
        f"{rows[-1]['epsilon_r_SCHA_acoustic']:.10g}"
    )


if __name__ == "__main__":
    main()
