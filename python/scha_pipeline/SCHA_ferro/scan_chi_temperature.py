#!/usr/bin/env python3
"""Cryogenic ferroelectric SCHA susceptibility scan (0--50 K by default).

The nonlinear broken-symmetry solution is followed by continuation in
temperature.  The intrinsic single-domain soft-mode susceptibility is

    chi = Z*^2 / Omega0 * A_local^{-1},

in the Gaussian convention used in the notes, with
``epsilon_r = epsilon_inf I + 4 pi chi``.

The historical experimental comparison is intentionally modest because the
published cryogenic response depends strongly on orientation and domains:

* B. Wul, J. Phys. USSR 10, 64--66 (1946): epsilon_r=100 at 4.2 K and
  d ln(epsilon_r)/dT=0.005--0.006 K^-1 between 2 and 4.2 K.
* J. C. Holste, W. N. Lawless, and G. A. Samara, Ferroelectrics 11,
  337--340 (1976), DOI 10.1080/00150197608236576: for a BaTiO3 single
  crystal, d ln(epsilon_r)/dT=9.4e-4 K^-1 between 1.5 and 4.5 K.

Only Wul's absolute datum is plotted as an absolute susceptibility.  The
Holste et al. result is compared in the normalized lower panel because its
abstract does not report the absolute permittivity or crystal orientation.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scha_ferro as scha


WUL_TEMPERATURE_K = 4.2
WUL_EPSILON_R = 100.0
WUL_ALPHA_MIN_PER_K = 0.005
WUL_ALPHA_MAX_PER_K = 0.006
HOLSTE_ALPHA_PER_K = 9.4e-4


def susceptibility_tensor(
    mass_matrix: np.ndarray,
    coefficients: scha.SchaCoefficients,
) -> np.ndarray:
    """Return the intrinsic static soft-mode susceptibility tensor."""

    return (
        coefficients.zstar**2
        / coefficients.omega0
        * np.linalg.inv(mass_matrix)
    )


def project_tensor(matrix: np.ndarray, direction: np.ndarray) -> float:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    return float(direction @ matrix @ direction)


def scan(
    temperatures: np.ndarray,
    coefficients: scha.SchaCoefficients,
    ngrid: int,
    cutoff: float,
    tolerance: float,
    max_nfev: int,
) -> list[dict[str, float]]:
    """Follow the rhombohedral solution using the previous state as a seed."""

    rows: list[dict[str, float]] = []
    previous_state: tuple[np.ndarray, np.ndarray] | None = None
    seed_direction = scha.parse_direction("generic")
    ex = np.array([1.0, 0.0, 0.0])

    for temperature_value in temperatures:
        temperature = float(temperature_value)
        result = scha.solve_ferro(
            temperature=temperature,
            ngrid=ngrid,
            cutoff=cutoff,
            direction=seed_direction,
            c=coefficients,
            a_t_guess=None,
            a_l_guess=None,
            p_guess=None,
            u_guess=None,
            tolerance=tolerance,
            max_nfev=max_nfev,
            initial_state=previous_state,
        )
        mass_matrix = np.asarray(result["A_local"], dtype=float)
        uvec = np.asarray(result["physical_displacement_bohr"], dtype=float)
        direction = np.asarray(result["direction"], dtype=float)
        previous_state = (mass_matrix, uvec)
        seed_direction = direction

        chi = susceptibility_tensor(mass_matrix, coefficients)
        epsilon = coefficients.eps_inf * np.eye(3) + 4.0 * np.pi * chi
        chi_l = project_tensor(chi, direction)
        chi_100 = project_tensor(chi, ex)
        chi_t = 0.5 * (float(np.trace(chi)) - chi_l)
        eps_l = project_tensor(epsilon, direction)
        eps_100 = project_tensor(epsilon, ex)
        eps_t = 0.5 * (float(np.trace(epsilon)) - eps_l)

        rows.append(
            {
                "temperature_K": temperature,
                "u_min_abs_bohr": float(result["physical_displacement_abs_bohr"]),
                "direction_x": float(direction[0]),
                "direction_y": float(direction[1]),
                "direction_z": float(direction[2]),
                "A_T_Ha_per_bohr2": float(result["A_T"]),
                "A_L_Ha_per_bohr2": float(result["A_L"]),
                "chi_transverse": chi_t,
                "chi_longitudinal": chi_l,
                "chi_100": chi_100,
                "epsilon_r_transverse": eps_t,
                "epsilon_r_longitudinal": eps_l,
                "epsilon_r_100": eps_100,
                "residual_norm": float(result["residual_norm"]),
                "min_static_eigenvalue_Ha": float(result["min_static_eigenvalue"]),
                "scipy_nfev": float(result["scipy_nfev"]),
            }
        )
        print(
            f"T={temperature:5.1f} K  |u|={rows[-1]['u_min_abs_bohr']:.8f} bohr  "
            f"chi_L={chi_l:.6f}  chi_T={chi_t:.6f}  "
            f"res={rows[-1]['residual_norm']:.2e}",
            flush=True,
        )
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    rows: list[dict[str, float]],
    coefficients: scha.SchaCoefficients,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperature = np.array([row["temperature_K"] for row in rows])
    chi_t = np.array([row["chi_transverse"] for row in rows])
    chi_l = np.array([row["chi_longitudinal"] for row in rows])
    chi_100 = np.array([row["chi_100"] for row in rows])
    eps_l = np.array([row["epsilon_r_longitudinal"] for row in rows])

    fig, (axis, normalized_axis) = plt.subplots(
        2,
        1,
        figsize=(7.4, 6.7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.15, 1.0]},
        constrained_layout=True,
    )
    axis.plot(temperature, chi_t, linewidth=2.0, label=r"SCHA transverse $\chi_T$")
    axis.plot(temperature, chi_100, linewidth=2.0, label=r"SCHA along $[100]_c$")
    axis.plot(temperature, chi_l, linewidth=2.2, label=r"SCHA longitudinal $\chi_L$")

    experimental_chi = (
        WUL_EPSILON_R - coefficients.eps_inf
    ) / (4.0 * np.pi)
    wul_temperature = np.linspace(2.0, WUL_TEMPERATURE_K, 80)
    wul_eps_low = WUL_EPSILON_R * np.exp(
        WUL_ALPHA_MIN_PER_K * (wul_temperature - WUL_TEMPERATURE_K)
    )
    wul_eps_high = WUL_EPSILON_R * np.exp(
        WUL_ALPHA_MAX_PER_K * (wul_temperature - WUL_TEMPERATURE_K)
    )
    wul_chi_low = (wul_eps_low - coefficients.eps_inf) / (4.0 * np.pi)
    wul_chi_high = (wul_eps_high - coefficients.eps_inf) / (4.0 * np.pi)
    axis.fill_between(
        wul_temperature,
        wul_chi_low,
        wul_chi_high,
        color="black",
        alpha=0.18,
        label=r"Wul (1946), inferred from $\epsilon_r(4.2\,K)=100$",
    )
    axis.plot(
        [WUL_TEMPERATURE_K],
        [experimental_chi],
        "ko",
        markersize=6.0,
    )
    axis.set_ylabel(r"Soft-mode susceptibility $\chi$")
    axis.grid(alpha=0.23)
    axis.legend(frameon=False, fontsize=8.6, ncol=2)
    axis.set_title(
        r"BaTiO$_3$: intrinsic rhombohedral SCHA response and cryogenic data"
    )

    model_reference = float(np.interp(WUL_TEMPERATURE_K, temperature, eps_l))
    normalized_axis.plot(
        temperature,
        eps_l / model_reference,
        linewidth=2.0,
        label=r"SCHA $\epsilon_L(T)/\epsilon_L(4.2\,K)$",
    )
    wul_ratio_min = np.exp(
        WUL_ALPHA_MIN_PER_K * (wul_temperature - WUL_TEMPERATURE_K)
    )
    wul_ratio_max = np.exp(
        WUL_ALPHA_MAX_PER_K * (wul_temperature - WUL_TEMPERATURE_K)
    )
    normalized_axis.fill_between(
        wul_temperature,
        wul_ratio_min,
        wul_ratio_max,
        color="black",
        alpha=0.18,
        label=r"Wul: $d\ln\epsilon_r/dT=0.005$--$0.006$ K$^{-1}$",
    )
    holste_temperature = np.linspace(1.5, 4.5, 80)
    normalized_axis.plot(
        holste_temperature,
        np.exp(HOLSTE_ALPHA_PER_K * (holste_temperature - WUL_TEMPERATURE_K)),
        "--",
        color="tab:red",
        linewidth=1.8,
        label=r"Holste et al. (1976): $9.4\times10^{-4}$ K$^{-1}$",
    )
    normalized_axis.axhline(1.0, color="0.65", linewidth=0.8)
    normalized_axis.set_xlabel("Temperature (K)")
    normalized_axis.set_ylabel(r"Normalized $\epsilon_r$")
    normalized_axis.set_xlim(float(temperature[0]), float(temperature[-1]))
    normalized_axis.grid(alpha=0.23)
    normalized_axis.legend(frameon=False, fontsize=8.2, ncol=2)

    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan the direction-free ferroelectric SCHA susceptibility."
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=50.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--ngrid", type=int, default=12)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument(
        "--csv",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "ferroelectric"
            / "chi_ferro_0_50K_n12.csv"
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "ferroelectric"
            / "chi_ferro_0_50K_n12.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step <= 0.0 or args.stop < args.start:
        raise SystemExit("require step > 0 and stop >= start")
    temperatures = np.arange(
        args.start,
        args.stop + 0.5 * args.step,
        args.step,
        dtype=float,
    )
    coefficients = scha.MATERIALS["nishimatsu"]
    rows = scan(
        temperatures=temperatures,
        coefficients=coefficients,
        ngrid=args.ngrid,
        cutoff=coefficients.kmax,
        tolerance=args.tolerance,
        max_nfev=args.max_nfev,
    )
    write_csv(rows, args.csv)
    write_plot(rows, coefficients, args.plot)
    print(f"wrote {args.csv}")
    print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
