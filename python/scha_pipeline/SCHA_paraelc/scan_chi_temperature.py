#!/usr/bin/env python3
"""Scan the paraelectric SCHA susceptibility as a function of temperature.

The full periodic Nishimatsu/FERAM harmonic kernel is diagonalized only once.
At each temperature, the scalar paraelectric SCHA equation is solved for the
renormalized local mass A1(T,p), after which the dc transverse susceptibility is

    chi_T(T) = Omega0 / A1_P(T),
    A1_P(T) = (Omega0 / Z*)**2 A1_u(T).

In Gaussian atomic units the corresponding relative permittivity is
epsilon_r = epsilon_inf + 4 pi chi_T.

The optional thermal-expansion pressure follows Nishimatsu et al. (2010),

    p(T) = -0.005 T GPa,

and enters as the isotropic local-mode mass shift obtained by analytically
eliminating the homogeneous strain.
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
import scha_paraelc as scha


GPA_TO_HARTREE_PER_BOHR3 = 3.398930921743e-5
EV_TO_HARTREE = 1.0 / 27.211386245988
ANGSTROM_PER_BOHR = 0.529177210903
EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2 = (
    EV_TO_HARTREE * ANGSTROM_PER_BOHR**2
)

# Wu--Cohen/Nishimatsu parameters from Table II of Phys. Rev. B 82, 134106
# (2010). Elastic constants are in eV and mode--strain couplings in eV/A^2.
NISHIMATSU_B11_EV = 126.73
NISHIMATSU_B12_EV = 41.76
NISHIMATSU_B1XX_EV_PER_ANGSTROM2 = -185.35
NISHIMATSU_B1YY_EV_PER_ANGSTROM2 = -3.2809
THERMAL_EXPANSION_PRESSURE_SLOPE_GPA_PER_K = -0.005
BARRETT_CURIE_CONSTANT_K = 1.5e5
BARRETT_CURIE_WEISS_TEMPERATURE_K = 390.0
BARRETT_REFERENCE_MIN_K = 420.0
BARRETT_REFERENCE_MAX_K = 560.0
WIECZOREK_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "inputs" / "paraelectric"
    / "wieczorek2006_batio3_fig1a_digitized.csv"
)


def temperature_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0 or stop < start:
        raise ValueError("require step > 0 and stop >= start")
    count = int(round((stop - start) / step))
    grid = start + step * np.arange(count + 1, dtype=float)
    if not np.isclose(grid[-1], stop, rtol=0.0, atol=1.0e-10):
        raise ValueError("the temperature interval must be an integer multiple of the step")
    return grid


def pressure_gpa(
    temperature: float,
    pressure_model: str,
    constant_pressure_gpa: float,
) -> float:
    if pressure_model == "zero":
        return 0.0
    if pressure_model == "constant":
        return constant_pressure_gpa
    if pressure_model == "thermal-expansion":
        return THERMAL_EXPANSION_PRESSURE_SLOPE_GPA_PER_K * temperature
    raise ValueError(f"unsupported pressure model: {pressure_model}")


def experimental_curie_weiss(temperature: float) -> tuple[float, float]:
    """Return the broad BaTiO3 experimental Curie--Weiss reference.

    Barrett, Phys. Rev. 86, 118 (1952), reports the experimental parameters
    C=1.5e5 K and T0=390 K and notes that BaTiO3 follows the Curie--Weiss law
    down to the transition.  The returned values are the total relative
    permittivity and the soft contribution in the convention used here.
    """

    epsilon_r = BARRETT_CURIE_CONSTANT_K / (
        temperature - BARRETT_CURIE_WEISS_TEMPERATURE_K
    )
    chi_soft = (epsilon_r - 6.0) / (4.0 * np.pi)
    return epsilon_r, chi_soft


def fit_pressure_to_curie_weiss(
    temperature: float,
    ngrid: int,
    coefficients: scha.SchaCoefficients,
) -> float:
    """Fit a constant pressure by matching the Barrett reference at one T.

    The target susceptibility fixes the dressed transverse curvature A1.  The
    zero-pressure SCHA residual evaluated at that curvature is exactly the
    pressure-induced quadratic shift required by the self-consistency equation,
    so no outer numerical pressure scan is needed.
    """

    _, target_chi = experimental_curie_weiss(temperature)
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    target_a1_p = coefficients.omega0 / target_chi
    target_a1_u = target_a1_p / scale2
    threshold = scha.stable_a1_threshold(
        ngrid, coefficients.kmax, coefficients
    )
    if target_a1_u <= threshold:
        raise RuntimeError(
            "the target Curie--Weiss susceptibility lies outside the stable "
            "centered SCHA branch"
        )
    required_shift_u = float(
        scha.residual(
            target_a1_u,
            temperature,
            ngrid,
            coefficients.kmax,
            coefficients,
        )[0]
    )
    shift_per_gpa = pressure_mass_shift_u(1.0, coefficients)
    return required_shift_u / shift_per_gpa


def pressure_mass_shift_u(
    pressure_gpa_value: float,
    coefficients: scha.SchaCoefficients,
) -> float:
    """Return the hydrostatic pressure kernel P^(p,u) in Ha/bohr^2.

    Cubic symmetry gives

        P^(p,u) = -p a0^3 (B1xx + 2 B1yy)/(B11 + 2 B12).
    """

    if coefficients.name != "nishimatsu_2010_bto":
        raise ValueError("the pressure parameters implemented here are Nishimatsu-specific")

    pressure_au = pressure_gpa_value * GPA_TO_HARTREE_PER_BOHR3
    b11 = NISHIMATSU_B11_EV * EV_TO_HARTREE
    b12 = NISHIMATSU_B12_EV * EV_TO_HARTREE
    b1xx = (
        NISHIMATSU_B1XX_EV_PER_ANGSTROM2
        * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    b1yy = (
        NISHIMATSU_B1YY_EV_PER_ANGSTROM2
        * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    return float(
        -pressure_au
        * coefficients.omega0
        * (b1xx + 2.0 * b1yy)
        / (b11 + 2.0 * b12)
    )


def solve_root(
    temperature: float,
    ngrid: int,
    cutoff: float,
    coefficients: scha.SchaCoefficients,
    pressure_shift_u: float,
) -> float:
    def scalar_residual(a1: float) -> float:
        zero_pressure_residual = float(
            scha.residual(a1, temperature, ngrid, cutoff, coefficients)[0]
        )
        return zero_pressure_residual - pressure_shift_u

    lower = max(
        1.0e-12,
        scha.stable_a1_threshold(ngrid, cutoff, coefficients) + 1.0e-12,
    )
    upper = 1.0
    lower_value = scalar_residual(lower)
    upper_value = scalar_residual(upper)
    if lower_value * upper_value <= 0.0:
        return float(brentq(scalar_residual, lower, upper, xtol=1.0e-13, rtol=1.0e-13))

    # Fallback for a non-monotonic residual with more than one branch.
    samples = np.concatenate(([lower], np.logspace(-10.0, 0.0, 120)))
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
    raise RuntimeError(f"no stable paraelectric SCHA root found at T={temperature:g} K")


def scan(
    temperatures: np.ndarray,
    ngrid: int,
    coefficients: scha.SchaCoefficients,
    pressure_model: str,
    constant_pressure_gpa: float,
    skip_unstable: bool = False,
) -> list[dict[str, float]]:
    cutoff = coefficients.kmax
    scha.cached_harmonic_eigensystem(ngrid, cutoff, coefficients)
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    a5_p = coefficients.A05 * scale2
    rows: list[dict[str, float]] = []

    for temperature in temperatures:
        temperature = float(temperature)
        current_pressure_gpa = pressure_gpa(
            temperature,
            pressure_model=pressure_model,
            constant_pressure_gpa=constant_pressure_gpa,
        )
        pressure_shift_u = pressure_mass_shift_u(current_pressure_gpa, coefficients)
        try:
            a1_u = solve_root(
                temperature,
                ngrid,
                cutoff,
                coefficients,
                pressure_shift_u=pressure_shift_u,
            )
        except RuntimeError:
            if not skip_unstable:
                raise
            epsilon_experiment, chi_soft_experiment = experimental_curie_weiss(
                temperature
            )
            rows.append(
                {
                    "temperature_K": temperature,
                    "pressure_GPa": current_pressure_gpa,
                    "pressure_Ha_per_bohr3": (
                        current_pressure_gpa * GPA_TO_HARTREE_PER_BOHR3
                    ),
                    "pressure_mass_shift_u": pressure_shift_u,
                    "A01_with_pressure_u": coefficients.A01 + pressure_shift_u,
                    "A1_u": np.nan,
                    "A1_P": np.nan,
                    "chi_T_SCHA": np.nan,
                    "chi_L_SCHA": np.nan,
                    "epsilon_r_SCHA": np.nan,
                    "chi_soft_experiment_CW": chi_soft_experiment,
                    "epsilon_r_experiment_CW": epsilon_experiment,
                    "G_trace_over_3": np.nan,
                    "pressure_corrected_residual": np.nan,
                }
            )
            continue
        residual = scha.residual(a1_u, float(temperature), ngrid, cutoff, coefficients)
        a1_p = scale2 * a1_u
        chi_t = coefficients.omega0 / a1_p
        chi_l = coefficients.omega0 / (a1_p + a5_p)
        epsilon_r = coefficients.eps_inf + 4.0 * np.pi * chi_t

        epsilon_experiment, chi_soft_experiment = experimental_curie_weiss(
            temperature
        )

        rows.append(
            {
                "temperature_K": temperature,
                "pressure_GPa": current_pressure_gpa,
                "pressure_Ha_per_bohr3": (
                    current_pressure_gpa * GPA_TO_HARTREE_PER_BOHR3
                ),
                "pressure_mass_shift_u": pressure_shift_u,
                "A01_with_pressure_u": coefficients.A01 + pressure_shift_u,
                "A1_u": a1_u,
                "A1_P": a1_p,
                "chi_T_SCHA": chi_t,
                "chi_L_SCHA": chi_l,
                "epsilon_r_SCHA": epsilon_r,
                "chi_soft_experiment_CW": chi_soft_experiment,
                "epsilon_r_experiment_CW": epsilon_experiment,
                "G_trace_over_3": float(np.trace(residual[1]) / 3.0),
                "pressure_corrected_residual": float(residual[0] - pressure_shift_u),
            }
        )
    return rows


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_wieczorek_response() -> tuple[np.ndarray, np.ndarray]:
    """Load the BaTiO3 paraelectric branch digitized from Wieczorek Fig. 1(a)."""

    with WIECZOREK_DATA_PATH.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    temperatures = np.array([float(row["temperature_K"]) for row in rows])
    susceptibilities = np.array([float(row["chi_soft"]) for row in rows])
    return temperatures, susceptibilities


def write_plot(
    rows: list[dict[str, float]],
    path: Path,
    pressure_model: str,
    constant_pressure_gpa: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperatures = np.array([row["temperature_K"] for row in rows])
    chi_scha = np.array([row["chi_T_SCHA"] for row in rows])
    chi_experiment = np.array([row["chi_soft_experiment_CW"] for row in rows])
    wieczorek_temperature, wieczorek_chi = load_wieczorek_response()
    experimental_fit_domain = (
        (temperatures >= BARRETT_REFERENCE_MIN_K)
        & (temperatures <= BARRETT_REFERENCE_MAX_K)
    )
    stable = np.isfinite(chi_scha)

    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    if pressure_model == "thermal-expansion":
        scha_label = r"SCHA, $p(T)=-0.005T$ GPa"
    elif pressure_model == "constant":
        scha_label = rf"SCHA, $p={constant_pressure_gpa:.3f}$ GPa"
    else:
        scha_label = r"SCHA, $p=0$ GPa"
    if np.any(stable):
        axis.plot(temperatures[stable], chi_scha[stable], "o-", label=scha_label)
    if np.any(~stable):
        axis.plot(
            temperatures[~stable],
            np.zeros(np.count_nonzero(~stable)),
            "x",
            color="0.45",
            label="no stable centered SCHA root",
        )
    axis.plot(
        temperatures[experimental_fit_domain],
        chi_experiment[experimental_fit_domain],
        "--",
        label="experimental Curie--Weiss fit (Barrett, 1952)",
    )
    axis.plot(
        wieczorek_temperature,
        wieczorek_chi,
        "s--",
        markerfacecolor="white",
        label=r"Wieczorek et al. (2006), 1 kHz (digitized)",
    )
    axis.axvline(411.0, color="0.45", linestyle=":", linewidth=1.2, label=r"$T_C=411$ K (MD, thermal expansion)")
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"Soft-mode susceptibility $\chi_T$")
    axis.set_xlim(405.0, max(520.0, float(temperatures[-1]) + 5.0))
    axis.set_ylim(
        0.0,
        1.12
        * max(
            float(np.nanmax(chi_scha)) if np.any(stable) else 0.0,
            float(chi_experiment[experimental_fit_domain].max()),
            float(wieczorek_chi.max()),
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan the paraelectric SCHA susceptibility.")
    parser.add_argument("--start", type=float, default=415.0, help="First temperature in K.")
    parser.add_argument("--stop", type=float, default=515.0, help="Last temperature in K.")
    parser.add_argument("--step", type=float, default=5.0, help="Temperature step in K.")
    parser.add_argument("--ngrid", type=int, default=40, help="Even Gauss-Legendre order per direction.")
    parser.add_argument(
        "--pressure-model",
        choices=("zero", "constant", "thermal-expansion"),
        default="zero",
        help="Pressure protocol. Use thermal-expansion for p(T)=-0.005*T GPa.",
    )
    parser.add_argument(
        "--pressure-gpa",
        type=float,
        default=-2.0,
        help="Constant pressure in GPa when --pressure-model=constant.",
    )
    parser.add_argument(
        "--fit-pressure-temperature",
        type=float,
        default=None,
        metavar="K",
        help=(
            "For constant pressure, override --pressure-gpa by matching the "
            "Barrett Curie--Weiss susceptibility at this temperature."
        ),
    )
    parser.add_argument(
        "--skip-unstable",
        action="store_true",
        help="Write NaN rows instead of stopping where the centered SCHA root is unstable.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--plot", type=Path, default=None, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ngrid % 2 != 0:
        raise SystemExit("--ngrid must be even")
    temperatures = temperature_grid(args.start, args.stop, args.step)
    coefficients = scha.MATERIALS["nishimatsu"]
    constant_pressure_gpa = args.pressure_gpa
    if args.fit_pressure_temperature is not None:
        if args.pressure_model != "constant":
            raise SystemExit(
                "--fit-pressure-temperature requires --pressure-model=constant"
            )
        constant_pressure_gpa = fit_pressure_to_curie_weiss(
            args.fit_pressure_temperature,
            args.ngrid,
            coefficients,
        )
    rows = scan(
        temperatures,
        args.ngrid,
        coefficients,
        pressure_model=args.pressure_model,
        constant_pressure_gpa=constant_pressure_gpa,
        skip_unstable=args.skip_unstable,
    )

    if args.pressure_model == "thermal-expansion":
        pressure_tag = "pT"
    elif args.pressure_model == "constant":
        pressure_tag = f"p{constant_pressure_gpa:g}GPa".replace("-", "m").replace(".", "p")
    else:
        pressure_tag = "p0"
    stem = (
        f"chi_vs_temperature_{args.start:g}_{args.stop:g}K_"
        f"n{args.ngrid}_{pressure_tag}"
    )
    output_dir = (
        Path(__file__).resolve().parents[1] / "outputs" / "paraelectric"
    )
    csv_path = args.csv or output_dir / f"{stem}.csv"
    plot_path = args.plot or output_dir / f"{stem}.png"
    write_csv(rows, csv_path)
    write_plot(
        rows,
        plot_path,
        pressure_model=args.pressure_model,
        constant_pressure_gpa=constant_pressure_gpa,
    )

    print(f"computed {len(rows)} temperatures from {args.start:g} to {args.stop:g} K")
    print(f"pressure model = {args.pressure_model}")
    if args.fit_pressure_temperature is not None:
        print(
            "pressure fitted to Barrett at "
            f"{args.fit_pressure_temperature:g} K = {constant_pressure_gpa:.12g} GPa"
        )
    print(
        f"pressure range = {rows[0]['pressure_GPa']:.6g} to "
        f"{rows[-1]['pressure_GPa']:.6g} GPa"
    )
    stable_count = sum(np.isfinite(row["chi_T_SCHA"]) for row in rows)
    print(f"stable centered roots = {stable_count}/{len(rows)}")
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")
    print(f"chi_T({rows[0]['temperature_K']:g} K) = {rows[0]['chi_T_SCHA']:.10g}")
    print(f"chi_T({rows[-1]['temperature_K']:g} K) = {rows[-1]['chi_T_SCHA']:.10g}")


if __name__ == "__main__":
    main()
