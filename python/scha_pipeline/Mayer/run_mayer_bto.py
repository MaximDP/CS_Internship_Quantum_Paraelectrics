#!/usr/bin/env python3
"""Provisional Mayer-PBEsol BaTiO3 SCHA scans.

This script imports, but never modifies, the shared SCHA kernels.
It compares the Mayer anharmonic parameter set at zero pressure with the same
set under a pointwise hydrostatic pressure chosen to reproduce the weighted
Nakatani cubic lattice fit.  An affine compression of that pressure is also
reported as a diagnostic, but the pointwise result is the lattice-matched
protocol requested for this provisional study.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
PARAELECTRIC_DIR = PIPELINE_DIR / "SCHA_paraelc"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(PARAELECTRIC_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import parameters as mayer
import scan_chi_temperature as reference_scan
import scan_chi_temperature_inhomogeneous as inhomogeneous
import scha_paraelc as scha


LATTICE_DATA = MODULE_DIR.parent / "inputs" / "mayer" / "nakatani2016_batio3_cubic_lattice.csv"
WIECZOREK_DATA = (
    MODULE_DIR.parent / "inputs" / "mayer" / "wieczorek2006_batio3_fig1a_digitized.csv"
)
PREVIOUS_NISHIMATSU_DATA = (
    MODULE_DIR.parent / "outputs" / "reference" / "mayer" / "nishimatsu_previous_predictions_400_800K_n40.csv"
)
MAYER_EXPERIMENT_DATA = (
    MODULE_DIR.parent / "inputs" / "mayer" / "mayer2022_batio3_fig3a_experiment_digitized.csv"
)
RESULTS_DIR = MODULE_DIR.parent / "outputs" / "mayer"


def pressure_mass_shift_u(
    pressure_gpa: float,
    coefficients: scha.SchaCoefficients,
    parameters: inhomogeneous.AcousticQuarticParameters,
) -> float:
    """Return Mayer's hydrostatic quadratic shift in Ha/bohr^2."""

    if coefficients.name != parameters.coefficients_name:
        raise ValueError("the Mayer polar and acoustic parameter sets do not match")
    pressure_au = pressure_gpa * reference_scan.GPA_TO_HARTREE_PER_BOHR3
    b11 = parameters.b11_ev * reference_scan.EV_TO_HARTREE
    b12 = parameters.b12_ev * reference_scan.EV_TO_HARTREE
    b1xx = (
        parameters.b1xx_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    b1yy = (
        parameters.b1yy_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    return float(
        -pressure_au
        * coefficients.omega0
        * (b1xx + 2.0 * b1yy)
        / (b11 + 2.0 * b12)
    )


def load_lattice_data(path: Path = LATTICE_DATA) -> tuple[np.ndarray, ...]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return tuple(
        np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in (
            "temperature_K",
            "lattice_parameter_angstrom",
            "standard_uncertainty_angstrom",
        )
    )


def load_wieczorek_response(
    path: Path = WIECZOREK_DATA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load digitized 1-kHz permittivity and convert it for Mayer's model."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    temperatures = np.asarray(
        [float(row["temperature_K"]) for row in rows], dtype=float
    )
    epsilon_r = np.asarray(
        [float(row["epsilon_r"]) for row in rows], dtype=float
    )
    chi_soft = (
        epsilon_r - mayer.MAYER_ANHARMONIC_COEFFICIENTS.eps_inf
    ) / (4.0 * np.pi)
    return temperatures, epsilon_r, chi_soft


def load_previous_nishimatsu_predictions(
    path: Path = PREVIOUS_NISHIMATSU_DATA,
) -> tuple[np.ndarray, ...]:
    """Load the former Nishimatsu curves used in the reference Nishimatsu comparison."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return tuple(
        np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in (
            "temperature_K",
            "nishimatsu_lattice_matched_chi",
            "nishimatsu_response_fitted_chi",
            "nishimatsu_zero_pressure_chi",
        )
    )


def load_mayer_experimental_response(
    path: Path = MAYER_EXPERIMENT_DATA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Mayer Fig. 3(a) points and convert them to soft susceptibility."""

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    temperatures = np.asarray(
        [float(row["temperature_K"]) for row in rows], dtype=float
    )
    epsilon_r = np.asarray(
        [float(row["epsilon_r"]) for row in rows], dtype=float
    )
    chi_soft = (
        epsilon_r - mayer.MAYER_ANHARMONIC_COEFFICIENTS.eps_inf
    ) / (4.0 * np.pi)
    return temperatures, epsilon_r, chi_soft


def fit_lattice_reference(
    fit_start: float,
    fit_stop: float,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    temperatures, lattice, uncertainties = load_lattice_data()
    selected = (temperatures >= fit_start) & (temperatures <= fit_stop)
    if np.count_nonzero(selected) < 2:
        raise ValueError("the lattice reference fit needs at least two points")
    slope, intercept = np.polyfit(
        temperatures[selected],
        lattice[selected],
        1,
        w=1.0 / uncertainties[selected],
    )
    return (
        float(slope),
        float(intercept),
        temperatures[selected],
        lattice[selected],
        uncertainties[selected],
    )


def evaluate_state(
    temperature: float,
    pressure_gpa: float,
    coefficients: scha.SchaCoefficients,
    grid: inhomogeneous.InhomogeneousQuarticGrid,
) -> dict[str, float]:
    shift = pressure_mass_shift_u(
        pressure_gpa, coefficients, grid.parameters
    )
    a1 = inhomogeneous.solve_root(
        temperature, shift, coefficients, grid
    )
    evaluated = inhomogeneous.residual(a1, temperature, coefficients, grid)
    lattice, strain, variance = inhomogeneous.cubic_lattice_parameter(
        temperature,
        pressure_gpa,
        evaluated[1],
        coefficients,
        grid.parameters,
    )
    chi = coefficients.zstar**2 / (coefficients.omega0 * a1)
    epsilon = coefficients.eps_inf + 4.0 * np.pi * chi
    return {
        "pressure_GPa": float(pressure_gpa),
        "pressure_mass_shift_Ha_per_bohr2": shift,
        "A1_Ha_per_bohr2": a1,
        "chi_soft": float(chi),
        "epsilon_r": float(epsilon),
        "lattice_parameter_angstrom": lattice,
        "isotropic_strain": strain,
        "mean_u_component_variance_bohr2": variance,
        "delta_A1_quartic_local": evaluated[2],
        "delta_A1_quartic_homogeneous": evaluated[3],
        "delta_A1_quartic_inhomogeneous": evaluated[4],
        "delta_A1_sextic": evaluated[5],
        "delta_A1_octic": evaluated[6],
        "root_residual": evaluated[0] - shift,
    }


def calibrate_pointwise_pressure(
    temperature: float,
    target_lattice_angstrom: float,
    coefficients: scha.SchaCoefficients,
    grid: inhomogeneous.InhomogeneousQuarticGrid,
    initial_pressure_gpa: float,
    bounds_gpa: tuple[float, float] = (-8.0, 3.0),
) -> float:
    """Invert the centered cubic equation of state at one temperature."""

    def objective(pressure: float) -> float:
        return (
            evaluate_state(temperature, pressure, coefficients, grid)[
                "lattice_parameter_angstrom"
            ]
            - target_lattice_angstrom
        )

    lower, upper = bounds_gpa
    pressure = float(np.clip(initial_pressure_gpa, lower, upper))
    try:
        value = objective(pressure)
    except RuntimeError:
        pressure = 0.0
        value = objective(pressure)
    if abs(value) < 1.0e-13:
        return pressure

    # Increasing conventional hydrostatic pressure decreases the lattice
    # parameter.  Continuation from the previous temperature supplies a close
    # starting value and avoids an expensive global pressure scan.
    direction = -1.0 if value < 0.0 else 1.0
    previous_pressure = pressure
    previous_value = value
    step = 0.20
    for _ in range(120):
        candidate = float(
            np.clip(previous_pressure + direction * step, lower, upper)
        )
        if candidate == previous_pressure:
            break
        try:
            candidate_value = objective(candidate)
        except RuntimeError:
            step *= 0.5
            if step < 1.0e-6:
                break
            continue
        if previous_value * candidate_value <= 0.0:
            left, right = sorted((previous_pressure, candidate))
            return float(
                brentq(
                    objective,
                    left,
                    right,
                    xtol=1.0e-12,
                    rtol=1.0e-12,
                )
            )
        previous_pressure = candidate
        previous_value = candidate_value
        step = min(1.25 * step, 0.75)
    raise RuntimeError(
        f"no stable Mayer pressure reproduces a={target_lattice_angstrom:g} "
        f"angstrom at T={temperature:g} K in [{lower:g},{upper:g}] GPa"
    )


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def run_scan(
    temperatures: np.ndarray,
    ngrid: int,
    lattice_fit_start: float,
    lattice_fit_stop: float,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
    acoustic = mayer.MAYER_ANHARMONIC_ACOUSTIC_PARAMETERS
    grid = inhomogeneous.build_inhomogeneous_quartic_grid(
        ngrid,
        coefficients,
        acoustic_mass_amu=acoustic.acoustic_mass_amu,
        parameters=acoustic,
    )
    (
        lattice_slope,
        lattice_intercept,
        fit_temperatures,
        fit_lattice,
        fit_uncertainties,
    ) = fit_lattice_reference(lattice_fit_start, lattice_fit_stop)
    targets = lattice_slope * temperatures + lattice_intercept

    zero_states: list[dict[str, float]] = []
    matched_states: list[dict[str, float]] = []
    pointwise_pressures: list[float] = []
    previous_pressure = 0.0
    for temperature_value, target_value in zip(temperatures, targets):
        temperature = float(temperature_value)
        target = float(target_value)
        zero_states.append(evaluate_state(temperature, 0.0, coefficients, grid))
        pressure = calibrate_pointwise_pressure(
            temperature,
            target,
            coefficients,
            grid,
            initial_pressure_gpa=previous_pressure,
        )
        pointwise_pressures.append(pressure)
        matched_states.append(
            evaluate_state(temperature, pressure, coefficients, grid)
        )
        previous_pressure = pressure

    pointwise_pressures_array = np.asarray(pointwise_pressures, dtype=float)
    affine_slope, affine_intercept = np.polyfit(
        temperatures, pointwise_pressures_array, 1
    )
    affine_states = [
        evaluate_state(
            float(temperature),
            float(affine_slope * temperature + affine_intercept),
            coefficients,
            grid,
        )
        for temperature in temperatures
    ]

    rows: list[dict[str, float]] = []
    for temperature, target, pressure, zero, matched, affine in zip(
        temperatures,
        targets,
        pointwise_pressures_array,
        zero_states,
        matched_states,
        affine_states,
    ):
        row: dict[str, float] = {
            "temperature_K": float(temperature),
            "lattice_target_angstrom": float(target),
            "pointwise_pressure_GPa": float(pressure),
        }
        row.update(prefixed("zero", zero))
        row.update(prefixed("lattice_matched", matched))
        row.update(prefixed("affine_pressure", affine))
        row["lattice_matched_error_angstrom"] = (
            matched["lattice_parameter_angstrom"] - float(target)
        )
        row["affine_lattice_error_angstrom"] = (
            affine["lattice_parameter_angstrom"] - float(target)
        )
        rows.append(row)

    lattice_data_residuals = (
        lattice_slope * fit_temperatures + lattice_intercept - fit_lattice
    )
    matched_errors = np.asarray(
        [row["lattice_matched_error_angstrom"] for row in rows]
    )
    affine_errors = np.asarray(
        [row["affine_lattice_error_angstrom"] for row in rows]
    )
    root_residuals = np.asarray(
        [
            abs(row[f"{protocol}_root_residual"])
            for row in rows
            for protocol in ("zero", "lattice_matched", "affine_pressure")
        ]
    )
    wieczorek_temperatures, _, _ = load_wieczorek_response()
    mayer_experimental_temperatures, _, _ = load_mayer_experimental_response()
    summary: dict[str, object] = {
        "status": "Mayer PBEsol effective anharmonic model",
        "model": coefficients.name,
        "ngrid": ngrid,
        "temperature_grid_K": [
            float(temperatures[0]),
            float(temperatures[-1]),
            float(temperatures[1] - temperatures[0]),
        ],
        "lattice_reference": {
            "source": "Nakatani et al., Acta Cryst. B 72, 151-159 (2016)",
            "fit_domain_K": [lattice_fit_start, lattice_fit_stop],
            "slope_angstrom_per_K": lattice_slope,
            "intercept_angstrom": lattice_intercept,
            "data_rmse_angstrom": float(
                np.sqrt(np.mean(lattice_data_residuals**2))
            ),
        },
        "dielectric_reference": {
            "source": (
                "Wieczorek et al., Ferroelectrics 336, 61-67 (2006), "
                "Fig. 1(a)"
            ),
            "doi": "10.1080/00150190600695743",
            "status": "digitized experimental points; not raw author table",
            "sample": "BaTiO3 single crystal with evaporated gold electrodes",
            "frequency_Hz": 1000.0,
            "field_kV_per_cm": 0.1,
            "number_of_points": int(wieczorek_temperatures.size),
            "temperature_interval_K": [
                float(np.min(wieczorek_temperatures)),
                float(np.max(wieczorek_temperatures)),
            ],
            "conversion": (
                "chi_soft=(epsilon_r-eps_inf_Mayer)/(4*pi), "
                f"eps_inf_Mayer={coefficients.eps_inf:g}"
            ),
            "used_in_pressure_fit": False,
        },
        "mayer_dielectric_reference": {
            "source": "Mayer et al., Phys. Rev. B 106, 064108 (2022), Fig. 3(a)",
            "doi": "10.1103/PhysRevB.106.064108",
            "status": "digitized experimental points; not raw author table",
            "sample": "BaTiO3 single crystal with silver-paint electrodes",
            "measurement_frequency_range_Hz": [20.0, 1.0e6],
            "plotted_frequency_status": "not specified in the Fig. 3 caption",
            "number_of_points": int(mayer_experimental_temperatures.size),
            "temperature_interval_K": [
                float(np.min(mayer_experimental_temperatures)),
                float(np.max(mayer_experimental_temperatures)),
            ],
            "conversion": (
                "chi_soft=(epsilon_r-eps_inf_Mayer)/(4*pi), "
                f"eps_inf_Mayer={coefficients.eps_inf:g}"
            ),
            "used_in_pressure_fit": False,
        },
        "previous_nishimatsu_comparison": {
            "source": PREVIOUS_NISHIMATSU_DATA.name,
            "status": (
                "former n=40 SCHA curves shown in the reference Nishimatsu comparison, "
                "recomputed to 800 K with the historical pressure laws held fixed"
            ),
            "protocols": [
                "zero pressure",
                "lattice-matched pressure",
                "pressure fitted to the Wieczorek response",
            ],
        },
        "pointwise_lattice_pressure": {
            "minimum_GPa": float(np.min(pointwise_pressures_array)),
            "maximum_GPa": float(np.max(pointwise_pressures_array)),
            "maximum_abs_lattice_error_angstrom": float(
                np.max(np.abs(matched_errors))
            ),
        },
        "affine_pressure_diagnostic": {
            "slope_GPa_per_K": float(affine_slope),
            "intercept_GPa": float(affine_intercept),
            "pressure_rmse_GPa": float(
                np.sqrt(
                    np.mean(
                        (
                            affine_slope * temperatures
                            + affine_intercept
                            - pointwise_pressures_array
                        )
                        ** 2
                    )
                )
            ),
            "maximum_abs_lattice_error_angstrom": float(
                np.max(np.abs(affine_errors))
            ),
        },
        "numerical_checks": {
            "maximum_abs_root_residual": float(np.max(root_residuals)),
            "all_values_finite": bool(
                np.all(
                    np.isfinite(
                        [
                            value
                            for row in rows
                            for value in row.values()
                        ]
                    )
                )
            ),
        },
        "parameter_snapshot": mayer.serializable_parameter_snapshot(),
    }
    return rows, summary


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    rows: list[dict[str, float]],
    summary: dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperatures = np.asarray([row["temperature_K"] for row in rows])
    target = np.asarray([row["lattice_target_angstrom"] for row in rows])
    zero_a = np.asarray([row["zero_lattice_parameter_angstrom"] for row in rows])
    matched_a = np.asarray(
        [row["lattice_matched_lattice_parameter_angstrom"] for row in rows]
    )
    affine_a = np.asarray(
        [row["affine_pressure_lattice_parameter_angstrom"] for row in rows]
    )
    zero_chi = np.asarray([row["zero_chi_soft"] for row in rows])
    matched_chi = np.asarray([row["lattice_matched_chi_soft"] for row in rows])
    affine_chi = np.asarray([row["affine_pressure_chi_soft"] for row in rows])
    pressures = np.asarray([row["pointwise_pressure_GPa"] for row in rows])
    affine_summary = summary["affine_pressure_diagnostic"]
    affine_pressure = (
        float(affine_summary["slope_GPa_per_K"]) * temperatures
        + float(affine_summary["intercept_GPa"])
    )
    experimental_t, experimental_a, experimental_sigma = load_lattice_data()
    wieczorek_t, _, wieczorek_chi = load_wieczorek_response()
    mayer_experimental_t, _, mayer_experimental_chi = (
        load_mayer_experimental_response()
    )
    (
        old_t,
        old_lattice_chi,
        old_response_fit_chi,
        old_zero_chi,
    ) = load_previous_nishimatsu_predictions()

    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 5.25), constrained_layout=True
    )
    axes[0].errorbar(
        experimental_t,
        experimental_a,
        yerr=experimental_sigma,
        fmt="s",
        color="0.15",
        markerfacecolor="white",
        markersize=4.0,
        capsize=2.0,
        label="Nakatani et al.",
    )
    axes[0].plot(temperatures, target, "--", color="tab:green", label="weighted $a_{exp}(T)$ fit")
    axes[0].plot(temperatures, zero_a, "D-.", color="tab:purple", markersize=3.5, label="$p_{eff}=0$")
    axes[0].plot(temperatures, matched_a, "o-", color="tab:blue", markersize=3.5, label="pointwise lattice match")
    axes[0].plot(temperatures, affine_a, ":", color="tab:orange", linewidth=1.8, label="affine pressure diagnostic")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel(r"Cubic lattice parameter $a$ ($\AA$)")
    axes[0].legend(frameon=False, fontsize=7.5)

    axes[1].plot(temperatures, zero_chi, "D-.", color="tab:purple", markersize=3.5, label="Mayer: $p_{eff}=0$")
    axes[1].plot(temperatures, matched_chi, "o-", color="tab:blue", markersize=3.5, label="Mayer: lattice match")
    axes[1].plot(temperatures, affine_chi, ":", color="tab:orange", linewidth=1.8, label="Mayer: affine lattice")
    axes[1].plot(
        old_t,
        old_zero_chi,
        "--",
        color="tab:purple",
        linewidth=1.35,
        label="Nishimatsu: $p_{eff}=0$",
    )
    axes[1].plot(
        old_t,
        old_lattice_chi,
        "--",
        color="tab:blue",
        linewidth=1.35,
        label="Nishimatsu: lattice match",
    )
    fit_start = float(np.min(wieczorek_t))
    fit_stop = float(np.max(wieczorek_t))
    fit_t = np.unique(
        np.concatenate(
            ([fit_start], old_t[(old_t > fit_start) & (old_t < fit_stop)], [fit_stop])
        )
    )
    fit_chi = np.interp(fit_t, old_t, old_response_fit_chi)
    extrapolated_t = np.unique(
        np.concatenate(([fit_stop], old_t[old_t > fit_stop]))
    )
    extrapolated_chi = np.interp(extrapolated_t, old_t, old_response_fit_chi)
    axes[1].plot(
        fit_t,
        fit_chi,
        "-",
        color="tab:red",
        linewidth=1.55,
        label=r"Nishimatsu: fitted $p_\chi(T)$",
    )
    axes[1].plot(
        extrapolated_t,
        extrapolated_chi,
        "--",
        color="tab:red",
        linewidth=1.35,
        label=r"same $p_\chi(T)$, extrapolated",
    )
    barrett_t = old_t[old_t >= fit_start]
    barrett_chi = (1.5e5 / (barrett_t - 390.0) - 6.0) / (4.0 * np.pi)
    axes[1].plot(
        barrett_t,
        barrett_chi,
        "--",
        color="tab:green",
        linewidth=1.35,
        label="Barrett, empirical Curie--Weiss",
    )
    axes[1].plot(
        mayer_experimental_t,
        mayer_experimental_chi,
        linestyle="none",
        marker="o",
        color="tab:cyan",
        markersize=2.5,
        label="Mayer et al., experiment (digitized)",
    )
    axes[1].plot(
        wieczorek_t,
        wieczorek_chi,
        linestyle="none",
        marker="s",
        color="black",
        markerfacecolor="white",
        markersize=4.5,
        label="Wieczorek et al., 1 kHz (digitized)",
    )
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel(r"Soft-mode susceptibility $\chi_T$")
    axes[1].set_yscale("log")
    axes[1].legend(
        frameon=False,
        fontsize=6.0,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
    )

    axes[2].plot(temperatures, pressures, "o-", color="tab:blue", markersize=3.5, label="pointwise $p_a(T)$")
    axes[2].plot(temperatures, affine_pressure, "--", color="tab:orange", label="affine compression")
    axes[2].axhline(0.0, color="0.4", linewidth=0.9)
    axes[2].set_xlabel("Temperature (K)")
    axes[2].set_ylabel(r"Effective pressure $p_{eff}$ (GPa)")
    axes[2].legend(frameon=False, fontsize=7.5)
    for axis in axes:
        axis.grid(alpha=0.22)
    fig.suptitle("Provisional Mayer-PBEsol anharmonic BTO SCHA scan", fontsize=12)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=400.0)
    parser.add_argument("--stop", type=float, default=600.0)
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--ngrid", type=int, default=40)
    parser.add_argument("--lattice-fit-start", type=float, default=413.0)
    parser.add_argument("--lattice-fit-stop", type=float, default=598.0)
    parser.add_argument("--stem", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = reference_scan.temperature_grid(
        args.start, args.stop, args.step
    )
    if temperatures.size < 2:
        raise SystemExit("the Mayer scan needs at least two temperatures")
    rows, summary = run_scan(
        temperatures,
        args.ngrid,
        args.lattice_fit_start,
        args.lattice_fit_stop,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.stem or (
        f"mayer_bto_{args.start:g}_{args.stop:g}K_n{args.ngrid}"
    )
    csv_path = RESULTS_DIR / f"{stem}.csv"
    summary_path = RESULTS_DIR / f"{stem}.json"
    parameters_path = RESULTS_DIR / "mayer_parameter_snapshot.json"
    figure_path = RESULTS_DIR / f"{stem}.png"
    write_csv(rows, csv_path)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    parameters_path.write_text(
        json.dumps(mayer.serializable_parameter_snapshot(), indent=2) + "\n",
        encoding="utf-8",
    )
    write_plot(rows, summary, figure_path)
    pointwise = summary["pointwise_lattice_pressure"]
    affine = summary["affine_pressure_diagnostic"]
    print(f"model: {summary['model']}")
    print(
        "pointwise p_a(T) range: "
        f"{pointwise['minimum_GPa']:.9g} to "
        f"{pointwise['maximum_GPa']:.9g} GPa"
    )
    print(
        "pointwise max |a-a_target|: "
        f"{pointwise['maximum_abs_lattice_error_angstrom']:.9g} angstrom"
    )
    print(
        "affine p_a(T): "
        f"{affine['slope_GPa_per_K']:+.12g} T"
        f"{affine['intercept_GPa']:+.12g} GPa; "
        "max lattice error="
        f"{affine['maximum_abs_lattice_error_angstrom']:.9g} angstrom"
    )
    print(f"CSV: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"parameters: {parameters_path}")
    print(f"figure: {figure_path}")


if __name__ == "__main__":
    main()
