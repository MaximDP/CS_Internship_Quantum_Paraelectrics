#!/usr/bin/env python3
"""Translation-invariant VCA/SCHA dielectric scans for Ba_x Sr_(1-x) TiO3.

The Ba and Sr end-member effective-Hamiltonian parameters are interpolated at
fixed Ba fraction x.  The dipolar Gamma contact and non-analytic coefficient
are then rebuilt from the interpolated microscopic inputs.  This preserves a
periodic 3x3 kernel at every k; no quenched chemical field is introduced.

The experimental target is the 10 kHz Curie--Weiss envelope reported for
high-density BST ceramics by Weerasinghe et al. (arXiv:1211.6970).  Their
experimental T0(x) curve was digitized at x=0.4, 0.5, and 0.6, while the Curie
constant was reported only to order 1e5 K.  Consequently the green curve is a
literature-constrained reference envelope, not raw point-by-point data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import nishimatsu_harmonic as harmonic
import scan_chi_temperature as reference_scan
import scan_chi_temperature_inhomogeneous as inhomogeneous
import scha_paraelc as scha


REFERENCE_PATH = PIPELINE_DIR / "inputs" / "paraelectric" / "bst_curie_weiss_reference.csv"
DEFAULT_COMPOSITIONS = (0.4, 0.5, 0.6)
CENTRAL_MODE_CROSSOVER_K = 365.0


def _linear(sto_value: float, bto_value: float, x_ba: float) -> float:
    return float((1.0 - x_ba) * sto_value + x_ba * bto_value)


def vca_acoustic_parameters(
    x_ba: float,
) -> inhomogeneous.AcousticQuarticParameters:
    """Linearly interpolate the independent elastic/mode--strain inputs."""

    if not 0.0 <= x_ba <= 1.0:
        raise ValueError("the Ba fraction must lie in [0, 1]")
    sto = inhomogeneous.STO_ACOUSTIC_PARAMETERS
    bto = inhomogeneous.BTO_ACOUSTIC_PARAMETERS
    values: dict[str, float | str] = {
        "coefficients_name": f"bst_vca_x{x_ba:.6f}",
    }
    for field in fields(inhomogeneous.AcousticQuarticParameters):
        if field.name == "coefficients_name":
            continue
        values[field.name] = _linear(
            float(getattr(sto, field.name)),
            float(getattr(bto, field.name)),
            x_ba,
        )
    return inhomogeneous.AcousticQuarticParameters(**values)


def vca_coefficients(x_ba: float) -> scha.SchaCoefficients:
    """Build a periodic VCA parameter set from the Nishimatsu end members.

    Primary lattice, local, short-range, charge, and dielectric inputs are
    interpolated.  A01 is retained as the independently calibrated Gamma mass
    (this also reproduces both tabulated end members exactly), whereas A05 is
    rebuilt from its analytic non-analytic dipolar prefactor.  A02--A04 are
    only unused continuum diagnostics in this scan and retain their end-member
    interpolation; all SCHA integrals use the rebuilt periodic lattice kernel.
    """

    if not 0.0 <= x_ba <= 1.0:
        raise ValueError("the Ba fraction must lie in [0, 1]")
    sto = scha.MATERIALS["nishimatsu_sto"]
    bto = scha.MATERIALS["nishimatsu"]
    acoustic = vca_acoustic_parameters(x_ba)
    values: dict[str, float | str] = {"name": acoustic.coefficients_name}
    for field in fields(scha.SchaCoefficients):
        if field.name == "name":
            continue
        values[field.name] = _linear(
            float(getattr(sto, field.name)),
            float(getattr(bto, field.name)),
            x_ba,
        )

    # Rebuild the projected quartic from independently interpolated bare and
    # homogeneous-strain tensors, rather than interpolating it a second time.
    values["b1_eff"] = acoustic.bare_b1 - 0.5 * acoustic.homogeneous_lambda_b1
    values["b2_eff"] = acoustic.bare_b2 - 0.5 * acoustic.homogeneous_lambda_b2
    provisional = scha.SchaCoefficients(**values)
    gamma = harmonic.raw_harmonic_kernel(np.zeros(3), provisional)
    gamma_anisotropy = gamma - float(np.trace(gamma) / 3.0) * np.eye(3)
    if np.linalg.norm(gamma_anisotropy) > 1.0e-10:
        raise RuntimeError("the VCA Gamma kernel is not cubic")
    a05 = float(
        4.0
        * np.pi
        * provisional.zstar**2
        / (provisional.eps_inf * provisional.omega0)
    )
    return replace(provisional, A05=a05)


def pressure_mass_shift_u(
    pressure_gpa: float,
    coefficients: scha.SchaCoefficients,
    parameters: inhomogeneous.AcousticQuarticParameters,
) -> float:
    """Hydrostatic mass shift using the composition-interpolated tensors."""

    if coefficients.name != parameters.coefficients_name:
        raise ValueError("the VCA polar and acoustic parameter sets do not match")
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


def load_references(path: Path = REFERENCE_PATH) -> dict[float, dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    references: dict[float, dict[str, float | str]] = {}
    for row in rows:
        x_ba = float(row["x_ba"])
        references[x_ba] = {
            "x_ba": x_ba,
            "T0_K": float(row["T0_K"]),
            "T0_uncertainty_K": float(row["T0_uncertainty_K"]),
            "curie_constant_K": float(row["curie_constant_K"]),
            "curie_constant_relative_uncertainty": float(
                row["curie_constant_relative_uncertainty"]
            ),
            "measurement_frequency_Hz": float(row["measurement_frequency_Hz"]),
            "sample_type": row["sample_type"],
            "source": row["source"],
            "provenance": row["provenance"],
        }
    return references


def experimental_reference(
    temperature: float,
    coefficients: scha.SchaCoefficients,
    reference: dict[str, float | str],
) -> tuple[float, float]:
    """Return total epsilon_r and the corresponding soft susceptibility."""

    t0 = float(reference["T0_K"])
    if temperature <= t0:
        raise ValueError("Curie--Weiss reference requires T > T0")
    epsilon_r = float(reference["curie_constant_K"]) / (temperature - t0)
    chi_soft = (epsilon_r - coefficients.eps_inf) / (4.0 * np.pi)
    if chi_soft <= 0.0:
        raise ValueError("the reference soft-mode susceptibility is non-positive")
    return float(epsilon_r), float(chi_soft)


def _target_a1_u(
    temperature: float,
    coefficients: scha.SchaCoefficients,
    reference: dict[str, float | str],
) -> float:
    _, chi_target = experimental_reference(temperature, coefficients, reference)
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    return float(coefficients.omega0 / (scale2 * chi_target))


def fit_affine_pressure(
    temperatures: np.ndarray,
    coefficients: scha.SchaCoefficients,
    grid: inhomogeneous.InhomogeneousQuarticGrid,
    reference: dict[str, float | str],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit p(T)=aT+b to the 10 kHz Curie--Weiss reference envelope."""

    parameters = grid.parameters
    shift_per_gpa = pressure_mass_shift_u(1.0, coefficients, parameters)
    exact_pressures = []
    relative_sensitivities = []
    for temperature_value in temperatures:
        temperature = float(temperature_value)
        target_a1 = _target_a1_u(temperature, coefficients, reference)
        threshold = scha.stable_a1_threshold(
            grid.ngrid, coefficients.kmax, coefficients
        )
        if target_a1 <= threshold:
            raise RuntimeError(
                f"the experimental target at {temperature:g} K is outside "
                "the stable centered branch"
            )
        exact_pressures.append(
            inhomogeneous.residual(
                target_a1, temperature, coefficients, grid
            )[0]
            / shift_per_gpa
        )
        delta = max(1.0e-9, 1.0e-4 * target_a1)
        derivative = (
            inhomogeneous.residual(
                target_a1 + delta, temperature, coefficients, grid
            )[0]
            - inhomogeneous.residual(
                target_a1 - delta, temperature, coefficients, grid
            )[0]
        ) / (2.0 * delta)
        relative_sensitivities.append(
            abs(shift_per_gpa / (target_a1 * derivative))
        )

    exact = np.asarray(exact_pressures, dtype=float)
    weights = np.asarray(relative_sensitivities, dtype=float)
    weights /= np.max(weights)
    t_ref = float(np.mean(temperatures))
    centered = temperatures - t_ref
    design = np.column_stack((centered, np.ones_like(centered)))
    slope, p_at_ref = np.linalg.lstsq(
        design * weights[:, None], exact * weights, rcond=None
    )[0]

    # Gauss--Newton refinement of the relative permittivity error.
    for _ in range(5):
        log_errors = []
        jacobian = []
        for temperature_value in temperatures:
            temperature = float(temperature_value)
            pressure = slope * (temperature - t_ref) + p_at_ref
            a1 = inhomogeneous.solve_root(
                temperature,
                shift_per_gpa * pressure,
                coefficients,
                grid,
            )
            scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
            chi = coefficients.omega0 / (scale2 * a1)
            epsilon = coefficients.eps_inf + 4.0 * np.pi * chi
            epsilon_target, _ = experimental_reference(
                temperature, coefficients, reference
            )
            log_errors.append(np.log(epsilon / epsilon_target))
            delta = max(1.0e-9, 1.0e-4 * a1)
            derivative = (
                inhomogeneous.residual(
                    a1 + delta, temperature, coefficients, grid
                )[0]
                - inhomogeneous.residual(
                    a1 - delta, temperature, coefficients, grid
                )[0]
            ) / (2.0 * delta)
            dchi_dp = -chi * shift_per_gpa / (a1 * derivative)
            dlogepsilon_dp = 4.0 * np.pi * dchi_dp / epsilon
            jacobian.append(
                dlogepsilon_dp * np.array((temperature - t_ref, 1.0))
            )
        correction = np.linalg.lstsq(
            np.asarray(jacobian), -np.asarray(log_errors), rcond=None
        )[0]
        slope += correction[0]
        p_at_ref += correction[1]
        if np.max(np.abs(correction)) < 1.0e-10:
            break

    intercept = p_at_ref - slope * t_ref
    fitted = slope * temperatures + intercept
    return float(slope), float(intercept), exact, fitted


def scan_composition(
    x_ba: float,
    temperatures: np.ndarray,
    fit_temperatures: np.ndarray,
    ngrid: int,
    reference: dict[str, float | str],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    coefficients = vca_coefficients(x_ba)
    parameters = vca_acoustic_parameters(x_ba)
    grid = inhomogeneous.build_inhomogeneous_quartic_grid(
        ngrid,
        coefficients,
        acoustic_mass_amu=parameters.acoustic_mass_amu,
        parameters=parameters,
    )
    slope, intercept, exact, fitted = fit_affine_pressure(
        fit_temperatures, coefficients, grid, reference
    )
    shift_per_gpa = pressure_mass_shift_u(1.0, coefficients, parameters)
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    rows: list[dict[str, float]] = []
    for temperature_value in temperatures:
        temperature = float(temperature_value)
        epsilon_target, chi_target = experimental_reference(
            temperature, coefficients, reference
        )
        values: dict[str, float] = {
            "x_ba": x_ba,
            "temperature_K": temperature,
            "T0_reference_K": float(reference["T0_K"]),
            "curie_constant_reference_K": float(
                reference["curie_constant_K"]
            ),
            "epsilon_r_reference_10kHz": epsilon_target,
            "chi_soft_reference_10kHz": chi_target,
            "p_fit_GPa": slope * temperature + intercept,
        }
        for label, pressure in (
            ("zero", 0.0),
            ("fit", values["p_fit_GPa"]),
        ):
            try:
                a1 = inhomogeneous.solve_root(
                    temperature,
                    shift_per_gpa * pressure,
                    coefficients,
                    grid,
                )
                chi = coefficients.omega0 / (scale2 * a1)
                epsilon = coefficients.eps_inf + 4.0 * np.pi * chi
            except RuntimeError:
                a1 = chi = epsilon = float("nan")
            values[f"A1_u_{label}"] = a1
            values[f"chi_soft_{label}"] = chi
            values[f"epsilon_r_{label}"] = epsilon
        values["relative_error_fit"] = (
            values["epsilon_r_fit"] / epsilon_target - 1.0
        )
        rows.append(values)

    fit_errors = np.asarray(
        [
            row["relative_error_fit"]
            for row in rows
            if float(fit_temperatures[0])
            <= row["temperature_K"]
            <= float(fit_temperatures[-1])
        ],
        dtype=float,
    )
    summary = {
        "x_ba": x_ba,
        "ngrid": float(ngrid),
        "T0_reference_K": float(reference["T0_K"]),
        "curie_constant_reference_K": float(reference["curie_constant_K"]),
        "pressure_slope_GPa_per_K": slope,
        "pressure_intercept_GPa": intercept,
        "exact_pressure_min_GPa": float(np.min(exact)),
        "exact_pressure_max_GPa": float(np.max(exact)),
        "pressure_fit_rmse_GPa": float(np.sqrt(np.mean((fitted - exact) ** 2))),
        "epsilon_relative_error_rms_fit": float(
            np.sqrt(np.nanmean(fit_errors**2))
        ),
        "epsilon_relative_error_max_fit": float(np.nanmax(np.abs(fit_errors))),
    }
    return rows, summary


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    rows_by_composition: dict[float, list[dict[str, float]]],
    summaries: dict[float, dict[str, float]],
    fit_start: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compositions = sorted(rows_by_composition)
    fig, axes = plt.subplots(
        1,
        len(compositions),
        figsize=(12.0, 4.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(compositions) == 1:
        axes = np.array([axes])
    all_values: list[np.ndarray] = []
    for axis, x_ba in zip(axes, compositions):
        rows = rows_by_composition[x_ba]
        temperatures = np.asarray([row["temperature_K"] for row in rows])
        chi_zero = np.asarray([row["chi_soft_zero"] for row in rows])
        chi_fit = np.asarray([row["chi_soft_fit"] for row in rows])
        chi_reference = np.asarray(
            [row["chi_soft_reference_10kHz"] for row in rows]
        )
        all_values.extend((chi_zero, chi_fit, chi_reference))
        axis.axvspan(
            float(temperatures[0]),
            CENTRAL_MODE_CROSSOVER_K,
            color="0.92",
            zorder=0,
        )
        axis.axvline(
            CENTRAL_MODE_CROSSOVER_K,
            color="0.45",
            linestyle=":",
            linewidth=1.1,
        )
        axis.axvline(
            fit_start,
            color="tab:orange",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.plot(
            temperatures,
            chi_zero,
            "D-.",
            color="tab:purple",
            markersize=3.1,
            linewidth=1.35,
            label=r"VCA--SCHA, $p_{\rm eff}=0$",
        )
        axis.plot(
            temperatures,
            chi_fit,
            "^-",
            color="tab:orange",
            markersize=3.5,
            linewidth=1.5,
            label=r"VCA--SCHA, $p_{\rm fit}(T)$",
        )
        axis.plot(
            temperatures,
            chi_reference,
            "--",
            color="tab:green",
            linewidth=1.7,
            label="reconstructed Curie--Weiss envelope (not raw data)",
        )
        summary = summaries[x_ba]
        axis.set_title(rf"Ba fraction $x={x_ba:.1f}$")
        axis.text(
            0.97,
            0.95,
            (
                rf"$T_0={summary['T0_reference_K']:.0f}$ K" "\n"
                rf"$p_{{\rm fit}}={summary['pressure_slope_GPa_per_K']:.4f}T"
                rf"{summary['pressure_intercept_GPa']:+.2f}$ GPa" "\n"
                rf"max. err.={100.0 * summary['epsilon_relative_error_max_fit']:.2f}\%"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
        )
        axis.grid(alpha=0.22)
        axis.set_xlabel("Temperature (K)")
    axes[0].set_ylabel(r"Soft-mode susceptibility $\chi_T$")
    axes[0].text(
        0.03,
        0.04,
        "central-mode\nregime",
        transform=axes[0].transAxes,
        fontsize=7.8,
        color="0.35",
    )
    finite = np.concatenate(
        [values[np.isfinite(values)] for values in all_values]
    )
    axes[0].set_ylim(0.0, 1.08 * float(np.max(finite)))
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=8.6,
    )
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def _temperature_grid(start: float, stop: float, step: float) -> np.ndarray:
    return reference_scan.temperature_grid(start, stop, step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compositions", default="0.4,0.5,0.6")
    parser.add_argument("--start", type=float, default=300.0)
    parser.add_argument("--stop", type=float, default=600.0)
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--fit-start", type=float, default=400.0)
    parser.add_argument("--fit-stop", type=float, default=600.0)
    parser.add_argument("--fit-step", type=float, default=20.0)
    parser.add_argument("--ngrid", type=int, default=40)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compositions = tuple(float(value) for value in args.compositions.split(","))
    temperatures = _temperature_grid(args.start, args.stop, args.step)
    fit_temperatures = _temperature_grid(
        args.fit_start, args.fit_stop, args.fit_step
    )
    if args.fit_start < CENTRAL_MODE_CROSSOVER_K:
        raise SystemExit(
            f"fit-start must be at least {CENTRAL_MODE_CROSSOVER_K:g} K "
            "to exclude the resolved central-mode regime"
        )
    references = load_references(args.reference)
    rows_by_composition: dict[float, list[dict[str, float]]] = {}
    summaries: dict[float, dict[str, float]] = {}
    for x_ba in compositions:
        if x_ba not in references:
            raise SystemExit(
                f"no experimental reference is available for x={x_ba:g}"
            )
        rows, summary = scan_composition(
            x_ba,
            temperatures,
            fit_temperatures,
            args.ngrid,
            references[x_ba],
        )
        rows_by_composition[x_ba] = rows
        summaries[x_ba] = summary
        print(
            f"x={x_ba:.3f}: p_fit(T)="
            f"{summary['pressure_slope_GPa_per_K']:+.9g} T "
            f"{summary['pressure_intercept_GPa']:+.9g} GPa; "
            f"max relative epsilon error="
            f"{100.0 * summary['epsilon_relative_error_max_fit']:.4f}%"
        )

    output_dir = PIPELINE_DIR / "outputs" / "paraelectric"
    stem = f"bst_vca_x040_060_{args.start:g}_{args.stop:g}K_n{args.ngrid}"
    csv_path = args.csv or output_dir / f"{stem}.csv"
    plot_path = args.plot or output_dir / f"{stem}.png"
    summary_path = args.summary or output_dir / f"{stem}.json"
    all_rows = [row for x_ba in compositions for row in rows_by_composition[x_ba]]
    write_csv(all_rows, csv_path)
    write_plot(rows_by_composition, summaries, args.fit_start, plot_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "temperature_display_K": [args.start, args.stop, args.step],
                "pressure_fit_domain_K": [
                    args.fit_start,
                    args.fit_stop,
                    args.fit_step,
                ],
                "central_mode_crossover_K": CENTRAL_MODE_CROSSOVER_K,
                "reference": str(args.reference),
                "compositions": [summaries[x] for x in compositions],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"CSV: {csv_path}")
    print(f"plot: {plot_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
