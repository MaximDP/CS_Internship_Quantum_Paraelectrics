#!/usr/bin/env python3
"""Build the SrTiO3 analogue of the BaTiO3 pressure comparison.

The calculation uses the Nishimatsu et al. WC-GGA SrTiO3 Hamiltonian and the
same separated local/homogeneous/inhomogeneous quantum-SCHA closure as the
BaTiO3 scan.  Two affine pressure protocols are constructed independently:

* ``p_a(T)`` is a least-squares affine representation of the pointwise
  pressures reproducing the experimental cubic lattice expansion;
* ``p_chi(T)`` minimizes the vertical squared distance between the calculated
  soft-mode susceptibility and the Mueller--Burkard single-crystal Barrett
  reference on the requested interval.

The Barrett curve is experimentally calibrated, but it is an extrapolation
above the 300 K upper limit of the classic Neville et al. data.  The output
therefore labels it as a fitted/extrapolated reference rather than as direct
high-temperature points.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_chi_temperature as reference_scan
import scan_chi_temperature_inhomogeneous as acoustic_scan
import scha_paraelc as scha


BARRETT_C_K = 8.0e4
BARRETT_T0_K = 35.5
BARRETT_T1_K = 80.0

# Absolute room-temperature lattice constant (Schmidbauer et al., 2012) and
# the essentially constant volumetric expansion measured by de Ligny & Richet
# between 300 and 1800 K.  The small 298/300 K distinction is immaterial here.
LATTICE_AT_300_K_ANGSTROM = 3.905268
VOLUMETRIC_EXPANSION_PER_K = 3.23e-5
LATTICE_REFERENCE_TEMPERATURE_K = 300.0


def temperature_grid(start: float, stop: float, step: float) -> np.ndarray:
    if start <= 0.0 or step <= 0.0 or stop < start:
        raise ValueError("require start > 0, step > 0 and stop >= start")
    count = int(round((stop - start) / step))
    values = start + step * np.arange(count + 1, dtype=float)
    if not np.isclose(values[-1], stop, rtol=0.0, atol=1.0e-10):
        raise ValueError("the interval must be an integer multiple of the step")
    return values


def barrett_permittivity(temperature: np.ndarray | float) -> np.ndarray:
    values = np.asarray(temperature, dtype=float)
    denominator = 0.5 * BARRETT_T1_K / np.tanh(
        BARRETT_T1_K / (2.0 * values)
    ) - BARRETT_T0_K
    return BARRETT_C_K / denominator


def experimental_lattice_parameter(
    temperature: np.ndarray | float,
) -> np.ndarray:
    values = np.asarray(temperature, dtype=float)
    return LATTICE_AT_300_K_ANGSTROM * np.exp(
        VOLUMETRIC_EXPANSION_PER_K
        * (values - LATTICE_REFERENCE_TEMPERATURE_K)
        / 3.0
    )


def pressure_mass_shift_u(
    pressure_gpa: float,
    coefficients: scha.SchaCoefficients,
    parameters: acoustic_scan.AcousticQuarticParameters,
) -> float:
    """Hydrostatic pressure contribution to the local-mode quadratic kernel."""

    pressure_au = pressure_gpa * reference_scan.GPA_TO_HARTREE_PER_BOHR3
    elastic_bulk_au = (
        parameters.b11_ev + 2.0 * parameters.b12_ev
    ) * reference_scan.EV_TO_HARTREE
    mode_strain_bulk_au = (
        parameters.b1xx_ev_per_angstrom2
        + 2.0 * parameters.b1yy_ev_per_angstrom2
    ) * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    return float(
        -pressure_au
        * coefficients.omega0
        * mode_strain_bulk_au
        / elastic_bulk_au
    )


class Calculation:
    def __init__(self, ngrid: int) -> None:
        self.coefficients = scha.MATERIALS["nishimatsu_sto"]
        self.parameters = acoustic_scan.STO_ACOUSTIC_PARAMETERS
        self.grid = acoustic_scan.build_inhomogeneous_quartic_grid(
            ngrid,
            self.coefficients,
            parameters=self.parameters,
        )
        self.scale2 = (self.coefficients.omega0 / self.coefficients.zstar) ** 2
        self.shift_per_gpa = pressure_mass_shift_u(
            1.0, self.coefficients, self.parameters
        )

    def state(self, temperature: float, pressure_gpa: float) -> dict[str, float]:
        shift = self.shift_per_gpa * pressure_gpa
        a1_u = acoustic_scan.solve_root(
            temperature, shift, self.coefficients, self.grid
        )
        evaluated = acoustic_scan.residual(
            a1_u, temperature, self.coefficients, self.grid
        )
        lattice, strain, variance = acoustic_scan.cubic_lattice_parameter(
            temperature,
            pressure_gpa,
            evaluated[1],
            self.coefficients,
            self.parameters,
        )
        a1_p = self.scale2 * a1_u
        chi = self.coefficients.omega0 / a1_p
        epsilon = self.coefficients.eps_inf + 4.0 * np.pi * chi
        return {
            "A1_u_Ha_per_bohr2": float(a1_u),
            "A1_P": float(a1_p),
            "chi_soft": float(chi),
            "epsilon_r": float(epsilon),
            "lattice_parameter_angstrom": float(lattice),
            "isotropic_strain": float(strain),
            "variance_u_component_bohr2": float(variance),
            "delta_A1_local": float(evaluated[2]),
            "delta_A1_homogeneous": float(evaluated[3]),
            "delta_A1_inhomogeneous": float(evaluated[4]),
            "delta_A1_sextic": float(evaluated[5]),
            "delta_A1_octic": float(evaluated[6]),
        }

    def pointwise_lattice_pressure(
        self,
        temperature: float,
        target_lattice: float,
        initial_pressure: float,
    ) -> float:
        def objective(pressure: float) -> float:
            return (
                self.state(temperature, pressure)["lattice_parameter_angstrom"]
                - target_lattice
            )

        # The solution is close to the previous-temperature value.  Establish
        # a small stable bracket first, then widen it only if required.
        for half_width in (0.35, 0.75, 1.5, 3.0, 6.0):
            lower = max(-8.0, initial_pressure - half_width)
            upper = min(3.0, initial_pressure + half_width)
            try:
                lower_value = objective(lower)
                upper_value = objective(upper)
            except RuntimeError:
                continue
            if lower_value * upper_value <= 0.0:
                return float(
                    brentq(
                        objective,
                        lower,
                        upper,
                        xtol=2.0e-10,
                        rtol=2.0e-10,
                    )
                )
        raise RuntimeError(
            f"no stable lattice-matching pressure at T={temperature:g} K"
        )

    def pointwise_dielectric_pressure(
        self, temperature: float, target_chi: float
    ) -> float:
        target_a1_u = (
            self.coefficients.omega0 / target_chi / self.scale2
        )
        threshold = scha.stable_a1_threshold(
            self.grid.ngrid,
            self.coefficients.kmax,
            self.coefficients,
        )
        if target_a1_u <= threshold:
            raise RuntimeError("the dielectric target is outside the stable branch")
        required_shift = acoustic_scan.residual(
            target_a1_u,
            temperature,
            self.coefficients,
            self.grid,
        )[0]
        return float(required_shift / self.shift_per_gpa)


def fit_lattice_pressure(
    calculation: Calculation,
    temperatures: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    pressures = []
    initial = 0.0
    for temperature, target in zip(temperatures, targets):
        initial = calculation.pointwise_lattice_pressure(
            float(temperature), float(target), initial
        )
        pressures.append(initial)
    pointwise = np.asarray(pressures, dtype=float)
    reference_temperature = float(np.mean(temperatures))
    slope, pressure_at_reference = np.polyfit(
        temperatures - reference_temperature, pointwise, 1
    )
    return float(slope), float(pressure_at_reference), pointwise


def fit_dielectric_pressure(
    calculation: Calculation,
    temperatures: np.ndarray,
    target_chi: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Minimize the unweighted vertical chi-distance by Gauss--Newton."""

    pointwise = np.asarray(
        [
            calculation.pointwise_dielectric_pressure(float(t), float(chi))
            for t, chi in zip(temperatures, target_chi)
        ],
        dtype=float,
    )
    reference_temperature = float(np.mean(temperatures))

    # At the exact pointwise targets, dchi/dp supplies the correct weights for
    # the affine line in the vertical least-squares objective.
    sensitivities = []
    for temperature, pressure in zip(temperatures, pointwise):
        state = calculation.state(float(temperature), float(pressure))
        a1 = state["A1_u_Ha_per_bohr2"]
        delta = max(1.0e-9, 1.0e-4 * a1)
        derivative = (
            acoustic_scan.residual(
                a1 + delta,
                float(temperature),
                calculation.coefficients,
                calculation.grid,
            )[0]
            - acoustic_scan.residual(
                a1 - delta,
                float(temperature),
                calculation.coefficients,
                calculation.grid,
            )[0]
        ) / (2.0 * delta)
        sensitivities.append(
            abs(
                -state["chi_soft"]
                * calculation.shift_per_gpa
                / (a1 * derivative)
            )
        )
    weights = np.asarray(sensitivities, dtype=float)
    design = np.column_stack(
        (temperatures - reference_temperature, np.ones_like(temperatures))
    )
    slope, pressure_at_reference = np.linalg.lstsq(
        design * weights[:, None], pointwise * weights, rcond=None
    )[0]

    # Refine the nonlinear susceptibility objective itself.  The implicit
    # derivative of the SCHA root gives an analytic two-column Jacobian.
    for _ in range(6):
        errors = []
        jacobian = []
        for temperature, target in zip(temperatures, target_chi):
            centered_temperature = float(temperature - reference_temperature)
            pressure = slope * centered_temperature + pressure_at_reference
            state = calculation.state(float(temperature), float(pressure))
            a1 = state["A1_u_Ha_per_bohr2"]
            delta = max(1.0e-9, 1.0e-4 * a1)
            derivative = (
                acoustic_scan.residual(
                    a1 + delta,
                    float(temperature),
                    calculation.coefficients,
                    calculation.grid,
                )[0]
                - acoustic_scan.residual(
                    a1 - delta,
                    float(temperature),
                    calculation.coefficients,
                    calculation.grid,
                )[0]
            ) / (2.0 * delta)
            dchi_dp = (
                -state["chi_soft"]
                * calculation.shift_per_gpa
                / (a1 * derivative)
            )
            errors.append(state["chi_soft"] - float(target))
            jacobian.append(
                dchi_dp * np.array((centered_temperature, 1.0), dtype=float)
            )
        correction = np.linalg.lstsq(
            np.asarray(jacobian), -np.asarray(errors), rcond=None
        )[0]
        slope += correction[0]
        pressure_at_reference += correction[1]
        if np.max(np.abs(design @ correction)) < 1.0e-10:
            break
    return float(slope), float(pressure_at_reference), pointwise


def line_values(
    temperatures: np.ndarray,
    slope: float,
    value_at_reference: float,
) -> np.ndarray:
    return slope * (temperatures - np.mean(temperatures)) + value_at_reference


def evaluate_protocol(
    calculation: Calculation,
    temperatures: np.ndarray,
    pressures: np.ndarray,
) -> list[dict[str, float]]:
    return [
        calculation.state(float(temperature), float(pressure))
        for temperature, pressure in zip(temperatures, pressures)
    ]


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperature = np.asarray([row["temperature_K"] for row in rows])
    fig, axis = plt.subplots(figsize=(7.7, 4.8), constrained_layout=True)
    axis.plot(
        temperature,
        [row["chi_soft_lattice_fit"] for row in rows],
        "o-",
        color="#1f77b4",
        markersize=3.2,
        linewidth=1.5,
        label=r"SCHA, lattice-matched $p_a(T)$",
    )
    axis.plot(
        temperature,
        [row["chi_soft_dielectric_fit"] for row in rows],
        "^-",
        color="#ff7f0e",
        markersize=3.2,
        linewidth=1.5,
        label=r"SCHA, dielectric-fitted $p_\chi(T)$",
    )
    axis.plot(
        temperature,
        [row["chi_soft_zero_pressure"] for row in rows],
        "D-",
        color="#9467bd",
        markersize=3.0,
        linewidth=1.4,
        label=r"SCHA, $p_{\rm eff}=0$",
    )
    dense_temperature = np.linspace(temperature[0], temperature[-1], 800)
    dense_chi = (
        barrett_permittivity(dense_temperature)
        - scha.MATERIALS["nishimatsu_sto"].eps_inf
    ) / (4.0 * np.pi)
    axis.plot(
        dense_temperature,
        dense_chi,
        "--",
        color="#2ca02c",
        linewidth=1.7,
        label="Müller–Burkard Barrett fit (extrapolated)",
    )
    axis.set_xlim(float(temperature[0]), float(temperature[-1]))
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"Soft-mode susceptibility $\chi_{\rm soft}$")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8.3)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=400.0)
    parser.add_argument("--stop", type=float, default=565.0)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--ngrid", type=int, default=40)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    if args.ngrid % 2:
        raise SystemExit("--ngrid must be even")

    temperatures = temperature_grid(args.start, args.stop, args.step)
    calculation = Calculation(args.ngrid)
    lattice_targets = experimental_lattice_parameter(temperatures)
    epsilon_reference = barrett_permittivity(temperatures)
    chi_reference = (
        epsilon_reference - calculation.coefficients.eps_inf
    ) / (4.0 * np.pi)

    lattice_slope, lattice_at_reference, lattice_pointwise = fit_lattice_pressure(
        calculation, temperatures, lattice_targets
    )
    chi_slope, chi_at_reference, chi_pointwise = fit_dielectric_pressure(
        calculation, temperatures, chi_reference
    )
    zero_pressures = np.zeros_like(temperatures)
    lattice_pressures = line_values(
        temperatures, lattice_slope, lattice_at_reference
    )
    chi_pressures = line_values(temperatures, chi_slope, chi_at_reference)

    zero_states = evaluate_protocol(calculation, temperatures, zero_pressures)
    lattice_states = evaluate_protocol(
        calculation, temperatures, lattice_pressures
    )
    chi_states = evaluate_protocol(calculation, temperatures, chi_pressures)

    rows: list[dict[str, float]] = []
    for index, temperature in enumerate(temperatures):
        rows.append(
            {
                "temperature_K": float(temperature),
                "epsilon_r_Barrett_reference": float(epsilon_reference[index]),
                "chi_soft_Barrett_reference": float(chi_reference[index]),
                "lattice_target_angstrom": float(lattice_targets[index]),
                "pressure_lattice_pointwise_GPa": float(lattice_pointwise[index]),
                "pressure_lattice_affine_GPa": float(lattice_pressures[index]),
                "pressure_dielectric_pointwise_GPa": float(chi_pointwise[index]),
                "pressure_dielectric_affine_GPa": float(chi_pressures[index]),
                "chi_soft_zero_pressure": zero_states[index]["chi_soft"],
                "epsilon_r_zero_pressure": zero_states[index]["epsilon_r"],
                "lattice_zero_pressure_angstrom": zero_states[index][
                    "lattice_parameter_angstrom"
                ],
                "chi_soft_lattice_fit": lattice_states[index]["chi_soft"],
                "epsilon_r_lattice_fit": lattice_states[index]["epsilon_r"],
                "lattice_lattice_fit_angstrom": lattice_states[index][
                    "lattice_parameter_angstrom"
                ],
                "chi_soft_dielectric_fit": chi_states[index]["chi_soft"],
                "epsilon_r_dielectric_fit": chi_states[index]["epsilon_r"],
                "lattice_dielectric_fit_angstrom": chi_states[index][
                    "lattice_parameter_angstrom"
                ],
                "A1_P_zero_pressure": zero_states[index]["A1_P"],
                "A1_P_lattice_fit": lattice_states[index]["A1_P"],
                "A1_P_dielectric_fit": chi_states[index]["A1_P"],
            }
        )

    reference_temperature = float(np.mean(temperatures))
    chi_fit_values = np.asarray(
        [row["chi_soft_dielectric_fit"] for row in rows]
    )
    zero_values = np.asarray([row["chi_soft_zero_pressure"] for row in rows])
    lattice_values = np.asarray([row["chi_soft_lattice_fit"] for row in rows])
    lattice_fit_values = np.asarray(
        [row["lattice_lattice_fit_angstrom"] for row in rows]
    )

    def metrics(values: np.ndarray) -> dict[str, float]:
        errors = values - chi_reference
        return {
            "rmse_chi": float(np.sqrt(np.mean(errors**2))),
            "mean_absolute_relative_error_percent": float(
                100.0 * np.mean(np.abs(errors / chi_reference))
            ),
            "max_absolute_relative_error_percent": float(
                100.0 * np.max(np.abs(errors / chi_reference))
            ),
        }

    metadata = {
        "material": "nishimatsu_2016_sto",
        "ngrid": args.ngrid,
        "temperature_start_K": args.start,
        "temperature_stop_K": args.stop,
        "temperature_step_K": args.step,
        "pressure_reference_temperature_K": reference_temperature,
        "lattice_pressure_slope_GPa_per_K": lattice_slope,
        "lattice_pressure_at_reference_GPa": lattice_at_reference,
        "dielectric_pressure_slope_GPa_per_K": chi_slope,
        "dielectric_pressure_at_reference_GPa": chi_at_reference,
        "lattice_fit_rmse_angstrom": float(
            np.sqrt(np.mean((lattice_fit_values - lattice_targets) ** 2))
        ),
        "lattice_fit_max_abs_error_angstrom": float(
            np.max(np.abs(lattice_fit_values - lattice_targets))
        ),
        "zero_pressure_metrics": metrics(zero_values),
        "lattice_pressure_metrics": metrics(lattice_values),
        "dielectric_pressure_metrics": metrics(chi_fit_values),
        "barrett_parameters": {
            "C_K": BARRETT_C_K,
            "T0_K": BARRETT_T0_K,
            "T1_K": BARRETT_T1_K,
        },
        "lattice_reference": {
            "a_at_300_K_angstrom": LATTICE_AT_300_K_ANGSTROM,
            "volumetric_expansion_per_K": VOLUMETRIC_EXPANSION_PER_K,
        },
    }

    stem = f"srtio3_pressure_comparison_{args.start:g}_{args.stop:g}K_n{args.ngrid}"
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "paraelectric"
    csv_path = args.csv or output_dir / f"{stem}.csv"
    json_path = args.json or output_dir / f"{stem}.json"
    plot_path = args.plot or output_dir / f"{stem}.png"
    write_csv(rows, csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_plot(rows, plot_path)

    print(json.dumps(metadata, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
