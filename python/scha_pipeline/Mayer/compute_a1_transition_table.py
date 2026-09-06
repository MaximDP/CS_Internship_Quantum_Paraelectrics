#!/usr/bin/env python3
"""Compute paraelectric spinodals from ``A1(T) -> 0+`` on the n=40 grid.

The table covers every pressure protocol displayed in the Nishimatsu and
Mayer susceptibility benchmarks.  Pointwise lattice matching is evaluated at
fixed ``A1`` from the cubic equation of state; affine laws are extrapolated as
written in the report.  These roots are local-stability limits of the centered
paraelectric branch, not first-order coexistence temperatures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from scipy.optimize import brentq


MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
PARAELECTRIC_DIR = PIPELINE_DIR / "SCHA_paraelc"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(PARAELECTRIC_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import parameters as mayer
import run_mayer_bto as mayer_scan
import scan_chi_temperature as reference_scan
import scan_chi_temperature_inhomogeneous as inhomogeneous
import scha_paraelc as scha


NGRID = 40
A1_LIMIT = 1.0e-10
OUTPUT = MODULE_DIR.parent / "outputs" / "mayer" / "a1_transition_temperatures_n40.csv"
MAYER_SUMMARY = MODULE_DIR.parent / "outputs" / "reference" / "mayer" / "mayer_bto_400_800K_n40.json"

NISHIMATSU_LATTICE_AFFINE = (-0.004859728409, 0.250696172704)
NISHIMATSU_RESPONSE_AFFINE = (0.006347464062, -8.194337108)


PressureFunction = Callable[[float, float], float]
ShiftFunction = Callable[[float], float]


def affine_pressure(slope: float, intercept: float) -> PressureFunction:
    return lambda temperature, _a1: slope * temperature + intercept


def lattice_matching_pressure(
    temperature: float,
    a1: float,
    coefficients: scha.SchaCoefficients,
    grid: inhomogeneous.InhomogeneousQuarticGrid,
    target_lattice_angstrom: float,
) -> float:
    """Invert the cubic equation of state at a fixed trial mass."""

    gmat = inhomogeneous.residual(
        a1, temperature, coefficients, grid
    )[1]
    parameters = grid.parameters
    component_variance = float(
        scha.KB_HARTREE_PER_K * temperature * np.trace(gmat) / 3.0
    )
    elastic_bulk = (
        parameters.b11_ev + 2.0 * parameters.b12_ev
    ) * reference_scan.EV_TO_HARTREE
    mode_strain_bulk = (
        parameters.b1xx_ev_per_angstrom2
        + 2.0 * parameters.b1yy_ev_per_angstrom2
    ) * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    target_strain = (
        target_lattice_angstrom
        / (coefficients.a0 * reference_scan.ANGSTROM_PER_BOHR)
        - 1.0
    )
    pressure_energy = (
        -target_strain * elastic_bulk
        - 0.5 * mode_strain_bulk * component_variance
    )
    return float(
        pressure_energy
        / (
            reference_scan.GPA_TO_HARTREE_PER_BOHR3
            * coefficients.a0**3
        )
    )


def transition_root(
    coefficients: scha.SchaCoefficients,
    grid: inhomogeneous.InhomogeneousQuarticGrid,
    pressure: PressureFunction,
    pressure_shift: ShiftFunction,
    a1_limit: float,
) -> tuple[float, float]:
    def stability_residual(temperature: float) -> float:
        pressure_gpa = pressure(temperature, a1_limit)
        return float(
            inhomogeneous.residual(
                a1_limit, temperature, coefficients, grid
            )[0]
            - pressure_shift(pressure_gpa)
        )

    temperature = float(
        brentq(
            stability_residual,
            20.0,
            500.0,
            xtol=1.0e-9,
            rtol=1.0e-12,
        )
    )
    return temperature, float(pressure(temperature, a1_limit))


def main() -> None:
    nishimatsu_coefficients = scha.MATERIALS["nishimatsu"]
    nishimatsu_grid = inhomogeneous.build_inhomogeneous_quartic_grid(
        NGRID, nishimatsu_coefficients
    )
    mayer_coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
    mayer_acoustic = mayer.MAYER_ANHARMONIC_ACOUSTIC_PARAMETERS
    mayer_grid = inhomogeneous.build_inhomogeneous_quartic_grid(
        NGRID,
        mayer_coefficients,
        acoustic_mass_amu=mayer_acoustic.acoustic_mass_amu,
        parameters=mayer_acoustic,
    )

    lattice_slope, lattice_intercept, *_ = mayer_scan.fit_lattice_reference(
        413.0, 598.0
    )
    with MAYER_SUMMARY.open(encoding="utf-8") as stream:
        mayer_summary = json.load(stream)
    mayer_affine = mayer_summary["affine_pressure_diagnostic"]

    def pointwise_lattice(
        coefficients: scha.SchaCoefficients,
        grid: inhomogeneous.InhomogeneousQuarticGrid,
    ) -> PressureFunction:
        return lambda temperature, a1: lattice_matching_pressure(
            temperature,
            a1,
            coefficients,
            grid,
            lattice_slope * temperature + lattice_intercept,
        )

    nishimatsu_shift = lambda pressure: reference_scan.pressure_mass_shift_u(
        pressure, nishimatsu_coefficients
    )
    mayer_shift = lambda pressure: mayer_scan.pressure_mass_shift_u(
        pressure, mayer_coefficients, mayer_grid.parameters
    )

    protocols = [
        (
            "Nishimatsu WC-GGA",
            "zero pressure",
            nishimatsu_coefficients,
            nishimatsu_grid,
            affine_pressure(0.0, 0.0),
            nishimatsu_shift,
            "none",
        ),
        (
            "Nishimatsu WC-GGA",
            "pointwise lattice match",
            nishimatsu_coefficients,
            nishimatsu_grid,
            pointwise_lattice(nishimatsu_coefficients, nishimatsu_grid),
            nishimatsu_shift,
            "Nakatani cubic a(T), 413-598 K; formal extrapolation",
        ),
        (
            "Nishimatsu WC-GGA",
            "affine lattice pressure",
            nishimatsu_coefficients,
            nishimatsu_grid,
            affine_pressure(*NISHIMATSU_LATTICE_AFFINE),
            nishimatsu_shift,
            "affine compression fitted over 400-565 K; extrapolated",
        ),
        (
            "Nishimatsu WC-GGA",
            "Wieczorek response fit",
            nishimatsu_coefficients,
            nishimatsu_grid,
            affine_pressure(*NISHIMATSU_RESPONSE_AFFINE),
            nishimatsu_shift,
            "fit over 415.65-448.15 K; extrapolated",
        ),
        (
            "Mayer PBEsol anharmonic",
            "zero pressure",
            mayer_coefficients,
            mayer_grid,
            affine_pressure(0.0, 0.0),
            mayer_shift,
            "none",
        ),
        (
            "Mayer PBEsol anharmonic",
            "pointwise lattice match",
            mayer_coefficients,
            mayer_grid,
            pointwise_lattice(mayer_coefficients, mayer_grid),
            mayer_shift,
            "Nakatani cubic a(T), 413-598 K; formal extrapolation",
        ),
        (
            "Mayer PBEsol anharmonic",
            "affine lattice pressure",
            mayer_coefficients,
            mayer_grid,
            affine_pressure(
                float(mayer_affine["slope_GPa_per_K"]),
                float(mayer_affine["intercept_GPa"]),
            ),
            mayer_shift,
            "affine compression fitted over 400-800 K; extrapolated",
        ),
    ]

    rows = []
    for (
        parameter_set,
        protocol,
        coefficients,
        grid,
        pressure,
        pressure_shift,
        provenance,
    ) in protocols:
        transition, pressure_at_transition = transition_root(
            coefficients,
            grid,
            pressure,
            pressure_shift,
            A1_LIMIT,
        )
        convergence_transition, _ = transition_root(
            coefficients,
            grid,
            pressure,
            pressure_shift,
            1.0e-12,
        )
        rows.append(
            {
                "parameter_set": parameter_set,
                "protocol": protocol,
                "T_A1_to_zero_K": f"{transition:.7f}",
                "p_eff_at_transition_GPa": f"{pressure_at_transition:.7f}",
                "A1_limit_Ha_per_bohr2": f"{A1_LIMIT:.1e}",
                "limit_check_abs_delta_T_K": (
                    f"{abs(transition - convergence_transition):.3e}"
                ),
                "pressure_provenance": provenance,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['parameter_set']}: {row['protocol']}: "
            f"T={row['T_A1_to_zero_K']} K, "
            f"p={row['p_eff_at_transition_GPa']} GPa"
        )


if __name__ == "__main__":
    main()
