"""Mayer et al. (PRB 106, 064108, 2022) BaTiO3 parameters.

This module is deliberately isolated from the production material registry.
It stores both the conventional PBEsol fit from Table I and the effective
anharmonic fit obtained after minimizing the higher optical modes v1 and v2.

The full periodic FERAM kernel is used by the surrounding SCHA pipeline, so
the continuum coefficients A02--A04 are unused diagnostics here and are set to
zero.  A01 is the harmonic curvature 2*kappa in the kernel convention, while
A05 is rebuilt from Z*, epsilon_infinity, and the unit-cell volume.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import itertools
from pathlib import Path
import sys

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
PARAELECTRIC_DIR = PIPELINE_DIR / "SCHA_paraelc"
sys.path.insert(0, str(PARAELECTRIC_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import scha_paraelc as scha
import scan_chi_temperature_inhomogeneous as inhomogeneous


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def ev_per_angstrom_power_to_hartree_per_bohr_power(
    value: float, power: int
) -> float:
    return float(value / HARTREE_TO_EV * BOHR_TO_ANGSTROM**power)


# Values transcribed from Mayer et al., Table I.  The k1/k4 entries in this
# dictionary are the conventional PBEsol values before the v1/v2 reduction.
MAYER_TABLE_I = {
    "a0_angstrom": 3.987,
    "polar_mass_amu": 38.148,
    "kappa_ev_per_angstrom2": -1.965,
    "kappa2_ev_per_angstrom2": 8.007,
    "alpha_ev_per_angstrom4": 123.492,
    "gamma_ev_per_angstrom4": -165.344,
    "k1_ev_per_angstrom6": -528.388,
    "k2_ev_per_angstrom6": 123.688,
    "k3_ev_per_angstrom6": 307.317,
    "k4_ev_per_angstrom8": 3370.229,
    "b11_ev": 126.137,
    "b12_ev": 42.391,
    "b44_ev": 50.046,
    "b1xx_ev_per_angstrom2": -235.064,
    "b1yy_ev_per_angstrom2": -19.341,
    "b4yz_ev_per_angstrom2": -15.333,
    "zstar_e": 10.267,
    "eps_inf": 6.847,
    "j1_ev_per_angstrom2": -2.060,
    "j2_ev_per_angstrom2": -1.173,
    "j3_ev_per_angstrom2": 0.680,
    "j4_ev_per_angstrom2": -0.610,
    "j5_ev_per_angstrom2": 0.000,
    "j6_ev_per_angstrom2": 0.277,
    "j7_ev_per_angstrom2": 0.000,
}


# Mayer et al., Sec. II D: Eq. (4) is minimized over v1 and v2 and the
# resulting E_001(u) is refitted by changing only k1 and k4.
MAYER_ANHARMONIC_REDUCTION = {
    "k1_conventional_ev_per_angstrom6": -528.388,
    "k4_conventional_ev_per_angstrom8": 3370.229,
    "k1_effective_ev_per_angstrom6": -1443.850,
    "k4_effective_ev_per_angstrom8": 17216.816,
    "u_fit_min_angstrom": 0.0,
    "u_fit_max_angstrom": 0.25,
}


# Value used for the delivered curves. Nishimatsu 2016 Table I lists
# 46.64 amu for BTO; 46.44 is retained here for their exact reproduction.
# It is distinct from the polar mass and is not a Mayer Table I fit parameter.
FERAM_ACOUSTIC_MASS_AMU = 46.44


def _bare_quartic_invariants() -> tuple[float, float]:
    alpha = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["alpha_ev_per_angstrom4"], 4
    )
    gamma = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["gamma_ev_per_angstrom4"], 4
    )
    return (2.0 / 3.0) * (2.0 * alpha + gamma), -2.0 * gamma


def _homogeneous_lambda_invariants() -> tuple[float, float]:
    """Return the cubic invariants of B C^-1 B in pipeline conventions."""

    b11 = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["b11_ev"], 0
    )
    b12 = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["b12_ev"], 0
    )
    b44 = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["b44_ev"], 0
    )
    elastic = np.array(
        [
            [b11, b12, b12, 0.0, 0.0, 0.0],
            [b12, b11, b12, 0.0, 0.0, 0.0],
            [b12, b12, b11, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, b44, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, b44, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, b44],
        ],
        dtype=float,
    )
    b1xx = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["b1xx_ev_per_angstrom2"], 2
    )
    b1yy = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["b1yy_ev_per_angstrom2"], 2
    )
    b4yz = ev_per_angstrom_power_to_hartree_per_bohr_power(
        MAYER_TABLE_I["b4yz_ev_per_angstrom2"], 2
    )
    coupling = np.zeros((6, 3, 3), dtype=float)
    for voigt, axis in ((0, 0), (1, 1), (2, 2)):
        coupling[voigt, axis, axis] = b1xx
    for voigt, axis in (
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ):
        coupling[voigt, axis, axis] = b1yy
    for voigt, first, second in ((3, 1, 2), (4, 0, 2), (5, 0, 1)):
        coupling[voigt, first, second] = b4yz
        coupling[voigt, second, first] = b4yz
    tensor = np.einsum(
        "lab,lm,mcd->abcd",
        coupling,
        np.linalg.inv(elastic),
        coupling,
        optimize=True,
    )
    coefficient_x4 = float(tensor[0, 0, 0, 0])
    coefficient_x2y2 = sum(
        float(tensor[index])
        for index in set(itertools.permutations((0, 0, 1, 1), 4))
    )
    b1 = coefficient_x2y2 / 6.0
    b2 = coefficient_x4 - 3.0 * b1
    return float(b1), float(b2)


BARE_B1, BARE_B2 = _bare_quartic_invariants()
HOMOGENEOUS_LAMBDA_B1, HOMOGENEOUS_LAMBDA_B2 = (
    _homogeneous_lambda_invariants()
)
EFFECTIVE_B1 = BARE_B1 - 0.5 * HOMOGENEOUS_LAMBDA_B1
EFFECTIVE_B2 = BARE_B2 - 0.5 * HOMOGENEOUS_LAMBDA_B2


def _coefficients(
    name: str,
    k1_ev_per_angstrom6: float,
    k4_ev_per_angstrom8: float,
) -> scha.SchaCoefficients:
    a0 = MAYER_TABLE_I["a0_angstrom"] / BOHR_TO_ANGSTROM
    omega0 = a0**3
    zstar = MAYER_TABLE_I["zstar_e"]
    eps_inf = MAYER_TABLE_I["eps_inf"]
    return scha.SchaCoefficients(
        name=name,
        a0=a0,
        # Mayer's kappa multiplies u^2 in E_001; the trial kernel is a
        # curvature, hence the factor two.
        A01=ev_per_angstrom_power_to_hartree_per_bohr_power(
            2.0 * MAYER_TABLE_I["kappa_ev_per_angstrom2"], 2
        ),
        # Unused by the full periodic lattice quadrature in this branch.
        A02=0.0,
        A03=0.0,
        A04=0.0,
        A05=4.0 * np.pi * zstar**2 / (eps_inf * omega0),
        b1_eff=EFFECTIVE_B1,
        b2_eff=EFFECTIVE_B2,
        k1_6=ev_per_angstrom_power_to_hartree_per_bohr_power(
            k1_ev_per_angstrom6, 6
        ),
        k2_6=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["k2_ev_per_angstrom6"], 6
        ),
        k3_6=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["k3_ev_per_angstrom6"], 6
        ),
        k4_8=ev_per_angstrom_power_to_hartree_per_bohr_power(
            k4_ev_per_angstrom8, 8
        ),
        mass_amu=MAYER_TABLE_I["polar_mass_amu"],
        zstar=zstar,
        eps_inf=eps_inf,
        kappa2=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["kappa2_ev_per_angstrom2"], 2
        ),
        j1=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["j1_ev_per_angstrom2"], 2
        ),
        j2=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["j2_ev_per_angstrom2"], 2
        ),
        j3=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["j3_ev_per_angstrom2"], 2
        ),
        j4=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["j4_ev_per_angstrom2"], 2
        ),
        j5=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["j5_ev_per_angstrom2"], 2
        ),
        j6=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["j6_ev_per_angstrom2"], 2
        ),
        j7=ev_per_angstrom_power_to_hartree_per_bohr_power(
            MAYER_TABLE_I["j7_ev_per_angstrom2"], 2
        ),
    )


MAYER_CONVENTIONAL_COEFFICIENTS = _coefficients(
    "mayer_2022_pbesol_conventional_bto",
    MAYER_ANHARMONIC_REDUCTION["k1_conventional_ev_per_angstrom6"],
    MAYER_ANHARMONIC_REDUCTION["k4_conventional_ev_per_angstrom8"],
)
MAYER_ANHARMONIC_COEFFICIENTS = _coefficients(
    "mayer_2022_pbesol_anharmonic_bto",
    MAYER_ANHARMONIC_REDUCTION["k1_effective_ev_per_angstrom6"],
    MAYER_ANHARMONIC_REDUCTION["k4_effective_ev_per_angstrom8"],
)


def _acoustic_parameters(name: str) -> inhomogeneous.AcousticQuarticParameters:
    return inhomogeneous.AcousticQuarticParameters(
        coefficients_name=name,
        acoustic_mass_amu=FERAM_ACOUSTIC_MASS_AMU,
        b11_ev=MAYER_TABLE_I["b11_ev"],
        b12_ev=MAYER_TABLE_I["b12_ev"],
        b44_ev=MAYER_TABLE_I["b44_ev"],
        b1xx_ev_per_angstrom2=MAYER_TABLE_I["b1xx_ev_per_angstrom2"],
        b1yy_ev_per_angstrom2=MAYER_TABLE_I["b1yy_ev_per_angstrom2"],
        b4yz_ev_per_angstrom2=MAYER_TABLE_I["b4yz_ev_per_angstrom2"],
        bare_b1=BARE_B1,
        bare_b2=BARE_B2,
        homogeneous_lambda_b1=HOMOGENEOUS_LAMBDA_B1,
        homogeneous_lambda_b2=HOMOGENEOUS_LAMBDA_B2,
    )


MAYER_CONVENTIONAL_ACOUSTIC_PARAMETERS = _acoustic_parameters(
    MAYER_CONVENTIONAL_COEFFICIENTS.name
)
MAYER_ANHARMONIC_ACOUSTIC_PARAMETERS = _acoustic_parameters(
    MAYER_ANHARMONIC_COEFFICIENTS.name
)


def serializable_parameter_snapshot() -> dict[str, object]:
    return {
        "source": {
            "citation": "F. Mayer et al., Phys. Rev. B 106, 064108 (2022)",
            "doi": "10.1103/PhysRevB.106.064108",
            "exchange_correlation": "PBEsol",
            "table_I": dict(MAYER_TABLE_I),
            "anharmonic_reduction": dict(MAYER_ANHARMONIC_REDUCTION),
        },
        "explicit_assumptions": {
            "acoustic_mass_amu": FERAM_ACOUSTIC_MASS_AMU,
            "acoustic_mass_provenance": (
                "reference-curve value 46.44 amu; Nishimatsu 2016 Table I gives 46.64 amu; "
                "not a refitted anharmonic coefficient from Mayer Table I"
            ),
            "A01_mapping": "A01 = 2*kappa after unit conversion",
            "A02_A04": "set to zero because the full periodic FERAM kernel is used",
        },
        "derived_quartic_invariants_hartree_bohr": {
            "bare_b1": BARE_B1,
            "bare_b2": BARE_B2,
            "homogeneous_lambda_b1": HOMOGENEOUS_LAMBDA_B1,
            "homogeneous_lambda_b2": HOMOGENEOUS_LAMBDA_B2,
            "effective_b1": EFFECTIVE_B1,
            "effective_b2": EFFECTIVE_B2,
        },
        "converted_conventional": asdict(MAYER_CONVENTIONAL_COEFFICIENTS),
        "converted_anharmonic": asdict(MAYER_ANHARMONIC_COEFFICIENTS),
        "acoustic_parameters": asdict(MAYER_ANHARMONIC_ACOUSTIC_PARAMETERS),
    }
