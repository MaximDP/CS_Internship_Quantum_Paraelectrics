"""Constants and tensor helpers for cubic-perovskite quartic tensors.

Source: Zhong, Vanderbilt and Rabe, Phys. Rev. B 52, 6301 (1995),
Table II and Eqs. (3), (12), (14).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np


HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903


def ev_per_angstrom_power_to_hartree_per_bohr_power(value: float, power: int) -> float:
    return value / HARTREE_TO_EV * (BOHR_TO_ANGSTROM**power)


@dataclass(frozen=True)
class QuarticMaterial:
    elastic: dict[str, float]
    electrostrictive: dict[str, float]
    onsite: dict[str, float]


VOIGT_LABELS = ("xx", "yy", "zz", "yz", "xz", "xy")

# Energies are in Hartree, as stated in Table II of the paper.
ELASTIC = {
    "B11": 4.64,
    "B12": 1.65,
    "B44": 1.85,
}

ELECTROSTRICTIVE_VANDERBILT = {
    "B1xx": -2.18,
    "B1yy": -0.20,
    "B4yz": -0.08,
}

ONSITE_QUARTIC_VANDERBILT = {
    "alpha": 0.320,
    "gamma": -0.473,
}

NISHIMATSU_BTO = QuarticMaterial(
    elastic={
        # B_ij multiply dimensionless strains and are energies.
        "B11": ev_per_angstrom_power_to_hartree_per_bohr_power(126.731671475652, 0),
        "B12": ev_per_angstrom_power_to_hartree_per_bohr_power(41.7582963902597, 0),
        "B44": ev_per_angstrom_power_to_hartree_per_bohr_power(49.2408864348646, 0),
    },
    electrostrictive={
        # B_1ab multiply eta*u_a*u_b and are in eV/Angstrom^2.
        "B1xx": ev_per_angstrom_power_to_hartree_per_bohr_power(-185.347187551195, 2),
        "B1yy": ev_per_angstrom_power_to_hartree_per_bohr_power(-3.28092949275452, 2),
        "B4yz": ev_per_angstrom_power_to_hartree_per_bohr_power(-14.5501738943852, 2),
    },
    onsite={
        "alpha": ev_per_angstrom_power_to_hartree_per_bohr_power(78.9866142426711, 4),
        "gamma": ev_per_angstrom_power_to_hartree_per_bohr_power(-115.484148812671, 4),
    },
)

NISHIMATSU_STO = QuarticMaterial(
    elastic={
        # Nishimatsu et al., JPSJ 85, 114714 (2016), Table I.
        "B11": ev_per_angstrom_power_to_hartree_per_bohr_power(131.33, 0),
        "B12": ev_per_angstrom_power_to_hartree_per_bohr_power(36.26, 0),
        "B44": ev_per_angstrom_power_to_hartree_per_bohr_power(41.30, 0),
    },
    electrostrictive={
        "B1xx": ev_per_angstrom_power_to_hartree_per_bohr_power(-102.09, 2),
        "B1yy": ev_per_angstrom_power_to_hartree_per_bohr_power(+0.5299, 2),
        "B4yz": ev_per_angstrom_power_to_hartree_per_bohr_power(-15.494, 2),
    },
    onsite={
        "alpha": ev_per_angstrom_power_to_hartree_per_bohr_power(22.39, 4),
        "gamma": ev_per_angstrom_power_to_hartree_per_bohr_power(-28.88, 4),
    },
)

MATERIAL_SETS = {
    "zvr_bto": QuarticMaterial(
        elastic=ELASTIC,
        electrostrictive=ELECTROSTRICTIVE_VANDERBILT,
        onsite=ONSITE_QUARTIC_VANDERBILT,
    ),
    "nishimatsu_bto": NISHIMATSU_BTO,
    "nishimatsu_sto": NISHIMATSU_STO,
}


def elastic_matrix(elastic: Optional[dict[str, float]] = None) -> np.ndarray:
    """Return the 6x6 elastic matrix C in Vanderbilt's Voigt convention."""
    elastic = ELASTIC if elastic is None else elastic
    b11 = elastic["B11"]
    b12 = elastic["B12"]
    b44 = elastic["B44"]
    return np.array(
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


def electrostrictive_tensor_vanderbilt(electrostrictive: Optional[dict[str, float]] = None) -> np.ndarray:
    """Return B^V_{l alpha beta} from Eq. (14) and Table II.

    Voigt order is (xx, yy, zz, yz, xz, xy). Cartesian order is (x, y, z).
    Vanderbilt writes E_int = +1/2 sum B^V_{l alpha beta} eta_l u_alpha u_beta.
    """
    electrostrictive = ELECTROSTRICTIVE_VANDERBILT if electrostrictive is None else electrostrictive
    b1xx = electrostrictive["B1xx"]
    b1yy = electrostrictive["B1yy"]
    b4yz = electrostrictive["B4yz"]

    tensor = np.zeros((6, 3, 3), dtype=float)

    for voigt, axis in [(0, 0), (1, 1), (2, 2)]:
        tensor[voigt, axis, axis] = b1xx

    for voigt, axis in [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]:
        tensor[voigt, axis, axis] = b1yy

    for voigt, a, b in [(3, 1, 2), (4, 0, 2), (5, 0, 1)]:
        tensor[voigt, a, b] = b4yz
        tensor[voigt, b, a] = b4yz

    return tensor


def strain_quartic_tensor(b_tensor: np.ndarray, e_matrix: np.ndarray) -> np.ndarray:
    """Compute T_abcd = B_l_ab (E^-1)_lm B_m_cd."""
    e_inv = np.linalg.inv(e_matrix)
    return np.einsum("lab,lm,mcd->abcd", b_tensor, e_inv, b_tensor)


def quartic_polynomial_coefficients(tensor: np.ndarray) -> tuple[float, float]:
    """Return polynomial coefficients A, C for A sum x^4 + C sum x^2 y^2.

    The input only needs to be pair-symmetric. The coefficient is obtained by
    summing all index permutations contributing to each monomial.
    """
    a_x4 = float(tensor[0, 0, 0, 0])
    c_x2y2 = 0.0
    for indices in set(itertools.permutations((0, 0, 1, 1), 4)):
        c_x2y2 += float(tensor[indices])
    return a_x4, c_x2y2


def cubic_invariants_from_polynomial(a_x4: float, c_x2y2: float) -> tuple[float, float]:
    """Return b1, b2 for beta_abcd = b1 S_abcd + b2 delta_abcd."""
    b1 = c_x2y2 / 6.0
    b2 = a_x4 - 3.0 * b1
    return b1, b2


def cubic_invariants_from_tensor(tensor: np.ndarray) -> tuple[float, float]:
    """Return the fully symmetrized cubic invariants represented by tensor."""
    return cubic_invariants_from_polynomial(*quartic_polynomial_coefficients(tensor))


def bare_beta_invariants(onsite: Optional[dict[str, float]] = None) -> tuple[float, float]:
    """Return b1, b2 for the bare quartic beta tensor.

    Vanderbilt's on-site quartic energy is
        alpha (u^2)^2 + gamma (ux^2 uy^2 + uy^2 uz^2 + uz^2 ux^2).

    In untilted.tex the same term is written as
        1/4 beta_abcd u_a u_b u_c u_d,
    with beta_abcd = b1 S_abcd + b2 delta_abcd.
    """
    onsite = ONSITE_QUARTIC_VANDERBILT if onsite is None else onsite
    alpha = onsite["alpha"]
    gamma = onsite["gamma"]
    return (2.0 / 3.0) * (2.0 * alpha + gamma), -2.0 * gamma
