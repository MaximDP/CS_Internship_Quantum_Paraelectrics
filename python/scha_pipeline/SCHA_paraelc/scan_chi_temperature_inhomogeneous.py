#!/usr/bin/env python3
"""Temperature scan with separate local, homogeneous, and acoustic quartics.

This is deliberately kept separate from ``scan_chi_temperature.py``.  The
original script remains the projected-local reference calculation, whereas
this script evaluates the thermodynamic-limit centered SCHA equation (133)
with three distinct extensive terms,

    Sigma_loc = 3 kBT beta G,
    Sigma_hom = -1/2 kBT Lambda_hom G,
    Sigma_inh(K) = -kBT sum_Q H(K-Q)^* G_w(K-Q) H(K-Q) G_P(Q).

The finite-size homogeneous exchange term in Eq. (133) is proportional to
``1/N`` without an extensive internal momentum sum.  It therefore vanishes in
the thermodynamic limit represented by the Brillouin-zone quadrature and must
not be evaluated by identifying ``N`` with the quadrature order.

The Matsubara sum in ``Sigma_inh(0)`` is evaluated analytically, mode by mode.

The acoustic elastic kernel and the mode--strain vertex are Eqs. (17)--(23) of
Nishimatsu et al., Phys. Rev. B 78, 104104 (2008), as implemented in FERAM's
``elastic.F``.  The acoustic propagator retains the additional dynamical term
``m_w omega_n^2``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_chi_temperature as reference_scan
import scha_paraelc as scha


DEFAULT_LINEAR_PRESSURE_SLOPE_GPA_PER_K = -0.005
EXPERIMENTAL_LATTICE_DATA = (
    Path(__file__).resolve().parents[1]
    / "inputs" / "paraelectric"
    / "nakatani2016_batio3_cubic_lattice.csv"
)
EXPERIMENTAL_LATTICE_FIT_MIN_K = 413.0
EXPERIMENTAL_LATTICE_FIT_MAX_K = 598.0


@dataclass(frozen=True)
class AcousticQuarticParameters:
    """Elastic and mode--strain data entering H^dagger G_w H."""

    coefficients_name: str
    acoustic_mass_amu: float
    b11_ev: float
    b12_ev: float
    b44_ev: float
    b1xx_ev_per_angstrom2: float
    b1yy_ev_per_angstrom2: float
    b4yz_ev_per_angstrom2: float
    bare_b1: float
    bare_b2: float
    homogeneous_lambda_b1: float
    homogeneous_lambda_b2: float


# Nishimatsu/Wu--Cohen parameters used by FERAM.  B_ij are energies, while
# B_1ab are energies per squared local-mode amplitude.  The rank-four values
# are in Ha/bohr^4 and obey beta_proj=beta_bare-Lambda_hom/2.
BTO_ACOUSTIC_PARAMETERS = AcousticQuarticParameters(
    coefficients_name="nishimatsu_2010_bto",
    # Reference-curve value; Nishimatsu 2016 Table I reports 46.64 amu.
    acoustic_mass_amu=46.44,
    b11_ev=126.731671475652,
    b12_ev=41.7582963902598,
    b44_ev=49.2408864348646,
    b1xx_ev_per_angstrom2=-185.347187551195,
    b1yy_ev_per_angstrom2=-3.28092949275457,
    b4yz_ev_per_angstrom2=-14.5501738943852,
    bare_b1=0.0816281455,
    bare_b2=0.6655891570,
    homogeneous_lambda_b1=-0.0605593324,
    homogeneous_lambda_b2=1.0993850582,
)

STO_ACOUSTIC_PARAMETERS = AcousticQuarticParameters(
    coefficients_name="nishimatsu_2016_sto",
    acoustic_mass_amu=36.70,
    b11_ev=131.33,
    b12_ev=36.26,
    b44_ev=41.30,
    b1xx_ev_per_angstrom2=-102.09,
    b1yy_ev_per_angstrom2=+0.5299,
    b4yz_ev_per_angstrom2=-15.494,
    bare_b1=0.0305463786,
    bare_b2=0.1664489460,
    homogeneous_lambda_b1=-0.0082678896,
    homogeneous_lambda_b2=0.2857065946,
)

ACOUSTIC_PARAMETERS = {
    parameters.coefficients_name: parameters
    for parameters in (BTO_ACOUSTIC_PARAMETERS, STO_ACOUSTIC_PARAMETERS)
}
ACOUSTIC_MASS_AMU = BTO_ACOUSTIC_PARAMETERS.acoustic_mass_amu


@dataclass(frozen=True)
class InhomogeneousQuarticGrid:
    """Precomputed k-dependent data entering Delta beta."""

    ngrid: int
    weights: np.ndarray
    prefactor: float
    polar_offset_eigenvalues: np.ndarray
    polar_eigenvectors: np.ndarray
    acoustic_omega: np.ndarray
    contracted_vertex: np.ndarray
    acoustic_mass_au: float
    parameters: AcousticQuarticParameters


def _wavevectors_and_weights(ngrid: int, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(ngrid)
    points = cutoff * nodes
    scaled_weights = cutoff * weights
    kx, ky, kz = np.meshgrid(points, points, points, indexing="ij")
    wx, wy, wz = np.meshgrid(
        scaled_weights, scaled_weights, scaled_weights, indexing="ij"
    )
    wavevectors = np.stack((kx.ravel(), ky.ravel(), kz.ravel()), axis=1)
    integration_weights = (wx * wy * wz).ravel()
    return wavevectors, integration_weights


def _acoustic_kernel(
    wavevectors: np.ndarray,
    parameters: AcousticQuarticParameters = BTO_ACOUSTIC_PARAMETERS,
) -> np.ndarray:
    """Return Phi_ab(k) in Ha/bohr^2, as in FERAM ``elastic.F``."""

    b11 = parameters.b11_ev * reference_scan.EV_TO_HARTREE
    b12 = parameters.b12_ev * reference_scan.EV_TO_HARTREE
    b44 = parameters.b44_ev * reference_scan.EV_TO_HARTREE
    k2 = np.einsum("na,na->n", wavevectors, wavevectors)
    kernel = (b12 + b44) * np.einsum(
        "na,nb->nab", wavevectors, wavevectors, optimize=True
    )
    diagonal = b44 * k2[:, None] + (b11 - b44) * wavevectors**2
    axes = np.arange(3)
    kernel[:, axes, axes] = diagonal
    return kernel


def _mixed_tensor(
    wavevectors: np.ndarray,
    parameters: AcousticQuarticParameters = BTO_ACOUSTIC_PARAMETERS,
) -> np.ndarray:
    """Return H_abc(k) in Ha/bohr^3 in the convention of rapportCS.tex.

    The physical Fourier vertex is imaginary because strain is a derivative of
    the acoustic displacement.  Only H^* G_w H is needed here, so its real
    amplitude is stored.
    """

    b1xx = (
        parameters.b1xx_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    b1yy = (
        parameters.b1yy_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    b4yz = (
        parameters.b4yz_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )

    count = wavevectors.shape[0]
    tensor = np.zeros((count, 3, 3, 3), dtype=float)

    # Normal strains: H_{a ii}=k_a B1xx for a=i and k_a B1yy otherwise.
    for acoustic_axis in range(3):
        for polar_axis in range(3):
            coupling = b1xx if acoustic_axis == polar_axis else b1yy
            tensor[:, acoustic_axis, polar_axis, polar_axis] = (
                wavevectors[:, acoustic_axis] * coupling
            )

    # Engineering shear strains (yz, xz, xy).  Both polar-index orderings are
    # present because H is symmetric in its last two indices.
    for acoustic_axis, polar_a, polar_b, derivative_axis in (
        (1, 1, 2, 2),
        (2, 1, 2, 1),
        (0, 0, 2, 2),
        (2, 0, 2, 0),
        (0, 0, 1, 1),
        (1, 0, 1, 0),
    ):
        value = wavevectors[:, derivative_axis] * b4yz
        tensor[:, acoustic_axis, polar_a, polar_b] = value
        tensor[:, acoustic_axis, polar_b, polar_a] = value

    return tensor


def _validate_nishimatsu_mixed_tensor(
    wavevectors: np.ndarray,
    tensor: np.ndarray,
    parameters: AcousticQuarticParameters = BTO_ACOUSTIC_PARAMETERS,
) -> None:
    """Check the symmetric H tensor against Nishimatsu's Eq. (23)."""

    b1xx = (
        parameters.b1xx_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    b1yy = (
        parameters.b1yy_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    b4yz = (
        parameters.b4yz_ev_per_angstrom2
        * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    )
    expected = np.zeros((wavevectors.shape[0], 3, 6), dtype=float)
    for acoustic_axis in range(3):
        for polar_axis in range(3):
            expected[:, acoustic_axis, polar_axis] = wavevectors[
                :, acoustic_axis
            ] * (b1xx if acoustic_axis == polar_axis else b1yy)
    expected[:, 1, 3] = 2.0 * wavevectors[:, 2] * b4yz
    expected[:, 2, 3] = 2.0 * wavevectors[:, 1] * b4yz
    expected[:, 0, 4] = 2.0 * wavevectors[:, 2] * b4yz
    expected[:, 2, 4] = 2.0 * wavevectors[:, 0] * b4yz
    expected[:, 0, 5] = 2.0 * wavevectors[:, 1] * b4yz
    expected[:, 1, 5] = 2.0 * wavevectors[:, 0] * b4yz

    # y=(u_x^2,u_y^2,u_z^2,u_yu_z,u_zu_x,u_xu_y).  Off-diagonal
    # entries acquire a factor two when the symmetric H_ab tensor is mapped
    # back to Nishimatsu's 3x6 matrix.
    actual = np.stack(
        (
            tensor[:, :, 0, 0],
            tensor[:, :, 1, 1],
            tensor[:, :, 2, 2],
            2.0 * tensor[:, :, 1, 2],
            2.0 * tensor[:, :, 2, 0],
            2.0 * tensor[:, :, 0, 1],
        ),
        axis=2,
    )
    if not np.allclose(actual, expected, rtol=1.0e-13, atol=1.0e-14):
        raise RuntimeError("the mixed tensor does not reproduce Nishimatsu Eq. (23)")


def build_inhomogeneous_quartic_grid(
    ngrid: int,
    coefficients: scha.SchaCoefficients,
    acoustic_mass_amu: float | None = None,
    parameters: AcousticQuarticParameters | None = None,
) -> InhomogeneousQuarticGrid:
    """Precompute acoustic modes and tensor contractions on the SCHA grid."""

    if parameters is None:
        try:
            parameters = ACOUSTIC_PARAMETERS[coefficients.name]
        except KeyError as error:
            raise ValueError(
                f"no inhomogeneous acoustic parameters for {coefficients.name}"
            ) from error
    if coefficients.name != parameters.coefficients_name:
        raise ValueError("the acoustic and polar parameter sets do not match")
    if acoustic_mass_amu is None:
        acoustic_mass_amu = parameters.acoustic_mass_amu
    if ngrid % 2 != 0:
        raise ValueError("ngrid must be even so the acoustic zero mode is not sampled")
    projected = np.array(
        (
            parameters.bare_b1 - 0.5 * parameters.homogeneous_lambda_b1,
            parameters.bare_b2 - 0.5 * parameters.homogeneous_lambda_b2,
        )
    )
    if not np.allclose(
        projected,
        (coefficients.b1_eff, coefficients.b2_eff),
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise RuntimeError("bare beta and Lambda_hom do not reproduce beta_proj")

    harmonic_grid, polar_offset_eigenvalues, polar_eigenvectors = (
        scha.cached_harmonic_eigensystem(
            ngrid, coefficients.kmax, coefficients
        )
    )
    wavevectors, weights = _wavevectors_and_weights(ngrid, coefficients.kmax)
    if not np.allclose(weights, harmonic_grid.weights, rtol=1.0e-14, atol=0.0):
        raise RuntimeError("the reconstructed k grid does not match the harmonic grid")

    acoustic_eigenvalues, acoustic_eigenvectors = np.linalg.eigh(
        _acoustic_kernel(wavevectors, parameters)
    )
    if np.any(acoustic_eigenvalues <= 0.0):
        raise RuntimeError("the sampled acoustic kernel is not positive definite")

    acoustic_mass_au = acoustic_mass_amu * scha.AMU_TO_ELECTRON_MASS
    acoustic_omega = np.sqrt(acoustic_eigenvalues / acoustic_mass_au)
    mixed_tensor = _mixed_tensor(wavevectors, parameters)
    _validate_nishimatsu_mixed_tensor(wavevectors, mixed_tensor, parameters)

    # For the dc mass, one Wick channel transfers q=0 through the acoustic
    # propagator and vanishes because H(0)=0.  The two crossed channels transfer
    # the loop (k, omega_n).  For one acoustic eigenvector e_l, define
    # M_l,bc=e_l,a H_abc.  Combining the two crossed channels and projecting the
    # external mass onto delta_ab/3 leaves
    #
    #   -(1/3) v_j^T M_l^2 v_j.
    #
    # This channel-resolved contraction is essential: replacing all three Wick
    # channels by one symmetrized beta(k, omega_n) would incorrectly assign the
    # loop momentum to the q=0 channel.  The remaining acoustic and polar
    # denominators are supplied by the Matsubara product below.
    contracted_vertex = np.empty(
        (wavevectors.shape[0], 3, 3), dtype=float
    )
    for acoustic_mode in range(3):
        matrix = np.einsum(
            "na,nabc->nbc",
            acoustic_eigenvectors[:, :, acoustic_mode],
            mixed_tensor,
            optimize=True,
        )
        matrix_times_polar_modes = np.einsum(
            "nab,nbj->naj", matrix, polar_eigenvectors, optimize=True
        )
        v_m2_v = np.einsum(
            "naj,naj->nj",
            matrix_times_polar_modes,
            matrix_times_polar_modes,
            optimize=True,
        )
        contracted_vertex[:, acoustic_mode, :] = -v_m2_v / 3.0

    return InhomogeneousQuarticGrid(
        ngrid=ngrid,
        weights=harmonic_grid.weights,
        prefactor=harmonic_grid.prefactor,
        polar_offset_eigenvalues=polar_offset_eigenvalues,
        polar_eigenvectors=polar_eigenvectors,
        acoustic_omega=acoustic_omega,
        contracted_vertex=contracted_vertex,
        acoustic_mass_au=acoustic_mass_au,
        parameters=parameters,
    )


def _bosonic_sum(omega: np.ndarray, kbt: float) -> np.ndarray:
    """Return sum_n 1/(omega_n^2+omega^2)."""

    return 1.0 / np.tanh(omega / (2.0 * kbt)) / (2.0 * kbt * omega)


def _bosonic_product_sum(
    polar_omega: np.ndarray,
    acoustic_omega: np.ndarray,
    kbt: float,
) -> np.ndarray:
    """Return sum_n of the product of polar and acoustic denominators.

    The output order is ``(k, acoustic mode, polar mode)`` and includes both
    effective masses.
    """

    polar = polar_omega[:, None, :]
    acoustic = acoustic_omega[:, :, None]
    polar_sum = _bosonic_sum(polar, kbt)
    acoustic_sum = _bosonic_sum(acoustic, kbt)
    denominator = acoustic**2 - polar**2
    result = np.empty_like(denominator)

    near_degenerate = np.abs(denominator) <= 1.0e-9 * (
        acoustic**2 + polar**2
    )
    np.divide(
        polar_sum - acoustic_sum,
        denominator,
        out=result,
        where=~near_degenerate,
    )
    if np.any(near_degenerate):
        omega = 0.5 * (polar + acoustic)
        argument = omega / (2.0 * kbt)
        coth = 1.0 / np.tanh(argument)
        csch2 = np.zeros_like(argument)
        moderate = argument < 40.0
        csch2[moderate] = 1.0 / np.sinh(argument[moderate]) ** 2
        limit = coth / (4.0 * kbt * omega**3) + csch2 / (
            8.0 * kbt**2 * omega**2
        )
        result[near_degenerate] = np.broadcast_to(
            limit, result.shape
        )[near_degenerate]
    return result


def _cubic_rank4_component(
    a: int,
    b: int,
    g: int,
    d: int,
    b1: float,
    b2: float,
) -> float:
    delta_ab = 1.0 if a == b else 0.0
    delta_gd = 1.0 if g == d else 0.0
    delta_ag = 1.0 if a == g else 0.0
    delta_bd = 1.0 if b == d else 0.0
    delta_ad = 1.0 if a == d else 0.0
    delta_bg = 1.0 if b == g else 0.0
    delta_abgd = 1.0 if a == b == g == d else 0.0
    return b1 * (
        delta_ab * delta_gd
        + delta_ag * delta_bd
        + delta_ad * delta_bg
    ) + b2 * delta_abgd


def _rank4_scalar_contraction(
    gmat: np.ndarray, b1: float, b2: float
) -> float:
    contraction_trace = 0.0
    for a in range(3):
        for g in range(3):
            for d in range(3):
                contraction_trace += (
                    _cubic_rank4_component(a, a, g, d, b1, b2)
                    * gmat[g, d]
                )
    return contraction_trace / 3.0


def local_quartic_self_energy_mass(
    gmat: np.ndarray,
    temperature: float,
    parameters: AcousticQuarticParameters = BTO_ACOUSTIC_PARAMETERS,
) -> float:
    """Return 3 kBT beta_abgd G_gd, projected onto the cubic mass."""

    kbt = scha.KB_HARTREE_PER_K * temperature
    return float(
        3.0
        * kbt
        * _rank4_scalar_contraction(
            gmat, parameters.bare_b1, parameters.bare_b2
        )
    )


def homogeneous_quartic_self_energy_mass(
    gmat: np.ndarray,
    temperature: float,
    parameters: AcousticQuarticParameters = BTO_ACOUSTIC_PARAMETERS,
) -> float:
    """Return -kBT Lambda_hom_abgd G_gd/2 for the all-to-all term."""

    kbt = scha.KB_HARTREE_PER_K * temperature
    return float(
        -0.5
        * kbt
        * _rank4_scalar_contraction(
            gmat,
            parameters.homogeneous_lambda_b1,
            parameters.homogeneous_lambda_b2,
        )
    )


def inhomogeneous_quartic_self_energy_mass(
    a1: float,
    temperature: float,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
) -> float:
    """Return Sigma_inh(0,0) with the dynamic acoustic propagator."""

    polar_mu = grid.polar_offset_eigenvalues + a1
    if np.any(polar_mu <= 1.0e-14):
        raise ValueError("non-positive static polar eigenvalue is not allowed")
    kbt = scha.KB_HARTREE_PER_K * temperature
    polar_omega = np.sqrt(polar_mu / coefficients.mass_au)
    product_sum = _bosonic_product_sum(
        polar_omega, grid.acoustic_omega, kbt
    ) / (coefficients.mass_au * grid.acoustic_mass_au)
    contracted_loop = grid.prefactor * np.einsum(
        "n,nlj,nlj->",
        grid.weights,
        grid.contracted_vertex,
        product_sum,
        optimize=True,
    )
    return float(kbt * contracted_loop)


def residual(
    a1: float,
    temperature: float,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
) -> tuple[float, np.ndarray, float, float, float, float, float]:
    """SCHA residual with beta, Lambda_hom, and M_inh kept separate."""

    gmat = scha.integrate_g(
        a1,
        temperature=temperature,
        ngrid=grid.ngrid,
        cutoff=coefficients.kmax,
        c=coefficients,
    )
    delta_local = local_quartic_self_energy_mass(
        gmat, temperature, grid.parameters
    )
    delta_homogeneous = homogeneous_quartic_self_energy_mass(
        gmat, temperature, grid.parameters
    )
    delta_inhomogeneous = inhomogeneous_quartic_self_energy_mass(
        a1, temperature, coefficients, grid
    )
    delta_sextic = scha.sextic_self_energy_mass(
        gmat, temperature=temperature, c=coefficients
    )
    delta_octic = scha.octic_self_energy_mass(
        gmat, temperature=temperature, c=coefficients
    )
    return (
        a1
        - (
            coefficients.A01
            + delta_local
            + delta_homogeneous
            + delta_inhomogeneous
            + delta_sextic
            + delta_octic
        ),
        gmat,
        delta_local,
        delta_homogeneous,
        delta_inhomogeneous,
        delta_sextic,
        delta_octic,
    )


def solve_root(
    temperature: float,
    pressure_shift_u: float,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
) -> float:
    def scalar_residual(a1: float) -> float:
        return float(residual(a1, temperature, coefficients, grid)[0]) - pressure_shift_u

    lower = max(
        1.0e-12,
        scha.stable_a1_threshold(grid.ngrid, coefficients.kmax, coefficients)
        + 1.0e-12,
    )
    upper = 1.0
    lower_value = scalar_residual(lower)
    upper_value = scalar_residual(upper)
    if lower_value * upper_value <= 0.0:
        return float(
            brentq(
                scalar_residual,
                lower,
                upper,
                xtol=1.0e-13,
                rtol=1.0e-13,
            )
        )

    # Fallback for a possible non-monotonic centered branch.
    samples = np.concatenate(
        ([lower], lower + np.logspace(-10.0, -1.0, 40), np.logspace(-1.0, 0.0, 8))
    )
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


def cubic_lattice_parameter(
    temperature: float,
    pressure_gpa: float,
    gmat: np.ndarray,
    coefficients: scha.SchaCoefficients,
    parameters: AcousticQuarticParameters = BTO_ACOUSTIC_PARAMETERS,
) -> tuple[float, float, float]:
    """Return ``(a, eta, <u_x^2>)`` for the centered cubic SCHA solution.

    The propagator convention used in this module is

        <u_a u_b> = k_B T G_ab.

    Minimizing the homogeneous-strain enthalpy then gives

        eta = -[ (B1xx + 2 B1yy)<u_x^2>/2 + p a0^3 ]
                /(B11 + 2 B12),

    with ``a = a0 (1 + eta)`` to first order in the isotropic strain.
    """

    if coefficients.name != parameters.coefficients_name:
        raise ValueError("the elastic and polar parameter sets do not match")

    component_variance_bohr2 = float(
        scha.KB_HARTREE_PER_K * temperature * np.trace(gmat) / 3.0
    )
    elastic_bulk_au = (
        parameters.b11_ev + 2.0 * parameters.b12_ev
    ) * reference_scan.EV_TO_HARTREE
    mode_strain_bulk_au = (
        parameters.b1xx_ev_per_angstrom2
        + 2.0 * parameters.b1yy_ev_per_angstrom2
    ) * reference_scan.EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    pressure_energy_au = (
        pressure_gpa
        * reference_scan.GPA_TO_HARTREE_PER_BOHR3
        * coefficients.a0**3
    )
    strain = -(
        0.5 * mode_strain_bulk_au * component_variance_bohr2
        + pressure_energy_au
    ) / elastic_bulk_au
    lattice_parameter_angstrom = (
        coefficients.a0
        * (1.0 + strain)
        * reference_scan.ANGSTROM_PER_BOHR
    )
    return (
        float(lattice_parameter_angstrom),
        float(strain),
        component_variance_bohr2,
    )


def load_experimental_cubic_lattice_parameter(
    path: Path = EXPERIMENTAL_LATTICE_DATA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Nakatani et al.'s cubic BaTiO3 lattice parameters.

    The tabulated standard uncertainties are used as regression weights.  The
    source spans 413--778 K; the default fit below is restricted to 413--598 K
    because those points bracket the temperature range of the susceptibility
    figure without allowing the high-temperature curvature to bias it.
    """

    temperatures = []
    lattice_parameters = []
    uncertainties = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            temperatures.append(float(row["temperature_K"]))
            lattice_parameters.append(float(row["lattice_parameter_angstrom"]))
            uncertainties.append(float(row["standard_uncertainty_angstrom"]))
    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (temperatures, lattice_parameters, uncertainties)
    )
    if arrays[0].size < 2 or np.any(arrays[2] <= 0.0):
        raise ValueError("invalid experimental cubic-lattice dataset")
    return arrays


def fit_experimental_cubic_lattice_parameter(
    fit_min_k: float = EXPERIMENTAL_LATTICE_FIT_MIN_K,
    fit_max_k: float = EXPERIMENTAL_LATTICE_FIT_MAX_K,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the weighted line ``a_exp(T)=slope*T+intercept`` in angstrom."""

    temperatures, lattice_parameters, uncertainties = (
        load_experimental_cubic_lattice_parameter()
    )
    selected = (temperatures >= fit_min_k) & (temperatures <= fit_max_k)
    if np.count_nonzero(selected) < 2:
        raise ValueError("the experimental lattice fit needs at least two points")
    slope, intercept = np.polyfit(
        temperatures[selected],
        lattice_parameters[selected],
        1,
        w=1.0 / uncertainties[selected],
    )
    return (
        float(slope),
        float(intercept),
        temperatures[selected],
        lattice_parameters[selected],
        uncertainties[selected],
    )


def calibrate_pressure_to_lattice_parameter(
    temperature: float,
    target_lattice_parameter_angstrom: float,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
    pressure_bounds_gpa: tuple[float, float] = (-3.0, 0.0),
    initial_pressure_gpa: float = 0.0,
) -> float:
    """Fit a constant pressure to ``a(T)=target`` within the cubic SCHA model."""

    def lattice_residual(pressure_gpa: float) -> float:
        pressure_shift = reference_scan.pressure_mass_shift_u(
            pressure_gpa, coefficients
        )
        a1_u = solve_root(temperature, pressure_shift, coefficients, grid)
        evaluated = residual(a1_u, temperature, coefficients, grid)
        lattice_parameter, _, _ = cubic_lattice_parameter(
            temperature,
            pressure_gpa,
            evaluated[1],
            coefficients,
            grid.parameters,
        )
        return lattice_parameter - target_lattice_parameter_angstrom

    lower, upper = pressure_bounds_gpa
    pressure = float(np.clip(initial_pressure_gpa, lower, upper))
    try:
        value = lattice_residual(pressure)
    except RuntimeError:
        pressure = float(np.clip(0.0, lower, upper))
        value = lattice_residual(pressure)
    if value == 0.0:
        return pressure

    # The cubic lattice parameter decreases monotonically with increasing
    # hydrostatic pressure.  Walk from a stable initial point in the required
    # direction, then use Brent only once a stable sign-changing bracket has
    # been found.  Continuation in T makes this substantially cheaper than a
    # fresh pressure scan at every temperature.
    direction = -1.0 if value < 0.0 else 1.0
    step = 0.25
    previous_pressure = pressure
    previous_value = value
    for _ in range(80):
        candidate = float(
            np.clip(previous_pressure + direction * step, lower, upper)
        )
        if candidate == previous_pressure:
            break
        try:
            candidate_value = lattice_residual(candidate)
        except RuntimeError:
            step *= 0.5
            if step < 1.0e-5:
                break
            continue
        if previous_value * candidate_value <= 0.0:
            left_pressure, right_pressure = sorted(
                (previous_pressure, candidate)
            )
            return float(
                brentq(
                    lattice_residual,
                    left_pressure,
                    right_pressure,
                    xtol=1.0e-12,
                    rtol=1.0e-12,
                )
            )
        previous_pressure = candidate
        previous_value = candidate_value
        step = min(1.25 * step, 0.75)
    raise RuntimeError(
        f"no stable pressure reproduces a={target_lattice_parameter_angstrom:g} "
        f"angstrom at T={temperature:g} K in [{lower:g}, {upper:g}] GPa"
    )


def fit_affine_lattice_matching_pressure(
    temperatures: np.ndarray,
    lattice_fit_slope_angstrom_per_k: float,
    lattice_fit_intercept_angstrom: float,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit an affine pressure to the pointwise inverse cubic equation of state."""

    targets = (
        lattice_fit_slope_angstrom_per_k * temperatures
        + lattice_fit_intercept_angstrom
    )
    pressures_list = []
    initial_pressure = 0.0
    for temperature, target in zip(temperatures, targets):
        pressure = calibrate_pressure_to_lattice_parameter(
            float(temperature),
            float(target),
            coefficients,
            grid,
            pressure_bounds_gpa=(-5.0, 1.0),
            initial_pressure_gpa=initial_pressure,
        )
        pressures_list.append(pressure)
        initial_pressure = pressure
    required_pressures = np.asarray(pressures_list, dtype=float)
    slope, intercept = np.polyfit(temperatures, required_pressures, 1)
    fitted_pressures = slope * temperatures + intercept
    return (
        float(slope),
        float(intercept),
        np.asarray(fitted_pressures, dtype=float),
        targets,
        required_pressures,
    )


def fit_constant_pressure(
    temperature: float,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
) -> float:
    """Fit a constant pressure to Barrett at one temperature."""

    _, target_chi = reference_scan.experimental_curie_weiss(temperature)
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    target_a1_u = coefficients.omega0 / target_chi / scale2
    threshold = scha.stable_a1_threshold(
        grid.ngrid, coefficients.kmax, coefficients
    )
    if target_a1_u <= threshold:
        raise RuntimeError("the Barrett target is outside the stable centered branch")
    required_shift = residual(
        target_a1_u, temperature, coefficients, grid
    )[0]
    return float(
        required_shift
        / reference_scan.pressure_mass_shift_u(1.0, coefficients)
    )


def fit_linear_pressure_to_barrett(
    temperatures: np.ndarray,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
) -> tuple[float, float, np.ndarray]:
    """Fit p(T)=a*T+b to the pressure required by the Barrett target.

    At fixed temperature the target susceptibility fixes ``A1``.  Since the
    pressure contribution to Eq. (133) is linear, its required value follows
    directly from the zero-pressure residual.  A sensitivity-weighted linear
    regression supplies the initial line, which is refined by minimizing the
    squared logarithmic susceptibility error.
    """

    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    shift_per_gpa = reference_scan.pressure_mass_shift_u(1.0, coefficients)
    required_pressures = []
    relative_chi_sensitivities = []
    target_chis = []
    for temperature_value in temperatures:
        temperature = float(temperature_value)
        _, target_chi = reference_scan.experimental_curie_weiss(temperature)
        target_chis.append(target_chi)
        target_a1_u = coefficients.omega0 / target_chi / scale2
        threshold = scha.stable_a1_threshold(
            grid.ngrid, coefficients.kmax, coefficients
        )
        if target_a1_u <= threshold:
            raise RuntimeError(
                f"the Barrett target at T={temperature:g} K is outside the "
                "stable centered branch"
            )
        target_residual = residual(
            target_a1_u, temperature, coefficients, grid
        )[0]
        required_pressures.append(target_residual / shift_per_gpa)

        # Linearize ln(chi)=const-ln(A1) around the exact Barrett solution.
        # This makes the straight-line regression minimize the relative error
        # in susceptibility to first order, rather than an unweighted error in
        # pressure that underemphasizes the very pressure-sensitive low-T end.
        delta_a1_u = max(1.0e-9, 1.0e-4 * target_a1_u)
        derivative = (
            residual(
                target_a1_u + delta_a1_u,
                temperature,
                coefficients,
                grid,
            )[0]
            - residual(
                target_a1_u - delta_a1_u,
                temperature,
                coefficients,
                grid,
            )[0]
        ) / (2.0 * delta_a1_u)
        relative_chi_sensitivities.append(
            abs(shift_per_gpa / (target_a1_u * derivative))
        )

    required = np.asarray(required_pressures, dtype=float)
    weights = np.asarray(relative_chi_sensitivities, dtype=float)
    weights /= np.max(weights)
    reference_temperature = float(np.mean(temperatures))
    centered_design = np.column_stack(
        (temperatures - reference_temperature, np.ones_like(temperatures))
    )
    slope, pressure_at_reference = np.linalg.lstsq(
        centered_design * weights[:, None], required * weights, rcond=None
    )[0]

    # Refine the first-order result with Gauss--Newton steps on ln(chi).  The
    # logarithm gives a dimensionless relative-error objective and avoids the
    # large low-temperature susceptibility dominating merely through its scale.
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    target_chis_array = np.asarray(target_chis, dtype=float)
    for _ in range(3):
        log_errors = []
        jacobian = []
        for temperature_value, target_chi in zip(
            temperatures, target_chis_array
        ):
            temperature = float(temperature_value)
            pressure = (
                slope * (temperature - reference_temperature)
                + pressure_at_reference
            )
            a1_u = solve_root(
                temperature,
                shift_per_gpa * pressure,
                coefficients,
                grid,
            )
            chi = coefficients.omega0 / (scale2 * a1_u)
            log_errors.append(np.log(chi / target_chi))

            delta_a1_u = max(1.0e-9, 1.0e-4 * a1_u)
            derivative = (
                residual(
                    a1_u + delta_a1_u,
                    temperature,
                    coefficients,
                    grid,
                )[0]
                - residual(
                    a1_u - delta_a1_u,
                    temperature,
                    coefficients,
                    grid,
                )[0]
            ) / (2.0 * delta_a1_u)
            dlogchi_dp = -shift_per_gpa / (a1_u * derivative)
            jacobian.append(
                dlogchi_dp
                * np.array(
                    (temperature - reference_temperature, 1.0), dtype=float
                )
            )

        correction = np.linalg.lstsq(
            np.asarray(jacobian), -np.asarray(log_errors), rcond=None
        )[0]
        slope += correction[0]
        pressure_at_reference += correction[1]
        pressure_correction = (
            correction[0] * (temperatures - reference_temperature)
            + correction[1]
        )
        if np.max(np.abs(pressure_correction)) < 1.0e-9:
            break

    intercept = pressure_at_reference - slope * reference_temperature
    fitted = slope * temperatures + intercept
    if np.any(fitted >= 0.0):
        raise RuntimeError(
            "the unconstrained Barrett pressure fit is not negative over the "
            "requested temperature interval"
        )
    return float(slope), float(intercept), required


def fit_affine_pressure_to_wieczorek(
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``p(T)=a*T+b`` by vertical least squares to Wieczorek's data.

    The digitized temperatures are used directly.  At each point the target
    susceptibility first fixes the exact pressure required by the centered
    SCHA equation.  A sensitivity-weighted regression provides the initial
    affine pressure, which is then refined by Gauss--Newton iterations on

        sum_i [chi_SCHA(T_i, p(T_i)) - chi_Wieczorek(T_i)]**2.

    Thus the fitted quantity is the vertical distance in the susceptibility
    plot, not a pressure error or a logarithmic/relative error.
    """

    temperatures, target_chis = reference_scan.load_wieczorek_response()
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    shift_per_gpa = reference_scan.pressure_mass_shift_u(1.0, coefficients)
    threshold = scha.stable_a1_threshold(
        grid.ngrid, coefficients.kmax, coefficients
    )
    required_pressures = []
    chi_pressure_sensitivities = []

    for temperature_value, target_chi_value in zip(temperatures, target_chis):
        temperature = float(temperature_value)
        target_chi = float(target_chi_value)
        target_a1_u = coefficients.omega0 / (scale2 * target_chi)
        if target_a1_u <= threshold:
            raise RuntimeError(
                f"the Wieczorek target at T={temperature:g} K is outside the "
                "stable centered branch"
            )
        target_residual = residual(
            target_a1_u, temperature, coefficients, grid
        )[0]
        required_pressures.append(target_residual / shift_per_gpa)

        delta_a1_u = max(1.0e-9, 1.0e-4 * target_a1_u)
        residual_derivative = (
            residual(
                target_a1_u + delta_a1_u,
                temperature,
                coefficients,
                grid,
            )[0]
            - residual(
                target_a1_u - delta_a1_u,
                temperature,
                coefficients,
                grid,
            )[0]
        ) / (2.0 * delta_a1_u)
        chi_pressure_sensitivities.append(
            abs(
                -target_chi
                * shift_per_gpa
                / (target_a1_u * residual_derivative)
            )
        )

    required = np.asarray(required_pressures, dtype=float)
    weights = np.asarray(chi_pressure_sensitivities, dtype=float)
    weights /= np.max(weights)
    reference_temperature = float(np.mean(temperatures))
    centered_temperatures = temperatures - reference_temperature
    centered_design = np.column_stack(
        (centered_temperatures, np.ones_like(temperatures))
    )
    slope, pressure_at_reference = np.linalg.lstsq(
        centered_design * weights[:, None], required * weights, rcond=None
    )[0]

    predicted_chis = np.empty_like(target_chis)
    for _ in range(8):
        chi_errors = []
        jacobian = []
        for index, (temperature_value, target_chi_value) in enumerate(
            zip(temperatures, target_chis)
        ):
            temperature = float(temperature_value)
            pressure = (
                slope * (temperature - reference_temperature)
                + pressure_at_reference
            )
            a1_u = solve_root(
                temperature,
                shift_per_gpa * pressure,
                coefficients,
                grid,
            )
            chi = coefficients.omega0 / (scale2 * a1_u)
            predicted_chis[index] = chi
            chi_errors.append(chi - float(target_chi_value))

            delta_a1_u = max(1.0e-9, 1.0e-4 * a1_u)
            residual_derivative = (
                residual(
                    a1_u + delta_a1_u,
                    temperature,
                    coefficients,
                    grid,
                )[0]
                - residual(
                    a1_u - delta_a1_u,
                    temperature,
                    coefficients,
                    grid,
                )[0]
            ) / (2.0 * delta_a1_u)
            dchi_dp = (
                -chi * shift_per_gpa / (a1_u * residual_derivative)
            )
            jacobian.append(
                dchi_dp
                * np.array(
                    (temperature - reference_temperature, 1.0), dtype=float
                )
            )

        correction = np.linalg.lstsq(
            np.asarray(jacobian), -np.asarray(chi_errors), rcond=None
        )[0]
        slope += correction[0]
        pressure_at_reference += correction[1]
        pressure_correction = (
            correction[0] * centered_temperatures + correction[1]
        )
        if np.max(np.abs(pressure_correction)) < 1.0e-10:
            break

    intercept = pressure_at_reference - slope * reference_temperature

    # Re-evaluate once at the final line so the reported residuals correspond
    # exactly to the returned coefficients.
    for index, temperature_value in enumerate(temperatures):
        temperature = float(temperature_value)
        pressure = slope * temperature + intercept
        a1_u = solve_root(
            temperature,
            shift_per_gpa * pressure,
            coefficients,
            grid,
        )
        predicted_chis[index] = coefficients.omega0 / (scale2 * a1_u)

    return (
        float(slope),
        float(intercept),
        required,
        temperatures,
        target_chis,
        predicted_chis,
    )


def scan_protocol(
    temperatures: np.ndarray,
    pressure_model: str,
    constant_pressure_gpa: float,
    coefficients: scha.SchaCoefficients,
    grid: InhomogeneousQuarticGrid,
    linear_pressure_slope_gpa_per_k: float = DEFAULT_LINEAR_PRESSURE_SLOPE_GPA_PER_K,
    linear_pressure_intercept_gpa: float = 0.0,
    prescribed_pressures_gpa: np.ndarray | None = None,
) -> list[dict[str, float]]:
    if pressure_model == "prescribed":
        if prescribed_pressures_gpa is None:
            raise ValueError("prescribed pressure model needs one pressure per T")
        if len(prescribed_pressures_gpa) != len(temperatures):
            raise ValueError("prescribed pressure and temperature grids differ")
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    rows: list[dict[str, float]] = []
    for index, temperature_value in enumerate(temperatures):
        temperature = float(temperature_value)
        if pressure_model == "prescribed":
            pressure = float(prescribed_pressures_gpa[index])
        elif pressure_model == "linear":
            pressure = (
                linear_pressure_slope_gpa_per_k * temperature
                + linear_pressure_intercept_gpa
            )
        else:
            pressure = reference_scan.pressure_gpa(
                temperature, pressure_model, constant_pressure_gpa
            )
        pressure_shift = reference_scan.pressure_mass_shift_u(
            pressure, coefficients
        )
        try:
            a1_u = solve_root(
                temperature, pressure_shift, coefficients, grid
            )
        except RuntimeError:
            rows.append(
                {
                    "temperature_K": temperature,
                    "pressure_GPa": pressure,
                    "pressure_mass_shift_u": pressure_shift,
                    "A1_u": np.nan,
                    "chi_T_SCHA": np.nan,
                    "lattice_parameter_angstrom": np.nan,
                    "isotropic_strain": np.nan,
                    "mean_u_component_variance_bohr2": np.nan,
                    "delta_A1_quartic_local": np.nan,
                    "delta_A1_quartic_homogeneous": np.nan,
                    "delta_A1_quartic_inhomogeneous": np.nan,
                    "delta_A1_sextic": np.nan,
                    "delta_A1_octic": np.nan,
                    "residual": np.nan,
                }
            )
            continue

        evaluated = residual(a1_u, temperature, coefficients, grid)
        a1_p = scale2 * a1_u
        lattice_parameter, isotropic_strain, component_variance = (
            cubic_lattice_parameter(
                temperature,
                pressure,
                evaluated[1],
                coefficients,
                grid.parameters,
            )
        )
        rows.append(
            {
                "temperature_K": temperature,
                "pressure_GPa": pressure,
                "pressure_mass_shift_u": pressure_shift,
                "A1_u": a1_u,
                "chi_T_SCHA": coefficients.omega0 / a1_p,
                "lattice_parameter_angstrom": lattice_parameter,
                "isotropic_strain": isotropic_strain,
                "mean_u_component_variance_bohr2": component_variance,
                "delta_A1_quartic_local": evaluated[2],
                "delta_A1_quartic_homogeneous": evaluated[3],
                "delta_A1_quartic_inhomogeneous": evaluated[4],
                "delta_A1_sextic": evaluated[5],
                "delta_A1_octic": evaluated[6],
                "residual": evaluated[0] - pressure_shift,
            }
        )
    return rows


def write_comparison_csv(
    lattice_rows: list[dict[str, float]],
    linear_rows: list[dict[str, float]],
    zero_pressure_rows: list[dict[str, float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["temperature_K"]
    protocols = (
        ("lattice_matched", lattice_rows),
        ("linear", linear_rows),
        ("zero_pressure", zero_pressure_rows),
    )
    for prefix, rows in protocols:
        fields.extend(
            f"{prefix}_{name}" for name in rows[0] if name != "temperature_K"
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        if not (
            len(lattice_rows) == len(linear_rows) == len(zero_pressure_rows)
        ):
            raise ValueError("the pressure scans must use the same temperature grid")
        for lattice, linear, zero_pressure in zip(
            lattice_rows, linear_rows, zero_pressure_rows
        ):
            row: dict[str, float] = {"temperature_K": lattice["temperature_K"]}
            row.update(
                {
                    f"lattice_matched_{name}": value
                    for name, value in lattice.items()
                    if name != "temperature_K"
                }
            )
            row.update(
                {
                    f"linear_{name}": value
                    for name, value in linear.items()
                    if name != "temperature_K"
                }
            )
            row.update(
                {
                    f"zero_pressure_{name}": value
                    for name, value in zero_pressure.items()
                    if name != "temperature_K"
                }
            )
            writer.writerow(row)


def write_plot(
    lattice_rows: list[dict[str, float]],
    linear_rows: list[dict[str, float]],
    zero_pressure_rows: list[dict[str, float]],
    lattice_pressure_description: str,
    linear_pressure_slope_gpa_per_k: float,
    linear_pressure_intercept_gpa: float,
    linear_pressure_description: str,
    path: Path,
    fitted_response_temperatures: np.ndarray | None = None,
    fitted_response_chis: np.ndarray | None = None,
    fitted_response_extrapolation_max_temperature: float = 560.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperatures = np.array([row["temperature_K"] for row in lattice_rows])
    lattice_chi = np.array([row["chi_T_SCHA"] for row in lattice_rows])
    linear_chi = np.array([row["chi_T_SCHA"] for row in linear_rows])
    zero_pressure_chi = np.array(
        [row["chi_T_SCHA"] for row in zero_pressure_rows]
    )
    barrett = np.array(
        [reference_scan.experimental_curie_weiss(t)[1] for t in temperatures]
    )
    wieczorek_temperature, wieczorek_chi = reference_scan.load_wieczorek_response()
    reference_domain = (
        (temperatures >= reference_scan.BARRETT_REFERENCE_MIN_K)
        & (temperatures <= reference_scan.BARRETT_REFERENCE_MAX_K)
    )
    lattice_stable = np.isfinite(lattice_chi)
    zero_pressure_stable = np.isfinite(zero_pressure_chi)
    if fitted_response_temperatures is not None:
        if fitted_response_chis is None:
            raise ValueError("fitted response temperatures need susceptibilities")
        displayed_fit_temperatures = np.asarray(
            fitted_response_temperatures, dtype=float
        )
        displayed_fit_chis = np.asarray(fitted_response_chis, dtype=float)
        if displayed_fit_temperatures.shape != displayed_fit_chis.shape:
            raise ValueError("fitted response temperature and chi grids differ")
        # Continue the same affine pressure law beyond the measured points.
        # Keep that continuation separate so the plot can distinguish the
        # inverse-calibration interval from its unconstrained extrapolation.
        extrapolated_fit = np.isfinite(linear_chi) & (
            temperatures > float(np.max(displayed_fit_temperatures))
        ) & (
            temperatures <= fitted_response_extrapolation_max_temperature
        )
        extrapolated_fit_temperatures = np.concatenate(
            (
                displayed_fit_temperatures[-1:],
                temperatures[extrapolated_fit],
            )
        )
        extrapolated_fit_chis = np.concatenate(
            (
                displayed_fit_chis[-1:],
                linear_chi[extrapolated_fit],
            )
        )
    else:
        linear_displayed = np.isfinite(linear_chi) & (
            temperatures >= float(np.min(wieczorek_temperature))
        ) & (
            temperatures <= fitted_response_extrapolation_max_temperature
        )
        displayed_fit_temperatures = temperatures[linear_displayed]
        displayed_fit_chis = linear_chi[linear_displayed]
        extrapolated_fit_temperatures = np.empty(0, dtype=float)
        extrapolated_fit_chis = np.empty(0, dtype=float)

    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(
        temperatures[lattice_stable],
        lattice_chi[lattice_stable],
        "o-",
        color="tab:blue",
        linewidth=1.7,
        markersize=4.5,
        label=lattice_pressure_description,
    )
    displayed_fit_stable = np.isfinite(displayed_fit_chis)
    if np.any(displayed_fit_stable):
        axis.plot(
            displayed_fit_temperatures[displayed_fit_stable],
            displayed_fit_chis[displayed_fit_stable],
            "^-",
            color="tab:orange",
            linewidth=1.7,
            markersize=5.0,
            label=r"SCHA, fitted $p_\chi(T)$ (Wieczorek interval)",
        )
    extrapolated_fit_stable = np.isfinite(extrapolated_fit_chis)
    if np.any(extrapolated_fit_stable):
        axis.plot(
            extrapolated_fit_temperatures[extrapolated_fit_stable],
            extrapolated_fit_chis[extrapolated_fit_stable],
            "--",
            color="tab:orange",
            linewidth=1.7,
            label=rf"same $p_\chi(T)$, extrapolated to "
            rf"${fitted_response_extrapolation_max_temperature:g}$ K",
        )
    linear_unstable = ~displayed_fit_stable
    if np.any(linear_unstable):
        axis.plot(
            displayed_fit_temperatures[linear_unstable],
            np.zeros(np.count_nonzero(linear_unstable)),
            "x",
            color="0.42",
            markersize=5.0,
            label=r"no stable centered root for $p(T)$",
        )
    axis.plot(
        temperatures[zero_pressure_stable],
        zero_pressure_chi[zero_pressure_stable],
        "D-.",
        color="tab:purple",
        linewidth=1.6,
        markersize=3.8,
        label=r"SCHA, $p_{\rm eff}=0$",
    )
    axis.plot(
        temperatures[reference_domain],
        barrett[reference_domain],
        "--",
        color="tab:green",
        linewidth=1.8,
        label="Barrett (1952), empirical Curie--Weiss",
    )
    axis.plot(
        wieczorek_temperature,
        wieczorek_chi,
        "s",
        color="0.15",
        markerfacecolor="white",
        markersize=5.0,
        label=r"Wieczorek et al. (2006), 1 kHz",
    )
    axis.axvline(
        411.0,
        color="0.5",
        linestyle=":",
        linewidth=1.2,
        label=r"$T_C=411$ K (MD)",
    )
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"Soft-mode susceptibility $\chi_T$")
    axis.set_xlim(
        float(temperatures[0]) - 5.0,
        max(570.0, float(temperatures[-1]) + 5.0),
    )
    finite_values = np.concatenate(
        (
            lattice_chi[lattice_stable],
            displayed_fit_chis[displayed_fit_stable],
            extrapolated_fit_chis[extrapolated_fit_stable],
            zero_pressure_chi[zero_pressure_stable],
            barrett[reference_domain],
            wieczorek_chi,
        )
    )
    axis.set_ylim(0.0, 1.12 * float(np.max(finite_values)))
    axis.grid(alpha=0.23)
    axis.legend(frameon=False, fontsize=8.3)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare lattice-matched, dielectric-fitted, and zero pressure "
            "after including the k- and Matsubara-dependent acoustic "
            "quartic vertex."
        )
    )
    parser.add_argument("--start", type=float, default=400.0)
    parser.add_argument("--stop", type=float, default=565.0)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--ngrid", type=int, default=40)
    parser.add_argument(
        "--constant-pressure-gpa",
        type=float,
        default=None,
        help=(
            "Replace the default temperature-dependent lattice match by an "
            "explicit constant pressure."
        ),
    )
    parser.add_argument(
        "--lattice-calibration-temperature",
        type=float,
        default=None,
        help="Legacy single-point lattice calibration temperature.",
    )
    parser.add_argument(
        "--lattice-calibration-target-angstrom",
        type=float,
        default=None,
        help="Legacy single-point lattice calibration target.",
    )
    parser.add_argument(
        "--lattice-fit-start",
        type=float,
        default=EXPERIMENTAL_LATTICE_FIT_MIN_K,
        help="Lower temperature of the weighted experimental a(T) fit.",
    )
    parser.add_argument(
        "--lattice-fit-stop",
        type=float,
        default=EXPERIMENTAL_LATTICE_FIT_MAX_K,
        help="Upper temperature of the weighted experimental a(T) fit.",
    )
    parser.add_argument(
        "--fit-pressure-temperature",
        type=float,
        default=None,
        help=(
            "Legacy alternative: fit the constant pressure to Barrett at this "
            "temperature instead of calibrating the lattice parameter."
        ),
    )
    parser.add_argument(
        "--linear-pressure-slope",
        type=float,
        default=DEFAULT_LINEAR_PRESSURE_SLOPE_GPA_PER_K,
        help="Slope a in p(T)=a*T+b (default: -0.005 GPa/K).",
    )
    parser.add_argument(
        "--linear-pressure-intercept",
        type=float,
        default=0.0,
        help="Intercept b in p(T)=a*T+b GPa (default: 0 GPa).",
    )
    parser.add_argument(
        "--fit-linear-pressure",
        action="store_true",
        help=(
            "Legacy alternative: fit a and b to Barrett over the selected "
            "fit interval instead of fitting the Wieczorek data."
        ),
    )
    parser.add_argument(
        "--use-explicit-linear-pressure",
        action="store_true",
        help=(
            "Do not fit the affine pressure; use --linear-pressure-slope and "
            "--linear-pressure-intercept directly."
        ),
    )
    parser.add_argument("--linear-fit-start", type=float, default=420.0)
    parser.add_argument("--linear-fit-stop", type=float, default=560.0)
    parser.add_argument("--acoustic-mass-amu", type=float, default=ACOUSTIC_MASS_AMU)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = reference_scan.temperature_grid(args.start, args.stop, args.step)
    coefficients = scha.MATERIALS["nishimatsu"]
    grid = build_inhomogeneous_quartic_grid(
        args.ngrid, coefficients, acoustic_mass_amu=args.acoustic_mass_amu
    )
    single_point_lattice = (
        args.lattice_calibration_temperature is not None
        or args.lattice_calibration_target_angstrom is not None
    )
    if single_point_lattice and (
        args.lattice_calibration_temperature is None
        or args.lattice_calibration_target_angstrom is None
    ):
        raise SystemExit(
            "single-point lattice calibration requires both its temperature "
            "and target lattice parameter"
        )
    lattice_targets = None
    lattice_fit_data = None
    if args.constant_pressure_gpa is not None:
        fitted_pressure = float(args.constant_pressure_gpa)
        lattice_pressures = np.full_like(temperatures, fitted_pressure)
        calibration_description = "explicit constant pressure"
        lattice_pressure_description = (
            rf"SCHA, constant $p={fitted_pressure:.3f}$ GPa"
        )
    elif args.fit_pressure_temperature is not None:
        fitted_pressure = fit_constant_pressure(
            args.fit_pressure_temperature, coefficients, grid
        )
        lattice_pressures = np.full_like(temperatures, fitted_pressure)
        calibration_description = (
            f"Barrett susceptibility at {args.fit_pressure_temperature:g} K"
        )
        lattice_pressure_description = (
            rf"SCHA, constant $p={fitted_pressure:.3f}$ GPa"
        )
    elif single_point_lattice:
        fitted_pressure = calibrate_pressure_to_lattice_parameter(
            args.lattice_calibration_temperature,
            args.lattice_calibration_target_angstrom,
            coefficients,
            grid,
        )
        lattice_pressures = np.full_like(temperatures, fitted_pressure)
        calibration_description = (
            f"a({args.lattice_calibration_temperature:g} K)="
            f"{args.lattice_calibration_target_angstrom:.6f} angstrom"
        )
        lattice_pressure_description = (
            rf"SCHA, single-point lattice $p={fitted_pressure:.3f}$ GPa"
        )
    else:
        (
            lattice_fit_slope,
            lattice_fit_intercept,
            lattice_fit_temperatures,
            lattice_fit_parameters,
            lattice_fit_uncertainties,
        ) = fit_experimental_cubic_lattice_parameter(
            args.lattice_fit_start, args.lattice_fit_stop
        )
        (
            lattice_pressure_slope,
            lattice_pressure_intercept,
            lattice_pressures,
            lattice_targets,
            lattice_required_pressures,
        ) = fit_affine_lattice_matching_pressure(
            temperatures,
            lattice_fit_slope,
            lattice_fit_intercept,
            coefficients,
            grid,
        )
        lattice_fit_data = (
            lattice_fit_slope,
            lattice_fit_intercept,
            lattice_fit_temperatures,
            lattice_fit_parameters,
            lattice_fit_uncertainties,
            lattice_pressure_slope,
            lattice_pressure_intercept,
            lattice_required_pressures,
        )
        calibration_description = (
            f"Nakatani a_exp(T) fit over {args.lattice_fit_start:g}-"
            f"{args.lattice_fit_stop:g} K"
        )
        lattice_pressure_description = (
            r"SCHA, lattice-matched $p_a(T)$"
        )
    linear_slope = args.linear_pressure_slope
    linear_intercept = args.linear_pressure_intercept
    required_pressures = None
    fit_target_temperatures = None
    fit_target_chis = None
    fit_predicted_chis = None
    linear_pressure_description = "explicit"
    fit_temperatures = temperatures[
        (temperatures >= args.linear_fit_start)
        & (temperatures <= args.linear_fit_stop)
    ]
    if args.fit_linear_pressure and args.use_explicit_linear_pressure:
        raise SystemExit(
            "--fit-linear-pressure and --use-explicit-linear-pressure "
            "cannot be used together"
        )
    if args.fit_linear_pressure:
        if fit_temperatures.size < 2:
            raise SystemExit("the linear-pressure fit needs at least two temperatures")
        linear_slope, linear_intercept, required_pressures = (
            fit_linear_pressure_to_barrett(fit_temperatures, coefficients, grid)
        )
        linear_pressure_description = "Barrett fit"
    elif not args.use_explicit_linear_pressure:
        (
            linear_slope,
            linear_intercept,
            required_pressures,
            fit_target_temperatures,
            fit_target_chis,
            fit_predicted_chis,
        ) = fit_affine_pressure_to_wieczorek(coefficients, grid)
        linear_pressure_description = "fit to Wieczorek"
    lattice_rows = scan_protocol(
        temperatures,
        "prescribed",
        0.0,
        coefficients,
        grid,
        prescribed_pressures_gpa=lattice_pressures,
    )
    linear_rows = scan_protocol(
        temperatures,
        "linear",
        0.0,
        coefficients,
        grid,
        linear_pressure_slope_gpa_per_k=linear_slope,
        linear_pressure_intercept_gpa=linear_intercept,
    )
    zero_pressure_rows = scan_protocol(
        temperatures, "constant", 0.0, coefficients, grid
    )

    output_dir = (
        Path(__file__).resolve().parents[1] / "outputs" / "paraelectric"
    )
    stem = (
        f"chi_vs_temperature_{args.start:g}_{args.stop:g}K_n{args.ngrid}_"
        "inhomogeneous_dynamic"
    )
    csv_path = args.csv or output_dir / f"{stem}.csv"
    plot_path = args.plot or output_dir / f"{stem}.png"
    write_comparison_csv(
        lattice_rows, linear_rows, zero_pressure_rows, csv_path
    )
    write_plot(
        lattice_rows,
        linear_rows,
        zero_pressure_rows,
        lattice_pressure_description,
        linear_slope,
        linear_intercept,
        linear_pressure_description,
        plot_path,
        fitted_response_temperatures=fit_target_temperatures,
        fitted_response_chis=fit_predicted_chis,
    )

    lattice_stable = sum(
        np.isfinite(row["chi_T_SCHA"]) for row in lattice_rows
    )
    linear_stable = sum(
        np.isfinite(row["chi_T_SCHA"]) for row in linear_rows
    )
    zero_pressure_stable = sum(
        np.isfinite(row["chi_T_SCHA"]) for row in zero_pressure_rows
    )
    print(f"lattice protocol: {calibration_description}")
    print(
        f"lattice-protocol stable roots = {lattice_stable}/{len(temperatures)}"
    )
    if lattice_targets is not None:
        reconstructed_lattice = np.asarray(
            [row["lattice_parameter_angstrom"] for row in lattice_rows]
        )
        lattice_errors = reconstructed_lattice - lattice_targets
        print(
            f"lattice fit p(T) range = {np.min(lattice_pressures):.12g} to "
            f"{np.max(lattice_pressures):.12g} GPa, max |a-a_target| = "
            f"{np.nanmax(np.abs(lattice_errors)):.6g} angstrom"
        )
        if lattice_fit_data is not None:
            (
                slope,
                intercept,
                fit_t,
                fit_a,
                fit_sigma,
                pressure_slope,
                pressure_intercept,
                required_lattice_pressures,
            ) = lattice_fit_data
            fit_residuals = slope * fit_t + intercept - fit_a
            pressure_residuals = lattice_pressures - required_lattice_pressures
            print(
                f"a_exp_fit(T)={slope:.12g}*T{intercept:+.12g} angstrom, "
                f"weighted-data RMSE={np.sqrt(np.mean(fit_residuals**2)):.6g} "
                "angstrom"
            )
            print(
                f"p_a(T)={pressure_slope:+.12g}*T"
                f"{pressure_intercept:+.12g} GPa, pointwise-pressure RMSE="
                f"{np.sqrt(np.mean(pressure_residuals**2)):.6g} GPa"
            )
    else:
        print(
            f"constant lattice-protocol pressure = {lattice_pressures[0]:.12g} GPa"
        )
    print(
        f"zero-pressure stable roots = {zero_pressure_stable}/{len(temperatures)}"
    )
    print(
        f"p(T)={linear_slope:+.12g}*T{linear_intercept:+.12g} GPa stable roots = "
        f"{linear_stable}/{len(temperatures)}"
    )
    if required_pressures is not None and fit_target_temperatures is not None:
        fitted_pressures = (
            linear_slope * fit_target_temperatures + linear_intercept
        )
        pressure_rmse = float(
            np.sqrt(np.mean((fitted_pressures - required_pressures) ** 2))
        )
        chi_errors = fit_predicted_chis - fit_target_chis
        chi_rmse = float(np.sqrt(np.mean(chi_errors**2)))
        print(
            "affine Wieczorek fit over "
            f"{fit_target_temperatures[0]:g}-{fit_target_temperatures[-1]:g} K: "
            f"a={linear_slope:.12g} GPa/K, "
            f"b={linear_intercept:.12g} GPa, "
            f"pressure RMSE={pressure_rmse:.12g} GPa, "
            f"vertical chi RMSE={chi_rmse:.12g}"
        )
    elif required_pressures is not None:
        fitted_pressures = linear_slope * fit_temperatures + linear_intercept
        pressure_rmse = float(
            np.sqrt(np.mean((fitted_pressures - required_pressures) ** 2))
        )
        fit_rows = [
            row
            for row in linear_rows
            if args.linear_fit_start
            <= row["temperature_K"]
            <= args.linear_fit_stop
        ]
        relative_errors = np.array(
            [
                abs(
                    row["chi_T_SCHA"]
                    / reference_scan.experimental_curie_weiss(
                        row["temperature_K"]
                    )[1]
                    - 1.0
                )
                for row in fit_rows
            ]
        )
        print(
            f"negative linear Barrett fit over {fit_temperatures[0]:g}-"
            f"{fit_temperatures[-1]:g} K: a={linear_slope:.12g} GPa/K, "
            f"b={linear_intercept:.12g} GPa, pressure RMSE={pressure_rmse:.12g} "
            f"GPa, chi MAPE={100.0 * np.mean(relative_errors):.6g}%"
        )
    for target in (420.0, 500.0, 560.0):
        matching = [
            row for row in lattice_rows if np.isclose(row["temperature_K"], target)
        ]
        if matching:
            row = matching[0]
            print(
                f"lattice p(T): chi_T({target:g} K) = {row['chi_T_SCHA']:.10g}, "
                f"Delta A1_loc = {row['delta_A1_quartic_local']:.10g}, "
                f"Delta A1_hom = {row['delta_A1_quartic_homogeneous']:.10g}, "
                f"Delta A1_inh = {row['delta_A1_quartic_inhomogeneous']:.10g}"
            )
    print(f"wrote {csv_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
