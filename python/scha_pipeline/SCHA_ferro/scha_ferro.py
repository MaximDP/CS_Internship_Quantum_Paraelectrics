#!/usr/bin/env python3
"""Ferroelectric SCHA solver for Eqs. (159)--(160).

The implementation mirrors ``SCHA_paraelc/scha_paraelc.py`` but adds a
static homogeneous broken-symmetry background u_min. Brillouin-zone loops use
the full periodic Nishimatsu/FERAM harmonic kernel. All six independent
components of the positive local mass matrix and all three components of
u_min are varied. The resulting implementation therefore solves nine scalar
equations for nine unknowns and does not prescribe the polarization direction.
The zero-mode background is never included in the BZ loop; it enters only
through the contractions of the stationarity equations. The local quartic,
homogeneous-strain, and dynamic inhomogeneous-strain sectors are evaluated
separately, together with the local sixth- and eighth-order terms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))
from nishimatsu_harmonic import HarmonicGrid, build_harmonic_grid


KB_HARTREE_PER_K = 3.166811563e-6
AMU_TO_ELECTRON_MASS = 1822.888486217313
EV_TO_HARTREE = 1.0 / 27.211386245988
ANGSTROM_PER_BOHR = 0.529177210903
EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2 = (
    EV_TO_HARTREE * ANGSTROM_PER_BOHR**2
)
ACOUSTIC_MASS_AMU = 46.44

# Nishimatsu/Wu--Cohen elastic and mode--strain coefficients.
B11_EV = 126.731671475652
B12_EV = 41.7582963902598
B44_EV = 49.2408864348646
B1XX_EV_PER_ANGSTROM2 = -185.347187551195
B1YY_EV_PER_ANGSTROM2 = -3.28092949275457
B4YZ_EV_PER_ANGSTROM2 = -14.5501738943852

# Nishimatsu bare local quartic and homogeneous-strain tensors. Their
# thermodynamic contractions must remain separate: the fluctuation term is
# 3*beta_bare*G - Lambda_hom*G/2, whereas the uniform-background curvature
# contains the additional two homogeneous exchange channels.
BARE_B1 = 0.0816281455
BARE_B2 = 0.6655891570
HOMOGENEOUS_LAMBDA_B1 = -0.0605593324
HOMOGENEOUS_LAMBDA_B2 = 1.0993850582


@dataclass(frozen=True)
class SchaCoefficients:
    """Coefficients from Appendix A, in Hartree and bohr conventions."""

    name: str
    a0: float = 7.46
    A01: float = -0.035058109
    A02: float = +0.155207480
    A03: float = -5.348957200
    A04: float = +0.905359992
    A05: float = +0.572574326
    b1_eff: float = 1.36449e-1
    b2_eff: float = 2.93874e-1
    k1_6: float = 0.0
    k2_6: float = 0.0
    k3_6: float = 0.0
    k4_8: float = 0.0
    mass_amu: float = 39.05
    zstar: float = 9.956
    eps_inf: float = 5.24
    kappa2: float = 0.0568
    j1: float = -0.02734
    j2: float = +0.04020
    j3: float = +0.00927
    j4: float = -0.00815
    j5: float = +0.00580
    j6: float = +0.00370
    j7: float = +0.00185

    @property
    def omega0(self) -> float:
        return self.a0**3

    @property
    def kmax(self) -> float:
        return math.pi / self.a0

    @property
    def mass_au(self) -> float:
        return self.mass_amu * AMU_TO_ELECTRON_MASS


MATERIALS: dict[str, SchaCoefficients] = {
    "vanderbilt": SchaCoefficients(name="vanderbilt_zvr"),
    "nishimatsu": SchaCoefficients(
        name="nishimatsu_2010_bto",
        a0=7.532372743713335,
        A01=-0.031248507143898818,
        A02=+0.13891652381255355,
        A03=-2.6304665896796883,
        A04=+1.5738147329579442,
        A05=+0.4567855127021116,
        b1_eff=0.1119078117465313,
        b2_eff=0.11589662794120481,
        k1_6=-0.2162513067289395,
        k2_6=+0.15937669276170113,
        k3_6=+0.669944535531705,
        k4_8=+0.14506807378350556,
        mass_amu=38.24,
        zstar=10.33,
        eps_inf=6.8691464565,
        kappa2=0.08782224891906101,
        j1=-0.021446406452141763,
        j2=-0.011618803008767388,
        j3=+0.007095114406006146,
        j4=-0.006291221721453839,
        j5=0.0,
        j6=+0.0028495045061522105,
        j7=0.0,
    ),
}


def cubic_rank4_component(
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
        delta_ab * delta_gd + delta_ag * delta_bd + delta_ad * delta_bg
    ) + b2 * delta_abgd


def gamma_component(indices: tuple[int, int, int, int, int, int], c: SchaCoefficients) -> float:
    counts = sorted([indices.count(axis) for axis in range(3) if indices.count(axis) > 0], reverse=True)
    if counts == [6]:
        return 6.0 * c.k1_6
    if counts == [4, 2]:
        return 2.0 * (3.0 * c.k1_6 + c.k2_6) / 5.0
    if counts == [2, 2, 2]:
        return (6.0 * c.k1_6 + c.k3_6) / 15.0
    return 0.0


def rho_component(indices: tuple[int, int, int, int, int, int, int, int], c: SchaCoefficients) -> float:
    counts = sorted([indices.count(axis) for axis in range(3) if indices.count(axis) > 0], reverse=True)
    if counts == [8]:
        return 8.0 * c.k4_8
    if counts == [6, 2]:
        return 8.0 * c.k4_8 / 7.0
    if counts == [4, 4]:
        return 24.0 * c.k4_8 / 35.0
    if counts == [4, 2, 2]:
        return 8.0 * c.k4_8 / 35.0
    return 0.0


@dataclass(frozen=True)
class SchaTensors:
    beta: np.ndarray
    lambda_hom: np.ndarray
    gamma: np.ndarray
    rho: np.ndarray


def build_tensors(c: SchaCoefficients) -> SchaTensors:
    beta = np.zeros((3, 3, 3, 3), dtype=float)
    lambda_hom = np.zeros((3, 3, 3, 3), dtype=float)
    gamma = np.zeros((3, 3, 3, 3, 3, 3), dtype=float)
    rho = np.zeros((3, 3, 3, 3, 3, 3, 3, 3), dtype=float)

    if c.name == "nishimatsu_2010_bto":
        beta_coefficients = (BARE_B1, BARE_B2)
        lambda_coefficients = (
            HOMOGENEOUS_LAMBDA_B1,
            HOMOGENEOUS_LAMBDA_B2,
        )
        projected = np.array(beta_coefficients) - 0.5 * np.array(
            lambda_coefficients
        )
        if not np.allclose(
            projected,
            (c.b1_eff, c.b2_eff),
            rtol=1.0e-8,
            atol=1.0e-10,
        ):
            raise RuntimeError("bare beta and Lambda_hom do not reproduce beta_eff")
    else:
        beta_coefficients = (c.b1_eff, c.b2_eff)
        lambda_coefficients = (0.0, 0.0)

    for a in range(3):
        for b in range(3):
            for g in range(3):
                for d in range(3):
                    beta[a, b, g, d] = cubic_rank4_component(
                        a, b, g, d, *beta_coefficients
                    )
                    lambda_hom[a, b, g, d] = cubic_rank4_component(
                        a, b, g, d, *lambda_coefficients
                    )

    for indices in np.ndindex(gamma.shape):
        gamma[indices] = gamma_component(indices, c)
    for indices in np.ndindex(rho.shape):
        rho[indices] = rho_component(indices, c)

    return SchaTensors(
        beta=beta,
        lambda_hom=lambda_hom,
        gamma=gamma,
        rho=rho,
    )


def longitudinal_projector(direction: np.ndarray) -> np.ndarray:
    return np.outer(direction, direction)


def transverse_projector(direction: np.ndarray) -> np.ndarray:
    return np.eye(3) - longitudinal_projector(direction)


def local_mass_matrix(a_t: float, a_l: float, direction: np.ndarray) -> np.ndarray:
    """Local split mass A_T P_T + A_L P_L, with P_L parallel to p_min."""

    return a_t * transverse_projector(direction) + a_l * longitudinal_projector(direction)


def static_kernel(kvec: np.ndarray, mass_matrix: np.ndarray, c: SchaCoefficients) -> np.ndarray:
    """Static trial matrix mu(k), without the mass times omega_n^2 term."""

    k2 = float(np.dot(kvec, kvec))
    mat = mass_matrix + c.A02 * k2 * np.eye(3)
    mat += c.A03 * np.outer(kvec, kvec)
    mat += c.A04 * np.diag(kvec * kvec)
    if k2 > 0.0:
        mat += c.A05 * np.outer(kvec, kvec) / k2
    return mat


def coth(x: float) -> float:
    if abs(x) < 1.0e-5:
        return 1.0 / x + x / 3.0 - x**3 / 45.0
    if x > 50.0:
        return 1.0
    return 1.0 / math.tanh(x)


def mode_covariance(mu: float, temperature: float, c: SchaCoefficients) -> float:
    """Physical equal-time covariance, including zero-point motion at T=0."""

    if mu <= 1.0e-14:
        raise ValueError("non-positive static eigenvalue is not allowed")
    omega = math.sqrt(mu / c.mass_au)
    if temperature == 0.0:
        return 1.0 / (2.0 * c.mass_au * omega)
    kbt = KB_HARTREE_PER_K * temperature
    return coth(omega / (2.0 * kbt)) / (2.0 * c.mass_au * omega)


def equal_time_covariance(
    kvec: np.ndarray,
    mass_matrix: np.ndarray,
    temperature: float,
    c: SchaCoefficients,
) -> np.ndarray:
    mu = static_kernel(kvec, mass_matrix, c)
    eigvals, eigvecs = np.linalg.eigh(mu)
    diag = np.array([mode_covariance(float(value), temperature, c) for value in eigvals])
    return (eigvecs * diag) @ eigvecs.T


@lru_cache(maxsize=16)
def cached_harmonic_grid(ngrid: int, cutoff: float, c: SchaCoefficients) -> HarmonicGrid:
    return build_harmonic_grid(c, ngrid=ngrid, cutoff=cutoff)


def wavevectors_and_weights(
    ngrid: int, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(ngrid)
    points = cutoff * nodes
    scaled_weights = cutoff * weights
    kx, ky, kz = np.meshgrid(points, points, points, indexing="ij")
    wx, wy, wz = np.meshgrid(
        scaled_weights, scaled_weights, scaled_weights, indexing="ij"
    )
    return (
        np.stack((kx.ravel(), ky.ravel(), kz.ravel()), axis=1),
        (wx * wy * wz).ravel(),
    )


def acoustic_kernel(wavevectors: np.ndarray) -> np.ndarray:
    """Nishimatsu acoustic kernel Phi_ab(k), in Ha/bohr^2."""

    b11 = B11_EV * EV_TO_HARTREE
    b12 = B12_EV * EV_TO_HARTREE
    b44 = B44_EV * EV_TO_HARTREE
    k2 = np.einsum("na,na->n", wavevectors, wavevectors)
    kernel = (b12 + b44) * np.einsum(
        "na,nb->nab", wavevectors, wavevectors, optimize=True
    )
    axes = np.arange(3)
    kernel[:, axes, axes] = (
        b44 * k2[:, None] + (b11 - b44) * wavevectors**2
    )
    return kernel


def mixed_tensor(wavevectors: np.ndarray) -> np.ndarray:
    """Nishimatsu mode--strain vertex H_sab(k), symmetric in a,b."""

    b1xx = B1XX_EV_PER_ANGSTROM2 * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    b1yy = B1YY_EV_PER_ANGSTROM2 * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    b4yz = B4YZ_EV_PER_ANGSTROM2 * EV_PER_ANGSTROM2_TO_HARTREE_PER_BOHR2
    tensor = np.zeros((wavevectors.shape[0], 3, 3, 3), dtype=float)
    for acoustic_axis in range(3):
        for polar_axis in range(3):
            coupling = b1xx if acoustic_axis == polar_axis else b1yy
            tensor[:, acoustic_axis, polar_axis, polar_axis] = (
                wavevectors[:, acoustic_axis] * coupling
            )
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


def bosonic_sum(omega: np.ndarray, kbt: float) -> np.ndarray:
    return 1.0 / np.tanh(omega / (2.0 * kbt)) / (2.0 * kbt * omega)


def bosonic_product_sum(
    polar_omega: np.ndarray,
    acoustic_omega: np.ndarray,
    kbt: float,
) -> np.ndarray:
    """Sum two bosonic propagator denominators over Matsubara frequency."""

    polar = polar_omega[:, None, :]
    acoustic = acoustic_omega[:, :, None]
    denominator = acoustic**2 - polar**2
    result = np.empty_like(denominator)
    near = np.abs(denominator) <= 1.0e-9 * (acoustic**2 + polar**2)
    np.divide(
        bosonic_sum(polar, kbt) - bosonic_sum(acoustic, kbt),
        denominator,
        out=result,
        where=~near,
    )
    if np.any(near):
        omega = 0.5 * (polar + acoustic)
        argument = omega / (2.0 * kbt)
        coth_value = 1.0 / np.tanh(argument)
        csch2 = np.zeros_like(argument)
        moderate = argument < 40.0
        csch2[moderate] = 1.0 / np.sinh(argument[moderate]) ** 2
        limit = coth_value / (4.0 * kbt * omega**3)
        limit += csch2 / (8.0 * kbt**2 * omega**2)
        result[near] = np.broadcast_to(limit, result.shape)[near]
    return result


def thermal_weighted_bosonic_product_sum(
    polar_omega: np.ndarray,
    acoustic_omega: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return kBT times the two-propagator Matsubara sum.

    The explicit zero-temperature limit avoids the singular scaled variables
    used in the finite-temperature derivation:

        lim_T->0 kBT sum_n 1/[(wn^2+wp^2)(wn^2+wa^2)]
          = 1/[2 wp wa (wp+wa)].
    """

    polar = polar_omega[:, None, :]
    acoustic = acoustic_omega[:, :, None]
    if temperature == 0.0:
        return 1.0 / (2.0 * polar * acoustic * (polar + acoustic))
    kbt = KB_HARTREE_PER_K * temperature
    return kbt * bosonic_product_sum(polar_omega, acoustic_omega, kbt)


@dataclass(frozen=True)
class AcousticGrid:
    weights: np.ndarray
    prefactor: float
    acoustic_omega: np.ndarray
    acoustic_eigenvectors: np.ndarray
    mixed_tensor: np.ndarray
    acoustic_mass_au: float


@lru_cache(maxsize=8)
def cached_acoustic_grid(
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
) -> AcousticGrid:
    """Precompute the acoustic modes entering the Gamma-point convolution."""

    if c.name != "nishimatsu_2010_bto":
        raise ValueError("the inhomogeneous strain kernel is Nishimatsu-specific")
    wavevectors, weights = wavevectors_and_weights(ngrid, cutoff)
    harmonic = cached_harmonic_grid(ngrid, cutoff, c)
    if not np.allclose(weights, harmonic.weights, rtol=1.0e-14, atol=0.0):
        raise RuntimeError("the acoustic and polar quadratures do not match")

    acoustic_eigenvalues, acoustic_eigenvectors = np.linalg.eigh(
        acoustic_kernel(wavevectors)
    )
    if np.any(acoustic_eigenvalues <= 0.0):
        raise ValueError("non-positive acoustic eigenvalue is not allowed")
    acoustic_mass_au = ACOUSTIC_MASS_AMU * AMU_TO_ELECTRON_MASS
    coupling = mixed_tensor(wavevectors)
    return AcousticGrid(
        weights=weights,
        prefactor=harmonic.prefactor,
        acoustic_omega=np.sqrt(acoustic_eigenvalues / acoustic_mass_au),
        acoustic_eigenvectors=acoustic_eigenvectors,
        mixed_tensor=coupling,
        acoustic_mass_au=acoustic_mass_au,
    )


def integrate_covariance(
    mass_matrix: np.ndarray,
    temperature: float,
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
) -> np.ndarray:
    """Integrate physical Q=<delta u delta u> over the full lattice kernel."""

    grid = cached_harmonic_grid(ngrid, cutoff, c)
    eigvals, eigvecs = np.linalg.eigh(grid.offsets + mass_matrix)
    if np.any(eigvals <= 1.0e-14):
        raise ValueError("non-positive static eigenvalue is not allowed")

    omega = np.sqrt(eigvals / c.mass_au)
    if temperature == 0.0:
        mode_sums = 1.0 / (2.0 * c.mass_au * omega)
    else:
        kbt = KB_HARTREE_PER_K * temperature
        mode_sums = 1.0 / np.tanh(omega / (2.0 * kbt))
        mode_sums /= 2.0 * c.mass_au * omega
    propagators = np.einsum(
        "nim,nm,njm->nij", eigvecs, mode_sums, eigvecs, optimize=True
    )
    return grid.prefactor * np.einsum(
        "n,nij->ij", grid.weights, propagators, optimize=True
    )


def inhomogeneous_self_energy_tensor(
    mass_matrix: np.ndarray,
    temperature: float,
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
) -> np.ndarray:
    """Return the dynamic inhomogeneous self-energy at K=(0,0).

    This is ``-kBT * G_inh_ab(0,0)/N`` from Eq. (159). The same tensor
    multiplies the static background in Eq. (160).
    """

    if c.name != "nishimatsu_2010_bto":
        return np.zeros((3, 3), dtype=float)
    harmonic = cached_harmonic_grid(ngrid, cutoff, c)
    acoustic = cached_acoustic_grid(ngrid, cutoff, c)
    polar_eigenvalues, polar_eigenvectors = np.linalg.eigh(
        harmonic.offsets + mass_matrix
    )
    if np.any(polar_eigenvalues <= 1.0e-14):
        raise ValueError("non-positive static polar eigenvalue is not allowed")

    polar_omega = np.sqrt(polar_eigenvalues / c.mass_au)
    weighted_product_sum = thermal_weighted_bosonic_product_sum(
        polar_omega,
        acoustic.acoustic_omega,
        temperature,
    ) / (c.mass_au * acoustic.acoustic_mass_au)

    # X_l,ag=e_l,s H_sag and y_lj,a=X_l,ag v_gj.  The external tensor is
    # y_a y_b; tracing it reproduces the scalar paraelectric implementation.
    projected_vertex = np.einsum(
        "nsl,nsag->nlag",
        acoustic.acoustic_eigenvectors,
        acoustic.mixed_tensor,
        optimize=True,
    )
    external_vectors = np.einsum(
        "nlag,ngj->nlja",
        projected_vertex,
        polar_eigenvectors,
        optimize=True,
    )
    sigma = -acoustic.prefactor * np.einsum(
        "n,nlj,nlja,nljb->ab",
        acoustic.weights,
        weighted_product_sum,
        external_vectors,
        external_vectors,
        optimize=True,
    )
    return 0.5 * (sigma + sigma.T)


def min_static_eigenvalue(
    mass_matrix: np.ndarray,
    ngrid: int,
    cutoff: float,
    c: SchaCoefficients,
) -> float:
    grid = cached_harmonic_grid(ngrid, cutoff, c)
    return float(np.linalg.eigvalsh(grid.offsets + mass_matrix).min())


def mass_self_energy_tensors(
    covariance: np.ndarray,
    uvec: np.ndarray,
    tensors: SchaTensors,
    sigma_inhomogeneous: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return Eq. (159)'s Gamma-point self-energy tensors by sector."""

    uu = np.outer(uvec, uvec)

    quartic_local = 3.0 * np.einsum(
        "abgd,gd->ab", tensors.beta, covariance + uu, optimize=False
    )
    quartic_homogeneous = -0.5 * np.einsum(
        "abgd,gd->ab", tensors.lambda_hom, covariance + uu, optimize=False
    )
    # At Gamma the uniform-background part of the homogeneous exchange channel
    # is finite because p_min^2/N is intensive. The fluctuation H/N term is
    # omitted in the thermodynamic limit.
    quartic_homogeneous -= np.einsum(
        "agbd,gd->ab", tensors.lambda_hom, uu, optimize=False
    )

    sextic_arg = (
        np.einsum("gd,er->gder", covariance, covariance, optimize=False)
        + 2.0 * np.einsum("gd,er->gder", uu, covariance, optimize=False)
        + (1.0 / 3.0) * np.einsum("gd,er->gder", uu, uu, optimize=False)
    )
    sextic_tensor = 15.0 * np.einsum("abgder,gder->ab", tensors.gamma, sextic_arg, optimize=False)

    octic_arg = (
        np.einsum("gd,er,hl->gderhl", covariance, covariance, covariance, optimize=False)
        + 3.0 * np.einsum("gd,er,hl->gderhl", uu, covariance, covariance, optimize=False)
        + np.einsum("gd,er,hl->gderhl", uu, uu, covariance, optimize=False)
        + (1.0 / 15.0) * np.einsum("gd,er,hl->gderhl", uu, uu, uu, optimize=False)
    )
    octic_tensor = 105.0 * np.einsum("abgderhl,gderhl->ab", tensors.rho, octic_arg, optimize=False)

    return {
        "quartic_local": quartic_local,
        "quartic_homogeneous": quartic_homogeneous,
        "quartic_inhomogeneous": sigma_inhomogeneous,
        "sextic": sextic_tensor,
        "octic": octic_tensor,
    }


def project_split_parts(parts: dict[str, np.ndarray], direction: np.ndarray) -> dict[str, dict[str, float]]:
    p_l = longitudinal_projector(direction)
    p_t = transverse_projector(direction)
    split: dict[str, dict[str, float]] = {}
    for name, tensor in parts.items():
        split[name] = {
            "T": float(np.einsum("ab,ab->", p_t, tensor) / 2.0),
            "L": float(np.einsum("ab,ab->", p_l, tensor)),
        }
    return split


def umin_gradient(
    covariance: np.ndarray,
    uvec: np.ndarray,
    c: SchaCoefficients,
    tensors: SchaTensors,
    sigma_inhomogeneous: np.ndarray,
) -> np.ndarray:
    """Return the physical-displacement form of the Eq. (160) gradient."""

    beta = tensors.beta
    gamma = tensors.gamma
    rho = tensors.rho
    u = uvec
    q = covariance
    grad = c.A01 * u.copy()

    grad += 3.0 * np.einsum("ebgd,b,gd->e", beta, u, q, optimize=False)
    grad += np.einsum("ebgd,b,g,d->e", beta, u, u, u, optimize=False)

    grad -= 0.5 * np.einsum(
        "ebgd,b,gd->e", tensors.lambda_hom, u, q, optimize=False
    )
    grad -= 0.5 * np.einsum(
        "ebgd,b,g,d->e", tensors.lambda_hom, u, u, u, optimize=False
    )
    grad += sigma_inhomogeneous @ u

    grad += 15.0 * np.einsum("ebgdrs,b,gd,rs->e", gamma, u, q, q, optimize=False)
    grad += 10.0 * np.einsum("ebgdrs,b,g,d,rs->e", gamma, u, u, u, q, optimize=False)
    grad += np.einsum("ebgdrs,b,g,d,r,s->e", gamma, u, u, u, u, u, optimize=False)

    grad += 105.0 * np.einsum("ebgdrshl,b,gd,rs,hl->e", rho, u, q, q, q, optimize=False)
    grad += 105.0 * np.einsum("ebgdrshl,b,g,d,rs,hl->e", rho, u, u, u, q, q, optimize=False)
    grad += 21.0 * np.einsum("ebgdrshl,b,g,d,r,s,hl->e", rho, u, u, u, u, u, q, optimize=False)
    grad += np.einsum("ebgdrshl,b,g,d,r,s,h,l->e", rho, u, u, u, u, u, u, u, optimize=False)

    return grad


def parse_direction(raw: str) -> np.ndarray:
    aliases = {
        "generic": np.array([1.0, 0.8, 0.6]),
        "001": np.array([0.0, 0.0, 1.0]),
        "010": np.array([0.0, 1.0, 0.0]),
        "100": np.array([1.0, 0.0, 0.0]),
        "110": np.array([1.0, 1.0, 0.0]),
        "111": np.array([1.0, 1.0, 1.0]),
    }
    key = raw.strip().lower().replace("[", "").replace("]", "")
    if key in aliases:
        vec = aliases[key]
    else:
        parts = [float(item) for item in key.replace(",", " ").split()]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "direction must be generic, 001, 110, 111, or three numbers"
            )
        vec = np.array(parts, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise argparse.ArgumentTypeError("direction vector must be non-zero")
    return vec / norm


@dataclass(frozen=True)
class FerroProblem:
    temperature: float
    ngrid: int
    cutoff: float
    seed_direction: np.ndarray
    c: SchaCoefficients
    tensors: SchaTensors

    def unpack(self, variables: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = np.array(
            [
                [math.exp(float(variables[0])), 0.0, 0.0],
                [float(variables[1]), math.exp(float(variables[2])), 0.0],
                [float(variables[3]), float(variables[4]), math.exp(float(variables[5]))],
            ],
            dtype=float,
        )
        mass_matrix = lower @ lower.T
        uvec = np.asarray(variables[6:9], dtype=float)
        return mass_matrix, uvec

    @staticmethod
    def pack_initial_mass(mass_matrix: np.ndarray, uvec: np.ndarray) -> np.ndarray:
        lower = np.linalg.cholesky(mass_matrix)
        return np.array(
            [
                math.log(lower[0, 0]),
                lower[1, 0],
                math.log(lower[1, 1]),
                lower[2, 0],
                lower[2, 1],
                math.log(lower[2, 2]),
                float(uvec[0]),
                float(uvec[1]),
                float(uvec[2]),
            ],
            dtype=float,
        )

    @staticmethod
    def symmetric_components(matrix: np.ndarray) -> np.ndarray:
        sqrt2 = math.sqrt(2.0)
        return np.array(
            [
                matrix[0, 0],
                matrix[1, 1],
                matrix[2, 2],
                sqrt2 * matrix[0, 1],
                sqrt2 * matrix[0, 2],
                sqrt2 * matrix[1, 2],
            ],
            dtype=float,
        )

    def residual_physical(
        self,
        mass_matrix: np.ndarray,
        uvec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, dict[str, float]]]:
        u_abs = float(np.linalg.norm(uvec))
        if u_abs <= 1.0e-10:
            raise ValueError("the nonzero ferroelectric branch requires |u_min| > 0")
        direction = uvec / u_abs
        covariance = integrate_covariance(
            mass_matrix,
            temperature=self.temperature,
            ngrid=self.ngrid,
            cutoff=self.cutoff,
            c=self.c,
        )
        sigma_inhomogeneous = inhomogeneous_self_energy_tensor(
            mass_matrix,
            temperature=self.temperature,
            ngrid=self.ngrid,
            cutoff=self.cutoff,
            c=self.c,
        )
        parts = mass_self_energy_tensors(
            covariance,
            uvec,
            tensors=self.tensors,
            sigma_inhomogeneous=sigma_inhomogeneous,
        )
        split_parts = project_split_parts(parts, direction)
        target_matrix = self.c.A01 * np.eye(3) + sum(parts.values())
        mass_residual = self.symmetric_components(mass_matrix - target_matrix)
        grad = umin_gradient(
            covariance,
            uvec,
            c=self.c,
            tensors=self.tensors,
            sigma_inhomogeneous=sigma_inhomogeneous,
        )
        u_residual = grad / u_abs
        return np.concatenate((mass_residual, u_residual)).astype(float), covariance, parts, split_parts

    def residual_solver(self, variables: np.ndarray) -> np.ndarray:
        try:
            mass_matrix, uvec = self.unpack(variables)
            residual, _, _, _ = self.residual_physical(mass_matrix, uvec)
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            return np.full(9, 1.0e6, dtype=float)
        return residual


def solve_with_scipy(
    problem: FerroProblem,
    guesses: list[tuple[float, float, float]],
    tolerance: float,
    max_nfev: int,
    initial_state: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    from scipy.optimize import least_squares

    best = None
    initial_conditions: list[tuple[np.ndarray, np.ndarray]] = []
    if initial_state is not None:
        initial_conditions.append(
            (np.asarray(initial_state[0], dtype=float), np.asarray(initial_state[1], dtype=float))
        )
    for a_t_guess, a_l_guess, u_guess in guesses:
        if a_t_guess <= 0.0 or a_l_guess <= 0.0 or u_guess <= 0.0:
            continue
        initial_conditions.append(
            (
                local_mass_matrix(a_t_guess, a_l_guess, problem.seed_direction),
                u_guess * problem.seed_direction,
            )
        )

    for initial_mass, initial_uvec in initial_conditions:
        x0 = problem.pack_initial_mass(initial_mass, initial_uvec)
        result = least_squares(
            problem.residual_solver,
            x0,
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
            max_nfev=max_nfev,
        )
        mass_matrix, uvec = problem.unpack(result.x)
        u_abs = float(np.linalg.norm(uvec))
        try:
            residual, covariance, parts, split_parts = problem.residual_physical(mass_matrix, uvec)
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            continue
        norm = float(np.linalg.norm(residual))
        candidate = (norm, result, mass_matrix, u_abs, uvec, residual, covariance, parts, split_parts)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if norm <= tolerance and u_abs > 1.0e-10:
            break

    if best is None:
        raise RuntimeError("no valid positive initial guess was provided for A_T, A_L, and |u_min|")

    norm, result, mass_matrix, u_abs, uvec, residual, covariance, parts, split_parts = best
    direction = uvec / u_abs
    p_l = longitudinal_projector(direction)
    p_t = transverse_projector(direction)
    a_l = float(np.einsum("ab,ab->", p_l, mass_matrix))
    a_t = float(np.einsum("ab,ab->", p_t, mass_matrix) / 2.0)
    if norm > tolerance:
        u_norm = float(np.linalg.norm(residual[6:]))
        raise RuntimeError(
            "ferroelectric solve did not converge; "
            f"best residual norm={norm:.6e}, A_T={a_t:.12g}, A_L={a_l:.12g}, |u_min|={u_abs:.12g}, "
            f"||mass residual||={np.linalg.norm(residual[:6]):.6e}, "
            f"||u residual / |u_min|||={u_norm:.6e}; "
            f"last scipy status={result.status}: {result.message}"
        )

    sigma_inhomogeneous = parts["quartic_inhomogeneous"]
    grad = umin_gradient(
        covariance,
        uvec,
        c=problem.c,
        tensors=problem.tensors,
        sigma_inhomogeneous=sigma_inhomogeneous,
    )
    u_residual_vector = residual[6:]
    min_mu = min_static_eigenvalue(mass_matrix, problem.ngrid, problem.cutoff, problem.c)
    transverse_basis = np.linalg.eigh(p_t)[1][:, 1:]
    transverse_masses = np.linalg.eigvalsh(transverse_basis.T @ mass_matrix @ transverse_basis)
    if problem.temperature > 0.0:
        physical_scale = math.sqrt(KB_HARTREE_PER_K * problem.temperature)
        pvec = uvec / physical_scale
        p_abs = u_abs / physical_scale
        gmat = covariance / (KB_HARTREE_PER_K * problem.temperature)
    else:
        pvec = None
        p_abs = None
        gmat = None
    quartic_names = (
        "quartic_local",
        "quartic_homogeneous",
        "quartic_inhomogeneous",
    )
    quartic_t = sum(split_parts[name]["T"] for name in quartic_names)
    quartic_l = sum(split_parts[name]["L"] for name in quartic_names)
    return {
        "material": problem.c.name,
        "temperature_K": problem.temperature,
        "A_T": float(a_t),
        "A_T1": float(transverse_masses[0]),
        "A_T2": float(transverse_masses[1]),
        "A_L": float(a_l),
        "A_local": mass_matrix.tolist(),
        "A01": problem.c.A01,
        "p_min_abs": None if p_abs is None else float(p_abs),
        "p_min": None if pvec is None else pvec.tolist(),
        "physical_displacement_abs_bohr": float(u_abs),
        "physical_displacement_bohr": uvec.tolist(),
        "direction": direction.tolist(),
        "initial_direction": problem.seed_direction.tolist(),
        "delta_A_T_quartic_local": split_parts["quartic_local"]["T"],
        "delta_A_T_quartic_homogeneous": split_parts["quartic_homogeneous"]["T"],
        "delta_A_T_quartic_inhomogeneous": split_parts["quartic_inhomogeneous"]["T"],
        "delta_A_T_quartic": quartic_t,
        "delta_A_T_sextic": split_parts["sextic"]["T"],
        "delta_A_T_octic": split_parts["octic"]["T"],
        "delta_A_T_total": sum(order["T"] for order in split_parts.values()),
        "delta_A_L_quartic_local": split_parts["quartic_local"]["L"],
        "delta_A_L_quartic_homogeneous": split_parts["quartic_homogeneous"]["L"],
        "delta_A_L_quartic_inhomogeneous": split_parts["quartic_inhomogeneous"]["L"],
        "delta_A_L_quartic": quartic_l,
        "delta_A_L_sextic": split_parts["sextic"]["L"],
        "delta_A_L_octic": split_parts["octic"]["L"],
        "delta_A_L_total": sum(order["L"] for order in split_parts.values()),
        "self_energy_parts": {key: value.tolist() for key, value in parts.items()},
        "covariance_Q_bohr2": covariance.tolist(),
        "covariance_Q_trace_over_3_bohr2": float(np.trace(covariance) / 3.0),
        "G": None if gmat is None else gmat.tolist(),
        "G_trace_over_3": None if gmat is None else float(np.trace(gmat) / 3.0),
        "mass_tensor_residual": float(np.linalg.norm(residual[:6])),
        "umin_residual_over_abs": float(np.linalg.norm(u_residual_vector)),
        "umin_residual_over_abs_vector": u_residual_vector.tolist(),
        "umin_gradient": grad.tolist(),
        "pmin_residual_over_abs": float(np.linalg.norm(u_residual_vector)),
        "pmin_residual_over_abs_vector": u_residual_vector.tolist(),
        "pmin_gradient": grad.tolist(),
        "residual_norm": norm,
        "ngrid": problem.ngrid,
        "cutoff_bohr_inv": problem.cutoff,
        "min_static_eigenvalue": float(min_mu),
        "scipy_nfev": int(result.nfev),
        "scipy_status": int(result.status),
        "coefficients": asdict(problem.c),
        "strain_coefficients": {
            "bare_b1": BARE_B1,
            "bare_b2": BARE_B2,
            "homogeneous_lambda_b1": HOMOGENEOUS_LAMBDA_B1,
            "homogeneous_lambda_b2": HOMOGENEOUS_LAMBDA_B2,
            "acoustic_mass_amu": ACOUSTIC_MASS_AMU,
        },
    }


def default_guesses(
    c: SchaCoefficients,
    user_a_t: float | None,
    user_a_l: float | None,
    user_u: float | None,
) -> list[tuple[float, float, float]]:
    if user_a_t is not None and user_a_l is not None and user_u is not None:
        return [(user_a_t, user_a_l, user_u)]
    mass_pairs = [
        (0.004, 0.03),
        (0.01, 0.04),
        (0.1, 0.1),
        (0.3, 0.3),
        (0.3, 0.6),
        (0.6, 0.3),
        (1.0, 1.0),
    ]
    if user_a_t is not None or user_a_l is not None:
        a_t_values = [user_a_t] if user_a_t is not None else [0.1, 0.3, 0.6, 1.0]
        a_l_values = [user_a_l] if user_a_l is not None else [0.1, 0.3, 0.6, 1.0]
        mass_pairs = [(a_t, a_l) for a_t in a_t_values for a_l in a_l_values]
    u_values = [0.25, 0.18, 0.32, 0.10, 0.45, 0.03, 0.70]
    if user_u is not None:
        u_values = [user_u]
    return [(a_t, a_l, u) for a_t, a_l in mass_pairs for u in u_values]


def solve_ferro(
    temperature: float,
    ngrid: int,
    cutoff: float,
    direction: np.ndarray,
    c: SchaCoefficients,
    a_t_guess: float | None,
    a_l_guess: float | None,
    p_guess: float | None,
    tolerance: float,
    max_nfev: int,
    u_guess: float | None = None,
    initial_state: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")
    if ngrid % 2 != 0:
        raise ValueError("ngrid must be even so no finite quadrature weight is placed at the non-analytic Gamma point")
    tensors = build_tensors(c)
    problem = FerroProblem(
        temperature=temperature,
        ngrid=ngrid,
        cutoff=cutoff,
        seed_direction=direction,
        c=c,
        tensors=tensors,
    )
    if u_guess is not None and p_guess is not None:
        raise ValueError("provide only one of u_guess and the legacy scaled p_guess")
    if p_guess is not None:
        if temperature == 0.0:
            raise ValueError("the scaled p_guess is undefined at T=0; use u_guess")
        u_guess = math.sqrt(KB_HARTREE_PER_K * temperature) * p_guess
    guesses = default_guesses(c=c, user_a_t=a_t_guess, user_a_l=a_l_guess, user_u=u_guess)
    return solve_with_scipy(
        problem,
        guesses,
        tolerance,
        max_nfev=max_nfev,
        initial_state=initial_state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve ferroelectric SCHA Eqs. (159)--(160).")
    parser.add_argument(
        "--material",
        choices=sorted(MATERIALS),
        default="nishimatsu",
        help="Coefficient set. Default: nishimatsu.",
    )
    parser.add_argument("--temperature", "-T", type=float, default=100.0, help="Temperature in kelvin.")
    parser.add_argument(
        "--direction",
        type=parse_direction,
        default=parse_direction("generic"),
        help="Initial p_min direction; the three Cartesian components are then optimized.",
    )
    parser.add_argument("--ngrid", type=int, default=12, help="Even Gauss-Legendre order per direction. Default: 12.")
    parser.add_argument("--cutoff", type=float, default=None, help="Cubic BZ half-width in bohr^-1. Default: pi/a0.")
    parser.add_argument("--a-t-guess", type=float, default=None, help="Optional initial guess for A_T.")
    parser.add_argument("--a-l-guess", type=float, default=None, help="Optional initial guess for A_L.")
    parser.add_argument("--u-guess", type=float, default=None, help="Optional physical initial guess for |u_min| in bohr.")
    parser.add_argument("--p-guess", type=float, default=None, help="Legacy finite-T initial guess for scaled |p_min|.")
    parser.add_argument("--tolerance", type=float, default=1.0e-10, help="Solver tolerance on the mass and p_min residuals.")
    parser.add_argument("--max-nfev", type=int, default=300, help="Maximum residual evaluations per initial guess.")
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coeffs = MATERIALS[args.material]
    cutoff = coeffs.kmax if args.cutoff is None else args.cutoff

    try:
        result = solve_ferro(
            temperature=args.temperature,
            ngrid=args.ngrid,
            cutoff=cutoff,
            direction=args.direction,
            c=coeffs,
            a_t_guess=args.a_t_guess,
            a_l_guess=args.a_l_guess,
            p_guess=args.p_guess,
            tolerance=args.tolerance,
            max_nfev=args.max_nfev,
            u_guess=args.u_guess,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"material = {result['material']}")
    print(f"T = {result['temperature_K']:.6g} K")
    print(f"initial direction = {np.array(result['initial_direction'])}")
    print(f"optimized direction = {np.array(result['direction'])}")
    print(f"A_T average(T) = {result['A_T']:.12g} Hartree/bohr^2")
    print(f"A_T1(T) = {result['A_T1']:.12g} Hartree/bohr^2")
    print(f"A_T2(T) = {result['A_T2']:.12g} Hartree/bohr^2")
    print(f"A_L(T) = {result['A_L']:.12g} Hartree/bohr^2")
    print(f"A_local(T) =\n{np.array(result['A_local'])}")
    print(f"A01 = {result['A01']:.12g} Hartree/bohr^2")
    if result["p_min_abs"] is not None:
        print(f"|p_min| = {result['p_min_abs']:.12g} bohr/sqrt(Hartree)")
        print(f"p_min = {np.array(result['p_min'])}")
    else:
        print("p_min is undefined at T=0; the solver uses physical u_min")
    print(f"physical |u_min| = {result['physical_displacement_abs_bohr']:.12g} bohr")
    print(f"physical u_min = {np.array(result['physical_displacement_bohr'])} bohr")
    print(f"local quartic delta A_T = {result['delta_A_T_quartic_local']:.12g} Hartree/bohr^2")
    print(f"homogeneous quartic delta A_T = {result['delta_A_T_quartic_homogeneous']:.12g} Hartree/bohr^2")
    print(f"inhomogeneous quartic delta A_T = {result['delta_A_T_quartic_inhomogeneous']:.12g} Hartree/bohr^2")
    print(f"total quartic delta A_T = {result['delta_A_T_quartic']:.12g} Hartree/bohr^2")
    print(f"sextic delta A_T = {result['delta_A_T_sextic']:.12g} Hartree/bohr^2")
    print(f"octic delta A_T = {result['delta_A_T_octic']:.12g} Hartree/bohr^2")
    print(f"total delta A_T = {result['delta_A_T_total']:.12g} Hartree/bohr^2")
    print(f"local quartic delta A_L = {result['delta_A_L_quartic_local']:.12g} Hartree/bohr^2")
    print(f"homogeneous quartic delta A_L = {result['delta_A_L_quartic_homogeneous']:.12g} Hartree/bohr^2")
    print(f"inhomogeneous quartic delta A_L = {result['delta_A_L_quartic_inhomogeneous']:.12g} Hartree/bohr^2")
    print(f"total quartic delta A_L = {result['delta_A_L_quartic']:.12g} Hartree/bohr^2")
    print(f"sextic delta A_L = {result['delta_A_L_sextic']:.12g} Hartree/bohr^2")
    print(f"octic delta A_L = {result['delta_A_L_octic']:.12g} Hartree/bohr^2")
    print(f"total delta A_L = {result['delta_A_L_total']:.12g} Hartree/bohr^2")
    print(
        "Tr(Q)/3 = "
        f"{result['covariance_Q_trace_over_3_bohr2']:.12g} bohr^2"
    )
    if result["G_trace_over_3"] is not None:
        print(f"Tr(G)/3 = {result['G_trace_over_3']:.12g} bohr^2/Hartree")
    print(f"M* = {coeffs.mass_amu:.6g} amu = {coeffs.mass_au:.12g} electron masses")
    print(f"mass tensor residual = {result['mass_tensor_residual']:.3e}")
    print(f"u_min residual / |u_min| = {result['umin_residual_over_abs']:.3e}")
    print(f"residual norm = {result['residual_norm']:.3e}")
    print(f"ngrid = {result['ngrid']}, cutoff = {result['cutoff_bohr_inv']:.12g} bohr^-1")
    print(
        "min static eigenvalue on grid = "
        f"{result['min_static_eigenvalue']:.12g} Hartree/bohr^2"
    )

    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
