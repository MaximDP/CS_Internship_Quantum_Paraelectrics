"""Full FERAM/ZVR harmonic local-mode kernel on the simple-cubic BZ."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import erfc


@dataclass(frozen=True)
class HarmonicGrid:
    offsets: np.ndarray
    weights: np.ndarray
    prefactor: float


def _short_range_neighbours(c) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    js = (c.j1, c.j2, c.j3, c.j4, c.j5, c.j6, c.j7)

    for ix in range(-1, 2):
        for iy in range(-1, 2):
            for iz in range(-1, 2):
                lattice_vector = np.array([ix, iy, iz], dtype=float)
                shell = ix * ix + iy * iy + iz * iz
                if shell == 0:
                    continue
                matrix = np.zeros((3, 3), dtype=float)
                for a in range(3):
                    for b in range(3):
                        if shell == 1 and a == b:
                            matrix[a, b] = js[0] if lattice_vector[a] == 0.0 else js[1]
                        elif shell == 2:
                            if a == b:
                                matrix[a, b] = js[3] if lattice_vector[a] == 0.0 else js[2]
                            else:
                                matrix[a, b] = js[4] * lattice_vector[a] * lattice_vector[b]
                        elif shell == 3:
                            if a == b:
                                matrix[a, b] = js[5]
                            else:
                                matrix[a, b] = js[6] * lattice_vector[a] * lattice_vector[b]
                vectors.append(c.a0 * lattice_vector)
                matrices.append(matrix)
    return np.asarray(vectors), np.asarray(matrices)


def _ewald_terms(
    c,
    shell: int = 4,
    eta: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Prepare real, reciprocal, and self terms for the dipolar Ewald sum.

    ``eta`` is exposed so convergence and splitting-parameter independence can
    be tested.  The production default remains ``2/a0``.
    """

    if shell < 1:
        raise ValueError("shell must be at least one")
    eta = 2.0 / c.a0 if eta is None else float(eta)
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be positive and finite")
    integers = np.array(
        [index for index in np.ndindex(*(2 * shell + 1,) * 3)],
        dtype=float,
    ) - shell

    real_vectors = c.a0 * integers
    real_vectors = real_vectors[np.any(integers != 0.0, axis=1)]
    radii = np.linalg.norm(real_vectors, axis=1)
    unit = real_vectors / radii[:, None]
    gaussian = np.exp(-(eta * radii) ** 2)
    bracket = erfc(eta * radii) + 2.0 * eta * radii * gaussian / math.sqrt(math.pi)
    real_tensors = (
        np.eye(3)[None, :, :] - 3.0 * unit[:, :, None] * unit[:, None, :]
    ) * (bracket / radii**3)[:, None, None]
    real_tensors -= (
        4.0
        * eta**3
        / math.sqrt(math.pi)
        * gaussian[:, None, None]
        * unit[:, :, None]
        * unit[:, None, :]
    )

    reciprocal_vectors = 2.0 * math.pi * integers / c.a0
    self_term = -4.0 * eta**3 * np.eye(3) / (3.0 * math.sqrt(math.pi))
    return real_vectors, real_tensors, reciprocal_vectors, eta, self_term


def raw_harmonic_kernel(kvec: np.ndarray, c, prepared=None) -> np.ndarray:
    if prepared is None:
        neighbours = _short_range_neighbours(c)
        ewald = _ewald_terms(c)
    else:
        neighbours, ewald = prepared

    neighbour_vectors, neighbour_matrices = neighbours
    real_vectors, real_tensors, reciprocal_vectors, eta, self_term = ewald

    short_range = 2.0 * c.kappa2 * np.eye(3)
    short_range += np.einsum(
        "r,rij->ij", np.cos(neighbour_vectors @ kvec), neighbour_matrices, optimize=True
    )

    dipolar = self_term.copy()
    dipolar += np.einsum("r,rij->ij", np.cos(real_vectors @ kvec), real_tensors, optimize=True)
    shifted = reciprocal_vectors + kvec
    shifted_norm2 = np.einsum("ri,ri->r", shifted, shifted)
    nonzero = shifted_norm2 > 1.0e-24
    shifted = shifted[nonzero]
    shifted_norm2 = shifted_norm2[nonzero]
    reciprocal_weights = np.exp(-shifted_norm2 / (4.0 * eta**2)) / shifted_norm2
    dipolar += (4.0 * math.pi / c.omega0) * np.einsum(
        "r,ri,rj->ij", reciprocal_weights, shifted, shifted, optimize=True
    )
    dipolar *= c.zstar**2 / c.eps_inf
    return short_range + dipolar


def build_harmonic_grid(c, ngrid: int, cutoff: float) -> HarmonicGrid:
    nodes, weights = np.polynomial.legendre.leggauss(ngrid)
    points = cutoff * nodes
    scaled_weights = cutoff * weights
    neighbours = _short_range_neighbours(c)
    ewald = _ewald_terms(c)
    prepared = (neighbours, ewald)

    gamma_raw = raw_harmonic_kernel(np.zeros(3), c, prepared=prepared)
    gamma_scalar = float(np.trace(gamma_raw) / 3.0)
    gamma_anisotropy = gamma_raw - gamma_scalar * np.eye(3)
    if np.linalg.norm(gamma_anisotropy) > 1.0e-10:
        raise RuntimeError("the Ewald Gamma kernel is not cubic within tolerance")

    offsets: list[np.ndarray] = []
    integration_weights: list[float] = []
    for ix, kx in enumerate(points):
        for iy, ky in enumerate(points):
            for iz, kz in enumerate(points):
                kvec = np.array([kx, ky, kz], dtype=float)
                raw = raw_harmonic_kernel(kvec, c, prepared=prepared)
                offsets.append(raw - gamma_scalar * np.eye(3))
                integration_weights.append(
                    scaled_weights[ix] * scaled_weights[iy] * scaled_weights[iz]
                )

    return HarmonicGrid(
        offsets=np.asarray(offsets),
        weights=np.asarray(integration_weights),
        prefactor=c.omega0 / (2.0 * math.pi) ** 3,
    )
