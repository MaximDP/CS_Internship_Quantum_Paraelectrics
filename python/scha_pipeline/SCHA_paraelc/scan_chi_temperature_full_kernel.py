#!/usr/bin/env python3
"""Test the full matrix SCHA equation A_ab(k, omega_n).

The local, homogeneous, sixth- and eighth-order self-energies are evaluated
from the local loop, while the inhomogeneous acoustic contribution is retained
as the matrix convolution

    Sigma_inh_ab(K) = -kBT < M_ag,bd(K-Q) G_gd(Q) >_Q.

The spatial convolution is evaluated by FFT on a uniform periodic BZ grid.
Explicit Matsubara frequencies are supplemented by the analytic harmonic tail
in the local loop.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scan_chi_temperature as reference_scan
import scan_chi_temperature_inhomogeneous as projected_scan
import scha_paraelc as scha
from nishimatsu_harmonic import (
    _ewald_terms,
    _short_range_neighbours,
    raw_harmonic_kernel,
)


@dataclass(frozen=True)
class FullKernelGrid:
    nspace: int
    wavevectors: np.ndarray
    offsets: np.ndarray
    acoustic_kernel: np.ndarray
    mixed_tensor: np.ndarray
    gamma_index: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nspace, self.nspace, self.nspace)

    @property
    def nk(self) -> int:
        return self.nspace**3


@dataclass(frozen=True)
class FullKernelSolution:
    temperature_K: float
    pressure_GPa: float
    pressure_shift_u: float
    kernel: np.ndarray
    local_loop: np.ndarray
    sigma_local_u: float
    sigma_inhomogeneous: np.ndarray
    iterations: int
    residual: float
    minimum_static_eigenvalue: float

    @property
    def gamma_mass_u(self) -> float:
        center = self.kernel[self.kernel.shape[0] // 2, 0, 0, 0]
        return float(np.trace(center) / 3.0)


def build_uniform_grid(
    nspace: int, coefficients: scha.SchaCoefficients
) -> FullKernelGrid:
    if nspace < 4 or nspace % 2 != 0:
        raise ValueError("nspace must be an even integer greater than or equal to four")

    components = 2.0 * np.pi * np.fft.fftfreq(nspace, d=coefficients.a0)
    kx, ky, kz = np.meshgrid(components, components, components, indexing="ij")
    wavevectors = np.stack((kx.ravel(), ky.ravel(), kz.ravel()), axis=1)

    prepared = (
        _short_range_neighbours(coefficients),
        _ewald_terms(coefficients),
    )
    gamma_raw = raw_harmonic_kernel(
        np.zeros(3), coefficients, prepared=prepared
    )
    gamma_scalar = float(np.trace(gamma_raw) / 3.0)
    offsets = np.stack(
        [
            raw_harmonic_kernel(kvec, coefficients, prepared=prepared)
            - gamma_scalar * np.eye(3)
            for kvec in wavevectors
        ]
    ).reshape(nspace, nspace, nspace, 3, 3)

    acoustic_kernel = projected_scan._acoustic_kernel(wavevectors).reshape(
        nspace, nspace, nspace, 3, 3
    )
    mixed_tensor = projected_scan._mixed_tensor(wavevectors).reshape(
        nspace, nspace, nspace, 3, 3, 3
    )
    projected_scan._validate_nishimatsu_mixed_tensor(
        wavevectors, mixed_tensor.reshape(-1, 3, 3, 3)
    )
    return FullKernelGrid(
        nspace=nspace,
        wavevectors=wavevectors.reshape(nspace, nspace, nspace, 3),
        offsets=offsets,
        acoustic_kernel=acoustic_kernel,
        mixed_tensor=mixed_tensor,
        gamma_index=(0, 0, 0),
    )


def acoustic_vertex_fft(
    grid: FullKernelGrid,
    temperature: float,
    nmatsubara: int,
    acoustic_mass_amu: float,
) -> np.ndarray:
    """Return FFT[M_ag,bd(q,nu)] for |nu| <= 2*nmatsubara."""

    kbt = scha.KB_HARTREE_PER_K * temperature
    mass = acoustic_mass_amu * scha.AMU_TO_ELECTRON_MASS
    identity = np.eye(3)
    transfers = np.arange(-2 * nmatsubara, 2 * nmatsubara + 1)
    result = np.empty(
        (transfers.size,) + grid.shape + (3, 3, 3, 3), dtype=complex
    )
    flat_h = grid.mixed_tensor.reshape(-1, 3, 3, 3)
    flat_phi = grid.acoustic_kernel.reshape(-1, 3, 3)

    for position, transfer in enumerate(transfers):
        omega = 2.0 * np.pi * float(transfer) * kbt
        inverse_kernel = flat_phi + mass * omega**2 * identity
        inverse_kernel = inverse_kernel.copy()
        inverse_kernel[0] = identity
        propagator = np.linalg.inv(inverse_kernel)
        propagator[0] = 0.0
        interaction = np.einsum(
            "nrag,nrs,nsbd->nagbd",
            flat_h,
            propagator,
            flat_h,
            optimize=True,
        ).reshape(grid.shape + (3, 3, 3, 3))
        result[position] = np.fft.fftn(interaction, axes=(0, 1, 2))
    return result


def inhomogeneous_self_energy(
    propagator: np.ndarray,
    vertex_fft: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Evaluate the nonlocal matrix self-energy by a spatial FFT convolution."""

    nfrequency = propagator.shape[0]
    nmatsubara = (nfrequency - 1) // 2
    grid_shape = propagator.shape[1:4]
    nk = int(np.prod(grid_shape))
    propagator_fft = np.fft.fftn(propagator, axes=(1, 2, 3))
    sigma_fft = np.zeros_like(propagator_fft, dtype=complex)
    for external_position in range(nfrequency):
        external_n = external_position - nmatsubara
        for internal_position in range(nfrequency):
            internal_n = internal_position - nmatsubara
            transfer_position = external_n - internal_n + 2 * nmatsubara
            sigma_fft[external_position] += np.einsum(
                "xyzagbd,xyzgd->xyzab",
                vertex_fft[transfer_position],
                propagator_fft[internal_position],
                optimize=True,
            )
    sigma = np.fft.ifftn(sigma_fft, axes=(1, 2, 3)).real
    sigma *= -scha.KB_HARTREE_PER_K * temperature / nk
    return 0.5 * (sigma + np.swapaxes(sigma, -1, -2))


def analytic_local_tail(
    grid: FullKernelGrid,
    temperature: float,
    nmatsubara: int,
    static_local_mass: float,
    coefficients: scha.SchaCoefficients,
) -> np.ndarray:
    """Harmonic |n|>nmatsubara tail used only in the local loop."""

    static_kernel = grid.offsets + static_local_mass * np.eye(3)
    eigenvalues, eigenvectors = np.linalg.eigh(static_kernel)
    if np.min(eigenvalues) <= 0.0:
        raise RuntimeError("non-positive static kernel in the Matsubara tail")
    kbt = scha.KB_HARTREE_PER_K * temperature
    mode_omega = np.sqrt(eigenvalues / coefficients.mass_au)
    exact = 1.0 / np.tanh(mode_omega / (2.0 * kbt))
    exact /= 2.0 * coefficients.mass_au * kbt * mode_omega
    finite = np.zeros_like(eigenvalues)
    for nvalue in range(-nmatsubara, nmatsubara + 1):
        omega_n = 2.0 * np.pi * float(nvalue) * kbt
        finite += 1.0 / (eigenvalues + coefficients.mass_au * omega_n**2)
    tail = np.maximum(exact - finite, 0.0)
    return np.einsum(
        "...ai,...i,...bi->...ab", eigenvectors, tail, eigenvectors, optimize=True
    )


def local_self_energy(
    local_loop: np.ndarray,
    temperature: float,
    coefficients: scha.SchaCoefficients,
) -> tuple[float, dict[str, float]]:
    terms = {
        "quartic_local": projected_scan.local_quartic_self_energy_mass(
            local_loop, temperature
        ),
        "quartic_homogeneous": projected_scan.homogeneous_quartic_self_energy_mass(
            local_loop, temperature
        ),
        "sextic": scha.sextic_self_energy_mass(
            local_loop, temperature, coefficients
        ),
        "octic": scha.octic_self_energy_mass(
            local_loop, temperature, coefficients
        ),
    }
    return float(sum(terms.values())), terms


def _initial_kernel(
    grid: FullKernelGrid,
    temperature: float,
    nmatsubara: int,
    gamma_mass: float,
    coefficients: scha.SchaCoefficients,
) -> np.ndarray:
    kbt = scha.KB_HARTREE_PER_K * temperature
    frequencies = np.arange(-nmatsubara, nmatsubara + 1)
    dynamic = coefficients.mass_au * (2.0 * np.pi * frequencies * kbt) ** 2
    return (
        grid.offsets[None, ...]
        + gamma_mass * np.eye(3)
        + dynamic[:, None, None, None, None, None] * np.eye(3)
    )


def solve_full_kernel(
    grid: FullKernelGrid,
    vertex_fft: np.ndarray,
    temperature: float,
    nmatsubara: int,
    coefficients: scha.SchaCoefficients,
    *,
    pressure_gpa: float | None = None,
    target_gamma_mass: float | None = None,
    initial_gamma_mass: float,
    mixing: float,
    tolerance: float,
    max_iterations: int,
) -> FullKernelSolution:
    if (pressure_gpa is None) == (target_gamma_mass is None):
        raise ValueError("specify exactly one of pressure_gpa or target_gamma_mass")

    kernel = _initial_kernel(
        grid, temperature, nmatsubara, initial_gamma_mass, coefficients
    )
    kbt = scha.KB_HARTREE_PER_K * temperature
    frequencies = np.arange(-nmatsubara, nmatsubara + 1)
    dynamic = coefficients.mass_au * (2.0 * np.pi * frequencies * kbt) ** 2
    pressure_shift = (
        reference_scan.pressure_mass_shift_u(float(pressure_gpa), coefficients)
        if pressure_gpa is not None
        else 0.0
    )
    residual = np.inf

    for iteration in range(1, max_iterations + 1):
        propagator = np.linalg.inv(kernel)
        static_local_mass = float(
            np.trace(kernel[nmatsubara][grid.gamma_index]) / 3.0
        )
        tail = analytic_local_tail(
            grid,
            temperature,
            nmatsubara,
            static_local_mass=static_local_mass,
            coefficients=coefficients,
        )
        local_loop_raw = np.mean(
            np.sum(propagator, axis=0) + tail, axis=(0, 1, 2)
        )
        local_loop = float(np.trace(local_loop_raw) / 3.0) * np.eye(3)
        sigma_local, _ = local_self_energy(local_loop, temperature, coefficients)
        sigma_inhomogeneous = inhomogeneous_self_energy(
            propagator, vertex_fft, temperature
        )

        if target_gamma_mass is not None:
            center_sigma = sigma_inhomogeneous[nmatsubara][grid.gamma_index]
            pressure_shift = float(
                target_gamma_mass
                - coefficients.A01
                - sigma_local
                - np.trace(center_sigma) / 3.0
            )

        updated = (
            grid.offsets[None, ...]
            + (coefficients.A01 + pressure_shift + sigma_local) * np.eye(3)
            + dynamic[:, None, None, None, None, None] * np.eye(3)
            + sigma_inhomogeneous
        )
        updated = 0.5 * (updated + np.swapaxes(updated, -1, -2))
        minimum_static = float(
            np.min(np.linalg.eigvalsh(updated[nmatsubara]))
        )
        if minimum_static <= 1.0e-12:
            raise RuntimeError(
                f"non-positive updated static kernel at T={temperature:g} K"
            )
        residual = float(np.max(np.abs(updated - kernel)))
        if residual < tolerance:
            kernel = updated
            break
        kernel = (1.0 - mixing) * kernel + mixing * updated
    else:
        raise RuntimeError(
            f"full-kernel SCHA did not converge at T={temperature:g} K; "
            f"residual={residual:.3e}"
        )

    propagator = np.linalg.inv(kernel)
    tail_mass = float(np.trace(kernel[nmatsubara][grid.gamma_index]) / 3.0)
    tail = analytic_local_tail(
        grid,
        temperature,
        nmatsubara,
        static_local_mass=tail_mass,
        coefficients=coefficients,
    )
    local_loop_raw = np.mean(
        np.sum(propagator, axis=0) + tail, axis=(0, 1, 2)
    )
    local_loop = float(np.trace(local_loop_raw) / 3.0) * np.eye(3)
    sigma_local, _ = local_self_energy(local_loop, temperature, coefficients)
    sigma_inhomogeneous = inhomogeneous_self_energy(
        propagator, vertex_fft, temperature
    )
    pressure_value = (
        float(pressure_gpa)
        if pressure_gpa is not None
        else pressure_shift / reference_scan.pressure_mass_shift_u(1.0, coefficients)
    )
    return FullKernelSolution(
        temperature_K=temperature,
        pressure_GPa=pressure_value,
        pressure_shift_u=pressure_shift,
        kernel=kernel,
        local_loop=local_loop,
        sigma_local_u=sigma_local,
        sigma_inhomogeneous=sigma_inhomogeneous,
        iterations=iteration,
        residual=residual,
        minimum_static_eigenvalue=float(
            np.min(np.linalg.eigvalsh(kernel[nmatsubara]))
        ),
    )


def scalar_initial_mass(
    temperature: float,
    pressure_gpa: float,
    coefficients: scha.SchaCoefficients,
    projected_grid: projected_scan.InhomogeneousQuarticGrid,
) -> float:
    pressure_shift = reference_scan.pressure_mass_shift_u(
        pressure_gpa, coefficients
    )
    target_chi = reference_scan.experimental_curie_weiss(temperature)[1]
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    barrett_mass = float(coefficients.omega0 / (scale2 * target_chi))
    try:
        scalar_mass = projected_scan.solve_root(
            temperature, pressure_shift, coefficients, projected_grid
        )
        return max(scalar_mass, barrett_mass)
    except RuntimeError:
        return barrett_mass


def solution_row(
    solution: FullKernelSolution,
    coefficients: scha.SchaCoefficients,
) -> dict[str, float]:
    scale2 = (coefficients.omega0 / coefficients.zstar) ** 2
    center_sigma = solution.sigma_inhomogeneous[
        solution.kernel.shape[0] // 2, 0, 0, 0
    ]
    return {
        "temperature_K": solution.temperature_K,
        "pressure_GPa": solution.pressure_GPa,
        "pressure_mass_shift_u": solution.pressure_shift_u,
        "A_Gamma_0_u": solution.gamma_mass_u,
        "chi_T_SCHA": coefficients.omega0 / (scale2 * solution.gamma_mass_u),
        "trace_G_local_over_3": float(np.trace(solution.local_loop) / 3.0),
        "sigma_local_u": solution.sigma_local_u,
        "sigma_inhomogeneous_Gamma_0_u": float(np.trace(center_sigma) / 3.0),
        "iterations": float(solution.iterations),
        "residual": solution.residual,
        "minimum_static_eigenvalue": solution.minimum_static_eigenvalue,
    }


def write_csv(
    constant_rows: list[dict[str, float]],
    thermal_rows: list[dict[str, float]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["temperature_K"]
    for prefix, rows in (("constant", constant_rows), ("thermal", thermal_rows)):
        fields.extend(
            f"{prefix}_{key}" for key in rows[0] if key != "temperature_K"
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for constant, thermal in zip(constant_rows, thermal_rows):
            row: dict[str, float] = {"temperature_K": constant["temperature_K"]}
            row.update(
                {
                    f"constant_{key}": value
                    for key, value in constant.items()
                    if key != "temperature_K"
                }
            )
            row.update(
                {
                    f"thermal_{key}": value
                    for key, value in thermal.items()
                    if key != "temperature_K"
                }
            )
            writer.writerow(row)


def write_plot(
    constant_rows: list[dict[str, float]],
    thermal_rows: list[dict[str, float]],
    fitted_pressure_gpa: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temperatures = np.array([row["temperature_K"] for row in constant_rows])
    constant_chi = np.array([row["chi_T_SCHA"] for row in constant_rows])
    thermal_chi = np.array([row["chi_T_SCHA"] for row in thermal_rows])
    reference_temperature = np.linspace(420.0, 560.0, 281)
    barrett = np.array(
        [reference_scan.experimental_curie_weiss(t)[1] for t in reference_temperature]
    )
    wieczorek_temperature, wieczorek_chi = reference_scan.load_wieczorek_response()

    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    axis.plot(
        temperatures,
        constant_chi,
        "o-",
        linewidth=1.8,
        markersize=5.0,
        label=rf"full $A_{{ab}}(k,\omega_n)$, $p={fitted_pressure_gpa:.3f}$ GPa",
    )
    axis.plot(
        temperatures,
        thermal_chi,
        "^-",
        linewidth=1.7,
        markersize=5.2,
        label=r"full $A_{ab}(k,\omega_n)$, $p(T)=-0.005T$ GPa",
    )
    axis.plot(
        reference_temperature,
        barrett,
        "--",
        color="0.15",
        linewidth=1.6,
        label="experimental Curie--Weiss fit (Barrett, 1952)",
    )
    axis.plot(
        wieczorek_temperature,
        wieczorek_chi,
        "s--",
        markerfacecolor="white",
        markersize=5.0,
        label=r"Wieczorek et al. (2006), 1 kHz (digitized)",
    )
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel(r"Soft-mode susceptibility $\chi_T$")
    axis.set_xlim(412.0, 568.0)
    axis.set_ylim(
        0.0,
        1.10
        * max(
            float(np.max(barrett)),
            float(np.max(constant_chi)),
            float(np.max(wieczorek_chi)),
        ),
    )
    axis.grid(alpha=0.23)
    axis.legend(frameon=False, fontsize=8.3)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=420.0)
    parser.add_argument("--stop", type=float, default=560.0)
    parser.add_argument("--step", type=float, default=20.0)
    parser.add_argument("--nspace", type=int, default=12)
    parser.add_argument("--nmatsubara", type=int, default=4)
    parser.add_argument("--scalar-ngrid", type=int, default=12)
    parser.add_argument(
        "--constant-pressure-gpa", type=float, default=-5.104768172668956
    )
    parser.add_argument(
        "--acoustic-mass-amu",
        type=float,
        default=projected_scan.ACOUSTIC_MASS_AMU,
    )
    parser.add_argument("--mixing", type=float, default=0.15)
    parser.add_argument("--tolerance", type=float, default=2.0e-9)
    parser.add_argument("--max-iterations", type=int, default=320)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = reference_scan.temperature_grid(args.start, args.stop, args.step)
    coefficients = scha.MATERIALS["nishimatsu"]
    grid = build_uniform_grid(args.nspace, coefficients)
    projected_grid = projected_scan.build_inhomogeneous_quartic_grid(
        args.scalar_ngrid,
        coefficients,
        acoustic_mass_amu=args.acoustic_mass_amu,
    )
    constant_pressure = args.constant_pressure_gpa

    constant_rows: list[dict[str, float]] = []
    thermal_rows: list[dict[str, float]] = []
    for temperature_value in temperatures:
        temperature = float(temperature_value)
        vertex_fft = acoustic_vertex_fft(
            grid, temperature, args.nmatsubara, args.acoustic_mass_amu
        )
        constant_initial = scalar_initial_mass(
            temperature, constant_pressure, coefficients, projected_grid
        )
        constant_solution = solve_full_kernel(
            grid,
            vertex_fft,
            temperature,
            args.nmatsubara,
            coefficients,
            pressure_gpa=constant_pressure,
            initial_gamma_mass=constant_initial,
            mixing=args.mixing,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        constant_rows.append(solution_row(constant_solution, coefficients))

        thermal_pressure = reference_scan.pressure_gpa(
            temperature, "thermal-expansion", constant_pressure
        )
        thermal_initial = scalar_initial_mass(
            temperature, thermal_pressure, coefficients, projected_grid
        )
        thermal_solution = solve_full_kernel(
            grid,
            vertex_fft,
            temperature,
            args.nmatsubara,
            coefficients,
            pressure_gpa=thermal_pressure,
            initial_gamma_mass=thermal_initial,
            mixing=args.mixing,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        thermal_rows.append(solution_row(thermal_solution, coefficients))

    stem = (
        f"chi_vs_temperature_full_kernel_n{args.nspace}_m{args.nmatsubara}_test"
    )
    output_dir = (
        Path(__file__).resolve().parents[1] / "outputs" / "paraelectric"
    )
    csv_path = args.csv or output_dir / f"{stem}.csv"
    plot_path = args.plot or output_dir / f"{stem}.png"
    write_csv(constant_rows, thermal_rows, csv_path)
    write_plot(constant_rows, thermal_rows, constant_pressure, plot_path)
    print(f"constant_pressure_GPa={constant_pressure:.12g}")
    print(f"csv={csv_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
