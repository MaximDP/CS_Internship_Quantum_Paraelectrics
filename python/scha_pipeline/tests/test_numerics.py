"""Independent mathematical and integration checks for the scientific pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
PARA_ROOT = PYTHON_ROOT / "SCHA_paraelc"
FERRO_ROOT = PYTHON_ROOT / "SCHA_ferro"
LATTICE_ROOT = PYTHON_ROOT / "lattice_sum"

for directory in (PYTHON_ROOT, PARA_ROOT, FERRO_ROOT, LATTICE_ROOT):
    sys.path.insert(0, str(directory))

import dielec_response
import lattice_sum
import materials
import nishimatsu_harmonic
import reciprocal_sum
import scan_chi_temperature_full_kernel as full_kernel
import scan_chi_temperature_inhomogeneous as acoustic
import scan_chi_temperature_bst_vca as bst_vca
import scha_paraelc as scha


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tensor_c = load_module("validation_tensor_c", PYTHON_ROOT / "tenseur_C" / "constants.py")
tensor_beta = load_module(
    "validation_tensor_beta", PYTHON_ROOT / "tenseur_beta" / "constants.py"
)
ferro_scha = load_module("validation_ferro_scha", FERRO_ROOT / "scha_ferro.py")


class RepositoryLayoutTests(unittest.TestCase):
    def test_required_entry_points_and_source_data_exist(self) -> None:
        required = (
            PYTHON_ROOT / "README.md",
            PYTHON_ROOT / "requirements.txt",
            PYTHON_ROOT / "validate_pipeline.py",
            PYTHON_ROOT / "nishimatsu_harmonic.py",
            PYTHON_ROOT / "dielec_response.py",
            PARA_ROOT / "scha_paraelc.py",
            PARA_ROOT / "scan_chi_temperature.py",
            PARA_ROOT / "scan_chi_temperature_inhomogeneous.py",
            PARA_ROOT / "scan_chi_temperature_full_kernel.py",
            PARA_ROOT / "scan_chi_temperature_srtio3.py",
            PARA_ROOT / "scan_chi_temperature_bst_vca.py",
            PYTHON_ROOT / "inputs" / "paraelectric" / "wieczorek2006_batio3_fig1a_digitized.csv",
            PYTHON_ROOT / "inputs" / "paraelectric" / "nakatani2016_batio3_cubic_lattice.csv",
            PYTHON_ROOT / "inputs" / "paraelectric" / "bst_curie_weiss_reference.csv",
            FERRO_ROOT / "scha_ferro.py",
            FERRO_ROOT / "scan_chi_temperature.py",
        )
        missing = [str(path.relative_to(PYTHON_ROOT)) for path in required if not path.is_file()]
        self.assertEqual(missing, [])

    def test_active_tree_has_no_archived_or_generated_artifacts(self) -> None:
        forbidden_directories = {"ocr_pages", "results"}
        forbidden_suffixes = {".html", ".json", ".pdf", ".png", ".pyc"}
        violations = []
        for path in PYTHON_ROOT.rglob("*"):
            relative = path.relative_to(PYTHON_ROOT)
            if "__pycache__" in relative.parts:
                continue
            if relative.parts and relative.parts[0] == "inputs":
                continue
            if relative.parts and relative.parts[0] == "outputs":
                continue
            if path.is_dir() and path.name in forbidden_directories:
                violations.append(str(relative))
            elif path.is_file() and (
                path.name == ".DS_Store" or path.suffix.lower() in forbidden_suffixes
            ):
                violations.append(str(relative))
            elif path.is_file() and path.suffix.lower() == ".csv" and "inputs" not in relative.parts:
                violations.append(str(relative))
        self.assertEqual(violations, [])


def direct_short_range_gradient(coefficients: scha.SchaCoefficients) -> np.ndarray:
    """Independent definition C_abgd=-1/2 sum_R J_ab(R) R_g R_d."""

    vectors, matrices = nishimatsu_harmonic._short_range_neighbours(coefficients)
    return -0.5 * np.einsum(
        "rab,rg,rd->abgd", matrices, vectors, vectors, optimize=True
    )


def geometric_ewald_coefficients(a0: float, eta: float) -> np.ndarray:
    sr = lattice_sum.expansion_coefficients(a0, eta, tol=1.0e-8)
    lr = reciprocal_sum.lrp_quadratic_invariants(a0, eta, tol=1.0e-8)
    g0 = reciprocal_sum.g0_quadratic_correction(a0, eta)
    names = ("k2_delta", "k_a k_b", "delta_k_a2")
    return np.array([sr[name] + lr[name] + g0[name] for name in names])


class TensorTests(unittest.TestCase):
    def test_short_range_tensor_matches_direct_lattice_derivative(self) -> None:
        for material_name, scha_name in (
            ("zvr_bto", "vanderbilt"),
            ("nishimatsu_bto", "nishimatsu"),
            ("nishimatsu_sto", "nishimatsu_sto"),
        ):
            with self.subTest(material=material_name):
                direct = direct_short_range_gradient(scha.MATERIALS[scha_name])
                expected = np.array(
                    (
                        direct[0, 0, 0, 0],
                        2.0 * direct[0, 1, 0, 1],
                        direct[0, 0, 1, 1],
                    )
                )
                actual = np.array(
                    tensor_c.short_range_cartesian_coefficients(material_name)
                )
                np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)

    def test_short_range_taylor_expansion(self) -> None:
        coefficients = scha.MATERIALS["nishimatsu"]
        vectors, matrices = nishimatsu_harmonic._short_range_neighbours(coefficients)
        gradient = direct_short_range_gradient(coefficients)
        direction = np.array([0.37, -0.51, 0.23])

        def kernel(kvec: np.ndarray) -> np.ndarray:
            return np.einsum(
                "r,rab->ab", np.cos(vectors @ kvec), matrices, optimize=True
            )

        step = 1.0e-4
        numerical = kernel(step * direction) - kernel(np.zeros(3))
        quadratic = step**2 * np.einsum(
            "abgd,g,d->ab", gradient, direction, direction, optimize=True
        )
        np.testing.assert_allclose(numerical, quadratic, rtol=2.0e-7, atol=2.0e-15)

    def test_beta_symmetry_and_independent_strain_elimination(self) -> None:
        material = tensor_beta.MATERIAL_SETS["nishimatsu_bto"]
        elastic = tensor_beta.elastic_matrix(material.elastic)
        coupling = tensor_beta.electrostrictive_tensor_vanderbilt(
            material.electrostrictive
        )
        induced = tensor_beta.strain_quartic_tensor(coupling, elastic)
        # B_l,ab C^-1_lm B_m,cd is pair-symmetric and major-symmetric.  Only
        # its fully symmetrized polynomial representative enters beta_eff.
        for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
            np.testing.assert_allclose(
                induced,
                np.transpose(induced, permutation),
                rtol=2.0e-14,
                atol=2.0e-14,
            )

        displacement = np.array([0.13, -0.08, 0.19])
        source = 0.5 * np.einsum("lab,a,b->l", coupling, displacement, displacement)
        strain = -np.linalg.solve(elastic, source)
        minimized = 0.5 * strain @ elastic @ strain + strain @ source
        contracted = -0.125 * np.einsum(
            "abcd,a,b,c,d", induced, displacement, displacement, displacement, displacement
        )
        self.assertAlmostEqual(minimized, contracted, delta=2.0e-18)

    def test_sixth_and_eighth_tensors_reproduce_polynomials(self) -> None:
        coefficients = scha.MATERIALS["nishimatsu"]
        displacement = np.array([0.11, -0.07, 0.16])
        energy6_tensor = 0.0
        for indices in itertools.product(range(3), repeat=6):
            energy6_tensor += scha.gamma_component(indices, coefficients) * math.prod(
                displacement[index] for index in indices
            ) / 6.0
        squared = displacement**2
        energy6_polynomial = (
            coefficients.k1_6 * np.sum(squared) ** 3
            + coefficients.k2_6
            * sum(squared[a] ** 2 * squared[b] for a in range(3) for b in range(3) if a != b)
            + coefficients.k3_6 * np.prod(squared)
        )
        self.assertAlmostEqual(energy6_tensor, energy6_polynomial, delta=2.0e-17)

        energy8_tensor = 0.0
        for indices in itertools.product(range(3), repeat=8):
            energy8_tensor += scha.rho_component(indices, coefficients) * math.prod(
                displacement[index] for index in indices
            ) / 8.0
        energy8_polynomial = coefficients.k4_8 * np.sum(squared) ** 4
        self.assertAlmostEqual(energy8_tensor, energy8_polynomial, delta=2.0e-17)


class EwaldAndHarmonicTests(unittest.TestCase):
    def test_ewald_splitting_parameter_independence(self) -> None:
        a0 = materials.MATERIALS["nishimatsu_bto"]["a0_bohr"]
        reference = geometric_ewald_coefficients(a0, 2.0 / a0)
        for factor in (1.2, 1.5, 2.5, 3.0):
            with self.subTest(eta_a0=factor):
                actual = geometric_ewald_coefficients(a0, factor / a0)
                np.testing.assert_allclose(actual, reference, rtol=5.0e-13, atol=8.0e-14)

    def test_full_ewald_shell_convergence_and_eta_independence(self) -> None:
        coefficients = scha.MATERIALS["nishimatsu"]
        neighbours = nishimatsu_harmonic._short_range_neighbours(coefficients)
        kvec = np.array([0.07, -0.11, 0.05])
        reference = nishimatsu_harmonic.raw_harmonic_kernel(
            kvec,
            coefficients,
            prepared=(
                neighbours,
                nishimatsu_harmonic._ewald_terms(
                    coefficients, shell=5, eta=2.0 / coefficients.a0
                ),
            ),
        )
        shell3 = nishimatsu_harmonic.raw_harmonic_kernel(
            kvec,
            coefficients,
            prepared=(
                neighbours,
                nishimatsu_harmonic._ewald_terms(
                    coefficients, shell=3, eta=2.0 / coefficients.a0
                ),
            ),
        )
        np.testing.assert_allclose(shell3, reference, rtol=3.0e-13, atol=3.0e-14)
        for factor in (1.2, 1.5, 2.5, 3.0):
            actual = nishimatsu_harmonic.raw_harmonic_kernel(
                kvec,
                coefficients,
                prepared=(
                    neighbours,
                    nishimatsu_harmonic._ewald_terms(
                        coefficients, shell=5, eta=factor / coefficients.a0
                    ),
                ),
            )
            np.testing.assert_allclose(actual, reference, rtol=3.0e-13, atol=3.0e-14)

    def test_gamma_contact_and_nonanalytic_coefficient(self) -> None:
        coefficients = scha.MATERIALS["nishimatsu"]
        gamma = nishimatsu_harmonic.raw_harmonic_kernel(
            np.zeros(3), coefficients
        )
        np.testing.assert_allclose(
            gamma, coefficients.A01 * np.eye(3), rtol=2.0e-14, atol=2.0e-14
        )
        analytic_a5 = 4.0 * np.pi * coefficients.zstar**2 / (
            coefficients.eps_inf * coefficients.omega0
        )
        self.assertAlmostEqual(analytic_a5, coefficients.A05, delta=2.0e-15)

    def test_harmonic_diagonalization_residuals(self) -> None:
        coefficients = scha.MATERIALS["nishimatsu"]
        grid = nishimatsu_harmonic.build_harmonic_grid(
            coefficients, ngrid=4, cutoff=coefficients.kmax
        )
        eigenvalues, eigenvectors = np.linalg.eigh(grid.offsets)
        reconstructed = np.einsum(
            "nai,ni,nbi->nab", eigenvectors, eigenvalues, eigenvectors, optimize=True
        )
        residual = np.max(np.abs(reconstructed - grid.offsets))
        orthogonality = np.einsum(
            "nai,naj->nij", eigenvectors, eigenvectors, optimize=True
        )
        self.assertLess(residual, 2.0e-15)
        np.testing.assert_allclose(
            orthogonality,
            np.broadcast_to(np.eye(3), orthogonality.shape),
            rtol=0.0,
            atol=2.0e-15,
        )


class FourierTests(unittest.TestCase):
    def test_round_trip_parseval_and_hermitian_symmetry(self) -> None:
        indices = np.indices((4, 4, 4), dtype=float)
        field = (
            0.3
            + np.cos(2.0 * np.pi * indices[0] / 4.0)
            - 0.7 * np.sin(2.0 * np.pi * indices[1] / 4.0)
            + 0.2 * np.cos(2.0 * np.pi * (indices[0] + indices[2]) / 4.0)
        )
        transformed = np.fft.fftn(field)
        recovered = np.fft.ifftn(transformed)
        np.testing.assert_allclose(recovered.real, field, rtol=0.0, atol=3.0e-16)
        self.assertLess(np.max(np.abs(recovered.imag)), 3.0e-16)
        self.assertAlmostEqual(
            float(np.sum(field**2)),
            float(np.sum(np.abs(transformed) ** 2) / field.size),
            delta=2.0e-14,
        )
        for index in itertools.product(range(4), repeat=3):
            negative = tuple((-component) % 4 for component in index)
            self.assertAlmostEqual(
                abs(transformed[negative] - np.conjugate(transformed[index])),
                0.0,
                delta=2.0e-14,
            )

    def test_fft_self_energy_matches_direct_circular_convolution(self) -> None:
        nspace = 4
        nk = nspace**3
        shape = (nspace, nspace, nspace)
        coordinates = np.indices(shape, dtype=float)
        scalar_m = 0.2 + 0.03 * coordinates[0] - 0.02 * coordinates[1]
        scalar_g = 0.7 + 0.05 * coordinates[2] + 0.01 * coordinates[0]
        interaction = np.zeros(shape + (3, 3, 3, 3))
        propagator = np.zeros((1,) + shape + (3, 3))
        for a in range(3):
            interaction[..., a, a, a, a] = (a + 1.0) * scalar_m
            propagator[0, ..., a, a] = scalar_g / (a + 1.0)
        vertex_fft = np.fft.fftn(interaction, axes=(0, 1, 2))[None, ...]
        actual = full_kernel.inhomogeneous_self_energy(
            propagator, vertex_fft, temperature=300.0
        )[0]

        direct = np.zeros(shape + (3, 3))
        for kindex in itertools.product(range(nspace), repeat=3):
            for qindex in itertools.product(range(nspace), repeat=3):
                transfer = tuple((kindex[axis] - qindex[axis]) % nspace for axis in range(3))
                direct[kindex] += np.einsum(
                    "agbd,gd->ab", interaction[transfer], propagator[(0,) + qindex]
                )
        direct *= -scha.KB_HARTREE_PER_K * 300.0 / nk
        direct = 0.5 * (direct + np.swapaxes(direct, -1, -2))
        np.testing.assert_allclose(actual, direct, rtol=3.0e-15, atol=3.0e-17)


class PropagatorAndResponseTests(unittest.TestCase):
    def test_matsubara_sum_against_explicit_symmetric_sum(self) -> None:
        coefficients = scha.MATERIALS["nishimatsu"]
        mu = 0.017
        temperature = 500.0
        exact = scha.matsubara_mode_sum(mu, temperature, coefficients)
        kbt = scha.KB_HARTREE_PER_K * temperature
        frequencies = 2.0 * np.pi * np.arange(-100_000, 100_001) * kbt
        explicit = np.sum(1.0 / (coefficients.mass_au * frequencies**2 + mu))
        self.assertLess(abs(explicit - exact) / exact, 2.0e-6)

    def test_static_inverse_and_dielectric_limits(self) -> None:
        coefficients = dielec_response.SchaKernel()
        direction = np.array([1.0, 2.0, -1.0])
        kernel = dielec_response.limiting_static_kernel(direction, coefficients)
        response = dielec_response.chi_from_static_kernel(kernel, coefficients)
        condition = float(np.linalg.cond(kernel))
        absolute_roundoff = (
            3.0
            * np.finfo(float).eps
            * condition
            * coefficients.omega0
        )
        np.testing.assert_allclose(
            kernel @ response,
            coefficients.omega0 * np.eye(3),
            rtol=5.0e-14,
            atol=absolute_roundoff,
        )
        unit = direction / np.linalg.norm(direction)
        longitudinal = float(unit @ response @ unit)
        transverse_trace = float(np.trace(response) - longitudinal)
        self.assertAlmostEqual(
            longitudinal,
            coefficients.omega0 / (coefficients.a1 + coefficients.a5),
            delta=2.0e-12,
        )
        self.assertAlmostEqual(
            transverse_trace / 2.0,
            coefficients.omega0 / coefficients.a1,
            delta=2.0e-12,
        )

    def test_acoustic_modes_and_vertex_symmetry(self) -> None:
        coefficients = scha.MATERIALS["nishimatsu"]
        wavevectors = np.array([[0.13, -0.09, 0.04], [0.03, 0.08, -0.12]])
        phi = acoustic._acoustic_kernel(wavevectors)
        eigenvalues = np.linalg.eigvalsh(phi)
        self.assertTrue(np.all(eigenvalues > 0.0))
        vertex = acoustic._mixed_tensor(wavevectors)
        np.testing.assert_allclose(vertex, np.swapaxes(vertex, -1, -2))
        acoustic._validate_nishimatsu_mixed_tensor(wavevectors, vertex)
        inverses = np.linalg.inv(phi)
        np.testing.assert_allclose(
            phi @ inverses,
            np.broadcast_to(np.eye(3), phi.shape),
            rtol=2.0e-14,
            atol=2.0e-14,
        )


class SolverAndIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coefficients = scha.MATERIALS["nishimatsu"]

    def test_solver_residual_and_multiple_brackets(self) -> None:
        automatic = scha.solve_a1(
            500.0,
            12,
            self.coefficients.kmax,
            self.coefficients,
            tolerance=1.0e-11,
        )
        from_wide_bracket = scha.solve_a1(
            500.0,
            12,
            self.coefficients.kmax,
            self.coefficients,
            bracket=(1.0e-6, 0.02),
            tolerance=1.0e-11,
        )
        self.assertGreater(automatic["A1"], 0.0)
        self.assertLess(abs(automatic["residual"]), 1.0e-11)
        self.assertAlmostEqual(
            automatic["A1"], from_wide_bracket["A1"], delta=2.0e-10
        )

    def test_solver_rejects_false_grid_stability_and_invalid_inputs(self) -> None:
        with self.assertRaises(RuntimeError):
            scha.solve_a1(
                500.0,
                4,
                self.coefficients.kmax,
                self.coefficients,
                tolerance=1.0e-11,
            )
        with self.assertRaises(ValueError):
            scha.integrate_g(
                -1.0e-3,
                500.0,
                12,
                self.coefficients.kmax,
                self.coefficients,
            )
        with self.assertRaises(ValueError):
            scha.solve_a1(
                math.nan, 12, self.coefficients.kmax, self.coefficients
            )

    def test_projected_and_separated_scha_residuals(self) -> None:
        projected = scha.solve_a1(
            500.0, 12, self.coefficients.kmax, self.coefficients, tolerance=1.0e-11
        )
        grid = acoustic.build_inhomogeneous_quartic_grid(12, self.coefficients)
        separated_root = acoustic.solve_root(500.0, 0.0, self.coefficients, grid)
        separated = acoustic.residual(
            separated_root, 500.0, self.coefficients, grid
        )
        self.assertLess(abs(projected["residual"]), 1.0e-11)
        self.assertLess(abs(separated[0]), 1.0e-11)
        self.assertLess(separated[4], 0.0)

    def test_experimental_cubic_lattice_fit(self) -> None:
        slope, intercept, temperatures, lattice, uncertainties = (
            acoustic.fit_experimental_cubic_lattice_parameter()
        )
        fitted = slope * temperatures + intercept
        self.assertEqual(temperatures[0], 413.0)
        self.assertEqual(temperatures[-1], 598.0)
        self.assertTrue(np.all(uncertainties > 0.0))
        self.assertAlmostEqual(slope, 4.98674856788e-5, delta=1.0e-15)
        self.assertLess(float(np.max(np.abs(fitted - lattice))), 2.5e-4)

    def test_reduced_end_to_end_scha_json_to_response(self) -> None:
        result = scha.solve_a1(
            500.0, 12, self.coefficients.kmax, self.coefficients, tolerance=1.0e-11
        )
        with tempfile.TemporaryDirectory(prefix="scha-test-") as directory:
            path = Path(directory) / "scha.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            response_coefficients = dielec_response.kernel_from_scha_json(path)
        transverse = dielec_response.chi_from_static_kernel(
            dielec_response.limiting_static_kernel(
                np.array([1.0, 0.0, 0.0]), response_coefficients
            ),
            response_coefficients,
        )[1, 1]
        scale2 = (self.coefficients.omega0 / self.coefficients.zstar) ** 2
        expected = self.coefficients.omega0 / (result["A1"] * scale2)
        self.assertAlmostEqual(float(transverse), expected, delta=2.0e-12)

    def test_reduced_full_kernel_fixed_point_residual(self) -> None:
        grid = full_kernel.build_uniform_grid(4, self.coefficients)
        projected_grid = acoustic.build_inhomogeneous_quartic_grid(
            4, self.coefficients
        )
        vertex_fft = full_kernel.acoustic_vertex_fft(
            grid, 500.0, nmatsubara=1, acoustic_mass_amu=acoustic.ACOUSTIC_MASS_AMU
        )
        initial = full_kernel.scalar_initial_mass(
            500.0, 0.0, self.coefficients, projected_grid
        )
        solution = full_kernel.solve_full_kernel(
            grid,
            vertex_fft,
            500.0,
            1,
            self.coefficients,
            pressure_gpa=0.0,
            initial_gamma_mass=initial,
            mixing=0.2,
            tolerance=1.0e-7,
            max_iterations=120,
        )
        frequencies = np.arange(-1, 2)
        kbt = scha.KB_HARTREE_PER_K * 500.0
        dynamic = self.coefficients.mass_au * (
            2.0 * np.pi * frequencies * kbt
        ) ** 2
        target = (
            grid.offsets[None, ...]
            + (self.coefficients.A01 + solution.sigma_local_u) * np.eye(3)
            + dynamic[:, None, None, None, None, None] * np.eye(3)
            + solution.sigma_inhomogeneous
        )
        fixed_point_residual = float(np.max(np.abs(target - solution.kernel)))
        self.assertLess(solution.residual, 1.0e-7)
        self.assertLess(fixed_point_residual, 1.2e-7)
        self.assertGreater(solution.minimum_static_eigenvalue, 1.0e-3)

    def test_three_clean_processes_are_bitwise_deterministic(self) -> None:
        program = (
            "import hashlib,json,numpy as np; "
            "import scha_paraelc as s; "
            "c=s.MATERIALS['nishimatsu']; "
            "r=s.solve_a1(500.0,10,c.kmax,c,tolerance=1e-11); "
            "payload=json.dumps({'A1':r['A1'],'G':r['G'],'residual':r['residual']},"
            "sort_keys=True,separators=(',',':')).encode(); "
            "print(hashlib.sha256(payload).hexdigest())"
        )
        digests = []
        for seed in (1, 17, 101):
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONHASHSEED": str(seed),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join((str(PARA_ROOT), str(PYTHON_ROOT))),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            output = subprocess.check_output(
                [sys.executable, "-c", program],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
            ).strip()
            digests.append(output)
        self.assertEqual(len(set(digests)), 1, digests)

    @unittest.skipUnless(
        os.environ.get("SCHA_VALIDATION_FULL") == "1",
        "set SCHA_VALIDATION_FULL=1 for the production-grid convergence check",
    )
    def test_production_grid_convergence(self) -> None:
        results = {}
        for ngrid in (32, 40, 48):
            results[ngrid] = scha.solve_a1(
                500.0,
                ngrid,
                self.coefficients.kmax,
                self.coefficients,
                tolerance=1.0e-11,
            )["A1"]
        difference_32_40 = abs(results[40] - results[32])
        difference_40_48 = abs(results[48] - results[40])
        self.assertLess(difference_40_48, difference_32_40)
        self.assertLess(difference_40_48 / abs(results[48]), 2.0e-3)

    @unittest.skipUnless(
        os.environ.get("SCHA_VALIDATION_FULL") == "1",
        "set SCHA_VALIDATION_FULL=1 for the production-grid convergence check",
    )
    def test_separated_acoustic_grid_convergence(self) -> None:
        roots = {}
        for ngrid in (32, 40):
            grid = acoustic.build_inhomogeneous_quartic_grid(
                ngrid, self.coefficients
            )
            roots[ngrid] = acoustic.solve_root(
                500.0, 0.0, self.coefficients, grid
            )
        self.assertLess(
            abs(roots[40] - roots[32]) / abs(roots[40]), 1.0e-3
        )

    @unittest.skipUnless(
        os.environ.get("SCHA_VALIDATION_FULL") == "1",
        "set SCHA_VALIDATION_FULL=1 for the production-grid convergence check",
    )
    def test_ferroelectric_grid_convergence(self) -> None:
        coefficients = ferro_scha.MATERIALS["nishimatsu"]
        masses = {}
        for ngrid in (16, 20):
            result = ferro_scha.solve_ferro(
                temperature=10.0,
                ngrid=ngrid,
                cutoff=coefficients.kmax,
                direction=ferro_scha.parse_direction("generic"),
                c=coefficients,
                a_t_guess=None,
                a_l_guess=None,
                p_guess=None,
                u_guess=None,
                tolerance=1.0e-9,
                max_nfev=150,
            )
            self.assertLess(result["residual_norm"], 1.0e-9)
            masses[ngrid] = result["A_T"]
        self.assertLess(
            abs(masses[20] - masses[16]) / abs(masses[20]), 2.5e-3
        )

    @unittest.skipUnless(
        os.environ.get("SCHA_VALIDATION_FULL") == "1",
        "set SCHA_VALIDATION_FULL=1 for the Matsubara convergence check",
    )
    def test_full_matrix_matsubara_convergence(self) -> None:
        grid = full_kernel.build_uniform_grid(4, self.coefficients)
        projected_grid = acoustic.build_inhomogeneous_quartic_grid(
            4, self.coefficients
        )
        masses = {}
        for nmatsubara in (2, 4):
            vertex_fft = full_kernel.acoustic_vertex_fft(
                grid,
                500.0,
                nmatsubara=nmatsubara,
                acoustic_mass_amu=acoustic.ACOUSTIC_MASS_AMU,
            )
            initial = full_kernel.scalar_initial_mass(
                500.0, 0.0, self.coefficients, projected_grid
            )
            solution = full_kernel.solve_full_kernel(
                grid,
                vertex_fft,
                500.0,
                nmatsubara,
                self.coefficients,
                pressure_gpa=0.0,
                initial_gamma_mass=initial,
                mixing=0.2,
                tolerance=1.0e-7,
                max_iterations=120,
            )
            masses[nmatsubara] = solution.gamma_mass_u
        self.assertLess(
            abs(masses[4] - masses[2]) / abs(masses[4]), 1.0e-5
        )


class BstVcaTests(unittest.TestCase):
    def test_vca_end_members_and_gamma_contact(self) -> None:
        for x_ba, source_name in ((0.0, "nishimatsu_sto"), (1.0, "nishimatsu")):
            with self.subTest(x_ba=x_ba):
                actual = bst_vca.vca_coefficients(x_ba)
                expected = scha.MATERIALS[source_name]
                for name in (
                    "a0",
                    "A01",
                    "A05",
                    "mass_amu",
                    "zstar",
                    "eps_inf",
                    "kappa2",
                    "j1",
                    "j2",
                    "j3",
                    "j4",
                    "j5",
                    "j6",
                    "j7",
                ):
                    self.assertAlmostEqual(
                        getattr(actual, name), getattr(expected, name), delta=3.0e-12
                    )
                gamma = nishimatsu_harmonic.raw_harmonic_kernel(
                    np.zeros(3), actual
                )
                gamma_scalar = float(np.trace(gamma) / 3.0)
                np.testing.assert_allclose(
                    gamma,
                    gamma_scalar * np.eye(3),
                    rtol=0.0,
                    atol=2.0e-13,
                )
                self.assertAlmostEqual(
                    actual.b1_eff, expected.b1_eff, delta=1.0e-9
                )
                self.assertAlmostEqual(
                    actual.b2_eff, expected.b2_eff, delta=1.0e-9
                )

    def test_midpoint_preserves_projected_quartic_and_pressure_linearity(self) -> None:
        coefficients = bst_vca.vca_coefficients(0.5)
        parameters = bst_vca.vca_acoustic_parameters(0.5)
        self.assertAlmostEqual(
            coefficients.b1_eff,
            parameters.bare_b1 - 0.5 * parameters.homogeneous_lambda_b1,
            delta=2.0e-15,
        )
        self.assertAlmostEqual(
            coefficients.b2_eff,
            parameters.bare_b2 - 0.5 * parameters.homogeneous_lambda_b2,
            delta=2.0e-15,
        )
        unit_shift = bst_vca.pressure_mass_shift_u(
            1.0, coefficients, parameters
        )
        doubled = bst_vca.pressure_mass_shift_u(
            2.0, coefficients, parameters
        )
        self.assertAlmostEqual(doubled, 2.0 * unit_shift, delta=2.0e-15)

    def test_curie_weiss_reference(self) -> None:
        references = bst_vca.load_references()
        self.assertEqual(set(references), {0.4, 0.5, 0.6})
        self.assertEqual(references[0.5]["T0_K"], 230.0)
        coefficients = bst_vca.vca_coefficients(0.5)
        epsilon, chi = bst_vca.experimental_reference(
            400.0, coefficients, references[0.5]
        )
        self.assertAlmostEqual(epsilon, 1.0e5 / 170.0, delta=1.0e-12)
        self.assertAlmostEqual(
            epsilon, coefficients.eps_inf + 4.0 * np.pi * chi, delta=1.0e-12
        )


class CommandLineTests(unittest.TestCase):
    def test_symbolic_diagonalization_cli(self) -> None:
        output = subprocess.check_output(
            [sys.executable, str(PYTHON_ROOT / "diagonalisation_A" / "diagonalize_A.py")],
            cwd=REPOSITORY_ROOT,
            text=True,
        )
        self.assertIn("Characteristic polynomial", output)
        self.assertIn("[111], Cx=Cy=Cz=C eigenvalues", output)

    def test_tensor_clis(self) -> None:
        for relative in (
            Path("tenseur_beta") / "compute_beta_eff.py",
            Path("tenseur_C") / "compute_C_tensor.py",
        ):
            with self.subTest(command=str(relative)):
                subprocess.run(
                    [sys.executable, str(PYTHON_ROOT / relative), "--material", "nishimatsu_bto"],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    unittest.main()
