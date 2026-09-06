from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = MODULE_DIR.parent
PARAELECTRIC_DIR = PIPELINE_DIR / "SCHA_paraelc"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(PARAELECTRIC_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

import nishimatsu_harmonic as harmonic
import parameters as mayer
import post_scha_quartic as post
import run_mayer_bto as run


class MayerParameterTests(unittest.TestCase):
    def test_effective_reduction_changes_only_k1_and_k4(self) -> None:
        conventional = asdict(mayer.MAYER_CONVENTIONAL_COEFFICIENTS)
        effective = asdict(mayer.MAYER_ANHARMONIC_COEFFICIENTS)
        changed = {
            key for key in conventional if conventional[key] != effective[key]
        }
        self.assertEqual(changed, {"name", "k1_6", "k4_8"})

    def test_published_effective_coefficients_are_converted(self) -> None:
        coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
        self.assertAlmostEqual(coefficients.k1_6, -1.1651402574725753, places=13)
        self.assertAlmostEqual(coefficients.k4_8, 3.8905520945777727, places=13)

    def test_gamma_kernel_is_cubic(self) -> None:
        gamma = harmonic.raw_harmonic_kernel(
            np.zeros(3), mayer.MAYER_ANHARMONIC_COEFFICIENTS
        )
        scalar = float(np.trace(gamma) / 3.0)
        np.testing.assert_allclose(gamma, scalar * np.eye(3), atol=2.0e-13)

    def test_projected_quartic_is_consistent(self) -> None:
        coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
        parameters = mayer.MAYER_ANHARMONIC_ACOUSTIC_PARAMETERS
        self.assertAlmostEqual(
            coefficients.b1_eff,
            parameters.bare_b1 - 0.5 * parameters.homogeneous_lambda_b1,
            places=14,
        )
        self.assertAlmostEqual(
            coefficients.b2_eff,
            parameters.bare_b2 - 0.5 * parameters.homogeneous_lambda_b2,
            places=14,
        )

    def test_pressure_shift_is_linear(self) -> None:
        coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
        parameters = mayer.MAYER_ANHARMONIC_ACOUSTIC_PARAMETERS
        one = run.pressure_mass_shift_u(1.0, coefficients, parameters)
        two = run.pressure_mass_shift_u(2.0, coefficients, parameters)
        self.assertAlmostEqual(two, 2.0 * one, places=14)

    def test_wieczorek_conversion_uses_mayer_eps_inf(self) -> None:
        temperatures, epsilon_r, chi_soft = run.load_wieczorek_response()
        self.assertEqual(temperatures.size, 14)
        self.assertAlmostEqual(temperatures[0], 415.65, places=12)
        self.assertAlmostEqual(temperatures[-1], 448.15, places=12)
        np.testing.assert_allclose(
            chi_soft,
            (epsilon_r - mayer.MAYER_ANHARMONIC_COEFFICIENTS.eps_inf)
            / (4.0 * np.pi),
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_previous_nishimatsu_snapshot_is_complete(self) -> None:
        temperature, lattice, response_fit, zero = (
            run.load_previous_nishimatsu_predictions()
        )
        self.assertEqual(temperature.size, 81)
        self.assertAlmostEqual(temperature[0], 400.0, places=12)
        self.assertAlmostEqual(temperature[-1], 800.0, places=12)
        self.assertTrue(np.all(np.diff(temperature) > 0.0))
        self.assertTrue(np.all(lattice > 0.0))
        self.assertTrue(np.all(response_fit > 0.0))
        self.assertTrue(np.all(zero > 0.0))

    def test_mayer_experimental_digitization_is_complete(self) -> None:
        temperature, epsilon_r, chi_soft = (
            run.load_mayer_experimental_response()
        )
        self.assertEqual(temperature.size, 87)
        self.assertAlmostEqual(temperature[0], 302.0, places=12)
        self.assertAlmostEqual(temperature[-1], 470.0, places=12)
        self.assertTrue(np.all(np.diff(temperature) > 0.0))
        np.testing.assert_allclose(
            chi_soft,
            (epsilon_r - mayer.MAYER_ANHARMONIC_COEFFICIENTS.eps_inf)
            / (4.0 * np.pi),
            rtol=0.0,
            atol=1.0e-13,
        )

    def test_post_scha_brillouin_wrapping_is_periodic(self) -> None:
        cutoff = mayer.MAYER_ANHARMONIC_COEFFICIENTS.kmax
        vectors = np.asarray(
            [[-3.2 * cutoff, -cutoff, 2.7 * cutoff], [0.0, cutoff, 4.0 * cutoff]]
        )
        wrapped = post.wrap_to_bz(vectors, cutoff)
        self.assertTrue(np.all(wrapped >= -cutoff))
        self.assertTrue(np.all(wrapped < cutoff))
        np.testing.assert_allclose(
            post.wrap_to_bz(vectors + 2.0 * cutoff, cutoff),
            wrapped,
            atol=1.0e-15,
        )

    def test_post_scha_batch_kernel_matches_reference(self) -> None:
        coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
        cutoff = coefficients.kmax
        vectors = cutoff * np.asarray(
            [[0.13, -0.27, 0.41], [-0.73, 0.19, -0.05], [0.0, 0.0, 0.0]]
        )
        batch = post._batch_raw_harmonic_kernel(vectors, coefficients)
        reference = np.asarray(
            [harmonic.raw_harmonic_kernel(vector, coefficients) for vector in vectors]
        )
        np.testing.assert_allclose(batch, reference, rtol=2.0e-12, atol=2.0e-12)

    def test_post_scha_effective_vertex_is_fully_symmetric(self) -> None:
        vertex = post.effective_quartic_vertex(
            0.02, mayer.MAYER_ANHARMONIC_COEFFICIENTS
        )
        for permutation in (
            (1, 0, 2, 3),
            (2, 1, 0, 3),
            (3, 2, 1, 0),
        ):
            np.testing.assert_allclose(
                vertex, np.transpose(vertex, permutation), atol=2.0e-14
            )
        self.assertGreater(vertex[0, 0, 0, 0], 0.0)

    def test_post_scha_sunset_softens_the_cubic_mass(self) -> None:
        coefficients = mayer.MAYER_ANHARMONIC_COEFFICIENTS
        grid = post.build_sunset_grid(coefficients, sample_power=5, seed=17)
        delta_a1, diagnostic, _ = post.quartic_sunset_mass(
            a1=0.02,
            temperature=500.0,
            component_variance=0.02,
            coefficients=coefficients,
            grid=grid,
            tau_order=8,
        )
        self.assertLess(delta_a1, 0.0)
        self.assertGreater(diagnostic, 0.0)


if __name__ == "__main__":
    unittest.main()
