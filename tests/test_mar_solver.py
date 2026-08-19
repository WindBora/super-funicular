"""Focused tests for the coefficient-space analytical-regularization solver."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import hankel1, jv

from src2.geometry import LineSegment, SinusoidalStrip
from src2.solver import (
    DifferentiatedNystromSolver,
    MARSolution,
    MultiReflectorMAR,
    MultiReflectorMDS,
    MultiReflectorMoM,
    PlaneWave,
    total_scattering_cross_section,
)


def sinusoidal_strip(amplitude: float) -> SinusoidalStrip:
    """Return the publication's unit-half-span, five-period strip."""

    return SinusoidalStrip(
        x_center=0.0,
        y_base=0.0,
        length=2.0,
        amplitude=amplitude,
        frequency=2.5,
        phase_rad=np.pi / 2.0,
    )


class MethodOfMomentsConstructionTests(unittest.TestCase):
    """Lock down the panel and quadrature construction reported in the paper."""

    def test_default_panel_quadratures_match_reported_construction(self) -> None:
        solver = MultiReflectorMoM(
            [sinusoidal_strip(0.0)],
            PlaneWave(k=3.0, beta_rad=np.pi / 2.0),
            n=7,
        )

        self.assertEqual(solver.panel_quad_order, 12)
        self.assertEqual(solver.self_panel_quad_order, 20)
        np.testing.assert_allclose(
            solver.panel_widths,
            np.full(solver.n, np.pi / solver.n),
            rtol=0.0,
            atol=4.0 * np.finfo(float).eps,
        )

        self.assertEqual(solver._regular_panel_nodes.shape, (solver.n, 12))
        self.assertEqual(solver._regular_panel_weights.shape, (solver.n, 12))
        np.testing.assert_allclose(
            np.sum(solver._regular_panel_weights, axis=1),
            solver.panel_widths,
            rtol=0.0,
            atol=4.0 * np.finfo(float).eps,
        )

        self.assertEqual(solver._singular_panel_nodes.shape, (solver.n, 40))
        self.assertEqual(solver._singular_panel_weights.shape, (solver.n, 40))
        left_weights = solver._singular_panel_weights[:, :20]
        right_weights = solver._singular_panel_weights[:, 20:]
        np.testing.assert_allclose(
            np.sum(left_weights, axis=1),
            0.5 * solver.panel_widths,
            rtol=0.0,
            atol=4.0 * np.finfo(float).eps,
        )
        np.testing.assert_allclose(
            np.sum(right_weights, axis=1),
            0.5 * solver.panel_widths,
            rtol=0.0,
            atol=4.0 * np.finfo(float).eps,
        )

        singular_theta = np.arccos(solver._singular_panel_nodes)
        self.assertTrue(np.all(singular_theta[:, :20] < solver.theta_centers[:, None]))
        self.assertTrue(np.all(singular_theta[:, 20:] > solver.theta_centers[:, None]))


class AnalyticalRegularizationTests(unittest.TestCase):
    """Protect the static semi-inversion and independent modal API."""

    def test_projection_and_field_quadratures_are_independent(self) -> None:
        curve = sinusoidal_strip(0.1)
        incident = PlaneWave(k=6.0, beta_rad=0.8)
        default_solver = MultiReflectorMAR(
            [curve], incident, n=16, quadrature_order=64
        )
        field_solver = MultiReflectorMAR(
            [curve],
            incident,
            n=16,
            quadrature_order=64,
            field_order=97,
        )

        self.assertEqual(default_solver.field_order, 64)
        self.assertEqual(default_solver.t_nodes.shape, (64,))
        self.assertEqual(field_solver.projection_nodes.shape, (64,))
        self.assertEqual(field_solver.field_nodes.shape, (97,))
        self.assertEqual(field_solver.t_nodes.shape, (97,))
        self.assertEqual(field_solver.caches.x_t[0].shape, (97,))
        self.assertEqual(field_solver._basis_at_projection.shape, (64, 16))
        self.assertEqual(field_solver._basis_at_field.shape, (97, 16))
        self.assertAlmostEqual(float(np.sum(field_solver.field_weights)), 1.0)
        self.assertAlmostEqual(field_solver.near_field_weight(), 1.0 / 97.0)

        default_solution = default_solver.solve()
        field_solution = field_solver.solve()
        np.testing.assert_allclose(
            field_solution.coefficients,
            default_solution.coefficients,
            rtol=0.0,
            atol=2.0e-14,
        )
        self.assertEqual(field_solution.v_nodes.shape, (1, 97))
        self.assertEqual(field_solution.physical_current_nodes.shape, (1, 97))

    def test_flat_strip_far_field_matches_modal_bessel_reconstruction(self) -> None:
        curve = sinusoidal_strip(0.0)
        incident = PlaneWave(k=5.0, beta_rad=0.73)
        solver = MultiReflectorMAR(
            [curve],
            incident,
            n=24,
            quadrature_order=96,
            field_order=257,
        )
        solution = solver.solve()
        phi = np.linspace(0.0, 2.0 * np.pi, 73, endpoint=False)
        numerical = solution.far_field_pattern(phi, total=False)

        modes = np.arange(solver.n)
        basis_normalization = np.ones(solver.n)
        basis_normalization[1:] = np.sqrt(2.0)
        modal_integrals = (
            basis_normalization[:, None]
            * (-1j) ** modes[:, None]
            * jv(modes[:, None], solver.k * np.cos(phi)[None, :])
        )
        analytic = np.sqrt(2.0 / (np.pi * 1j)) * (
            solution.coefficients[0] @ modal_integrals
        )
        np.testing.assert_allclose(numerical, analytic, rtol=2.0e-13, atol=2.0e-13)

    def test_small_argument_remainder_and_flat_diagonal_are_stable(self) -> None:
        curve = sinusoidal_strip(0.0)
        solver = MultiReflectorMAR(
            [curve],
            PlaneWave(k=4.0, beta_rad=np.pi / 2.0),
            n=12,
            quadrature_order=64,
        )
        z = np.array([1.0e-3, 1.0e-2, 0.1])
        a = solver.log_singularity_coeff
        direct = hankel1(0, z) - 1.0 - a * (
            np.log(0.5 * z) + 0.5772156649015329
        )
        np.testing.assert_allclose(
            solver._dynamic_remainder(z), direct, rtol=2.0e-10, atol=2.0e-15
        )

        nodes = np.array([-0.9, -0.2, 0.3, 0.85])
        flat_remainder = solver._kernel_block_samples(0, 0, nodes, nodes)
        np.testing.assert_allclose(
            np.diag(flat_remainder), np.zeros(nodes.size), rtol=0.0, atol=0.0
        )

    def test_static_eigenvalues_and_corrugated_diagonal_are_exact(self) -> None:
        curve = sinusoidal_strip(0.1)
        solver = MultiReflectorMAR(
            [curve],
            PlaneWave(k=12.0, beta_rad=np.pi / 2.0),
            n=12,
            quadrature_order=64,
        )

        a = 2j / np.pi
        expected_zero = 1.0 + a * (
            np.log(12.0 / 4.0) + 0.5772156649015329
        )
        # Writing the constant directly makes the normalization under test
        # visible: C_k - a*log(2) = 1 + a*(log(k/4) + gamma).
        np.testing.assert_allclose(
            solver.static_eigenvalues[0], expected_zero, rtol=0.0, atol=1.0e-15
        )
        modes = np.arange(1, solver.n, dtype=float)
        np.testing.assert_allclose(
            solver.static_eigenvalues[1:], -a / modes, rtol=0.0, atol=0.0
        )

        nodes = np.array([-0.72, -0.1, 0.43, 0.81])
        remainder = solver._kernel_block_samples(0, 0, nodes, nodes)
        expected_diagonal = a * np.log(curve.speed(nodes))
        np.testing.assert_allclose(
            np.diag(remainder), expected_diagonal, rtol=0.0, atol=1.0e-15
        )
        np.testing.assert_allclose(
            remainder, remainder.T, rtol=2.0e-14, atol=2.0e-14
        )

    def test_flat_strip_pattern_and_tscs_agree_with_nystrom(self) -> None:
        curve = sinusoidal_strip(0.0)
        incident = PlaneWave(k=5.0, beta_rad=np.pi / 2.0)
        mar_solver = MultiReflectorMAR(
            [curve], incident, n=32, quadrature_order=128
        )
        mar = mar_solver.solve()
        nystrom = DifferentiatedNystromSolver(
            [curve], incident, n=128
        ).solve()

        self.assertIsInstance(mar, MARSolution)
        self.assertNotIsInstance(mar_solver, MultiReflectorMDS)
        self.assertEqual(mar.coefficients.shape, (1, 32))
        self.assertEqual(mar.v_nodes.shape, (1, 128))
        self.assertLess(mar.boundary_residual_max, 3.0e-5)

        phi = np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False)
        mar_pattern = mar.far_field_pattern(phi, total=False)
        nystrom_pattern = nystrom.far_field_pattern(phi, total=False)
        relative_pattern_error = np.linalg.norm(
            mar_pattern - nystrom_pattern
        ) / np.linalg.norm(nystrom_pattern)
        self.assertLess(relative_pattern_error, 1.0e-5)

        mar_ratio = total_scattering_cross_section(
            phi, mar_pattern, incident.k
        ) / 4.0
        nystrom_ratio = total_scattering_cross_section(
            phi, nystrom_pattern, incident.k
        ) / 4.0
        self.assertLess(abs(mar_ratio - nystrom_ratio), 5.0e-7)

    def test_corrugated_solver_preserves_and_breaks_parity_as_expected(self) -> None:
        curve = sinusoidal_strip(0.1)
        phi = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
        for beta, expected_odd_bound in ((np.pi / 2.0, 1.0e-12), (0.7, None)):
            with self.subTest(beta=beta):
                incident = PlaneWave(k=12.0, beta_rad=beta)
                mar = MultiReflectorMAR(
                    [curve], incident, n=64, quadrature_order=256
                ).solve()
                nystrom = DifferentiatedNystromSolver(
                    [curve], incident, n=256
                ).solve()
                mar_pattern = mar.far_field_pattern(phi, total=False)
                nystrom_pattern = nystrom.far_field_pattern(phi, total=False)
                relative_error = np.linalg.norm(
                    mar_pattern - nystrom_pattern
                ) / np.linalg.norm(nystrom_pattern)
                self.assertLess(relative_error, 1.5e-4)
                self.assertLess(mar.boundary_residual_max, 2.0e-3)

                odd_fraction = np.linalg.norm(
                    mar.coefficients[0, 1::2]
                ) / np.linalg.norm(mar.coefficients[0])
                if expected_odd_bound is not None:
                    self.assertLess(odd_fraction, expected_odd_bound)
                else:
                    self.assertGreater(odd_fraction, 0.1)

    def test_two_disjoint_reflectors_form_a_finite_block_system(self) -> None:
        curves = [
            LineSegment(-1.0, -0.5, 1.0, -0.5),
            LineSegment(-1.0, 0.5, 1.0, 0.5),
        ]
        solver = MultiReflectorMAR(
            curves,
            PlaneWave(k=3.0, beta_rad=np.pi / 2.0),
            n=16,
            quadrature_order=64,
        )
        solution = solver.solve()
        self.assertEqual(solution.coefficients.shape, (2, 16))
        self.assertEqual(solver.regularized_matrix.shape, (32, 32))
        self.assertTrue(np.all(np.isfinite(solution.coefficients)))
        self.assertLess(solution.boundary_residual_max, 2.0e-4)


if __name__ == "__main__":
    unittest.main()
