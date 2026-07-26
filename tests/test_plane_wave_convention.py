"""Regression tests for the repository-wide ``exp(-i omega t)`` convention."""

from __future__ import annotations

import unittest

import numpy as np

from src2.geometry import SinusoidalStrip
from src2.plotting import _plane_wave_arrow_endpoints
from src2.solver import PlaneWave
from ukraine_microwave_week.generate_figures import floquet_reference_angles


class PlaneWaveConventionTests(unittest.TestCase):
    """Check that phasor phase, derivatives, and plot arrows use one direction."""

    def test_constant_phase_front_moves_along_beta(self) -> None:
        """A full complex phase point must translate along +d as time advances."""

        k = 2.0 * np.pi
        omega = k  # Unit wave speed is sufficient for this convention check.
        position = np.array([0.31, -0.27])
        time = 0.19
        delta_time = 0.137

        for beta in np.deg2rad([-90.0, 0.0, 37.0, 145.0]):
            with self.subTest(beta_rad=float(beta)):
                wave = PlaneWave(k=k, beta_rad=float(beta))
                direction = np.array([wave.direction_x, wave.direction_y])
                advanced_position = position + (omega / k) * delta_time * direction

                phase_now = wave.field(*position) * np.exp(-1j * omega * time)
                phase_later = wave.field(*advanced_position) * np.exp(
                    -1j * omega * (time + delta_time)
                )
                np.testing.assert_allclose(phase_later, phase_now, rtol=1.0e-13, atol=1.0e-13)

    def test_boundary_derivative_matches_finite_difference(self) -> None:
        """The spatial ``+ik`` sign must also be present in dU/dt."""

        curve = SinusoidalStrip(
            x_center=0.2,
            y_base=-0.4,
            length=3.0,
            amplitude=0.35,
            frequency=0.4,
            phase_rad=0.3,
        )
        wave = PlaneWave(k=3.7, beta_rad=np.deg2rad(-63.0))
        nodes = np.array([-0.72, -0.13, 0.41, 0.83])
        step = 1.0e-6

        finite_difference = (
            wave.boundary_field(curve, nodes + step)
            - wave.boundary_field(curve, nodes - step)
        ) / (2.0 * step)
        analytic = wave.boundary_derivative(curve, nodes)
        np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-9, atol=2.0e-9)

    def test_arrow_displacement_matches_propagation_direction(self) -> None:
        """Every plotted arrow must point along the beta direction, never against it."""

        xs = np.linspace(-4.0, 6.0, 11)
        ys = np.linspace(-3.0, 5.0, 9)
        xg, yg = np.meshgrid(xs, ys)

        for beta in np.deg2rad([-135.0, -90.0, 0.0, 42.0, 90.0, 170.0]):
            with self.subTest(beta_rad=float(beta)):
                wave = PlaneWave(k=2.0 * np.pi, beta_rad=float(beta))
                endpoints = _plane_wave_arrow_endpoints(wave, xg, yg)
                self.assertIsNotNone(endpoints)
                start, end = endpoints  # type: ignore[misc]
                displacement = end - start
                displacement /= np.linalg.norm(displacement)
                expected = np.array([wave.direction_x, wave.direction_y])
                np.testing.assert_allclose(displacement, expected, rtol=0.0, atol=1.0e-12)

                self.assertTrue(xs.min() <= start[0] <= xs.max())
                self.assertTrue(ys.min() <= start[1] <= ys.max())
                self.assertTrue(xs.min() <= end[0] <= xs.max())
                self.assertTrue(ys.min() <= end[1] <= ys.max())

    def test_e_minus_iwt_demodulation_recovers_phasor(self) -> None:
        """FDTD-style demodulation with ``+i omega t`` returns U, not conj(U)."""

        omega = 2.3
        expected = np.exp(1j * 0.71)
        times = np.arange(4096) * (2.0 * np.pi / omega / 64.0)
        real_signal = np.real(expected * np.exp(-1j * omega * times))
        recovered = 2.0 * np.mean(real_signal * np.exp(1j * omega * times))
        np.testing.assert_allclose(recovered, expected, rtol=0.0, atol=1.0e-13)

    def test_floquet_markers_use_reflected_branch(self) -> None:
        """Publication markers must put reflected orders in the -y hemisphere."""

        orders, forward, backward = floquet_reference_angles(0.2, max_order=1)
        references = {
            order: (phi_forward, phi_backward)
            for order, phi_forward, phi_backward in zip(orders, forward, backward)
        }
        np.testing.assert_allclose(references[-1], (101.536959, 258.463041), atol=1.0e-6)
        np.testing.assert_allclose(references[0], (90.0, 270.0), atol=1.0e-12)
        np.testing.assert_allclose(references[1], (78.463041, 281.536959), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
