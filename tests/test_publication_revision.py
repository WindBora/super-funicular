"""Regression tests for the publication-facing numerical API."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
import unittest

import numpy as np

from src2.geometry import SinusoidalStrip
from src2.numerics import hankel1
from src2.solver import (
    DifferentiatedNystromSolver,
    MultiReflectorPaperMDS,
    PlaneWave,
    differential_scattering_cross_section,
    total_scattering_cross_section,
)


class ScatteringCrossSectionTests(unittest.TestCase):
    """Check physical normalization and periodic angular integration."""

    def test_constant_pattern_total_cross_section_on_both_grid_conventions(self) -> None:
        """A constant pattern integrates to ``2*pi*|Phi|^2/k``."""

        k = 4.25
        constant_pattern = 2.0 - 1.5j
        expected = 2.0 * np.pi * abs(constant_pattern) ** 2 / k

        for endpoint in (True, False):
            with self.subTest(endpoint=endpoint):
                phi = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=endpoint)
                pattern = np.full(phi.shape, constant_pattern, dtype=np.complex128)
                differential = differential_scattering_cross_section(phi, pattern, k)
                np.testing.assert_allclose(
                    differential,
                    np.full(phi.shape, abs(constant_pattern) ** 2 / k),
                    rtol=0.0,
                    atol=1.0e-14,
                )
                self.assertAlmostEqual(
                    total_scattering_cross_section(phi, pattern, k), expected, places=12
                )

    def test_cross_section_input_validation(self) -> None:
        """Invalid physical and angular inputs fail instead of yielding a metric."""

        phi = np.linspace(0.0, 2.0 * np.pi, 17)
        pattern = np.ones_like(phi, dtype=np.complex128)

        for invalid_k in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(k=invalid_k):
                with self.assertRaises(ValueError):
                    differential_scattering_cross_section(phi, pattern, invalid_k)

        with self.assertRaises(ValueError):
            differential_scattering_cross_section(phi[:-1], pattern, 1.0)
        with self.assertRaises(ValueError):
            differential_scattering_cross_section(phi.astype(complex) + 1j, pattern, 1.0)
        with self.assertRaises(ValueError):
            total_scattering_cross_section(phi[:8], pattern[:8], 1.0)

    def test_flat_strip_solver_has_physical_go_normalization(self) -> None:
        """The solver and TSCS helper together approach ``sigma=4L``."""

        half_length = 1.0
        curve = SinusoidalStrip(
            x_center=0.0,
            y_base=0.0,
            length=2.0 * half_length,
            amplitude=0.0,
            frequency=2.5,
            phase_rad=np.pi / 2.0,
        )
        k = 20.0 / half_length
        solution = DifferentiatedNystromSolver(
            reflectors=[curve],
            incident=PlaneWave(k=k, beta_rad=np.pi / 2.0),
            n=64,
        ).solve()
        phi = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
        sigma = total_scattering_cross_section(
            phi, solution.far_field_pattern(phi, total=False), k
        )
        self.assertLess(abs(sigma / (4.0 * half_length) - 1.0), 5.0e-3)


class PublicationSolverApiTests(unittest.TestCase):
    """Protect the canonical solver name and the historical import."""

    def test_historical_paper_mds_name_is_an_exact_alias(self) -> None:
        self.assertIs(DifferentiatedNystromSolver, MultiReflectorPaperMDS)

        incident = PlaneWave(k=2.0 * np.pi, beta_rad=np.pi / 2.0)
        solver = MultiReflectorPaperMDS(
            reflectors=[], incident=incident, n=4, aux_quad_order=4
        )
        self.assertIsInstance(solver, DifferentiatedNystromSolver)

    def test_differentiated_kernel_uses_exact_cauchy_coefficient(self) -> None:
        """The self-kernel split uses the analytic ``-2i/pi`` coefficient."""

        curve = SinusoidalStrip(
            x_center=0.0,
            y_base=0.0,
            length=2.0,
            amplitude=0.1,
            frequency=2.5,
            phase_rad=np.pi / 2.0,
        )
        solver = DifferentiatedNystromSolver(
            reflectors=[curve],
            incident=PlaneWave(k=20.0, beta_rad=np.pi / 2.0),
            n=8,
            aux_quad_order=16,
        )
        self.assertEqual(solver.cauchy_singularity_coeff, -2j / np.pi)

        regular = solver._paper_k_block(0, 0)
        diff = solver.t_nodes[:, None] - solver.tau_nodes[None, :]
        reconstructed = regular + solver.cauchy_singularity_coeff / diff

        x_source = solver.caches.x_t[0][:, None]
        y_source = solver.caches.y_t[0][:, None]
        x_target = solver.caches.x_tau[0][None, :]
        y_target = solver.caches.y_tau[0][None, :]
        dx_target = solver.caches.dx_tau[0][None, :]
        dy_target = solver.caches.dy_tau[0][None, :]
        dx = x_source - x_target
        dy = y_source - y_target
        distance = np.sqrt(dx * dx + dy * dy)
        distance_derivative = -(dx * dx_target + dy * dy_target) / distance

        raw_derivative = -solver.k * hankel1(1, solver.k * distance) * distance_derivative
        np.testing.assert_allclose(reconstructed, raw_derivative, rtol=2.0e-14, atol=2.0e-14)

        solution = solver.solve()
        expected_current = solution.v_nodes / (
            np.pi
            * np.vstack(solver.caches.speed_t)
            * np.sqrt(1.0 - solver.t_nodes**2)[None, :]
        )
        np.testing.assert_allclose(
            solution.physical_current_nodes, expected_current, rtol=0.0, atol=0.0
        )

    def test_flat_strip_near_diagonal_limit_has_cauchy_sign(self) -> None:
        """An independent Hankel limit tends to ``-2i/pi``."""

        k = 7.0
        delta = 1.0e-7
        derivative_kernel = k * hankel1(1, k * delta)
        limit = delta * derivative_kernel
        np.testing.assert_allclose(limit, -2j / np.pi, rtol=1.0e-6, atol=1.0e-7)


class PublicationGeometryTests(unittest.TestCase):
    """Lock down the revised finite, symmetric five-period profile."""

    def test_p5_cosine_strip_is_mirror_symmetric(self) -> None:
        """``x=L*t, y=h*cos(pi*P*t)`` is even about the y-axis for P=5."""

        half_length = 1.0
        period_count = 5
        height = 0.1 * half_length
        curve = SinusoidalStrip(
            x_center=0.0,
            y_base=0.0,
            length=2.0 * half_length,
            amplitude=height,
            frequency=period_count / (2.0 * half_length),
            phase_rad=np.pi / 2.0,
        )

        t = np.linspace(0.0, 1.0, 101)
        np.testing.assert_allclose(curve.x(-t), -curve.x(t), rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(curve.y(-t), curve.y(t), rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            curve.y(t), height * np.cos(np.pi * period_count * t), rtol=0.0, atol=1.0e-15
        )
        np.testing.assert_allclose(curve.x(np.array([-1.0, 1.0])), [-1.0, 1.0])
        np.testing.assert_allclose(curve.y(np.array([-1.0, 1.0])), [-height, -height])

    def test_p5_floquet_cutoff_and_reference_angles(self) -> None:
        """Normal-incidence first orders obey the requested strict cutoff."""

        period_count = 5
        cutoff = np.pi * period_count
        self.assertAlmostEqual(cutoff, 15.707963267948966, places=14)
        self.assertGreater(np.pi * period_count / 12.0, 1.0)
        self.assertEqual(abs(np.pi * period_count / cutoff) < 1.0, False)

        expected_degrees = {
            16.0: (10.963751239909339, 169.03624876009064),
            20.0: (38.24248148397803, 141.75751851602197),
        }
        for k_l, expected in expected_degrees.items():
            cosine = np.pi * period_count / k_l
            self.assertLess(abs(cosine), 1.0)
            angles = (np.degrees(np.arccos(cosine)), np.degrees(np.arccos(-cosine)))
            np.testing.assert_allclose(angles, expected, rtol=0.0, atol=1.0e-12)


class GeneratedPublicationDataTests(unittest.TestCase):
    """Ensure the CSV, LaTeX macros, and manuscript remain synchronized."""

    def test_generated_results_and_macros_are_consistent(self) -> None:
        paper_dir = Path(__file__).resolve().parents[1] / "ukraine_microwave_week"
        csv_path = paper_dir / "revision_results.csv"
        tex_path = paper_dir / "revision_results.tex"
        manuscript_path = paper_dir / "main.tex"

        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        tex = tex_path.read_text(encoding="utf-8")
        manuscript = manuscript_path.read_text(encoding="utf-8")

        by_dataset: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_dataset.setdefault(row["dataset"], []).append(row)
        self.assertEqual(len(by_dataset["density_convergence"]), 9)
        self.assertEqual(len(by_dataset["flat_tscs_sweep"]), 41 * 4)
        self.assertEqual(len(by_dataset["flat_order_doubling"]), 9)
        self.assertEqual(len(by_dataset["production_reference_check"]), 10)
        self.assertEqual(len(by_dataset["amplitude_polar"]), 3 * 4096)
        self.assertEqual(len(by_dataset["frequency_polar"]), 3 * 4096)

        metrics = {
            row["series"]: float(row["y_value"])
            for row in by_dataset["metric"]
        }
        expected_names = {
            "ConvErrorNFiveTwelve",
            "FlatTSCSRatioTwenty",
            "BackendMaxDiff",
            "AmpTSCSFlat",
            "AmpTSCSFive",
            "AmpTSCSTen",
            "FreqTSCSTwelve",
            "FreqTSCSSixteen",
            "FreqTSCSTwenty",
            "FirstOrderCutoff",
            "FlatOrderDoublingMaxChange",
            "ProductionMaxTSCSChange",
            "ProductionMaxPatternChange",
        }
        self.assertEqual(set(metrics), expected_names)

        def latex_number(value: float) -> str:
            magnitude = abs(value)
            if value == 0.0:
                return r"\ensuremath{0}"
            if 1.0e-3 <= magnitude < 1.0e4:
                return rf"\ensuremath{{{value:.7g}}}"
            exponent = math.floor(math.log10(magnitude))
            mantissa = value / (10.0**exponent)
            return rf"\ensuremath{{{mantissa:.6g}\times 10^{{{exponent}}}}}"

        for name, value in metrics.items():
            expected_line = rf"\newcommand{{\{name}}}{{{latex_number(value)}}}"
            self.assertIn(expected_line, tex)
            self.assertIn(rf"\{name}", manuscript)

        self.assertLess(metrics["ConvErrorNFiveTwelve"], 1.0e-4)
        self.assertLess(metrics["BackendMaxDiff"], 1.0e-3)
        self.assertLess(metrics["FlatOrderDoublingMaxChange"], 1.0e-3)
        self.assertLess(metrics["ProductionMaxTSCSChange"], 1.0e-3)
        self.assertLess(metrics["ProductionMaxPatternChange"], 1.0e-3)
        self.assertAlmostEqual(metrics["FirstOrderCutoff"], 5.0 * np.pi, places=13)


class IEEEStyleSourceTests(unittest.TestCase):
    """Prevent the corrected conference-paper style from regressing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manuscript = (
            Path(__file__).resolve().parents[1]
            / "ukraine_microwave_week"
            / "main.tex"
        ).read_text(encoding="utf-8")

    def test_abstract_is_one_math_free_paragraph(self) -> None:
        match = re.search(
            r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
            self.manuscript,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        abstract = match.group(1).strip()
        self.assertNotIn("$", abstract)
        self.assertNotIn(r"\cite", abstract)
        self.assertNotRegex(abstract, r"\n\s*\n")

    def test_ieee_callouts_and_float_hack_do_not_return(self) -> None:
        self.assertNotIn(r"Figure~\ref", self.manuscript)
        self.assertNotRegex(
            self.manuscript,
            r"\\newpage\s*\\null\s*\\newpage",
        )

    def test_index_terms_and_bibliography_order(self) -> None:
        keywords = re.search(
            r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
            self.manuscript,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(keywords)
        terms = [term.strip() for term in keywords.group(1).split(",")]
        self.assertEqual(len(terms), 5)

        expected_bibitems = (
            "nazarchuk1989",
            "nosich2005",
            "nosich2007",
            "nosich_josa2007",
            "gandel2010",
            "tong2006",
            "tsalamengas2006",
            "shapoval2011",
            "oguzer1995",
            "oguzer2001",
            "kobayashi1991",
            "eizawa2014",
            "vinogradova2019",
            "vinogradova2021",
        )
        bibitems = tuple(
            re.findall(
                r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}",
                self.manuscript,
            )
        )
        self.assertEqual(len(bibitems), 14)
        self.assertEqual(len(set(bibitems)), len(bibitems))
        self.assertEqual(bibitems, expected_bibitems)

    def test_revised_manuscript_source_contract(self) -> None:
        """Keep manuscript notation independent of code and equation numbers."""

        self.assertNotIn("v_nodes", self.manuscript)
        self.assertNotIn(r"\label{eq:potential}", self.manuscript)
        self.assertIn(r"\label{tab:tscs_sweeps}", self.manuscript)
        self.assertIn(r"|\Phi_{\rm sc}(\varphi)|^2/(kL)", self.manuscript)
        self.assertNotIn(r"d\sigma/d\varphi", self.manuscript)


if __name__ == "__main__":
    unittest.main()
