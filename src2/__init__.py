"""Refactored reflector-solver package extracted from ``ai_impl_all.py``."""

from .cli import main
from .solver import (
    DifferentiatedNystromSolver,
    MultiReflectorPaperMDS,
    differential_scattering_cross_section,
    total_scattering_cross_section,
)

__all__ = [
    "DifferentiatedNystromSolver",
    "MultiReflectorPaperMDS",
    "differential_scattering_cross_section",
    "main",
    "total_scattering_cross_section",
]
