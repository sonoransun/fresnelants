"""Solvers: physical optics (default) + cascade + optional full-wave adapters."""

from .base import Solver, SolverResult
from .cascade_po import CascadePOSolver
from .physical_optics import PhysicalOpticsSolver

__all__ = ["CascadePOSolver", "PhysicalOpticsSolver", "Solver", "SolverResult"]
