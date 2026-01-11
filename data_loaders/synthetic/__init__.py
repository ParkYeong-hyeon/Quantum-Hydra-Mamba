"""
Synthetic Benchmark Datasets

Synthetic tasks for testing long-range dependencies and quantum advantages.
"""

from . import forrelation
from . import adding_problem
from . import selective_copy

from .forrelation import *
from .adding_problem import *
from .selective_copy import *

__all__ = [
    # Re-export all from submodules
    *forrelation.__all__,
    *adding_problem.__all__,
    *selective_copy.__all__,
]
