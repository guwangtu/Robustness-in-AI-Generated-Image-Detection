"""Fake mpi4py stub for single-process training.
Drop-in replacement when you don't have a real MPI runtime installed.
"""
from . import MPI  # noqa: F401
