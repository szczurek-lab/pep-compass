"""Backends for executing independent runtime plan entries."""

from pep_compass.runtime.backends.subprocess import execute_subprocess_plan
from pep_compass.runtime.backends.slurm import write_slurm_array_script

__all__ = ["execute_subprocess_plan", "write_slurm_array_script"]
