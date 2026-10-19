"""Ordinary preset functions; configuration and defaults live in real constructors."""

from pep_compass.predefined.lebo import create as lebo
from pep_compass.predefined.simple import create as simple

__all__ = ["simple", "lebo"]
