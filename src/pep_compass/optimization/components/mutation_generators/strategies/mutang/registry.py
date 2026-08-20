"""Registries for interchangeable stages of the fixed MUTANG algorithm."""

from pep_compass.registry import Registry


geometry_registry: Registry[object] = Registry("MUTANG geometry")
direction_selection_registry: Registry[object] = Registry("MUTANG direction selection")
scoring_registry: Registry[object] = Registry("MUTANG scoring")
mutation_selection_registry: Registry[object] = Registry("MUTANG mutation selection")
combination_registry: Registry[object] = Registry("MUTANG combination")
