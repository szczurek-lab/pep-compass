"""Compatibility imports; implementation lives in the thematic subpackage."""

from proteolysis_data_manager.integration.merge.contract.builder import ContractBuilder
from proteolysis_data_manager.integration.merge.contract.dataset import UnifiedDataset

__all__ = ["ContractBuilder", "UnifiedDataset"]
