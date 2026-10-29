"""Compatibility imports; implementation lives in the thematic subpackage."""

from proteolysis_data_manager.integration.merge.storage.sqlite import (
    ProteolysisDataset,
    write_dataset,
)

__all__ = ["ProteolysisDataset", "write_dataset"]
