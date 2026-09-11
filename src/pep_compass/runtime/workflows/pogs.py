"""Reserved PoGS workflow integration boundary."""

from __future__ import annotations


class PogsWorkflow:
    """Declare the future PoGS workflow without coupling it to composable steps."""

    def build_pipeline(self, *args, **kwargs):
        """Reject execution until the PoGS pipeline contract is implemented."""
        raise NotImplementedError("The PoGS workflow is not implemented yet.")

    # TODO - dodać implementacje PoGS'a 