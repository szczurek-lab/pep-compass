"""Top-level composition facade for experiment analyses and visualizations."""

from __future__ import annotations

from dataclasses import dataclass

from pep_compass.analysis.analysis_types import LocalityAnalysis
from pep_compass.analysis.visualization import (
    LocalityVisualizer,
    PlotTheme,
)
from pep_compass.analysis.reader.selection import ExperimentSelection


@dataclass(frozen=True, slots=True)
class VisualizationCollection:
    """Collect visualizers corresponding to registered analysis types."""

    locality: LocalityVisualizer


class ExperimentAnalysis:
    """Expose registered analysis types for one lazy experiment selection."""

    def __init__(
        self,
        selection: ExperimentSelection,
        plot_theme: str | PlotTheme = "default",
        use_cache: bool = True,
    ) -> None:
        """Initialize numerical and visualization facades.

        :param selection: Lazy reader selection used by all analysis types.
        :param plot_theme: Named preset or explicit shared plotting theme.
        :param use_cache: Whether completed numerical results use SQLite cache.
        """
        theme = (
            PlotTheme.preset(plot_theme)
            if isinstance(plot_theme, str)
            else plot_theme
        )
        self.locality = LocalityAnalysis(selection, use_cache=use_cache)
        self.visualize = VisualizationCollection(LocalityVisualizer(theme))
