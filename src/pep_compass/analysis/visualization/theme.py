"""Shared plotting themes for consistent experiment figures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlotTheme:
    """Configure consistent figure dimensions, palette, and confidence styling."""

    palette: str = "colorblind"
    figure_size: tuple[float, float] = (8.0, 5.0)
    context: str = "notebook"
    style: str = "whitegrid"
    confidence_alpha: float = 0.2
    dpi: int = 120

    @classmethod
    def preset(cls, name: str) -> "PlotTheme":
        """Return a named notebook, paper, or presentation preset."""
        presets = {
            "default": cls(),
            "notebook": cls(),
            "paper": cls(figure_size=(7.0, 4.2), context="paper", dpi=300),
            "presentation": cls(
                figure_size=(11.0, 6.0), context="talk", dpi=150
            ),
        }
        try:
            return presets[name]
        except KeyError as error:
            raise ValueError(f"Unknown plot theme: {name}") from error
