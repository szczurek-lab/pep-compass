"""Consistent visualizations for locality analysis results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from pep_compass.analysis.result import AnalysisResult
from pep_compass.analysis.visualization.theme import PlotTheme


class LocalityVisualizer:
    """Render locality results using one shared theme."""

    def __init__(self, theme: PlotTheme) -> None:
        self.theme = theme

    @staticmethod
    def share_y_limits(figures: dict) -> None:
        """Apply one common y-range across every ``Axes`` in a figures dict.

        Plotting methods that return multiple independent figures (e.g.
        :meth:`latent_jump`, :meth:`latent_jump_by_iteration`) deliberately
        don't equalize axes themselves -- call this on the returned dict
        before displaying when the figures need to be visually comparable.

        :param figures: A ``{key: Axes}`` mapping, e.g. one method's return value.
        """
        axes_list = list(figures.values())
        if not axes_list:
            return
        bottom = min(axes.get_ylim()[0] for axes in axes_list)
        top = max(axes.get_ylim()[1] for axes in axes_list)
        for axes in axes_list:
            axes.set_ylim(bottom, top)

    def _axes(self):
        with sns.axes_style(self.theme.style), sns.plotting_context(self.theme.context):
            figure, axes = plt.subplots(
                figsize=self.theme.figure_size, dpi=self.theme.dpi
            )
        return figure, axes

    def latent_jump(self, result: AnalysisResult) -> dict[tuple[bool, object], "plt.Axes"]:
        """Plot Euclidean latent distance vs. Levenshtein distance, one figure
        per ``(sorbes_moved, grid_value)`` combination.

        Separate figures rather than facets/hues for two reasons: comparing
        "mutang acceptance" (grid) settings as whole plots was the explicit
        requirement, and pooling seeds whose SORBES walker never moved with
        seeds where it did would average away exactly the effect this plot is
        meant to show (see ``sorbes_moved`` in :func:`latent_jump`'s docstring).
        Keyed and iterated movement-first so stationary-seed and moving-seed
        figures are each contiguous rather than interleaved by grid value.

        :param result: Output of ``LocalityAnalysis.latent_jump``.
        :return: Mapping from ``(sorbes_moved, grid_value)`` to its ``Axes``.
        """
        data = result.data
        if data.empty:
            raise ValueError("Cannot visualize an empty latent-jump result")
        labels = {
            "candidate_to_origin": "candidate → trajectory origin",
            "candidate_to_sorbes_parent": "candidate → generating SORBES point",
            "sorbes_to_origin": "SORBES point → trajectory origin",
        }
        figures = {}
        # Grouped by movement first so stationary-seed figures and
        # moving-seed figures are each contiguous when iterated/displayed,
        # rather than interleaving by grid value.
        for (sorbes_moved, grid_value), panel in data.groupby(
            ["sorbes_moved", "grid_value"], dropna=False, sort=True
        ):
            with sns.axes_style(self.theme.style), sns.plotting_context(
                self.theme.context
            ):
                figure, axes = plt.subplots(
                    figsize=(
                        self.theme.figure_size[0] * 1.6,
                        self.theme.figure_size[1],
                    ),
                    dpi=self.theme.dpi,
                )
            plot_data = panel.copy()
            plot_data["distance_kind"] = plot_data["distance_kind"].map(labels)
            sns.violinplot(
                data=plot_data,
                x="levenshtein_distance",
                y="euclidean_distance",
                hue="distance_kind",
                palette=self.theme.palette,
                cut=0,
                density_norm="width",
                ax=axes,
            )
            movement_label = "moving" if sorbes_moved else "stationary"
            axes.set(
                title=(
                    "Euclidean latent distance vs. Levenshtein edit distance\n"
                    f"direction_significance_threshold={grid_value:g}, "
                    f"SORBES {movement_label} seeds"
                ),
                xlabel="Levenshtein distance",
                ylabel="Euclidean latent distance",
            )
            axes.legend(title="", loc="upper left", fontsize="small")
            figure.tight_layout()
            figures[(sorbes_moved, grid_value)] = axes
        return figures

    def latent_jump_by_iteration(
        self, result: AnalysisResult
    ) -> dict[tuple[bool, object], "plt.Axes"]:
        """Plot Euclidean latent distance vs. local-enumeration iteration.

        Same three distance kinds and same ``(sorbes_moved, grid_value)``
        figure split as :meth:`latent_jump`, but x is ``trajectory_step``
        (the generating SORBES iteration) instead of Levenshtein distance --
        i.e. how each distance's *distribution* evolves step by step, not how
        it relates to edit distance. Rows with no resolved ``trajectory_step``
        (unmatched SORBES parent) are dropped.

        :param result: Output of ``LocalityAnalysis.latent_jump``.
        :return: Mapping from ``(sorbes_moved, grid_value)`` to its ``Axes``.
        """
        data = result.data.dropna(subset=["trajectory_step"])
        if data.empty:
            raise ValueError("Cannot visualize an empty latent-jump result")
        labels = {
            "candidate_to_origin": "candidate → trajectory origin",
            "candidate_to_sorbes_parent": "candidate → generating SORBES point",
            "sorbes_to_origin": "SORBES point → trajectory origin",
        }
        figures = {}
        for (sorbes_moved, grid_value), panel in data.groupby(
            ["sorbes_moved", "grid_value"], dropna=False, sort=True
        ):
            with sns.axes_style(self.theme.style), sns.plotting_context(
                self.theme.context
            ):
                figure, axes = plt.subplots(
                    figsize=(
                        self.theme.figure_size[0] * 1.6,
                        self.theme.figure_size[1],
                    ),
                    dpi=self.theme.dpi,
                )
            plot_data = panel.copy()
            plot_data["trajectory_step"] = plot_data["trajectory_step"].astype(int)
            plot_data["distance_kind"] = plot_data["distance_kind"].map(labels)
            sns.violinplot(
                data=plot_data,
                x="trajectory_step",
                y="euclidean_distance",
                hue="distance_kind",
                palette=self.theme.palette,
                cut=0,
                density_norm="width",
                ax=axes,
            )
            movement_label = "moving" if sorbes_moved else "stationary"
            axes.set(
                title=(
                    "Euclidean latent distance vs. local-enumeration iteration\n"
                    f"direction_significance_threshold={grid_value:g}, "
                    f"SORBES {movement_label} seeds"
                ),
                xlabel="Local-enumeration iteration",
                ylabel="Euclidean latent distance",
            )
            axes.legend(title="", loc="upper left", fontsize="small")
            figure.tight_layout()
            figures[(sorbes_moved, grid_value)] = axes
        return figures

    def sorbes_trajectory_profile(self, result: AnalysisResult):
        """Plot each trajectory's latent drift from its origin, by iteration.

        One thin line per trajectory (``units="trajectory_id"``, unaggregated),
        colored by seed sequence, so stationary and moving seeds are visible
        as flat-vs-rising line bundles at a glance.

        :param result: Output of ``LocalityAnalysis.sorbes_trajectory_profile``.
        """
        data = result.data
        if data.empty:
            raise ValueError("Cannot visualize an empty SORBES trajectory profile")
        _, axes = self._axes()
        sns.lineplot(
            data=data,
            x="trajectory_step",
            y="euclidean_distance",
            hue="seed_sequence",
            units="trajectory_id",
            estimator=None,
            alpha=0.6,
            palette=self.theme.palette,
            ax=axes,
        )
        axes.set(
            title="SORBES latent drift from trajectory origin",
            xlabel="Local-enumeration iteration",
            ylabel="Euclidean latent distance",
        )
        axes.legend(title="seed", loc="upper left", fontsize="small")
        return axes
