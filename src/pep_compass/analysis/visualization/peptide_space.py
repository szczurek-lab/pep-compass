"""Visualizations for HydrAMP training-corpus group-locality diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pep_compass.analysis.result import AnalysisResult
from pep_compass.analysis.visualization.theme import PlotTheme


class PeptideSpaceVisualizer:
    """Render group_locality results using one shared theme."""

    def __init__(self, theme: PlotTheme) -> None:
        self.theme = theme

    def _axes(self):
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
        return figure, axes

    def _group_order(self, groups) -> list:
        """Order groups by descending size, so large/small groups are grouped visually."""
        return list(groups.value_counts().sort_values(ascending=False).index)

    def neighbor_distance(self, result: AnalysisResult):
        """Plot nearest-neighbor distance distribution, one violin per group.

        :param result: Output of ``group_locality`` -- ``data`` holds
            ``neighbor_distance`` rows (one per sampled member per neighbor).
        """
        data = result.data
        if data.empty:
            raise ValueError("Cannot visualize an empty neighbor-distance result")
        _, axes = self._axes()
        order = self._group_order(data["group"])
        sns.violinplot(
            data=data,
            x="group",
            y="neighbor_distance",
            order=order,
            palette=self.theme.palette,
            cut=0,
            density_norm="width",
            ax=axes,
        )
        axes.set(
            title=(
                f"Nearest-neighbor latent distance within group "
                f"(k={result.metadata.get('neighbor_count')})"
            ),
            xlabel="",
            ylabel="Euclidean latent distance",
        )
        axes.tick_params(axis="x", rotation=30)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
        return axes

    def centroid_distance(self, result: AnalysisResult):
        """Plot each member's distance to its group centroid, one violin per group.

        :param result: Output of ``group_locality`` -- reads
            ``diagnostics["centroid_distance"]``.
        """
        data = result.diagnostics.get("centroid_distance")
        if data is None or data.empty:
            raise ValueError("Cannot visualize an empty centroid-distance result")
        _, axes = self._axes()
        order = self._group_order(data["group"])
        sns.violinplot(
            data=data,
            x="group",
            y="centroid_distance",
            order=order,
            palette=self.theme.palette,
            cut=0,
            density_norm="width",
            ax=axes,
        )
        axes.set(
            title="Distance to group centroid (latent-space spread)",
            xlabel="",
            ylabel="Euclidean latent distance",
        )
        axes.tick_params(axis="x", rotation=30)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
        return axes

    def separation_heatmap(
        self, within_result: AnalysisResult, between_result: AnalysisResult
    ):
        """Plot median nearest-neighbor distance: diagonal = within group,
        off-diagonal = between groups.

        Directly answers whether groups are actually separated in latent
        space: if a row's off-diagonal cell is not clearly larger than that
        row's diagonal cell, the group is not separated from that other
        group -- a typical member has neighbors just as close in the other
        group as in its own.

        :param within_result: Output of ``group_locality``.
        :param between_result: Output of ``between_group_distance``.
        """
        within_median = within_result.data.groupby("group")["neighbor_distance"].median()
        between_median = (
            between_result.data.groupby(["group", "other_group"])["neighbor_distance"]
            .median()
            .unstack()
        )
        groups = sorted(within_median.index)
        matrix = between_median.reindex(index=groups, columns=groups)
        for group in groups:
            matrix.loc[group, group] = within_median[group]

        _, axes = self._axes()
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            ax=axes,
            cbar_kws={"label": "Median Euclidean latent distance"},
        )
        axes.set(
            title="Nearest-neighbor distance: within group (diagonal) vs. between groups",
            xlabel="neighbors found in group",
            ylabel="peptide sampled from group",
        )
        axes.tick_params(axis="x", rotation=45)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
        return axes

    def centroid_distance_heatmap(self, between_result: AnalysisResult):
        """Plot exact centroid-to-centroid distance, all ordered group pairs.

        Unlike :meth:`separation_heatmap` (nearest-neighbor based, subject to
        sampling), this uses one point per group -- the mean latent vector
        over ALL of its members -- and the exact distance between those
        points. Diagonal is 0 by definition (a group's centroid to itself).

        :param between_result: Output of ``between_group_distance`` --
            reads ``diagnostics["centroid_distance"]``.
        """
        data = between_result.diagnostics.get("centroid_distance")
        if data is None or data.empty:
            raise ValueError("Cannot visualize an empty centroid-distance result")
        matrix = data.pivot(index="group", columns="other_group", values="centroid_distance")
        groups = sorted(set(matrix.index) | set(matrix.columns))
        matrix = matrix.reindex(index=groups, columns=groups)
        for group in groups:
            matrix.loc[group, group] = 0.0

        _, axes = self._axes()
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            ax=axes,
            cbar_kws={"label": "Centroid-to-centroid Euclidean distance"},
        )
        axes.set(
            title="Distance between group centers (mean latent vector)",
            xlabel="",
            ylabel="",
        )
        axes.tick_params(axis="x", rotation=45)
        for label in axes.get_xticklabels():
            label.set_horizontalalignment("right")
        return axes

    def neighbor_distance_by_reference(
        self, within_result: AnalysisResult, between_result: AnalysisResult
    ):
        """One panel per group: full distance distribution, own group vs. every other.

        ``separation_heatmap`` only shows the median per pair -- this shows
        the whole spread behind it, so heavy overlap between a group and a
        seemingly "separated" neighbor is visible directly, not just implied
        by a close median.

        :param within_result: Output of ``group_locality``.
        :param between_result: Output of ``between_group_distance``, computed
            with the SAME ``neighbor_count`` as ``within_result`` -- otherwise
            the two are not comparable.
        """
        if within_result.metadata.get("neighbor_count") != between_result.metadata.get(
            "neighbor_count"
        ):
            raise ValueError(
                "within_result and between_result must share neighbor_count "
                f"({within_result.metadata.get('neighbor_count')} != "
                f"{between_result.metadata.get('neighbor_count')}) -- "
                "otherwise own-group and other-group distances are not comparable."
            )
        within = within_result.data.assign(other_group=within_result.data["group"])
        combined = pd.concat(
            [
                within[["group", "other_group", "neighbor_distance"]],
                between_result.data[["group", "other_group", "neighbor_distance"]],
            ],
            ignore_index=True,
        )
        groups = self._group_order(within_result.data["group"])
        ncols = 2
        nrows = -(-len(groups) // ncols)
        with sns.axes_style(self.theme.style), sns.plotting_context(self.theme.context):
            figure, axes_grid = plt.subplots(
                nrows,
                ncols,
                figsize=(self.theme.figure_size[0] * 2.2, self.theme.figure_size[1] * 0.9 * nrows),
                dpi=self.theme.dpi,
                squeeze=False,
            )
        for index, group in enumerate(groups):
            axes = axes_grid[index // ncols][index % ncols]
            subset = combined.loc[combined["group"] == group].copy()
            order = [group] + [other for other in groups if other != group]
            subset["kind"] = np.where(
                subset["other_group"] == group, "own group", "other group"
            )
            sns.violinplot(
                data=subset,
                x="other_group",
                y="neighbor_distance",
                order=order,
                hue="kind",
                hue_order=["own group", "other group"],
                dodge=False,
                palette=self.theme.palette,
                cut=0,
                density_norm="width",
                ax=axes,
            )
            axes.set(
                title=group,
                xlabel="",
                ylabel="distance" if index % ncols == 0 else "",
            )
            axes.tick_params(axis="x", rotation=60)
            for label in axes.get_xticklabels():
                label.set_horizontalalignment("right")
            axes.legend().remove()
        for index in range(len(groups), nrows * ncols):
            axes_grid[index // ncols][index % ncols].axis("off")
        handles, labels = axes_grid[0][0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper right", ncol=2)
        figure.suptitle(
            f"Nearest-neighbor distance: own group vs. every other group "
            f"(k={within_result.metadata.get('neighbor_count')})"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        return figure
