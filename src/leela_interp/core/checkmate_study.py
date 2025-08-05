"""Checkmate study analysis module.

This module provides specialized analysis tools for studying checkmate patterns in chess games
using the Leela chess engine. It extends the FifthMoveStudy class to analyze residual effects
and attention patterns specifically in checkmate scenarios.

The module focuses on:
- Residual effect analysis for checkmate vs non-checkmate positions
- Attention pattern visualization for different checkmate scenarios
- Comparative analysis of mate-in-N puzzle themes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from leela_interp.tools import figure_helpers as fh
import pickle
import os
import math

from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)
from leela_interp.core.general_study import GeneralStudy
from leela_interp.core.fifth_move_study import FifthMoveStudy
import re
from leela_interp.core.leela_board import LeelaBoard
import chess
from leela_interp.core.iceberg_board import palette
import iceberg as ice


class CheckmateStudy(FifthMoveStudy):
    """Specialized study class for analyzing checkmate patterns and effects.
    
    This class extends FifthMoveStudy to provide specific analysis tools for
    checkmate scenarios, including residual effect plotting and attention
    pattern visualization for mate-in-N puzzles.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize CheckmateStudy with same parameters as FifthMoveStudy.
        
        Args:
            *args: Variable length argument list passed to parent class.
            **kwargs: Arbitrary keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)

    def plot_residual_effects(self, tag, possibility, filename=None, plot_ci=True, plot_std=False, ax=None, row_col=None, log=False, clean_plot=False, y_min=1e-2, y_max=8, mate=True):
        """Plot residual effects for checkmate vs non-checkmate positions.
        
        Creates line plots showing the mean residual effects across neural network layers,
        with separate analysis for mate-in-N puzzles vs non-mate positions.
        
        Args:
            tag (str): Dataset tag identifier (e.g., 's' for standard).
            possibility (str): Specific possibility/scenario identifier.
            filename (str, optional): Output filename for saving the plot.
            plot_ci (bool): Whether to plot confidence intervals (50% and 90%). Defaults to True.
            plot_std (bool): Whether to plot standard deviation instead of confidence intervals. Defaults to False.
            ax (matplotlib.axes.Axes, optional): Existing axes to plot on.
            row_col (tuple, optional): (is_bottom_row, is_left_col, title) for grid plots.
            log (bool): Whether to use logarithmic y-axis scaling. Defaults to False.
            clean_plot (bool): Whether to use clean plotting style. Defaults to False.
            y_min (float): Minimum y-axis value. Defaults to 1e-2.
            y_max (float): Maximum y-axis value. Defaults to 8.
            mate (bool): Whether to analyze mate positions (True) or non-mate (False).
        """
        ax_init = None if ax is None else ax

        effects_data, nonskipped = self.get_effect_set_data(tag, possibility)
        n_turns = (len(possibility)+1) // 2
        non_skipped_puzzles = self.puzzle_sets[tag][possibility].loc[nonskipped]
        mask = non_skipped_puzzles["Themes"].apply(lambda x: f"mateIn{n_turns}" in x)
        mask = mask.to_numpy()
        
        max_length = len(possibility) // (2 if tag == "s" else 1)

        fh.set()

        line_styles = ["-"] * 2 + ["-", "--"] * ((max_length - 1) // 2) + ["-"]

        colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[:len(line_styles)]
        layers = list(range(15))

        line_styles += ["-.", ":"] * ((max_length - 1) // 2) + ["-."]
        colors += plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[:len(line_styles) - len(colors)]

        # Create plots using matplotlib
        if ax is None:
            fig, ax = plt.subplots()
            if not clean_plot:
                fig.set_figwidth(5)
                fig.set_figheight(3)
            else:
                fig.set_figwidth(5)
                fig.set_figheight(3)

        for i, effect_data in enumerate(effects_data):
            if mate:
                effects = effect_data["effects"][mask]
            else:
                effects = effect_data["effects"][~mask]
            if len(effects) == 0:
                continue
        
            mean_effects = np.mean(effects, axis=0)
            std_effects = np.std(effects, axis=0)

            ax.plot(
                layers,
                mean_effects,
                label=effect_data["name"],
                color=colors[i],
                linestyle=line_styles[i],
                linewidth= 3 * fh.LINE_WIDTH,
            )
            if plot_std:
                # Plot standard deviation
                ax.fill_between(
                    layers,
                    mean_effects - std_effects,
                    mean_effects + std_effects,
                    color=colors[i],
                    alpha=0.3,
                )
            elif plot_ci:
                # Plot confidence intervals
                ci_50 = np.quantile(effects, [0.25, 0.75], axis=0)
                ci_90 = np.quantile(effects, [0.05, 0.95], axis=0)
                if not clean_plot:
                    ax.fill_between(
                        layers,
                        ci_90[0],
                        ci_90[1],
                        color=colors[i],
                        alpha=0.1,
                    )
                ax.fill_between(
                    layers,
                    ci_50[0],
                    ci_50[1],
                    color=colors[i],
                    alpha=0.3,
                )

        # ax.set_title("Patching effects on different squares by layer")
        if row_col is not None:
            #ax.set_title(f"Possibility: {row_col[2]}")
            if row_col[0]:
                ax.set_xlabel("Layer")
            if row_col[1]:
                ax.set_ylabel("Log odds reduction")
        else:
            ax.set_xlabel("Layer")
            ax.set_ylabel("Log odds reduction")
        ax.set_xlim(0, 14)
        ax.set_ylim(y_min, y_max)
        #ax.set_ylim(-8, 8)
        if log:
            ax.set_yscale("symlog", linthresh=1e-2)
        if row_col is not None:
            if len(possibility) < 7:
                ax.legend(loc="upper left", title=f"Set {'M' if mate else 'N'}{row_col[2]}")
            else:
                ax.legend(loc="upper left", title=f"Set {'M' if mate else 'N'}{row_col[2]}", fontsize="small")
        else:
            if len(possibility) < 7:
                ax.legend(loc="upper left")
            else:
                ax.legend(loc="upper left", fontsize="small")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.set_facecolor(fh.PLOT_FACE_COLOR)

        if filename is not None and ax_init is None:
            fh.save('figures/' + filename, fig)

        if ax is None:
            plt.show()

    def plot_residual_effects_grid(self, tag, possibilities=None, n_cols=4, filename=None, log=False, y_min=1e-2, y_max=8, plot_ci=True, plot_std=False):
        """Create a grid of residual effect plots for multiple possibilities.
        
        Generates a grid layout with paired plots (mate vs non-mate) for each
        possibility, allowing for comprehensive comparison across scenarios.
        
        Args:
            tag (str or list): Dataset tag(s). If list, possibilities is ignored.
            possibilities (list, optional): List of possibility identifiers to plot.
            n_cols (int): Number of columns in the grid layout. Defaults to 4.
            filename (str, optional): Output filename for saving the plot.
            log (bool): Whether to use logarithmic y-axis scaling. Defaults to False.
            y_min (float): Minimum y-axis value. Defaults to 1e-2.
            y_max (float): Maximum y-axis value. Defaults to 8.
            plot_ci (bool): Whether to plot confidence intervals (50% and 90%). Defaults to True.
            plot_std (bool): Whether to plot standard deviation instead of confidence intervals. Defaults to False.
        """
        if possibilities is None:
            multiple_tags = True
            cases = tag.copy()
        else:
            multiple_tags = False
        n_plots = len(possibilities if not multiple_tags else cases)
        n_rows = math.ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(n_rows, 2*n_cols, figsize=(3*2*n_cols, 2*n_rows), sharex=True, sharey=True)
        
        for idx, (tag, possibility) in enumerate(cases if multiple_tags else zip([tag]*len(possibilities), possibilities)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, 2*col] if n_rows > 1 else axes[2*col]
            try:
                self.plot_residual_effects(
                    tag=tag,
                    possibility=possibility,
                    ax=ax,
                    row_col=(row == n_rows - 1, col == 0, possibility),
                    plot_ci=plot_ci,
                    plot_std=plot_std,
                    filename=None,
                    log=log,
                    y_min=y_min,
                    y_max=y_max,
                    mate=True
                )
            except ValueError:
                ax.set_visible(False)
            ax = axes[row, 2*col+1] if n_rows > 1 else axes[2*col+1]
            try:
                self.plot_residual_effects(
                    tag=tag,
                    possibility=possibility,
                    ax=ax,
                    row_col=(row == n_rows - 1, col == 0, possibility),
                    plot_ci=plot_ci,
                    plot_std=plot_std,
                    filename=None,
                    log=log,
                    y_min=y_min,
                    y_max=y_max,
                    mate=False
                )
            except ValueError:
                ax.set_visible(False)
        
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, 2*col].set_visible(False)
                axes[row, 2*col+1].set_visible(False)
            else:
                axes[2*col].set_visible(False)
                axes[2*col+1].set_visible(False)
        
        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save('figures/' + filename, fig)

    def plot_attention_grid(self, tag, possibilities, n_cols=4, vmax=0.5, filename=None):
        """Create a grid of attention heatmaps for checkmate analysis.
        
        Generates heatmaps showing attention patterns across neural network layers
        and heads, with overlaid piece movement indicators (Knight, Bishop, Rook).
        Creates paired plots for mate vs non-mate scenarios.
        
        Args:
            tag (str): Dataset tag identifier. Must be 'n' for this implementation.
            possibilities (list): List of possibility identifiers to analyze.
            n_cols (int): Number of columns in the grid layout. Defaults to 4.
            vmax (float): Maximum value for heatmap color scaling. Defaults to 0.5.
            filename (str, optional): Output filename for saving the plot.
            
        Raises:
            NotImplementedError: If tag is not 'n'.
        """
        if tag != "n":
            raise NotImplementedError("Only tag 'n' is supported for checkmate plots")

        n_plots = len(possibilities)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, 2*n_cols, figsize=(2*2*n_cols, 3*n_rows), sharex=True, sharey=True)

        n_layers, n_heads = 15, 24
        data = np.full((n_heads, n_layers), '', dtype=object)

        for layer, head in knight_heads:
            data[head, layer] = 'K'
        for layer, head in bishop_heads:
            data[head, layer] = 'B'
        for layer, head in rook_heads:
            data[head, layer] = 'R'

        for idx, possibility in enumerate(possibilities):
            row, col = divmod(idx, n_cols)

            n_turns = (len(possibility)+1) // 2
            mask_1 = self.puzzle_sets[tag][possibility]["Themes"].apply(lambda x: f"mateIn{n_turns}" in x)
            mask_2 = ~mask_1
            mask_1, mask_2 = mask_1.to_numpy(), mask_2.to_numpy()

            effects = self.attention_sets[tag][possibility]
            effects_1 = effects[mask_1]
            effects_2 = effects[mask_2]

            mean_effects_1 = -effects_1.mean(dim=0).cpu().numpy()
            mean_effects_2 = -effects_2.mean(dim=0).cpu().numpy()

            ax = axes[row, 2*col] if n_rows > 1 else axes[2*col]
            try:
                sns.heatmap(mean_effects_1.T, cmap=fh.EFFECTS_CMAP_2, ax=ax, cbar=False, vmin=0, vmax=vmax)
                #ax.set_title(f"{possibility}, Mate in {n_turns}, {mask_1.sum()}")
                ax.set_title(f"M{possibility}")
                for i in range(n_heads):
                    for j in range(n_layers):
                        if data[i, j]:
                            text_color = {'K': 'blue', 'B': 'green', 'R': 'red'}[data[i, j]]
                            ax.text(j+0.5, i+0.5, data[i, j], ha='center', va='center', color=text_color, fontsize='xx-small')
                if 2*col == 0:
                    ax.set_ylabel("Head")
                if row == n_rows - 1:
                    ax.set_xlabel("Layer")
            except ValueError:
                ax.axis('off')

            ax = axes[row, 2*col+1] if n_rows > 1 else axes[2*col+1]
            try:
                sns.heatmap(mean_effects_2.T, cmap=fh.EFFECTS_CMAP_2, ax=ax, cbar=False, vmin=0, vmax=vmax)
                #ax.set_title(f"{possibility}, No mate, {mask_2.sum()}")
                ax.set_title(f"N{possibility}")
                for i in range(n_heads):
                    for j in range(n_layers):
                        if data[i, j]:
                            text_color = {'K': 'blue', 'B': 'green', 'R': 'red'}[data[i, j]]
                            ax.text(j+0.5, i+0.5, data[i, j], ha='center', va='center', color=text_color, fontsize='xx-small')
                if 2*col+1 == 0:
                    ax.set_ylabel("Head")
                if row == n_rows - 1:
                    ax.set_xlabel("Layer")
            except ValueError:
                ax.axis('off')

        for idx in range(2*n_plots, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            if n_rows > 1:
                axes[row, 2*col].axis('off')
                axes[row, 2*col+1].axis('off')
            else:
                axes[2*col].axis('off')
                axes[2*col+1].axis('off')

        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save('figures/' + filename, fig)