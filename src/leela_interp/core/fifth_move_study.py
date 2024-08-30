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
import re
from leela_interp.core.leela_board import LeelaBoard
import chess
from leela_interp.core.iceberg_board import palette
import iceberg as ice


class FifthMoveStudy(GeneralStudy):
    def __init__(self, *args, load_all=True, **kwargs):
        super().__init__(*args, **kwargs, load_all=load_all)
        self.load_puzzle_sets()
        self.load_effect_sets()
        self.load_attention_sets()

    def get_effect_set_data(self, tag, possibility, verbose=False):
        effects = self.effect_sets[tag][possibility]
        include_starting = tag == "s"
        max_length = len(possibility) // (2 if include_starting else 1)

        candidate_effects = []
        follow_up_effects = {j: [] for j in range(2, max_length + 1)}
        starting_effects = {j: [] for j in range(1, max_length + 1)} if include_starting else {}
        
        patching_square_effects = []
        other_effects = []
        skipped = []
        non_skipped = []

        for i, (idx, puzzle) in enumerate(self.puzzle_sets[tag][possibility].iterrows()):
            board = LeelaBoard.from_puzzle(puzzle)
            corrupted_board = LeelaBoard.from_fen(puzzle.corrupted_fen)
            pv = puzzle.principal_variation

            patching_squares = self.get_patching_squares(board, corrupted_board)
            movs = [pv[j][2:4] for j in range(len(pv))]
            starts = [pv[j][0:2] for j in range(len(pv))] if include_starting else []

            candidate_squares = [movs[0]]
            follow_up_squares = {j: [movs[j-1]] for j in range(2, max_length + 1)}
            starting_squares = {j: [starts[j-1]] for j in range(1, max_length + 1)} if include_starting else {}

            if self.should_skip(patching_squares, movs, starts):
                skipped.append(idx)
                continue

            non_skipped.append(idx)
            self.process_effects(effects[i], board, candidate_squares, follow_up_squares, starting_squares, 
                                 patching_squares, candidate_effects, follow_up_effects, starting_effects, 
                                 patching_square_effects, other_effects, include_starting)

        if verbose:
            self.print_verbose_info(len(skipped), len(self.puzzle_sets[tag][possibility]))

        return self.prepare_effects_data(candidate_effects, follow_up_effects, starting_effects, 
                                         patching_square_effects, other_effects, max_length, 
                                         include_starting, verbose), non_skipped

    def prepare_effects_data(self, candidate_effects, follow_up_effects, starting_effects, 
                             patching_square_effects, other_effects, max_length, include_starting, verbose):
        candidate_effects = np.stack(candidate_effects)
        follow_up_effects = {j: np.stack(effects) if effects else np.array([]) for j, effects in follow_up_effects.items()}
        if include_starting:
            starting_effects = {j: np.stack(effects) if effects else np.array([]) for j, effects in starting_effects.items()}
        patching_square_effects = np.stack(patching_square_effects)
        other_effects = np.stack(other_effects)

        if verbose:
            print(f"Patching: {len(patching_square_effects)}, Other: {len(other_effects)}")
            self.print_effects({1: candidate_effects, **follow_up_effects}, "End square")
            if include_starting:
                self.print_effects(starting_effects, "Start square")

        effects_data = [
            {"effects": patching_square_effects, "name": "Corrupted"},
            {"effects": other_effects, "name": "Other"},
            {"effects": candidate_effects, "name": "Move 1"},
        ]
        effects_data.extend({"effects": effects, "name": f"Move {j}"} for j, effects in follow_up_effects.items())
        if include_starting:
            effects_data.extend({"effects": effects, "name": f"Move {j}S"} for j, effects in starting_effects.items())
        
        return effects_data

    def plot_residual_effects(self, tag, possibility, filename=None, plot_ci=True, ax=None, row_col=None, log=False, clean_plot=False, y_min=1e-2, y_max=8):
        ax_init = None if ax is None else ax

        effects_data, _ = self.get_effect_set_data(tag, possibility)
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
            effects = effect_data["effects"]
            if len(effects) == 0:
                continue
            
            mean_effects = np.mean(effects, axis=0)

            ax.plot(
                layers,
                mean_effects,
                label=effect_data["name"],
                color=colors[i],
                linestyle=line_styles[i],
                linewidth= 3 * fh.LINE_WIDTH,
            )
            if plot_ci:
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
                ax.legend(loc="upper left", title=f"Set {row_col[2]}")
            else:
                ax.legend(loc="upper left", title=f"Set {row_col[2]}", fontsize="small")
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

    def plot_residual_effects_grid(self, tag, possibilities=None, n_cols=4, filename=None, log=False, y_min=1e-2, y_max=8, plot_ci=True):
        if possibilities is None:
            multiple_tags = True
            cases = tag.copy()
        else:
            multiple_tags = False
        n_plots = len(possibilities if not multiple_tags else cases)
        n_rows = math.ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2*n_rows), sharex=True, sharey=True)
        
        for idx, (tag, possibility) in enumerate(cases if multiple_tags else zip([tag]*len(possibilities), possibilities)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            try:
                self.plot_residual_effects(
                    tag=tag,
                    possibility=possibility,
                    ax=ax,
                    row_col=(row == n_rows - 1, col == 0, possibility),
                    plot_ci=plot_ci,
                    filename=None,
                    log=log,
                    y_min=y_min,
                    y_max=y_max
                )
            except ValueError:
                ax.set_visible(False)
        
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save('figures/' + filename, fig)

    def plot_attention(self, tag, possibility, vmax=0.5, filename=None):
        # Create a single subplot
        fig, ax = plt.subplots(figsize=(3, 4))

        # Create a matrix to hold the data
        n_layers = 15
        n_heads = 24
        data = np.full((n_heads, n_layers), '', dtype=object)

        # Fill the matrix with values for each piece type
        for layer, head in knight_heads:
            data[head, layer] = 'K'
        for layer, head in bishop_heads:
            data[head, layer] = 'B'
        for layer, head in rook_heads:
            data[head, layer] = 'R'

        # Get the effects for the specific position
        effects = self.attention_sets[tag][possibility]

        # Calculate the effects for the single position
        position_effects = np.abs(effects[0].cpu().numpy())  # Use [0] to get the first (and only) item

        # Plot the heatmap
        sns.heatmap(position_effects.T, cmap=fh.EFFECTS_CMAP_2, ax=ax, cbar=True, vmin=0, vmax=vmax)
        ax.set_title(f"Set {possibility}")

        # Add text annotations
        for i in range(n_heads):
            for j in range(n_layers):
                if data[i, j]:
                    text_color = 'blue' if data[i, j] == 'K' else 'green' if data[i, j] == 'B' else 'red'
                    ax.text(j+0.5, i+0.5, data[i, j], ha='center', va='center', color=text_color, fontsize='xx-small')

        ax.set_ylabel("Head")
        ax.set_xlabel("Layer")

        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save('figures/' + filename, fig)

    def plot_attention_grid(self, tag, possibilities, n_cols=4, vmax=0.5, filename=None):

        n_plots = len(possibilities)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 3*n_rows), sharex=True, sharey=True)

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
            ax = axes[row, col] if n_rows > 1 else axes[col]

            effects = self.attention_sets[tag][possibility]

            mean_effects = -effects.mean(dim=0).cpu().numpy()

            try:
                sns.heatmap(mean_effects.T, cmap=fh.EFFECTS_CMAP_2, ax=ax, cbar=False, vmin=0, vmax=vmax)
                ax.set_title(f"Set {possibility}")
                for i in range(n_heads):
                    for j in range(n_layers):
                        if data[i, j]:
                            text_color = {'K': 'blue', 'B': 'green', 'R': 'red'}[data[i, j]]
                            ax.text(j+0.5, i+0.5, data[i, j], ha='center', va='center', color=text_color, fontsize='xx-small')
                if col == 0:
                    ax.set_ylabel("Head")
                if row == n_rows - 1:
                    ax.set_xlabel("Layer")
            except ValueError:
                ax.axis('off')

        for idx in range(n_plots, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            if n_rows > 1:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')

        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save('figures/' + filename, fig)