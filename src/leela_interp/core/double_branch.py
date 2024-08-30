import string
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from leela_interp import LeelaBoard
from leela_interp.tools import figure_helpers as fh
import pickle
import random
import os
import math

import chess
import iceberg as ice
from matplotlib.patches import Patch
import torch
from leela_interp import Lc0Model, Lc0sight, LeelaBoard
from leela_interp.core.iceberg_board import palette
from leela_interp.core.alternative_moves import check_if_double_game, check_if_double_game_fast
from leela_interp.core.effect_study import EffectStudy
from leela_interp.tools import figure_helpers as fh
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)
from scipy.stats import binned_statistic
from tqdm import tqdm
from leela_interp.core.general_study import GeneralStudy

class DoubleBranchStudy(GeneralStudy):
    def __init__(self, *args, load_sets=True, load_all=True, **kwargs):
        super().__init__(*args, **kwargs, load_all=load_all)
        if load_all:
            self.check_contains_double_branch()
        if load_sets:
            self.load_puzzle_sets()
            self.load_effect_sets()
            self.load_attention_sets()

    def check_contains_double_branch(self):
        # Check if "branch_1" and "branch_2" columns exist in self.puzzles
        required_columns = ["branch_1", "branch_2"]
        missing_columns = [col for col in required_columns if col not in self.puzzles.columns]
        
        if missing_columns:
            raise ValueError(f"The following required columns are missing from self.puzzles: {', '.join(missing_columns)}")

    @staticmethod
    def check_if_double_branch(model, puzzles_original, must_include_pv=True, end: int = 3, min_prob: float | list[float] = 0.1):

        puzzles = puzzles_original.copy()

        data = []
        for _, x in tqdm(puzzles.iterrows(), total=len(puzzles), desc="Checking double games"):
            data.append(check_if_double_game_fast(model, x, end=end, min_prob=min_prob))
        alt_puzzle_movesets = [x for x in data if x]
        mask = np.array([bool(x) for x in data])
        puzzles = puzzles[mask]
        print(f"Puzzles that are double games: {puzzles.shape[0]}")

        #print(alt_puzzle_movesets)

        def get_moves_probs(depth, end, moves, is_branch_1, branch_1_moves, branch_1_probs, branch_2_moves, branch_2_probs):

            if depth == end:
                return

            for key, value in moves.items():
                if key == 'prob':
                    if is_branch_1:
                        branch_1_probs.append(value)
                    else:
                        branch_2_probs.append(value)
                    continue
                elif is_branch_1:
                    branch_1_moves.append(key)
                else:
                    branch_2_moves.append(key)

                get_moves_probs(depth+1, end, value, is_branch_1, branch_1_moves, branch_1_probs, branch_2_moves, branch_2_probs)

        main_moves = []
        main_probs = []
        has_pv = np.empty(len(alt_puzzle_movesets), dtype=bool)
        has_pv[:] = True
        for i, (correct_moves, total_moveset) in enumerate(alt_puzzle_movesets):
            zeroth_move = list(total_moveset)[0]
            first_round_moves = total_moveset[zeroth_move]
            pv_length = len(correct_moves[1:])
            
            branch_1_moves, branch_2_moves = [], []
            branch_1_probs, branch_2_probs = [], []
            for first_move, second_round_moves in first_round_moves.items():
                if first_move == 'prob':
                    continue
                elif branch_1_moves == []:
                    branch_1_moves.append(first_move)
                    is_branch_1 = True
                else:
                    branch_2_moves.append(first_move)
                    is_branch_1 = False

                get_moves_probs(0, pv_length, second_round_moves, is_branch_1, branch_1_moves, branch_1_probs, branch_2_moves, branch_2_probs)

            #print(branch_1_moves, branch_2_moves)
            if must_include_pv:
                if not (correct_moves[1:] in [branch_1_moves, branch_2_moves]):
                    has_pv[i] = False
                    continue

            main_moves.append([branch_1_moves, branch_2_moves])
            main_probs.append([branch_1_probs, branch_2_probs])

        #print(main_moves)
        puzzles = puzzles[has_pv]
        print(f"Puzzles with PV: {puzzles.shape[0]}")

        puzzles["branch_1"] = [main_moves[i][0] for i in range(len(main_moves))]
        puzzles["branch_2"] = [main_moves[i][1] for i in range(len(main_moves))]
        puzzles["branch_1_probs"] = [main_probs[i][0] for i in range(len(main_probs))]
        puzzles["branch_2_probs"] = [main_probs[i][1] for i in range(len(main_probs))]
        
        mask = np.array([len(set([b1[0][2:4], b1[2][2:4], b2[0][2:4], b2[2][2:4]])) == 4 for b1, b2 in main_moves])
        puzzles = puzzles[mask]
        print(f"Puzzles with 4 distinct moves: {puzzles.shape[0]}")
        
        return puzzles

    def find_result_sets(self, include_starting=False, n_examples=100):
        all_results = DoubleBranchStudy.get_possibility_indices(self.puzzles, include_starting=include_starting)
        result_sets = {k: v for k, v in all_results.items() if len(v) >= n_examples}
        result_sets = {k: v for k, v in sorted(result_sets.items(), key=lambda item: len(item[1]), reverse=True)}
        result_masks = np.zeros((len(result_sets), len(self.puzzles)), dtype=bool)
        for i, (_, idx_list) in enumerate(result_sets.items()):
            result_masks[i, idx_list] = True
        self.result_sets = result_sets
        self.result_masks = result_masks
        self.include_starting = include_starting
        self.n_examples = n_examples
    
    def export_puzzle_set_info(self, tag='b'):
        super().export_puzzle_set_info(tag=tag)
        

    def load_results(self):
        if self.alt_puzzles is not None:
            self.results = EffectStudy.get_possibility_indices_alt(self.main_moves)
            self.results = {k: list(np.arange(len(self.puzzles))[self.puzzles.index.isin(self.alt_puzzles.iloc[v].index)]) for k, v in self.results.items()}
        else:
            self.results = EffectStudy.get_possibility_indices(self.puzzles, include_starting=self.include_starting)
        good_results = {k: v for k, v in self.results.items() if len(v) > self.n_examples}
        self.good_results = {k: v for k, v in sorted(good_results.items(), key=lambda item: len(item[1]), reverse=True)}
        self.good_mask = np.zeros((len(self.good_results), len(self.puzzles)), dtype=bool)
        for i, (_, idx_list) in enumerate(self.good_results.items()):
            self.good_mask[i, idx_list] = True

    @staticmethod
    def map_to_possibility(branch_1_squares, branch_2_squares):
        mapping = {}
        result = []
        counter = 1

        for square in branch_1_squares:
            if square not in mapping:
                mapping[square] = str(counter)  
                counter += 1
            result.append(mapping[square])
        for square in branch_2_squares:
            if square not in mapping:
                mapping[square] = str(counter)
                counter += 1
            result.append(mapping[square])

        return result

    @staticmethod
    def get_possibility_indices(puzzles, include_starting=False):
        if include_starting:
            raise NotImplementedError("Include starting not implemented")

        possibilities = []
        indices = {}

        for i, (idx, puzzle) in enumerate(puzzles.iterrows()):
            branch_1_moves = puzzle.branch_1
            branch_2_moves = puzzle.branch_2
            branch_1_squares = [move[2:4] for move in branch_1_moves]
            branch_2_squares = [move[2:4] for move in branch_2_moves]
            possibility = ''.join(DoubleBranchStudy.map_to_possibility(branch_1_squares, branch_2_squares))
            if possibility not in indices:
                indices[possibility] = []
            indices[possibility].append(i)
            possibilities.append(possibility)
        
        return indices
    
    def get_effect_set_data(self, tag, possibility, verbose=False):
        effects = self.effect_sets[tag][possibility]
        include_branch = tag == "b"
        max_length = len(possibility) // (2 if include_branch else 1)

        candidate_effects = []
        follow_up_effects = {j: [] for j in range(2, max_length + 1)}
        starting_effects = {j: [] for j in range(1, max_length + 1)} if include_branch else {}
        
        patching_square_effects = []
        other_effects = []
        skipped = []
        non_skipped = []

        for i, (idx, puzzle) in enumerate(self.puzzle_sets[tag][possibility].iterrows()):
            board = LeelaBoard.from_puzzle(puzzle)
            corrupted_board = LeelaBoard.from_fen(puzzle.corrupted_fen)
            pv = puzzle.branch_1
            pv_2 = puzzle.branch_2

            patching_squares = self.get_patching_squares(board, corrupted_board)
            movs = [pv[j][2:4] for j in range(len(pv))]
            starts = [pv_2[j][2:4] for j in range(len(pv_2))] if include_branch else []

            candidate_squares = [movs[0]]
            follow_up_squares = {j: [movs[j-1]] for j in range(2, max_length + 1)}
            starting_squares = {j: [starts[j-1]] for j in range(1, max_length + 1)} if include_branch else {}

            if self.should_skip(patching_squares, movs, starts):
                skipped.append(idx)
                continue

            non_skipped.append(idx)
            self.process_effects(effects[i], board, candidate_squares, follow_up_squares, starting_squares, 
                                 patching_squares, candidate_effects, follow_up_effects, starting_effects, 
                                 patching_square_effects, other_effects, include_branch)

        if verbose:
            self.print_verbose_info(len(skipped), len(self.puzzle_sets[tag][possibility]))

        return self.prepare_effects_data(candidate_effects, follow_up_effects, starting_effects, 
                                         patching_square_effects, other_effects, max_length, 
                                         include_branch, verbose), non_skipped

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
                self.print_effects(starting_effects, "B square")

        effects_data = [
            {"effects": patching_square_effects, "name": "Corrupted"},
            {"effects": other_effects, "name": "Other"},
            {"effects": candidate_effects, "name": "Move 1"},
        ]
        effects_data.extend({"effects": effects, "name": f"Move {j}"} for j, effects in follow_up_effects.items())
        if include_starting:
            effects_data.extend({"effects": effects, "name": f"Move {j}B"} for j, effects in starting_effects.items())
        
        return effects_data

    def plot_residual_effects(self, tag, possibility, filename=None, plot_ci=True, ax=None, row_col=None, log=False, clean_plot=False):
        ax_init = None if ax is None else ax

        branch_1_probs = np.vstack(self.puzzle_sets[tag][possibility].branch_1_probs.to_numpy())
        branch_2_probs = np.vstack(self.puzzle_sets[tag][possibility].branch_2_probs.to_numpy())
        branch_probs = np.vstack((branch_1_probs[:, 0], branch_2_probs[:, 0])).T
        sorted_indices = np.argsort(branch_probs[:, 0] - branch_probs[:, 1])

        effects_data, nonskipped = self.get_effect_set_data(tag, possibility)
        # Find the new indices corresponding to sorted_indices in the nonskipped subset
        nonskipped_indices = [self.puzzle_sets[tag][possibility].index.get_loc(idx) for idx in nonskipped]
        #print(sorted_indices, nonskipped_indices)
        new_sorted_indices = []
        for idx in sorted_indices:
            if idx in nonskipped_indices:
                new_idx = nonskipped_indices.index(idx)
                new_sorted_indices.append(new_idx)
        
        # Use new_sorted_indices instead of sorted_indices for indexing effects
        sorted_indices = new_sorted_indices[:10]
        #print(sorted_indices)
        max_length = len(possibility) // (2 if tag != "n" else 1)

        fh.set()

        line_styles = ["-"] * 2 + ["-", "--"] * ((max_length - 1) // 2) + ["-"]

        colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[:len(line_styles)]
        layers = list(range(15))

        line_styles += ["-", "--"] * ((max_length - 1) // 2) + ["-"]
        colors += plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[len(line_styles) - len(colors) + 5:]
        #line_styles += ["-.", ":"] * ((max_length - 1) // 2) + ["-."]
        #colors += plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[:len(line_styles) - len(colors)]

        # Create plots using matplotlib
        if ax is None:
            fig, ax = plt.subplots()
            if not clean_plot:
                fig.set_figwidth(6)
                fig.set_figheight(4)
            else:
                fig.set_figwidth(3)
                fig.set_figheight(2)

        for i, effect_data in enumerate(effects_data):
            if "B" in effect_data["name"]:
                effects = effect_data["effects"]
                len_effects = len(effects)
                effects = effects[sorted_indices]
            else:
                effects = effect_data["effects"]
                len_effects = len(effects)
                effects = effects[sorted_indices]
            #effects = np.abs(effect_data["effects"])[sorted_indices]
            if i==0:    
                print(possibility, len_effects, len(effects))
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
        _, y_max = ax.get_ylim()
        ax.set_xlim(0, 14)
        #ax.set_ylim(1e-2, 2)
        ax.set_ylim(-2, 2)
        if log:
            ax.set_yscale("symlog", linthresh=1e-2)
        if row_col is not None:
            ax.legend(loc="upper left", title=f"Set {row_col[2]}")
        else:
            ax.legend(loc="upper left")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.set_facecolor(fh.PLOT_FACE_COLOR)

        if filename is not None and ax_init is None:
            fh.save('figures/' + filename, fig)

        if ax is None:
            plt.show()

    def plot_residual_effects_grid(self, tag, possibilities, n_cols=4, filename=None, log=False):
        n_plots = len(possibilities)
        n_rows = math.ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2*n_rows), sharex=True, sharey=True)
        
        for idx, possibility in enumerate(possibilities):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            try:
                self.plot_residual_effects(
                    tag=tag,
                    possibility=possibility,
                    ax=ax,
                    row_col=(row == n_rows - 1, col == 0, possibility),
                    plot_ci=True,
                    filename=None,
                    log=log
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
        ax.set_title(f"Attention for position {possibility}")

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
                ax.set_title(f"{possibility}")
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