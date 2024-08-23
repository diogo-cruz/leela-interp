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
from leela_interp.core.alternative_moves import check_if_double_game
from leela_interp.tools import figure_helpers as fh
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)
from scipy.stats import binned_statistic
from tqdm import tqdm

class EffectStudy:
    def __init__(self, model, puzzlename='', device='cpu', include_starting=False, n_examples=100, alt_puzzles=False):
        self.model = model
        self.puzzlename = ("_" if puzzlename != "" else "") + puzzlename
        self.device = device
        self.include_starting = include_starting
        self.n_examples = n_examples
        self.load_puzzles()
        if alt_puzzles:
            self.load_alt_puzzles()
        else:
            self.alt_puzzles = None
        self.load_effects()
        self.load_ratings()
        self.load_results()
        self.load_attentions()
        self.apply_mask = True
        

    def load_puzzles(self):
        with open(f"interesting_puzzles{self.puzzlename}.pkl", "rb") as f:
            puzzles = pickle.load(f)
        self.puzzles = puzzles

    def load_alt_puzzles(self):

        diff_puzzles = self.puzzles.copy()

        print(f"Puzzles: {self.puzzles.shape[0]}")

        diff_puzzles = diff_puzzles[diff_puzzles["principal_variation"].apply(lambda x: len(x)) == 3]
        print(f"Puzzles with 3 moves in principal variation: {diff_puzzles.shape[0]}")

        diff_puzzles = diff_puzzles[diff_puzzles.principal_variation.apply(lambda x: x[0][2:4] != x[2][2:4])]
        print(f"Puzzles with different start and end squares in principal variation: {diff_puzzles.shape[0]}")

        # diff_puzzles = diff_puzzles[diff_puzzles["full_pv_probs"].apply(lambda x: 0.3 <= x[0] <= 0.5)]
        # print(f"Puzzles with full PV probabilities between 0.4 and 0.6: {diff_puzzles.shape[0]}")

        diff_puzzles = diff_puzzles[diff_puzzles["Themes"].apply(lambda x: "mateIn2" in x)]
        print(f"Puzzles that are mate in 2: {diff_puzzles.shape[0]}")

        data = []
        for _, x in tqdm(diff_puzzles.iterrows(), total=len(diff_puzzles), desc="Checking double games"):
            data.append(check_if_double_game(self.model, x))
        self.alt_puzzle_movesets = [x for x in data if x]
        mask = np.array([bool(x) for x in data])
        diff_puzzles = diff_puzzles[mask]
        print(f"Puzzles that are double games: {diff_puzzles.shape[0]}")

        #print(self.alt_puzzle_movesets)

        main_moves = []
        for correct_moves, total_moveset in self.alt_puzzle_movesets:
            zeroth_move = list(total_moveset)[0]
            first_round_moves = total_moveset[zeroth_move]
            
            correct_branch, incorrect_branch = [], []
            for first_move, second_round_moves in first_round_moves.items():
                if first_move == 'prob':
                    continue
                elif first_move == correct_moves[1]:
                    correct_branch.append(first_move)
                    is_correct = True
                else:
                    incorrect_branch.append(first_move)
                    is_correct = False

                for second_move, third_round_moves in second_round_moves.items():
                    if second_move == 'prob':
                        continue
                    elif is_correct:
                        correct_branch.append(second_move)
                    else:
                        incorrect_branch.append(second_move)

                    for third_move, fourth_round_moves in third_round_moves.items():
                        if third_move == 'prob':
                            continue
                        elif is_correct:
                            correct_branch.append(third_move)
                        else:
                            incorrect_branch.append(third_move)

            main_moves.append([correct_branch, incorrect_branch])

        
        #print(self.main_moves)

        mask = np.array([len(set([b1[0][2:4], b1[2][2:4], b2[0][2:4], b2[2][2:4]])) == 4 for b1, b2 in main_moves])
        diff_puzzles = diff_puzzles[mask]
        print(f"Puzzles with 4 distinct moves: {diff_puzzles.shape[0]}")
        
        self.main_moves = [main_moves[i] for i in np.where(mask)[0]]

        # diff_puzzles = diff_puzzles[diff_puzzles["principal_variation"].apply(lambda x: x[0][2:4]) != diff_puzzles["full_model_moves"].apply(lambda x: x[0][2:4])]
        # print(f"Puzzles with different first moves: {diff_puzzles.shape[0]}")
        
        # diff_puzzles = diff_puzzles[diff_puzzles["principal_variation"].apply(lambda x: len(x)) == 3]
        # print(f"Puzzles with 3 moves in principal variation: {diff_puzzles.shape[0]}")
        
        # diff_puzzles = diff_puzzles[diff_puzzles["principal_variation"].apply(lambda x: x[2][2:4]) != diff_puzzles["full_model_moves"].apply(lambda x: x[2][2:4])]
        # print(f"Puzzles with different third moves: {diff_puzzles.shape[0]}")
        
        # diff_puzzles = diff_puzzles[diff_puzzles["principal_variation"].apply(lambda x: x[0][2:4]) != diff_puzzles["principal_variation"].apply(lambda x: x[2][2:4])]
        # print(f"Puzzles with different start and end squares in principal variation: {diff_puzzles.shape[0]}")
        
        # diff_puzzles = diff_puzzles[diff_puzzles["full_model_moves"].apply(lambda x: x[0][2:4]) != diff_puzzles["full_model_moves"].apply(lambda x: x[2][2:4])]
        # print(f"Puzzles with different start and end squares in full model moves: {diff_puzzles.shape[0]}")
        
        self.alt_puzzles = diff_puzzles
        self.alt_mask = self.puzzles.index.isin(self.alt_puzzles.index)

    def load_effects(self):
        if os.path.exists(f"results/global_patching/interesting_puzzles{self.puzzlename}_residual_stream_results.pt"):
            self.all_effects = -torch.load(
                f"results/global_patching/interesting_puzzles{self.puzzlename}_residual_stream_results.pt",
                map_location=self.device
            )
        else:
            print("No residual stream results found.")

    def load_attentions(self):
        if os.path.exists(f"results/global_patching/interesting_puzzles{self.puzzlename}_attention_head_results.pt"):
            self.all_attentions = torch.load(
                f"results/global_patching/interesting_puzzles{self.puzzlename}_attention_head_results.pt",
                map_location=self.device
            )
        else:
            print("No attention head results found.")
        
    def load_ratings(self):
        self.puzzle_ratings = self.puzzles["Rating"].to_numpy()

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

    def export_puzzles(self, filename):
        with open(f"interesting_puzzles_{filename}.pkl", "wb") as f:
            pickle.dump(self.puzzles.iloc[self.good_results[filename]], f)

    def get_effects_data(self, mask=None, allowed_lengths=list(range(3, 17)), rating_range=(0, 5000), verbose=False):
        include_starting = self.include_starting
        include_alt = self.alt_puzzles is not None
        apply_mask = self.apply_mask
        if mask is None:
            mask = np.ones(len(self.puzzles), dtype=bool)

        effects = self.all_effects[mask]

        candidate_effects = []
        follow_up_effects = {}
        max_length = max(allowed_lengths)
        for j in range(2, max_length + 1):
            follow_up_effects[j] = []
        if include_starting:
            starting_effects = {}
            for j in range(1, max_length + 1):
                starting_effects[j] = []
        if include_alt:
            alt_effects = {}
            for j in range(1, max_length + 1):
                alt_effects[j] = []
        intersection_skips = 0
        patching_square_effects = []
        other_effects = []
        skipped = []
        non_skipped = []

        for i, (idx, puzzle) in enumerate(self.puzzles[mask].iterrows()):
            # Should never happen on hard puzzles
            #print(i, len(self.main_moves))
            # Get position of idx in self.alt_puzzles
            if self.alt_puzzles is not None:
                k = np.where(self.alt_puzzles.index == idx)[0][0]
                pv = self.main_moves[k][0]
                av = self.main_moves[k][1]
            else:
                pv = puzzle.principal_variation
                av = []
            #print(i, pv, av)
            if len(pv) not in allowed_lengths:# or "mateIn3" not in puzzle["Themes"]:
                skipped.append(idx)
                continue
            if not (rating_range[0] <= puzzle["Rating"] <= rating_range[1]):
                skipped.append(idx)
                continue
            # if puzzle.sparring_full_pv_probs[1] < 0.5:
            #     skipped.append(idx)
            #     continue
            board = LeelaBoard.from_puzzle(puzzle)
            corrupted_board = LeelaBoard.from_fen(puzzle.corrupted_fen)

            # Figure out which square(s) differ in the corrupted position
            patching_squares = []
            for square in chess.SQUARES:
                if board.pc_board.piece_at(square) != corrupted_board.pc_board.piece_at(
                    square
                ):
                    patching_squares.append(chess.SQUARE_NAMES[square])

            movs = [pv[j][2:4] for j in range(len(pv))]
            if include_starting:
                starts = [pv[j][0:2] for j in range(len(pv))]
            if include_alt:
                alt_moves = [av[j][2:4] for j in range(len(av))]
            if not apply_mask:
                if movs[0] == movs[2]:
                    skipped.append(idx)
                    continue

            candidate_squares = [movs[0]]
            follow_up_squares = {}
            for j in range(2, max_length + 1):
                follow_up_squares[j] = [movs[j-1]] if len(movs) >= j and ((movs[j-1] not in movs[:j-1]) or apply_mask) else []
                #follow_up_squares[j] = [movs[j-1]] if len(movs) >= j and movs[j-1] not in (movs[:j-1]+movs[j:]) else []

            if include_starting:
                starting_squares = {}
                for j in range(1, max_length + 1):
                    starting_squares[j] = [starts[j-1]] if len(starts) >= j and ((starts[j-1] not in starts[:j-1] + movs) or apply_mask) else []

            if include_alt:
                alt_squares = {}
                for j in range(1, max_length + 1):
                    alt_squares[j] = [alt_moves[j-1]] if len(alt_moves) >= j and ((alt_moves[j-1] not in alt_moves[:j-1]) or apply_mask) else []

            if set(patching_squares).intersection(set(movs + (starts if include_starting else []) + (alt_moves if include_alt else []))):
                skipped.append(idx)
                intersection_skips += 1
                continue

            non_skipped.append(idx)
            candidate_effects.append(
                effects[i, :, [board.sq2idx(square) for square in candidate_squares]]
                .amax(-1)
                .cpu()
                .numpy()
            )
            for j in range(2, max_length + 1):
                if len(follow_up_squares[j]) > 0:
                    follow_up_effects[j].append(
                        effects[i, :, [board.sq2idx(square) for square in follow_up_squares[j]]]
                        .amax(-1)
                        .cpu()
                        .numpy()
                    )
            if include_starting:
                for j in range(1, max_length + 1):
                    if len(starting_squares[j]) > 0:
                        starting_effects[j].append(
                                effects[i, :, [board.sq2idx(square) for square in starting_squares[j]]]
                                .amax(-1)
                                .cpu()
                                .numpy()
                            )
            if include_alt:
                for j in range(1, max_length + 1):
                    if len(alt_squares[j]) > 0:
                        alt_effects[j].append(
                            effects[i, :, [board.sq2idx(square) for square in alt_squares[j]]]
                            .amax(-1)
                            .cpu()
                            .numpy()
                        )
            patching_square_effects.append(
                effects[i, :, [board.sq2idx(square) for square in patching_squares]]
                .amax(-1)
                .cpu()
                .numpy()
            )
            if include_starting:
                covered_squares = set(candidate_squares + patching_squares + 
                                    sum([starting_squares[j] for j in range(1, max_length + 1)], []) +
                                    sum([follow_up_squares[j] for j in range(2, max_length + 1)], []))
            elif include_alt:
                covered_squares = set(candidate_squares + patching_squares + 
                                    sum([alt_squares[j] for j in range(1, max_length + 1)], []) +
                                    sum([follow_up_squares[j] for j in range(2, max_length + 1)], []))
            else:
                covered_squares = set(candidate_squares + patching_squares + sum([follow_up_squares[j] for j in range(2, max_length + 1)], []))
            other_effects.append(
                effects[
                    i,
                    :,
                    [idx for idx in range(64) if board.idx2sq(idx) not in covered_squares],
                ]
                .amax(-1)
                .cpu()
                .numpy()
            )

        if verbose:
            print(
                f"Skipped {len(skipped)} out of {mask.sum()} puzzles ({len(skipped)/mask.sum():.2%})"
            )
            print(f"Intersection skips: {intersection_skips} out of {mask.sum()} puzzles ({intersection_skips/mask.sum():.2%})")

        candidate_effects = np.stack(candidate_effects)
        for j in range(2, max_length + 1):
            if len(follow_up_effects[j]) > 0:
                follow_up_effects[j] = np.stack(follow_up_effects[j])
        if include_starting:
            for j in range(1, max_length + 1):
                if len(starting_effects[j]) > 0:
                    starting_effects[j] = np.stack(starting_effects[j])
        if include_alt:
            for j in range(1, max_length + 1):
                if len(alt_effects[j]) > 0:
                    alt_effects[j] = np.stack(alt_effects[j])
        patching_square_effects = np.stack(patching_square_effects)
        other_effects = np.stack(other_effects)
        if verbose:
            print(f"Patching: {len(patching_square_effects)}, Other: {len(other_effects)}")

        def print_effects(effects_dict, prefix):
            print(f"{prefix}::", end=" ")
            for i in range(1, max_length + 1):
                if i in effects_dict:
                    suffix = "st" if i == 1 else "nd" if i == 2 else "rd" if i == 3 else "th"
                    print(f"{i}{suffix}: {len(effects_dict[i])}", end=", ")
            print()

        if verbose:
            print_effects({1: candidate_effects, **{i: follow_up_effects[i] for i in range(2, max_length + 1)}}, "End square")
            if include_starting:
                print_effects(starting_effects, "Start square")
            if include_alt:
                print_effects(alt_effects, "Alt square")

        # Define lists for effects and their configurations
        effects_data = [
            {"effects": patching_square_effects, "name": "Patched"},
            {"effects": other_effects, "name": "Other"},
            {"effects": candidate_effects, "name": "1"},
        ]
        for j in range(2, max_length + 1):
            effects_data.append({"effects": follow_up_effects[j], "name": f"{j}"})
        if include_starting:
            for j in range(1, max_length + 1):
                effects_data.append({"effects": starting_effects[j], "name": f"{j}S"})
        if include_alt:
            for j in range(1, max_length + 1):
                effects_data.append({"effects": alt_effects[j], "name": f"{j}A"})
        return effects_data, non_skipped

    def plot_rating_histogram(self, filename=None):

        fig = plt.figure()
        plt.hist(self.puzzle_ratings, bins=30)
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.title("Histogram of Puzzle Ratings")
        plt.show()
        
        if filename is not None:
            fh.save(filename, fig)

    def plot_examples(self, mask=None, n=5):
        if mask is None:
            mask = np.ones(len(self.puzzles), dtype=bool)

        effects = self.all_effects[mask]

        plots = []

        # Don't plot all the layers, it's too much
        layers = [0, 6, 8, 10, 12, 14]

        for i in range(n):
            puzzle = self.puzzles[mask].iloc[i]
            # if "mateIn3" not in puzzle["Themes"]:
            #     continue
            # else:
            print(i, puzzle.principal_variation, puzzle.full_pv_probs)
            #print(i, puzzle.full_model_moves, puzzle.sparring_full_pv_probs)
            board = LeelaBoard.from_puzzle(puzzle)
            colormap_values, mappable = palette(
                effects[i][layers].cpu().numpy().ravel(),
                cmap="bwr",
                zero_center=True,
            )
            colormap_values = [
                colormap_values[j : j + 64] for j in range(0, 64 * len(layers), 64)
            ]
            new_plots = []
            for j, layer in enumerate(layers):
                max_effect_idx = effects[i, layer].abs().argmax()
                max_effect = effects[i, layer, max_effect_idx].item()
                new_plots.append(
                    board.plot(
                        heatmap=colormap_values[j],
                        caption=f"L{layer}, max log odds reduction: {max_effect:.2f}",
                    )
                )

            plots.append(ice.Arrange(new_plots, gap=10))

        return ice.Arrange(plots, gap=10, arrange_direction=ice.Arrange.Direction.VERTICAL)

    def plot_residual_effects(self, mask=None, save_path=None, allowed_lengths=[3, 4, 5, 6, 7], apply_mask=True, rating_range=(0, 3000), plot_ci=True, ax=None, row_col=None, log=False):
        if mask is None:
            mask = np.ones(len(self.puzzles), dtype=bool)

        effects_data, _ = self.get_effects_data(mask, allowed_lengths, rating_range)

        fh.set()

        line_styles = ["-"] * 2 + ["-", "--"] * ((allowed_lengths[-1] - 1) // 2) + ["-"]

        colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[:len(line_styles)]
        layers = list(range(15))

        if self.alt_puzzles is not None:
            line_styles += ["-", "--"] * ((allowed_lengths[-1] - 1) // 2) + ["-"]
            colors += plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[len(line_styles) - len(colors) + 5:]
        else:
            line_styles += ["-.", ":"] * ((allowed_lengths[-1] - 1) // 2) + ["-."]
            colors += plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[:len(line_styles) - len(colors)]

        # Create plots using matplotlib
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_figwidth(10)
            fig.set_figheight(5)

        #print(len(line_styles), len(effects_data))

        for i, effect_data in enumerate(effects_data):
            effects = effect_data["effects"]
            if len(effects) == 0:
                continue
            
            if self.alt_puzzles is not None:
                effects = np.abs(effects)
                #pass
            mean_effects = np.mean(effects, axis=0)
            
            # Calculate confidence intervals
            if plot_ci:
                ci_50 = np.quantile(effects, [0.25, 0.75], axis=0)
                ci_70 = np.quantile(effects, [0.15, 0.85], axis=0)
                ci_90 = np.quantile(effects, [0.05, 0.95], axis=0)
                ci_100 = np.quantile(effects, [0, 1], axis=0)

            ax.plot(
                layers,
                mean_effects,
                label=effect_data["name"],
                color=colors[i],
                linestyle=line_styles[i],
                linewidth= 3 * fh.LINE_WIDTH,
            )
            if plot_ci:
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
        #ax.set_ylim(1e-2, 8)
        ax.set_ylim(-8, 8)
        if log:
            ax.set_yscale("symlog", linthresh=1e-2)
        if row_col is not None:
            ax.legend(loc="upper left", title=f"Set {row_col[2]}")
        else:
            ax.legend(loc="upper left")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.set_facecolor(fh.PLOT_FACE_COLOR)

        if save_path is not None:
            fh.save(save_path, fig)

        if ax is None:
            plt.show()

    def allowed_possibilities_mask(self, allowed_lengths=[3, 4, 5, 6, 7]):
        mask = np.zeros(len(self.good_results), dtype=bool)
        for idx, possibility in enumerate(self.good_results):
            n_moves = len(possibility) // (2 if self.include_starting or self.alt_puzzles is not None else 1)
            if n_moves in allowed_lengths:
                mask[idx] = True
        return mask

    def plot_residual_effects_grid(self, n_cols=4, allowed_lengths=[3, 4, 5, 6, 7], rating_range=(0, 5000), filename=None, log=False):
        possibilities_mask = self.allowed_possibilities_mask(allowed_lengths)
        n_plots = np.sum(possibilities_mask)
        n_rows = math.ceil(n_plots / n_cols)
        #print(n_plots, n_rows, n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2*n_rows), sharex=True, sharey=True)
        #fig.suptitle("Residual Effects Grid", fontsize=16)
        
        masked_good_results = [good_result for good_result, possibility_mask in zip(self.good_results, possibilities_mask) if possibility_mask]
        if self.alt_puzzles is not None:
            masked_good_mask = [good_mask & self.alt_mask for good_mask, possibility_mask in zip(self.good_mask, possibilities_mask) if possibility_mask]
        else:
            masked_good_mask = [good_mask for good_mask, possibility_mask in zip(self.good_mask, possibilities_mask) if possibility_mask]
        for idx, (possibility, mask) in enumerate(zip(masked_good_results, masked_good_mask)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            print(sum(mask))
            try:
                self.plot_residual_effects(
                    mask=mask, 
                    allowed_lengths=allowed_lengths, 
                    rating_range=rating_range,
                    ax=ax,
                    row_col=(True if row == n_rows - 1 else False, True if col == 0 else False, possibility),
                    plot_ci=True,
                    save_path=None,
                    log=log
                )
            except ValueError:
                ax.set_visible(False)
        
        # Hide any unused subplots
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save(filename, fig)
        

    def plot_rating(self, mask=None, save_path=None, allowed_lengths=[3, 4, 5, 6, 7]):
        if mask is None:
            mask = np.ones(len(self.puzzles), dtype=bool)

        effects_data, non_skipped = self.get_effects_data(mask, allowed_lengths)
        relevant_ratings = self.puzzles.loc[non_skipped, "Rating"]

        fh.set()

        colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
        layers = list(range(15))

        # Create plots using matplotlib
        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(5)

        line_styles = ["-"] * 2 + ["-", "--"] * ((allowed_lengths[-1] - 1) // 2) + ["-"]
        line_styles += ["-.", ":"] * ((allowed_lengths[-1] - 1) // 2)

        for i, effect_data in enumerate(effects_data):
            effects = effect_data["effects"]
            if len(effects) == 0:
                continue

            # Group the ratings into bins and calculate the average and std per bin
            bin_means, bin_edges, binnumber = binned_statistic(
                relevant_ratings, np.max(effects, axis=1), statistic='mean', bins=10
            )
            bin_std, _, _ = binned_statistic(
                relevant_ratings, np.max(effects, axis=1), statistic='std', bins=10
            )
            
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            ax.errorbar(
                bin_centers,
                bin_means,
                yerr=bin_std,
                label=effect_data["name"],
                fmt='o-',
                color=colors[i]
            )

        # ax.set_title("Patching effects on different squares by layer")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Log odds reduction of correct move")
        _, y_max = ax.get_ylim()
        #ax.set_ylim(1e-2, 8)
        #ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.set_facecolor(fh.PLOT_FACE_COLOR)

        if save_path is not None:
            fh.save(save_path, fig)

        plt.show()

    def plot_rating_layer(self, mask=None, save_path=None, allowed_lengths=[3, 4, 5, 6, 7]):
        if mask is None:
            mask = np.ones(len(self.puzzles), dtype=bool)

        effects_data, non_skipped = self.get_effects_data(mask, allowed_lengths)
        relevant_ratings = self.puzzles.loc[non_skipped, "Rating"]

        fh.set()

        colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
        layers = list(range(15))

        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(5)

        line_styles = ["-"] * 2 + ["-", "--"] * ((allowed_lengths[-1] - 1) // 2) + ["-"]
        line_styles += ["-.", ":"] * ((allowed_lengths[-1] - 1) // 2)

        for i, effect_data in enumerate(effects_data):
            effects = effect_data["effects"]
            if len(effects) == 0:
                continue
            
            #mean_effects = np.mean(effects, axis=0)
            #print(effects.shape, relevant_ratings.shape)

            # Group the ratings into bins and calculate the average and std per bin
            bin_means, bin_edges, binnumber = binned_statistic(
                relevant_ratings, np.argmax(effects, axis=1), statistic='mean', bins=10
            )
            bin_std, _, _ = binned_statistic(
                relevant_ratings, np.argmax(effects, axis=1), statistic='std', bins=10
            )
            
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            
            ax.errorbar(
                bin_centers,
                bin_means,
                yerr=bin_std,
                label=effect_data["name"],
                fmt='o-',
                color=colors[i]
            )

        # ax.set_title("Patching effects on different squares by layer")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Layer")
        _, y_max = ax.get_ylim()
        #ax.set_ylim(1e-2, 8)
        #ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.set_facecolor(fh.PLOT_FACE_COLOR)

        if save_path is not None:
            fh.save(save_path, fig)

        plt.show()

    def plot_rating_grid(self, n_cols=4, allowed_lengths=[3, 4, 5, 6, 7], filename=None):
        n_plots = len(self.good_results)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), squeeze=True, sharex=True, sharey=True)
        
        for idx, (possibility, mask) in enumerate(zip(list(self.good_results), self.good_mask)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            effects_data, non_skipped = self.get_effects_data(mask, allowed_lengths)
            relevant_ratings = self.puzzles.loc[non_skipped, "Rating"]

            colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
            
            for i, effect_data in enumerate(effects_data):
                effects = effect_data["effects"]
                if len(effects) == 0:
                    continue

                bin_means, bin_edges, _ = binned_statistic(
                    relevant_ratings, np.max(effects, axis=1), statistic='mean', bins=7, range=(700, 2200)
                )
                bin_std, _, _ = binned_statistic(
                    relevant_ratings, np.max(effects, axis=1), statistic='std', bins=7, range=(700, 2200)
                )
                
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                
                # Calculate 50% confidence interval
                ci_lower = bin_means - 0.67448 * bin_std
                ci_upper = bin_means + 0.67448 * bin_std
                
                ax.plot(bin_centers, bin_means, label=effect_data["name"], color=colors[i])
                ax.fill_between(bin_centers, ci_lower, ci_upper, alpha=0.3, color=colors[i])

            ax.set_title(f"{possibility}")
            if col == 0:
                ax.set_ylabel("Log odds reduction")
            if row == n_rows - 1:
                ax.set_xlabel("Rating")
            ax.legend(fontsize='x-small', loc="upper right")
            ax.spines[["right", "top"]].set_visible(False)

        # Remove any unused subplots
        for idx in range(len(self.good_results), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()
        
        if filename is not None:
            fh.save(filename, fig)

    def plot_rating_layer_grid(self, n_cols=4, allowed_lengths=[3, 4, 5, 6, 7], filename=None):
        n_plots = len(self.good_results)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows), squeeze=True, sharex=True, sharey=True)
        
        for idx, (possibility, mask) in enumerate(zip(list(self.good_results), self.good_mask)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            effects_data, non_skipped = self.get_effects_data(mask, allowed_lengths)
            relevant_ratings = self.puzzles.loc[non_skipped, "Rating"]

            colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
            
            for i, effect_data in enumerate(effects_data):
                effects = effect_data["effects"]
                if len(effects) == 0:
                    continue

                bin_means, bin_edges, _ = binned_statistic(
                    relevant_ratings, np.argmax(effects, axis=1), statistic='mean', bins=7, range=(700, 2200)
                )
                bin_std, _, _ = binned_statistic(
                    relevant_ratings, np.argmax(effects, axis=1), statistic='std', bins=7, range=(700, 2200)
                )
                
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                
                # Calculate 50% confidence interval
                ci_lower = bin_means - 0.67448 * bin_std
                ci_upper = bin_means + 0.67448 * bin_std
                
                ax.plot(bin_centers, bin_means, label=effect_data["name"], color=colors[i])
                ax.fill_between(bin_centers, ci_lower, ci_upper, alpha=0.3, color=colors[i])

            ax.set_title(f"{possibility}")
            if col == 0:
                ax.set_ylabel("Layer")
            if row == n_rows - 1:
                ax.set_xlabel("Rating")
            ax.legend(fontsize='x-small', loc="upper right")
            ax.spines[["right", "top"]].set_visible(False)

        # Remove any unused subplots
        for idx in range(len(self.good_results), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save(filename, fig)

    def plot_attention_grid(self, allowed_lengths=[3, 4, 5, 6, 7], n_cols=4, vmax=0.5, topk=5, rating_range=None, filename=None):

        if rating_range is not None:
            rating_mask = np.array((self.puzzles["Rating"] >= rating_range[0]) & (self.puzzles["Rating"] <= rating_range[1]))
        else:
            rating_mask = np.ones(len(self.puzzles), dtype=bool)

        # Calculate the number of rows and columns for the grid
        possibilities_mask = self.allowed_possibilities_mask(allowed_lengths)
        n_plots = np.sum(possibilities_mask)
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Create a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 3*n_rows), sharex=True, sharey=True)

        best_heads = {}
        self.all_heads = {}

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

        masked_good_results = [good_result for good_result, possibility_mask in zip(self.good_results, possibilities_mask) if possibility_mask]
        if self.alt_puzzles is not None:
            masked_good_mask = [good_mask & self.alt_mask for good_mask, possibility_mask in zip(self.good_mask, possibilities_mask) if possibility_mask]
        else:
            masked_good_mask = [good_mask for good_mask, possibility_mask in zip(self.good_mask, possibilities_mask) if possibility_mask]
        
        for idx, (possibility, mask) in enumerate(zip(masked_good_results, masked_good_mask)):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]

            if mask is None:
                mask = np.ones(len(self.all_attentions), dtype=bool)
            #print(mask)
            #print(rating_mask)
            mask = mask & rating_mask

            effects = self.all_attentions[mask]
            #print(possibility, np.sum(mask))

            mean_effects = -effects.mean(dim=0).cpu().numpy()
            #mean_effects = np.abs(effects.cpu().numpy()).mean(axis=0)
            if np.sum(mask) >= self.n_examples:
                self.all_heads[possibility] = mean_effects

            # Find the 5 most important heads
            flat_indices = np.argsort(mean_effects.flatten())[-topk:][::-1]
            top_5_heads = [((idx // mean_effects.shape[1], idx % mean_effects.shape[1]), mean_effects.flatten()[idx]) for idx in flat_indices]
            best_heads[possibility] = top_5_heads

            try:
                sns.heatmap(mean_effects.T, cmap=fh.EFFECTS_CMAP_2, ax=ax, cbar=False, vmin=0, vmax=vmax)
                ax.set_title(f"{possibility}")
                # Add text annotations
                for i in range(n_heads):
                    for j in range(n_layers):
                        if data[i, j]:
                            text_color = 'blue' if data[i, j] == 'K' else 'green' if data[i, j] == 'B' else 'red'
                            ax.text(j+0.5, i+0.5, data[i, j], ha='center', va='center', color=text_color, fontsize='xx-small')
                if col == 0:
                    ax.set_ylabel("Head")
                if row == n_rows - 1:
                    ax.set_xlabel("Layer")
            except ValueError:
                ax.axis('off')

        # Remove any unused subplots
        for idx in range(len(self.good_results), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off') if n_rows > 1 else axes[col].axis('off')

        plt.tight_layout()
        plt.show()

        if filename is not None:
            fh.save(filename, fig)

        self.best_heads = best_heads

    def plot_attention(self, pos, index, vmax=0.5, filename=None):
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

        # Get the mask for the specific position
        mask = self.good_mask[index].copy()
        mask[:] = False
        mask[pos] = True

        # Get the effects for the specific position
        effects = self.all_attentions[mask]

        # Calculate the effects for the single position
        position_effects = np.abs(effects[0].cpu().numpy())  # Use [0] to get the first (and only) item

        # Plot the heatmap
        sns.heatmap(position_effects.T, cmap=fh.EFFECTS_CMAP_2, ax=ax, cbar=True, vmin=0, vmax=vmax)
        ax.set_title(f"Attention for position {pos}")

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
            fh.save(filename, fig)



    @staticmethod
    def map_to_possibility(moves):
        mapping = {}
        result = []
        counter = 1
        
        for move in moves:
            if move not in mapping:
                mapping[move] = str(counter)
                counter += 1
            result.append(mapping[move])
        
        return result

    def map_to_possibility_alt(correct_squares, incorrect_squares):
        mapping = {}
        result = []
        counter = 1

        for square in correct_squares:
            if square not in mapping:
                mapping[square] = str(counter)  
                counter += 1
            result.append(mapping[square])
        for square in incorrect_squares:
            if square not in mapping:
                mapping[square] = str(counter)
                counter += 1
            result.append(mapping[square])

        return result

    @staticmethod
    def get_possibility_indices(puzzles, include_starting=False):
        possibilities = []
        indices = {}

        for i, (_, puzzle) in enumerate(puzzles.iterrows()):
            pv = puzzle.principal_variation
            moves = []
            for j in range(len(pv)):
                if include_starting:
                    moves += [pv[j][0:2]]
                moves.append(pv[j][2:4])
            possibility = ''.join(EffectStudy.map_to_possibility(moves))
            if possibility not in indices:
                indices[possibility] = []
            indices[possibility].append(i)
            possibilities.append(possibility)
        
        return indices

    @staticmethod
    def get_possibility_indices_alt(main_moves):
        possibilities = []
        indices = {}

        for i, (correct_branch, incorrect_branch) in enumerate(main_moves):
            correct_squares = [move[2:4] for move in correct_branch]
            incorrect_squares = [move[2:4] for move in incorrect_branch]
            possibility = ''.join(EffectStudy.map_to_possibility_alt(correct_squares, incorrect_squares))
            if possibility not in indices:
                indices[possibility] = []
            indices[possibility].append(i)
            possibilities.append(possibility)
        
        return indices

    @staticmethod
    def check_no_common_elements(list_of_sublists, ignore_even=False):
        # Convert each sublist to a set
        set_list = [set(list_of_sublists[0]), set(list_of_sublists[1])] + [set(sublist) for i, sublist in enumerate(list_of_sublists[2:]) if i % 2 != 0 or not ignore_even]
        
        # Check each pair of sets for intersection
        for i in range(len(set_list)):
            for j in range(i + 1, len(set_list)):
                if set_list[i].intersection(set_list[j]):
                    return False  # Found common element(s)
        
        return True  # No common elements found

    def create_head_to_possibilities_dict(self):
        head_to_possibilities = {}
        for possibility, heads in self.best_heads.items():
            for head, effect in heads:
                if head not in head_to_possibilities:
                    head_to_possibilities[head] = []
                head_to_possibilities[head].append((possibility, effect))
        
        # Sort each list by effect (descending) and keep only the possibilities
        for head in head_to_possibilities:
            head_to_possibilities[head].sort(key=lambda x: x[1], reverse=True)
            head_to_possibilities[head] = [p for p, _ in head_to_possibilities[head]]
        
        # Sort the dictionary by the number of possibilities for each head
        sorted_dict = dict(sorted(head_to_possibilities.items(), key=lambda item: len(item[1]), reverse=True))
        
        self.head_to_possibilities = sorted_dict

    def create_head_to_possibilities_dict_with_effects(self):
        head_to_possibilities = {}
        for possibility, heads in self.best_heads.items():
            for head, effect in heads:
                if head not in head_to_possibilities:
                    head_to_possibilities[head] = []
                head_to_possibilities[head].append((possibility, effect))
        
        # Sort each list by effect (descending)
        for head in head_to_possibilities:
            head_to_possibilities[head].sort(key=lambda x: x[1], reverse=True)
        
        # Sort the dictionary by the number of possibilities for each head
        sorted_dict = dict(sorted(head_to_possibilities.items(), key=lambda item: len(item[1]), reverse=True))
        
        self.head_to_possibilities_with_effects = sorted_dict

    def plot_attention_effects(self, mask=None):
        if mask is None:
            mask = np.ones(len(self.all_attentions), dtype=bool)
        effects = self.all_attentions[mask]

        mean_effects = -effects.mean(dim=0)
        fh.set()
        plt.figure(figsize=(fh.get_width(0.3), 2))
        plt.imshow(mean_effects.cpu().numpy().T, cmap=fh.EFFECTS_CMAP_2)
        plt.title("Mean patching effects")
        plt.ylabel("Head")
        plt.xlabel("Layer")
        plt.colorbar(fraction=0.10)
        plt.show()


class AblationStudy:
    def __init__(self, folder_name='L12H17', device='cpu'):
        self.folder_name = folder_name
        self.device = device
        self.load_ablation_data()

    def load_ablation_data(self):
        ablation = {}
        for file in os.listdir("results/" + self.folder_name):
            if file.endswith("_ablation.pt"):
                # Subtract suffix _ablation.pt
                file_prefix = file[:-12]
                if file_prefix == 'single_weight':
                    continue
                if file_prefix != 'other':
                    file_prefix = AblationStudy.pretty_prefix(file_prefix)
                ablation[file_prefix] = torch.load("results/" + self.folder_name + "/" + file, map_location=self.device)
        self.ablation = ablation

    @staticmethod
    def pretty_prefix(prefix):
        first_number, _, second_number = prefix.split("_")
        first_number = AblationStudy.word_to_number(first_number) + first_number[-2:]
        second_number = AblationStudy.word_to_number(second_number) + second_number[-2:]
        return first_number + r"$\to$" + second_number + ' target'

    def plot_ablation_effects(self, mask=None, verbose=False, filename=None, puzzle_set=None, LH=None):
        if mask is None:
            mask = slice(None)

        colors = {
            "other": fh.COLORS[-1],
            r"3rd$\to$1st target": fh.COLORS[0],
            r"5th$\to$1st target": fh.COLORS[1],
            r"5th$\to$3rd target": fh.COLORS[2],
            r"7th$\to$1st target": fh.COLORS[3],
            r"7th$\to$3rd target": fh.COLORS[4],
            r"7th$\to$5th target": fh.COLORS[5],
        }

        # Sorted dictionary of ablation
        sorted_ablation = dict(sorted(self.ablation.items(), key=lambda x: x[0]))
        sorted_colors = [colors[key] for key in sorted_ablation.keys()]
        sorted_colors_dict = dict(zip(sorted_ablation.keys(), sorted_colors))

        if len(sorted_ablation) > 4:
            scale_factor = 1.5
        else:
            scale_factor = 1.5

        fh.set()
        fh.plot_percentiles(
            sorted_ablation,
            zoom_start=94,
            zoom_width_ratio=0.7,
            colors=sorted_colors_dict,
            title="Attention ablation effects" + ((" (" + puzzle_set + ", " + LH + ")") if puzzle_set is not None and LH is not None else ""),
            figsize=(fh.get_width(0.66) * scale_factor, 2 * scale_factor),
            tick_frequency=25,
            zoom_tick_frequency=2,
            y_lower=-1,
            verbose=verbose,
        )
        if filename is not None:
            fh.save(filename, plt.gcf())

    @staticmethod
    def plot_ablation_effects_grid(ablation_configs, n_cols=2, filename=None):
        n_rows = (len(ablation_configs) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fh.TEXT_WIDTH, n_rows * 4))
        axes = axes.flatten()

        for i, (cases, puzzle_set) in enumerate(ablation_configs):
            ax = axes[i]
            for case in cases:
                ablation_study = AblationStudy(folder_name=case + "_" + puzzle_set)
                ablation_study.plot_ablation_effects(ax=ax)
            ax.set_title(f"{puzzle_set} - {', '.join(cases)}")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()

        




    @staticmethod
    def word_to_number(word):
        ordinal_dict = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
            'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14, 'fifteenth': 15,
            'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20
        }
        
        return str(ordinal_dict.get(word.lower(), None))

def prob_to_logodds(prob):
    return np.log(prob / (1 - prob))

def logodds_to_prob(logodds):
    return 1 / (1 + np.exp(-logodds))