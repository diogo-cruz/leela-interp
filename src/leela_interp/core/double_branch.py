"""Double Branch Study Module.

This module provides the DoubleBranchStudy class for analyzing chess puzzles that have
two distinct solution branches. It extends the GeneralStudy class to handle puzzles
where the neural network can choose between two different but equally valid move
sequences to achieve the same objective.

The module includes functionality for:
- Detecting and validating double-branch puzzles
- Analyzing residual stream effects for both branches
- Plotting attention patterns and effects across layers
- Computing possibility indices for different move patterns
"""

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
import re
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
    """Study class for analyzing chess puzzles with two distinct solution branches.
    
    This class extends GeneralStudy to handle puzzles where the neural network
    can choose between two different but equally valid move sequences. It provides
    methods for analyzing the effects of interventions on both branches.
    
    Attributes:
        all_effects_b: Effects for branch B residual stream patching
        effect_sets_b: Dictionary of effect sets for branch B organized by tag and possibility
        result_sets: Dictionary mapping possibility patterns to puzzle indices
        result_masks: Boolean masks for different possibility patterns
    """
    
    def __init__(self, *args, load_sets=True, load_all=True, **kwargs):
        """Initialize DoubleBranchStudy.
        
        Args:
            *args: Variable arguments passed to parent class
            load_sets: Whether to load puzzle and effect sets
            load_all: Whether to load all effects and check for double branches
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs, load_all=load_all)
        if load_all:
            self.load_effects_b()
            self.check_contains_double_branch()
        if load_sets:
            self.load_puzzle_sets()
            self.load_effect_sets()
            self.load_effect_sets_b()
            self.load_attention_sets()

    def check_contains_double_branch(self):
        """Validate that puzzle data contains required double branch columns.
        
        Raises:
            ValueError: If required columns 'branch_1' or 'branch_2' are missing
        """
        # Check if "branch_1" and "branch_2" columns exist in self.puzzles
        required_columns = ["branch_1", "branch_2"]
        missing_columns = [col for col in required_columns if col not in self.puzzles.columns]
        
        if missing_columns:
            raise ValueError(f"The following required columns are missing from self.puzzles: {', '.join(missing_columns)}")
        
    def load_effects_b(self):
        """Load residual stream effects for branch B.
        
        Loads the saved tensor of effects for branch B from the global patching results.
        The effects are negated when loaded to maintain consistency with the analysis framework.
        """
        if os.path.exists(f"results/global_patching/interesting_puzzles{self.puzzlename}_residual_stream_results_b.pt"):
            self.all_effects_b = -torch.load(
                f"results/global_patching/interesting_puzzles{self.puzzlename}_residual_stream_results_b.pt",
                map_location=self.device
            )
        else:
            print("No residual stream results B found.")
        
    def load_effect_sets_b(self):
        """Load effect sets for branch B organized by tag and possibility.
        
        Scans the global patching results directory for files matching the pattern
        for branch B results and organizes them by tag and possibility identifier.
        """
        self.effect_sets_b = {}
        for filename in os.listdir("results/global_patching"):
            match = re.search(rf'interesting_puzzles{self.puzzlename}_([a-zA-Z])_(\d+)_residual_stream_results_b\.pt$', filename)
            if match:
                tag, possibility = match.group(1), match.group(2)
                if tag not in self.effect_sets_b:
                    self.effect_sets_b[tag] = {}
                with open(f"results/global_patching/{filename}", "rb") as f:
                    self.effect_sets_b[tag][possibility] = torch.load(f)

    def export_puzzle_set_info_b(self, tag='b'):
        """Export puzzle set information including branch B results.
        
        Saves puzzle data, attention results, and both branch A and B residual stream
        results for each possibility pattern to separate files.
        
        Args:
            tag: Tag identifier for the export (default 'b')
        """
        tag = 's' if hasattr(self, 'include_starting') and self.include_starting else tag
        for (possibility, idx_list), mask in zip(self.result_sets.items(), self.result_masks):
            with open(f"puzzles/interesting_puzzles{self.puzzlename}_{tag}_{possibility}.pkl", "wb") as f:
                pickle.dump(self.puzzles[mask], f)
            with open(f"results/global_patching/interesting_puzzles{self.puzzlename}_{tag}_{possibility}_attention_head_results.pt", "wb") as f:
                torch.save(self.all_attentions[mask], f)
            with open(f"results/global_patching/interesting_puzzles{self.puzzlename}_{tag}_{possibility}_residual_stream_results.pt", "wb") as f:
                torch.save(self.all_effects[mask], f)
            with open(f"results/global_patching/interesting_puzzles{self.puzzlename}_{tag}_{possibility}_residual_stream_results_b.pt", "wb") as f:
                torch.save(self.all_effects_b[mask], f)

    @staticmethod
    def check_if_double_branch(model, puzzles_original, must_include_pv=True, end: int = 3, min_prob: float | list[float] = 0.1):
        """Check if puzzles have double branch structure and extract branch information.
        
        Analyzes puzzles to identify those with two distinct solution branches,
        extracting the move sequences and probabilities for each branch.
        
        Args:
            model: The neural network model to analyze
            puzzles_original: DataFrame of original puzzles
            must_include_pv: Whether the principal variation must be included in branches
            end: Maximum depth to analyze (default 3)
            min_prob: Minimum probability threshold for moves
            
        Returns:
            DataFrame with added columns for branch_1, branch_2, and their probabilities
        """

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
            """Recursively extract moves and probabilities for each branch.
            
            Args:
                depth: Current depth in the move tree
                end: Maximum depth to explore
                moves: Dictionary of moves at current depth
                is_branch_1: Boolean indicating if we're processing branch 1
                branch_1_moves: List to store branch 1 moves
                branch_1_probs: List to store branch 1 probabilities
                branch_2_moves: List to store branch 2 moves
                branch_2_probs: List to store branch 2 probabilities
            """
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
        """Find and organize puzzle result sets by possibility patterns.
        
        Groups puzzles by their possibility patterns (move sequences) and filters
        to keep only those with sufficient examples for analysis.
        
        Args:
            include_starting: Whether to include starting positions in analysis
            n_examples: Minimum number of examples required for a set
        """
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
        """Export puzzle set information using parent class method.
        
        Args:
            tag: Tag identifier for the export (default 'b')
        """
        super().export_puzzle_set_info(tag=tag)
        

    def load_results(self):
        """Load and organize puzzle results by possibility patterns.
        
        Loads results either from alternative puzzles or main puzzles,
        filters for sufficient examples, and creates boolean masks for analysis.
        """
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
        """Map chess squares to possibility pattern identifiers.
        
        Creates a mapping from chess squares to numeric identifiers,
        processing branch 1 squares first, then branch 2 squares.
        
        Args:
            branch_1_squares: List of squares for branch 1
            branch_2_squares: List of squares for branch 2
            
        Returns:
            List of string identifiers representing the possibility pattern
        """
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
        """Get indices of puzzles organized by possibility patterns.
        
        Analyzes puzzles to extract move patterns and groups them by
        their possibility identifier strings.
        
        Args:
            puzzles: DataFrame of puzzles with branch information
            include_starting: Whether to include starting positions (not implemented)
            
        Returns:
            Dictionary mapping possibility patterns to lists of puzzle indices
            
        Raises:
            NotImplementedError: If include_starting is True
        """
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
    
    def get_effect_set_data(self, tag, possibility, verbose=False, b=False):
        """Get effect data for a specific tag and possibility pattern.
        
        Processes effects for puzzles matching a specific possibility pattern,
        organizing them by move types (candidate, follow-up, starting) and
        square types (patching, other).
        
        Args:
            tag: Tag identifier for the effect set
            possibility: Possibility pattern string
            verbose: Whether to print detailed information
            b: Whether to use branch B effects
            
        Returns:
            Tuple of (effects_data, non_skipped_indices)
        """
        effects = self.effect_sets[tag][possibility] if not b else self.effect_sets_b[tag][possibility]
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
        """Prepare effects data for visualization and analysis.
        
        Organizes different types of effects into a structured format suitable
        for plotting and analysis.
        
        Args:
            candidate_effects: Effects for first move candidates
            follow_up_effects: Dictionary of effects for follow-up moves
            starting_effects: Dictionary of effects for starting moves
            patching_square_effects: Effects for patching squares
            other_effects: Effects for other squares
            max_length: Maximum sequence length
            include_starting: Whether starting effects are included
            verbose: Whether to print detailed information
            
        Returns:
            List of dictionaries containing effects and names for each category
        """
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
        ]
        
        if include_starting:
            effects_data.extend([{"effects": candidate_effects, "name": "Move 1A"}])
            #effects_data.extend({"effects": effects, "name": f"Move {j}A"} for j, effects in follow_up_effects.items())
            #effects_data.extend({"effects": effects, "name": f"Move {j}B"} for j, effects in starting_effects.items())
            # Check if starting_effects[1] exists and is not empty
            if 1 in starting_effects and len(starting_effects[1]) > 0:
                effects_data.extend([{"effects": starting_effects[1], "name": "Move 1B"}])
            
            for j in range(2, max(follow_up_effects.keys()) + 1):
                if j in follow_up_effects and len(follow_up_effects[j]) > 0:
                    effects_data.extend([{"effects": follow_up_effects[j], "name": f"Move {j}A"}])
                if j in starting_effects and len(starting_effects[j]) > 0:
                    effects_data.extend([{"effects": starting_effects[j], "name": f"Move {j}B"}])
        else:
            effects_data.extend([{"effects": candidate_effects, "name": "Move 1"}])
            effects_data.extend({"effects": effects, "name": f"Move {j}"} 
                              for j, effects in follow_up_effects.items() 
                              if len(effects) > 0)
        
        return effects_data

    def plot_residual_effects(self, tag, possibility, filename=None, plot_ci=True, plot_std=False, ax=None, row_col=None, log=False, clean_plot=False, b=False):
        """Plot residual stream effects across layers for a specific possibility.
        
        Creates a line plot showing how residual stream effects vary across
        different layers of the neural network for different move types.
        
        Args:
            tag: Tag identifier for the effect set
            possibility: Possibility pattern string
            filename: Optional filename to save the plot
            plot_ci: Whether to plot confidence intervals (standard error)
            plot_std: Whether to plot standard deviation instead of confidence intervals
            ax: Optional matplotlib axis to plot on
            row_col: Tuple of (is_bottom_row, is_left_col, title) for subplot layout
            log: Whether to use log scale for y-axis
            clean_plot: Whether to use clean plot formatting
            b: Whether to plot branch B effects
        """
        ax_init = None if ax is None else ax

        branch_1_probs = np.vstack(self.puzzle_sets[tag][possibility].branch_1_probs.to_numpy())
        branch_2_probs = np.vstack(self.puzzle_sets[tag][possibility].branch_2_probs.to_numpy())
        branch_probs = np.vstack((branch_1_probs[:, 0], branch_2_probs[:, 0])).T
        sorted_indices = np.argsort(branch_probs[:, 0] - branch_probs[:, 1])

        effects_data, nonskipped = self.get_effect_set_data(tag, possibility, b=b)
        # Find the new indices corresponding to sorted_indices in the nonskipped subset
        nonskipped_indices = [self.puzzle_sets[tag][possibility].index.get_loc(idx) for idx in nonskipped]
        #print(sorted_indices, nonskipped_indices)
        new_sorted_indices = []
        for idx in sorted_indices:
            if idx in nonskipped_indices:
                new_idx = nonskipped_indices.index(idx)
                new_sorted_indices.append(new_idx)
        
        # Use new_sorted_indices instead of sorted_indices for indexing effects
        sorted_indices = new_sorted_indices[:]
        #print(sorted_indices)
        max_length = len(possibility) // (2 if tag != "n" else 1)

        fh.set()

        # line_styles = ["-"] * 2 + ["-", "--"] * ((max_length - 1) // 2) + ["-"]

        # colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[:len(line_styles)]
        # layers = list(range(15))

        # line_styles += ["-", "--"] * ((max_length - 1) // 2) + ["-"]
        # colors += plt.cm.tab20(np.linspace(0, 1, 20)).tolist()[len(line_styles) - len(colors) + 5:]

        layers = list(range(15))

        line_styles = ["-"] * 2 + ["-", "--"] * (max_length - 1) * 2
        colors = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
        colors = colors[:4] + colors[6:8] + colors[4:6] + colors[8:len(line_styles)]

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
            std_effects = np.std(effects, axis=0)
            stderr_effects = std_effects / np.sqrt(len(effects))

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
                    alpha=fh.ERROR_ALPHA,
                )
            elif plot_ci:
                # Plot standard error (confidence intervals)
                ax.fill_between(
                    layers,
                    mean_effects - stderr_effects,
                    mean_effects + stderr_effects,
                    color=colors[i],
                    alpha=fh.ERROR_ALPHA,
                )
                # ci_50 = np.quantile(effects, [0.25, 0.75], axis=0)
                # ci_90 = np.quantile(effects, [0.05, 0.95], axis=0)
                # if not clean_plot:
                #     ax.fill_between(
                #         layers,
                #         ci_90[0],
                #         ci_90[1],
                #         color=colors[i],
                #         alpha=0.1,
                #     )
                # ax.fill_between(
                #     layers,
                #     ci_50[0],
                #     ci_50[1],
                #     color=colors[i],
                #     alpha=0.3,
                # )

        # ax.set_title("Patching effects on different squares by layer")
        if row_col is not None:
            #ax.set_title(f"Possibility: {row_col[2]}")
            if row_col[0]:
                ax.set_xlabel("Layer")
            if row_col[1]:
                ax.set_ylabel("Log odds reduction")
        else:
            ax.set_xlabel("Layer")
            ax.set_ylabel(f"Log odds reduction for branch {'B' if b else 'A'}")
        _, y_max = ax.get_ylim()
        ax.set_xlim(0, 14)
        #ax.set_ylim(1e-2, 2)
        ax.set_ylim(-1., 1.5)
        if log:
            ax.set_yscale("symlog", linthresh=1e-2)
        if row_col is not None:
            ax.legend(loc="upper left", title=f"Set {row_col[2]}" if "Branch" not in row_col[2] else f"{row_col[2]}")
        else:
            ax.legend(loc="upper left")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.set_facecolor(fh.PLOT_FACE_COLOR)

        if filename is not None and ax_init is None:
            fh.save('figures/' + filename, fig)

        if ax is None:
            plt.show()

    def _plot_residual_effects_extended(self, tag, possibility, filename=None, plot_ci=True, ax=None, row_col=None, log=False, clean_plot=False):
        """Plot residual effects with extended confidence intervals.
        
        Similar to plot_residual_effects but shows multiple confidence interval
        bands for better visualization of uncertainty.
        
        Args:
            tag: Tag identifier for the effect set
            possibility: Possibility pattern string
            filename: Optional filename to save the plot
            plot_ci: Whether to plot confidence intervals
            ax: Optional matplotlib axis to plot on
            row_col: Tuple of (is_bottom_row, is_left_col, title) for subplot layout
            log: Whether to use log scale for y-axis
            clean_plot: Whether to use clean plot formatting
        """
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
                for k in range(1, 11):
                    ci = np.quantile(effects, [0.5 - 0.05 * k, 0.5 + 0.05 * k], axis=0)
                    ax.fill_between(
                        layers,
                        ci[0],
                        ci[1],
                        color=colors[i],
                        alpha=0.2*np.sqrt(1.1 - 0.1 * k),
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

    def plot_residual_effects_grid(self, tag, possibilities, n_cols=4, filename=None, log=False, plot_ci=True, plot_std=False):
        """Plot residual effects for multiple possibilities in a grid layout.
        
        Creates a grid of subplots, each showing residual effects for a different
        possibility pattern.
        
        Args:
            tag: Tag identifier for the effect set
            possibilities: List of possibility pattern strings
            n_cols: Number of columns in the grid
            filename: Optional filename to save the plot
            log: Whether to use log scale for y-axis
            plot_ci: Whether to plot confidence intervals (standard error)
            plot_std: Whether to plot standard deviation instead of confidence intervals
        """
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
                    plot_ci=plot_ci,
                    plot_std=plot_std,
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
        """Plot attention pattern heatmap for a specific possibility.
        
        Creates a heatmap showing attention patterns across layers and heads,
        with annotations for piece movement heads (Knight, Bishop, Rook).
        
        Args:
            tag: Tag identifier for the attention set
            possibility: Possibility pattern string
            vmax: Maximum value for color scale
            filename: Optional filename to save the plot
        """
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
        """Plot attention patterns for multiple possibilities in a grid layout.
        
        Creates a grid of attention heatmaps, each showing patterns for a different
        possibility with piece movement head annotations.
        
        Args:
            tag: Tag identifier for the attention set
            possibilities: List of possibility pattern strings
            n_cols: Number of columns in the grid
            vmax: Maximum value for color scale
            filename: Optional filename to save the plot
        """

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

    def plot_residual_effects_grid_A_vs_B(self, tag, possibilities, n_cols=2, filename=None, log=False, plot_ci=True, plot_std=False):
        """Plot residual effects comparing branch A and B in a grid layout.
        
        Creates a grid with paired plots showing residual effects for both
        branch A and branch B for each possibility pattern.
        
        Args:
            tag: Tag identifier for the effect set
            possibilities: List of possibility pattern strings
            n_cols: Number of columns in the grid (for A plots)
            filename: Optional filename to save the plot
            log: Whether to use log scale for y-axis
            plot_ci: Whether to plot confidence intervals (standard error)
            plot_std: Whether to plot standard deviation instead of confidence intervals
        """
        #TODO
        fix = 1
        nfix = 3 - fix
        n_plots = len(possibilities) * nfix
        n_rows = math.ceil(n_plots / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols * fix, figsize=(3*n_cols* fix, 2.2*n_rows), sharex=True, sharey=True)
        
        for idx, possibility in enumerate(possibilities):
            row = idx // n_cols
            col = (idx % n_cols) * fix
            
            # Get the axes for both A and B plots
            ax_a = axes[row, col] if n_rows > 1 else axes[col]
            ax_b = axes[row, col+1] if n_rows > 1 else axes[col+1]
            
            # Set legend titles based on number of possibilities
            legend_a = f"{possibility} A" if len(possibilities) > 1 else "Branch A"
            legend_b = f"{possibility} B" if len(possibilities) > 1 else "Branch B"
            
            try:
                # Plot A (b=False)
                self.plot_residual_effects(
                    tag=tag,
                    possibility=possibility,
                    ax=ax_a,
                    row_col=(row == n_rows - 1, col == 0, legend_a),
                    plot_ci=plot_ci,
                    plot_std=plot_std,
                    filename=None,
                    log=log,
                    b=False
                )
                
                # Plot B (b=True)
                self.plot_residual_effects(
                    tag=tag,
                    possibility=possibility,
                    ax=ax_b,
                    row_col=(row == n_rows - 1, False, legend_b),
                    plot_ci=plot_ci,
                    plot_std=plot_std,
                    filename=None,
                    log=log,
                    b=True
                )
            except ValueError:
                ax_a.set_visible(False)
                ax_b.set_visible(False)
        
        # Hide any unused subplots
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = (idx % n_cols) * fix
            if n_rows > 1:
                axes[row, col].set_visible(False)
                axes[row, col+1].set_visible(False)
            else:
                axes[col].set_visible(False)
                axes[col+1].set_visible(False)
        
        plt.tight_layout()

        if filename is not None:
            if fig is None:
                fig = plt.gcf()

            #plt.tight_layout()
            fig.savefig('figures/' + filename + '.pdf')
            fig.savefig('figures/' + filename + '.png', dpi=300)
        plt.show()

        # if filename is not None:
        #     fh.save('figures/' + filename, fig)