"""General study module for analyzing chess puzzles using neural network interpretability.

This module provides the GeneralStudy class for loading, filtering, and analyzing
chess puzzles with their corresponding neural network activations and attention patterns.
It supports various forms of patching analysis and visualization of model behavior
on chess positions.
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

import chess
import iceberg as ice
from matplotlib.patches import Patch
import torch
from leela_interp import Lc0Model, Lc0sight, LeelaBoard
from leela_interp.core.iceberg_board import palette
from leela_interp.tools import figure_helpers as fh
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)
from scipy.stats import binned_statistic
from tqdm import tqdm
import re

class GeneralStudy:
    """A class for analyzing chess puzzles using neural network interpretability techniques.
    
    This class provides methods to load puzzle data, neural network activations,
    attention patterns, and perform various analyses on chess positions.
    """
    
    def __init__(self, puzzlename='', device='cpu', load_all=True):
        """Initialize the GeneralStudy instance.
        
        Args:
            puzzlename (str): Optional suffix for puzzle files. Defaults to ''.
            device (str): PyTorch device to use ('cpu' or 'cuda'). Defaults to 'cpu'.
            load_all (bool): Whether to load all data files on initialization. Defaults to True.
        """
        self.puzzlename = ("_" if puzzlename != "" else "") + puzzlename
        self.device = device
        if load_all:
            self.load_puzzles()
            self.load_effects()
            self.load_attentions()
            #self.load_ratings()
        fh.set()

    def load_puzzles(self):
        """Load puzzle data from pickle file.
        
        Loads the puzzle dataset containing chess positions, solutions, and metadata.
        The file is expected to be in puzzles/interesting_puzzles{puzzlename}.pkl format.
        """
        with open(f"puzzles/interesting_puzzles{self.puzzlename}.pkl", "rb") as f:
            puzzles = pickle.load(f)
        self.puzzles = puzzles

    def load_effects(self):
        """Load residual stream patching effects from saved results.
        
        Loads the precomputed effects of patching different positions in the residual stream.
        The effects are negated upon loading. If no results file exists, prints a warning.
        """
        if os.path.exists(f"results/global_patching/interesting_puzzles{self.puzzlename}_residual_stream_results.pt"):
            self.all_effects = -torch.load(
                f"results/global_patching/interesting_puzzles{self.puzzlename}_residual_stream_results.pt",
                map_location=self.device
            )
        else:
            print("No residual stream results found.")

    def load_attentions(self):
        """Load attention head patching results from saved files.
        
        Loads the precomputed effects of patching different attention heads.
        If no results file exists, prints a warning.
        """
        if os.path.exists(f"results/global_patching/interesting_puzzles{self.puzzlename}_attention_head_results.pt"):
            self.all_attentions = torch.load(
                f"results/global_patching/interesting_puzzles{self.puzzlename}_attention_head_results.pt",
                map_location=self.device
            )
        else:
            print("No attention head results found.")
        
    def load_ratings(self):
        """Load puzzle rating data as numpy array.
        
        Extracts the 'Rating' column from the loaded puzzles and converts to numpy array.
        """
        self.puzzle_ratings = self.puzzles["Rating"].to_numpy()

    def load_puzzle_sets(self):
        """Load multiple puzzle sets organized by tag and possibility.
        
        Scans the puzzles directory for files matching the pattern:
        interesting_puzzles{puzzlename}_{tag}_{possibility}.pkl
        and organizes them into a nested dictionary structure.
        """
        self.puzzle_sets = {}
        for filename in os.listdir("puzzles"):
            match = re.search(rf'interesting_puzzles{self.puzzlename}_([a-zA-Z])_(\d+)\.pkl$', filename)
            if match:
                tag, possibility = match.group(1), match.group(2)
                if tag not in self.puzzle_sets:
                    self.puzzle_sets[tag] = {}
                with open(f"puzzles/{filename}", "rb") as f:
                    #print(f"Loading {filename}, with tag {tag} and possibility {possibility}")
                    self.puzzle_sets[tag][possibility] = pickle.load(f)

    def load_effect_sets(self):
        """Load multiple effect sets organized by tag and possibility.
        
        Scans the results directory for residual stream results files matching the pattern:
        interesting_puzzles{puzzlename}_{tag}_{possibility}_residual_stream_results.pt
        and organizes them into a nested dictionary structure.
        """
        self.effect_sets = {}
        for filename in os.listdir("results/global_patching"):
            match = re.search(rf'interesting_puzzles{self.puzzlename}_([a-zA-Z])_(\d+)_residual_stream_results\.pt$', filename)
            if match:
                tag, possibility = match.group(1), match.group(2)
                if tag not in self.effect_sets:
                    self.effect_sets[tag] = {}
                with open(f"results/global_patching/{filename}", "rb") as f:
                    self.effect_sets[tag][possibility] = torch.load(f)

    def load_attention_sets(self):
        """Load multiple attention sets organized by tag and possibility.
        
        Scans the results directory for attention head results files matching the pattern:
        interesting_puzzles{puzzlename}_{tag}_{possibility}_attention_head_results.pt
        and organizes them into a nested dictionary structure.
        """
        self.attention_sets = {}
        for filename in os.listdir("results/global_patching"):
            match = re.search(rf'interesting_puzzles{self.puzzlename}_([a-zA-Z])_(\d+)_attention_head_results\.pt$', filename)
            if match:
                tag, possibility = match.group(1), match.group(2)
                if tag not in self.attention_sets:
                    self.attention_sets[tag] = {}
                with open(f"results/global_patching/{filename}", "rb") as f:
                    self.attention_sets[tag][possibility] = torch.load(f)

    def filter_puzzles(self, rules):
        """Filter puzzles using a sequence of rules.
        
        Args:
            rules (list): List of functions that take a DataFrame and return a boolean mask.
                         Each rule is applied sequentially to filter the puzzles.
        
        The filtered puzzles are stored in self.filtered_puzzles.
        Prints progress information showing how many puzzles remain after each rule.
        """
        new_puzzles = self.puzzles.copy()
        print(f"Starting with {len(new_puzzles)} puzzles")
        for i, rule in enumerate(rules):
            new_puzzles = new_puzzles[rule(new_puzzles)]
            print(f"After applying rule {i}, we have {len(new_puzzles)} puzzles")
        print(f"Ending with {len(new_puzzles)} puzzles")
        self.filtered_puzzles = new_puzzles

    def export_puzzles(self, filename):
        """Export current puzzles to a pickle file.
        
        Args:
            filename (str): Name for the output file (without extension).
                           Will be saved as puzzles/interesting_puzzles_{filename}.pkl
        """
        with open(f"puzzles/interesting_puzzles_{filename}.pkl", "wb") as f:
            pickle.dump(self.puzzles, f)

    def export_with_puzzle_mask(self, filtered_puzzles, filename):
        """Export filtered puzzles along with corresponding effects and attention data.
        
        Args:
            filtered_puzzles (DataFrame): The filtered puzzle dataset to export.
            filename (str): Base name for output files (without extension).
        
        Exports three files:
        - puzzles/interesting_puzzles_{filename}.pkl (puzzle data)
        - results/global_patching/interesting_puzzles_{filename}_residual_stream_results.pt (effects)
        - results/global_patching/interesting_puzzles_{filename}_attention_head_results.pt (attention)
        """

        puzzle_mask = self.puzzles.index.isin(filtered_puzzles.index)

        with open(f"puzzles/interesting_puzzles_{filename}.pkl", "wb") as f:
            pickle.dump(filtered_puzzles, f)

        with open(f"results/global_patching/interesting_puzzles_{filename}_residual_stream_results.pt", "wb") as f:
            torch.save(-self.all_effects[puzzle_mask], f)

        with open(f"results/global_patching/interesting_puzzles_{filename}_attention_head_results.pt", "wb") as f:
            torch.save(self.all_attentions[puzzle_mask], f)

    def export_puzzle_set_info(self, tag='n'):
        """Export puzzle sets with their corresponding results organized by possibility.
        
        Args:
            tag (str): Tag to use for file naming. Defaults to 'n'.
                      Automatically changes to 's' if include_starting is True.
        
        Exports puzzle data, attention results, and residual stream results
        for each possibility in the result sets.
        """
        tag = 's' if hasattr(self, 'include_starting') and self.include_starting else tag
        for (possibility, idx_list), mask in zip(self.result_sets.items(), self.result_masks):
            with open(f"puzzles/interesting_puzzles{self.puzzlename}_{tag}_{possibility}.pkl", "wb") as f:
                pickle.dump(self.puzzles[mask], f)
            with open(f"results/global_patching/interesting_puzzles{self.puzzlename}_{tag}_{possibility}_attention_head_results.pt", "wb") as f:
                torch.save(self.all_attentions[mask], f)
            with open(f"results/global_patching/interesting_puzzles{self.puzzlename}_{tag}_{possibility}_residual_stream_results.pt", "wb") as f:
                torch.save(self.all_effects[mask], f)

    def find_result_sets(self, include_starting=False, n_examples=100):
        """Find and organize puzzle sets by move possibilities.
        
        Args:
            include_starting (bool): Whether to include starting squares in possibility mapping.
                                   Defaults to False.
            n_examples (int): Minimum number of examples required for a possibility to be included.
                            Defaults to 100.
        
        Sets self.result_sets, self.result_masks, self.include_starting, and self.n_examples.
        """
        all_results = GeneralStudy.get_possibility_indices(self.puzzles, include_starting=include_starting)
        result_sets = {k: v for k, v in all_results.items() if len(v) >= n_examples}
        result_sets = {k: v for k, v in sorted(result_sets.items(), key=lambda item: len(item[1]), reverse=True)}
        result_masks = np.zeros((len(result_sets), len(self.puzzles)), dtype=bool)
        for i, (_, idx_list) in enumerate(result_sets.items()):
            result_masks[i, idx_list] = True
        self.result_sets = result_sets
        self.result_masks = result_masks
        self.include_starting = include_starting
        self.n_examples = n_examples

    @staticmethod
    def map_to_possibility(moves):
        """Map a sequence of moves to numbered possibilities.
        
        Args:
            moves (list): List of move strings.
        
        Returns:
            list: List of possibility numbers (as strings) corresponding to each move.
                 Each unique move gets assigned a sequential number starting from '1'.
        
        Example:
            >>> GeneralStudy.map_to_possibility(['e4', 'e5', 'e4', 'Nf3'])
            ['1', '2', '1', '3']
        """
        mapping = {}
        result = []
        counter = 1
        
        for move in moves:
            if move not in mapping:
                mapping[move] = str(counter)
                counter += 1
            result.append(mapping[move])
        
        return result

    @staticmethod
    def map_to_possibility_alt(correct_squares, incorrect_squares):
        """Map correct and incorrect squares to numbered possibilities.
        
        Args:
            correct_squares (list): List of correct square names.
            incorrect_squares (list): List of incorrect square names.
        
        Returns:
            list: List of possibility numbers (as strings) for all squares.
                 Correct squares are processed first, then incorrect squares.
                 Each unique square gets assigned a sequential number starting from '1'.
        """
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
        """Get indices of puzzles grouped by their move possibility patterns.
        
        Args:
            puzzles (DataFrame): Puzzle dataset with 'principal_variation' column.
            include_starting (bool): Whether to include starting squares in the pattern.
                                   Defaults to False.
        
        Returns:
            dict: Dictionary mapping possibility strings to lists of puzzle indices.
                 Each possibility string represents a unique pattern of moves.
        """
        possibilities = []
        indices = {}

        for i, (_, puzzle) in enumerate(puzzles.iterrows()):
            pv = puzzle.principal_variation
            moves = []
            for j in range(len(pv)):
                if include_starting:
                    moves += [pv[j][0:2]]
                moves.append(pv[j][2:4])
            possibility = ''.join(GeneralStudy.map_to_possibility(moves))
            if possibility not in indices:
                indices[possibility] = []
            indices[possibility].append(i)
            possibilities.append(possibility)
        
        return indices

    @staticmethod
    def get_possibility_indices_alt(main_moves):
        """Get indices of puzzles grouped by correct/incorrect move patterns.
        
        Args:
            main_moves (list): List of (correct_branch, incorrect_branch) tuples.
                             Each branch is a list of move strings.
        
        Returns:
            dict: Dictionary mapping possibility strings to lists of puzzle indices.
                 Possibility strings are based on destination squares of moves.
        """
        possibilities = []
        indices = {}

        for i, (correct_branch, incorrect_branch) in enumerate(main_moves):
            correct_squares = [move[2:4] for move in correct_branch]
            incorrect_squares = [move[2:4] for move in incorrect_branch]
            possibility = ''.join(GeneralStudy.map_to_possibility_alt(correct_squares, incorrect_squares))
            if possibility not in indices:
                indices[possibility] = []
            indices[possibility].append(i)
            possibilities.append(possibility)
        
        return indices

    @staticmethod
    def check_no_common_elements(list_of_sublists, ignore_even=False):
        """Check if sublists have no common elements.
        
        Args:
            list_of_sublists (list): List of sublists to check for common elements.
            ignore_even (bool): Whether to ignore even-indexed sublists (after the first two).
                              Defaults to False.
        
        Returns:
            bool: True if no common elements exist between any pair of sublists,
                 False otherwise.
        """
        # Convert each sublist to a set
        set_list = [set(list_of_sublists[0]), set(list_of_sublists[1])] + [set(sublist) for i, sublist in enumerate(list_of_sublists[2:]) if i % 2 != 0 or not ignore_even]
        
        # Check each pair of sets for intersection
        for i in range(len(set_list)):
            for j in range(i + 1, len(set_list)):
                if set_list[i].intersection(set_list[j]):
                    return False  # Found common element(s)
        
        return True  # No common elements found

    def get_patching_squares(self, board, corrupted_board):
        """Get squares that differ between original and corrupted boards.
        
        Args:
            board (LeelaBoard): Original board position.
            corrupted_board (LeelaBoard): Corrupted board position.
        
        Returns:
            list: List of square names where pieces differ between boards.
        """
        return [chess.SQUARE_NAMES[square] for square in chess.SQUARES 
                if board.pc_board.piece_at(square) != corrupted_board.pc_board.piece_at(square)]

    def get_max_effects_int(self, effects, board, squares):
        """Get maximum effects across specified squares using integer indices.
        
        Args:
            effects (torch.Tensor): Tensor of effects with shape (..., 64).
            board (LeelaBoard): Board instance (not used in this method).
            squares (list): List of integer square indices.
        
        Returns:
            numpy.ndarray: Maximum effect values across the specified squares.
        """
        return effects[:, squares].amax(-1).cpu().numpy()

    def get_max_effects(self, effects, board, squares):
        """Get maximum effects across specified squares using square names.
        
        Args:
            effects (torch.Tensor): Tensor of effects with shape (..., 64).
            board (LeelaBoard): Board instance used to convert square names to indices.
            squares (list): List of square names (e.g., ['e4', 'e5']).
        
        Returns:
            numpy.ndarray: Maximum effect values across the specified squares.
        """
        return effects[:, [board.sq2idx(square) for square in squares]].amax(-1).cpu().numpy()

    def print_verbose_info(self, skipped_count, total_count):
        """Print information about skipped puzzles.
        
        Args:
            skipped_count (int): Number of puzzles that were skipped.
            total_count (int): Total number of puzzles processed.
        """
        print(f"Skipped {skipped_count} out of {total_count} puzzles ({skipped_count/total_count:.2%})")

    def print_effects(self, effects_dict, prefix):
        """Print summary of effects organized by move number.
        
        Args:
            effects_dict (dict): Dictionary mapping move numbers to lists of effects.
            prefix (str): Prefix string to print before the summary.
        """
        print(f"{prefix}::", end=" ")
        for i, effects in effects_dict.items():
            if len(effects) > 0:
                suffix = "st" if i == 1 else "nd" if i == 2 else "rd" if i == 3 else "th"
                print(f"{i}{suffix}: {len(effects)}", end=", ")
        print()

    def should_skip(self, patching_squares, movs, starts):
        """Check if a puzzle should be skipped based on square overlap.
        
        Args:
            patching_squares (list): List of squares being patched.
            movs (list): List of move destination squares.
            starts (list): List of move starting squares.
        
        Returns:
            bool: True if there's overlap between patching squares and move squares.
        """
        return set(patching_squares).intersection(set(movs + starts))

    def process_effects(self, effects, board, candidate_squares, follow_up_squares, starting_squares, 
                        patching_squares, candidate_effects, follow_up_effects, starting_effects, 
                        patching_square_effects, other_effects, include_starting):
        """Process and categorize effects across different types of squares.
        
        Args:
            effects (torch.Tensor): Tensor of effects to process.
            board (LeelaBoard): Board instance for square conversion.
            candidate_squares (list): List of candidate move squares.
            follow_up_squares (dict): Dictionary mapping move numbers to follow-up squares.
            starting_squares (dict): Dictionary mapping move numbers to starting squares.
            patching_squares (list): List of squares being patched.
            candidate_effects (list): List to append candidate effects to.
            follow_up_effects (dict): Dictionary to append follow-up effects to.
            starting_effects (dict): Dictionary to append starting effects to.
            patching_square_effects (list): List to append patching square effects to.
            other_effects (list): List to append other square effects to.
            include_starting (bool): Whether to include starting square effects.
        """
        candidate_effects.append(self.get_max_effects(effects, board, candidate_squares))
        
        for j, squares in follow_up_squares.items():
            if squares:
                follow_up_effects[j].append(self.get_max_effects(effects, board, squares))
        
        if include_starting:
            for j, squares in starting_squares.items():
                if squares:
                    starting_effects[j].append(self.get_max_effects(effects, board, squares))
        
        patching_square_effects.append(self.get_max_effects(effects, board, patching_squares))
        
        covered_squares = set(candidate_squares + patching_squares + 
                              sum(starting_squares.values(), []) + 
                              sum(follow_up_squares.values(), []))
        other_squares = [idx for idx in range(64) if board.idx2sq(idx) not in covered_squares]
        other_effects.append(self.get_max_effects_int(effects, board, other_squares))

    def get_possibility_list(self, tag, lengths=[3]):
        """Get list of possibilities filtered by move sequence length.
        
        Args:
            tag (str): Tag identifier for the puzzle set.
            lengths (list): List of acceptable move sequence lengths. Defaults to [3].
        
        Returns:
            list: List of possibility strings that match the specified lengths.
        """

        return [possibility for possibility in self.puzzle_sets[tag].keys() if len(possibility) // (2 if tag != "n" else 1) in lengths]

    def plot_rating_histogram(self, tag, possibility, filename=None):
        """Plot histogram of puzzle ratings for a specific possibility.
        
        Args:
            tag (str): Tag identifier for the puzzle set.
            possibility (str): Possibility string identifier.
            filename (str, optional): If provided, save figure to this filename.
        """

        fig = plt.figure()
        plt.hist(self.puzzle_sets[tag][possibility].Rating.to_numpy(), bins=30)
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.title("Histogram of Puzzle Ratings")
        plt.show()
        
        if filename is not None:
            fh.save('figures/' + filename, fig)

    def plot_examples(self, tag, possibility, n=5):
        """Plot visual examples of puzzle effects across different layers.
        
        Args:
            tag (str): Tag identifier for the puzzle set.
            possibility (str): Possibility string identifier.
            n (int): Number of examples to plot. Defaults to 5.
        
        Returns:
            ice.Arrange: Arranged plot showing heatmaps of effects across layers.
        """

        effects = self.effect_sets[tag][possibility]

        plots = []

        # Don't plot all the layers, it's too much
        layers = [0, 6, 8, 10, 12, 14]

        for i in range(n):
            puzzle = self.puzzle_sets[tag][possibility].iloc[i]
            print(i, puzzle.principal_variation, puzzle.full_pv_probs)
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

def prob_to_logodds(prob):
    """Convert probability to log odds.
    
    Args:
        prob (float or numpy.ndarray): Probability value(s) between 0 and 1.
    
    Returns:
        float or numpy.ndarray: Log odds value(s).
    """
    return np.log(prob / (1 - prob))

def logodds_to_prob(logodds):
    """Convert log odds to probability.
    
    Args:
        logodds (float or numpy.ndarray): Log odds value(s).
    
    Returns:
        float or numpy.ndarray: Probability value(s) between 0 and 1.
    """
    return 1 / (1 + np.exp(-logodds))