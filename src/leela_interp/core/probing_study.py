"""Probing study analysis module for chess neural network interpretability.

This module provides classes for analyzing probing results from chess neural networks,
including visualization of accuracy across layers and comparison between trained and
random models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from leela_interp.tools import figure_helpers as fh
from typing import Dict, Tuple

class ProbingStudy:
    """Analyzes probing results for chess neural network interpretability studies.
    
    This class loads and visualizes probing accuracy results across different layers
    of a neural network, comparing trained models against random baselines.
    
    Attributes:
        case_number: Identifier for the specific case study
        tag_name: Additional tag for file naming
        n_seeds: Number of random seeds used in experiments
        add_opponent: Whether to include opponent moves in analysis
        start_goal_square: Starting square index for goal analysis
        setting_to_pretty_name: Mapping of settings to display names
        results: Dictionary containing accuracy results by setting and goal square
        n_examples: Number of examples in the puzzle set
    """
    
    def __init__(self, case_number, tag_name='', add_opponent=False, start_goal_square=3):
        """Initialize a probing study analysis.
        
        Args:
            case_number: Identifier for the specific case study
            tag_name: Additional tag for file naming (default: '')
            add_opponent: Whether to include opponent moves (default: False)
            start_goal_square: Starting square index for goal analysis (default: 3)
        """
        self.case_number = case_number
        #self.puzzlename = ('' if puzzlename == '' else '_' + puzzlename)
        self.tag_name = ('' if tag_name == '' else '_' + tag_name)
        self.n_seeds = 5
        self.add_opponent = add_opponent
        self.start_goal_square = start_goal_square
        self.set_pretty_names()
        self.load_results()

    def set_pretty_names(self):
        """Set up pretty names for different experimental settings.
        
        Creates a mapping from (setting, goal_square) tuples to human-readable
        display names for use in plots and legends.
        """

        goal_squares = range(self.start_goal_square, len(self.case_number) + 1, 2 if not self.add_opponent else 1)

        self.setting_to_pretty_name: Dict[Tuple[str, int], str] = {}
        for goal_square in goal_squares:
            self.setting_to_pretty_name[("main", goal_square)] = f"trained, move {goal_square}"
            self.setting_to_pretty_name[("random_model", goal_square)] = f"random, move {goal_square}"

    def load_results(self):
        """Load probing results from pickle files.
        
        Loads accuracy results for both trained and random models across
        different goal squares and random seeds. Also loads the puzzle set
        to determine the number of examples.
        
        The results are stored in self.results as a dictionary with keys
        (setting, goal_square) and values as numpy arrays of shape (15, n_seeds)
        representing accuracies across 15 layers and multiple random seeds.
        """
        self.results = {}
        for i, setting in enumerate(["main", "random_model"]):
            for j, goal_square in enumerate(range(self.start_goal_square, len(self.case_number) + 1, 2 if not self.add_opponent else 1)):
                results = np.zeros((15, self.n_seeds))
                for seed in range(self.n_seeds):
                    with open(f"results/probing{self.tag_name}_{self.case_number}/all/{seed}/{goal_square}/{setting}.pkl", "rb") as f:
                        new_results = pickle.load(f)
                        results[:, seed] = new_results["accuracies"]
                self.results[(setting, goal_square)] = results

        filename = f'puzzles/interesting_puzzles{self.tag_name}_{self.case_number}.pkl'
        with open(filename, "rb") as f:
            puzzle_set = pickle.load(f)
        n_examples = len(puzzle_set)
        self.n_examples = n_examples

    def plot_probe_results(self, split="all", filename=None):
        """Plot probing accuracy results across layers.
        
        Creates a visualization showing accuracy curves for different settings
        (trained vs random models) and goal squares across all 15 layers of
        the neural network. Includes error bars accounting for both seed
        variation and accuracy estimation uncertainty.
        
        Args:
            split: Data split to use ('all', 'train', 'test', etc.) (default: 'all')
            filename: Optional filename to save the plot (default: None)
        
        The plot shows:
        - Solid lines for trained models, dashed lines for random models
        - Different colors for different goal squares
        - Error bars representing 2-sigma confidence intervals
        - Grid and formatting for publication-ready figures
        """
        case_number = self.case_number
        n_seeds = self.n_seeds
        setting_to_pretty_name = self.setting_to_pretty_name

        fh.set(fast=False)
        plt.figure(figsize=(fh.HALF_WIDTH*1.5, 2*1.5))

        for i, setting in enumerate(["main", "random_model"]):
            for j, goal_square in enumerate(range(self.start_goal_square, len(case_number) + 1, 2 if not self.add_opponent else 1)):
                results = np.zeros((15, n_seeds))
                for seed in range(n_seeds):
                    with open(f"results/probing{self.tag_name}_{self.case_number}/{split}/{seed}/{goal_square}/{setting}.pkl", "rb") as f:
                        new_results = pickle.load(f)
                        results[:, seed] = new_results["accuracies"]

                means = results.mean(-1)
                squared_seed_errors = results.var(-1) / results.shape[-1]
                # Size of the eval dataset is 30% of all puzzles
                squared_acc_errors = means * (1 - means) / (0.3 * self.n_examples)
                # 2 sigma errors
                errors = np.sqrt(squared_seed_errors + squared_acc_errors)

                plt.plot(
                    means,
                    label=setting_to_pretty_name[(setting, goal_square)],
                    color=fh.COLORS[2*j + i],
                    linewidth=fh.LINE_WIDTH,
                    linestyle='-' if setting == 'main' else '--'
                )
                plt.fill_between(
                    range(15),
                    means - errors,
                    means + errors,
                    color=fh.COLORS[2*j + i],
                    alpha=fh.ERROR_ALPHA,
                    linewidth=0,
                )

        plt.title(f"Probing set {case_number}")
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower left")

        plt.ylim(0, 1.0)
        plt.xlim(0, 14)
        plt.gca().spines[:].set_visible(False)
        plt.gca().set_facecolor(fh.PLOT_FACE_COLOR)
        plt.grid(linestyle="--")
        plt.grid(which="minor", alpha=0.3, linestyle="--")

        if filename is not None:
            fh.save(f"figures/{filename}")


class ProbingBranchStudy(ProbingStudy):
    """Specialized probing study for branch analysis in chess puzzles.
    
    Extends ProbingStudy with specific handling for branch analysis,
    including both primary moves and alternative branch moves (marked with -B).
    """
    
    def __init__(self, case_number, puzzlename=''):
        """Initialize a probing branch study.
        
        Args:
            case_number: Identifier for the specific case study
            puzzlename: Name of the puzzle set (default: '')
        """
        super().__init__(case_number, puzzlename)
    
    def set_pretty_names(self):
        """Set up pretty names for branch analysis settings.
        
        Creates specific naming for branch analysis including both primary
        moves and alternative branch moves (marked with -B suffix).
        """
        self.setting_to_pretty_name: Dict[Tuple[str, int], str] = {
            ("main", 1): "trained, 1st square",
            ("main", 2): "trained, 1st-B square",
            ("main", 3): "trained, 3rd square",
            ("main", 4): "trained, 3rd-B square",
            ("random_model", 1): "random, 1st square",
            ("random_model", 2): "random, 1st-B square",
            ("random_model", 3): "random, 3rd square",
            ("random_model", 4): "random, 3rd-B square",
        }

    def load_results(self):
        """Load probing results for branch analysis.
        
        Loads accuracy results specifically for branch analysis studies,
        using the branch-specific puzzle file naming convention.
        """
        self.results = {}
        for i, setting in enumerate(["main", "random_model"]):
            for j, goal_square in enumerate(range(3, len(self.case_number) + 1, 2)):
                results = np.zeros((15, self.n_seeds))
                for seed in range(self.n_seeds):
                    with open(f"results/probing{self.tag_name}_{self.case_number}/all/{seed}/{goal_square}/{setting}.pkl", "rb") as f:
                        new_results = pickle.load(f)
                        results[:, seed] = new_results["accuracies"]
                self.results[(setting, goal_square)] = results

        filename = f'puzzles/interesting_puzzles{self.puzzlename}_b_{self.case_number}.pkl'
        with open(filename, "rb") as f:
            puzzle_set = pickle.load(f)
        n_examples = len(puzzle_set)
        self.n_examples = n_examples