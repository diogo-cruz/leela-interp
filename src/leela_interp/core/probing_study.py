import numpy as np
import matplotlib.pyplot as plt
import pickle
from leela_interp.tools import figure_helpers as fh
from typing import Dict, Tuple

class ProbingStudy:
    def __init__(self, case_number, tag_name='', add_opponent=False, start_goal_square=3):
        self.case_number = case_number
        #self.puzzlename = ('' if puzzlename == '' else '_' + puzzlename)
        self.tag_name = ('' if tag_name == '' else '_' + tag_name)
        self.n_seeds = 5
        self.add_opponent = add_opponent
        self.start_goal_square = start_goal_square
        self.set_pretty_names()
        self.load_results()

    def set_pretty_names(self):

        goal_squares = range(self.start_goal_square, len(self.case_number) + 1, 2 if not self.add_opponent else 1)

        self.setting_to_pretty_name: Dict[Tuple[str, int], str] = {}
        for goal_square in goal_squares:
            self.setting_to_pretty_name[("main", goal_square)] = f"trained, move {goal_square}"
            self.setting_to_pretty_name[("random_model", goal_square)] = f"random, move {goal_square}"

    def load_results(self):
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
    def __init__(self, case_number, puzzlename=''):
        super().__init__(case_number, puzzlename)
    
    def set_pretty_names(self):
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