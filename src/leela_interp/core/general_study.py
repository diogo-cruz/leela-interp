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
    def __init__(self, puzzlename='', device='cpu', load_all=True):
        self.puzzlename = ("_" if puzzlename != "" else "") + puzzlename
        self.device = device
        if load_all:
            self.load_puzzles()
            self.load_effects()
            self.load_attentions()
            #self.load_ratings()
        fh.set()

    def load_puzzles(self):
        with open(f"puzzles/interesting_puzzles{self.puzzlename}.pkl", "rb") as f:
            puzzles = pickle.load(f)
        self.puzzles = puzzles

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

    def load_puzzle_sets(self):
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
        new_puzzles = self.puzzles.copy()
        print(f"Starting with {len(new_puzzles)} puzzles")
        for i, rule in enumerate(rules):
            new_puzzles = new_puzzles[rule(new_puzzles)]
            print(f"After applying rule {i}, we have {len(new_puzzles)} puzzles")
        print(f"Ending with {len(new_puzzles)} puzzles")
        self.filtered_puzzles = new_puzzles

    def export_puzzles(self, filename):
        with open(f"puzzles/interesting_puzzles_{filename}.pkl", "wb") as f:
            pickle.dump(self.puzzles, f)

    def export_with_puzzle_mask(self, filtered_puzzles, filename):

        puzzle_mask = self.puzzles.index.isin(filtered_puzzles.index)

        with open(f"puzzles/interesting_puzzles_{filename}.pkl", "wb") as f:
            pickle.dump(filtered_puzzles, f)

        with open(f"results/global_patching/interesting_puzzles_{filename}_residual_stream_results.pt", "wb") as f:
            torch.save(-self.all_effects[puzzle_mask], f)

        with open(f"results/global_patching/interesting_puzzles_{filename}_attention_head_results.pt", "wb") as f:
            torch.save(self.all_attentions[puzzle_mask], f)

    def export_puzzle_set_info(self, tag='n'):
        tag = 's' if hasattr(self, 'include_starting') and self.include_starting else tag
        for (possibility, idx_list), mask in zip(self.result_sets.items(), self.result_masks):
            with open(f"puzzles/interesting_puzzles{self.puzzlename}_{tag}_{possibility}.pkl", "wb") as f:
                pickle.dump(self.puzzles[mask], f)
            with open(f"results/global_patching/interesting_puzzles{self.puzzlename}_{tag}_{possibility}_attention_head_results.pt", "wb") as f:
                torch.save(self.all_attentions[mask], f)
            with open(f"results/global_patching/interesting_puzzles{self.puzzlename}_{tag}_{possibility}_residual_stream_results.pt", "wb") as f:
                torch.save(self.all_effects[mask], f)

    def find_result_sets(self, include_starting=False, n_examples=100):
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
            possibility = ''.join(GeneralStudy.map_to_possibility(moves))
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
            possibility = ''.join(GeneralStudy.map_to_possibility_alt(correct_squares, incorrect_squares))
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

    def get_patching_squares(self, board, corrupted_board):
        return [chess.SQUARE_NAMES[square] for square in chess.SQUARES 
                if board.pc_board.piece_at(square) != corrupted_board.pc_board.piece_at(square)]

    def get_max_effects_int(self, effects, board, squares):
        return effects[:, squares].amax(-1).cpu().numpy()

    def get_max_effects(self, effects, board, squares):
        return effects[:, [board.sq2idx(square) for square in squares]].amax(-1).cpu().numpy()

    def print_verbose_info(self, skipped_count, total_count):
        print(f"Skipped {skipped_count} out of {total_count} puzzles ({skipped_count/total_count:.2%})")

    def print_effects(self, effects_dict, prefix):
        print(f"{prefix}::", end=" ")
        for i, effects in effects_dict.items():
            if len(effects) > 0:
                suffix = "st" if i == 1 else "nd" if i == 2 else "rd" if i == 3 else "th"
                print(f"{i}{suffix}: {len(effects)}", end=", ")
        print()

    def should_skip(self, patching_squares, movs, starts):
        return set(patching_squares).intersection(set(movs + starts))

    def process_effects(self, effects, board, candidate_squares, follow_up_squares, starting_squares, 
                        patching_squares, candidate_effects, follow_up_effects, starting_effects, 
                        patching_square_effects, other_effects, include_starting):
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

        return [possibility for possibility in self.puzzle_sets[tag].keys() if len(possibility) // (2 if tag != "n" else 1) in lengths]

    def plot_rating_histogram(self, tag, possibility, filename=None):

        fig = plt.figure()
        plt.hist(self.puzzle_sets[tag][possibility].Rating.to_numpy(), bins=30)
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.title("Histogram of Puzzle Ratings")
        plt.show()
        
        if filename is not None:
            fh.save('figures/' + filename, fig)

    def plot_examples(self, tag, possibility, n=5):

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
    return np.log(prob / (1 - prob))

def logodds_to_prob(logodds):
    return 1 / (1 + np.exp(-logodds))