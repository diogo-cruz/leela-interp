
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from leela_interp import LeelaBoard
from leela_interp.tools import figure_helpers as fh
import os
import networkx as nx
import torch

from leela_interp import Lc0sight, LeelaBoard
from leela_interp.core.alternative_moves import check_if_double_game, get_top_moves, create_tree_graph, hierarchy_pos
from leela_interp.core.double_mate import DoubleMateStudy
from leela_interp.tools import figure_helpers as fh
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)
from datetime import datetime
from leela_interp.tools.attention import attention_attribution, top_k_attributions
from leela_interp.tools.patching import activation_patch

class DoubleThirdStudy(DoubleMateStudy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_handcrafted_puzzles(self, puzzles):

        boards = []

        if self.augment_data:
            raise NotImplementedError("Augmentation not implemented for double third study")

        for _, fen in puzzles.iterrows():
            board = LeelaBoard.from_puzzle(fen)
            boards.append(board)

        self.boards = boards