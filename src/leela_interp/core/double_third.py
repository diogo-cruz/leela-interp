
"""Double Third Study Module

This module provides functionality for analyzing chess positions where there are "double third" patterns.
It extends the DoubleMateStudy class to handle specific chess puzzle scenarios involving third-rank
piece configurations and patterns.

The module supports loading handcrafted puzzles and analyzing them using the Leela Chess Zero engine
for interpretability studies.
"""

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
    """Study class for analyzing double third chess patterns.
    
    This class extends DoubleMateStudy to focus specifically on chess positions
    involving double third patterns - positions where pieces on the third rank
    create tactical opportunities or threats.
    
    Inherits all functionality from DoubleMateStudy and adds specialized
    methods for handling double third specific puzzle scenarios.
    
    Attributes:
        boards (list): List of LeelaBoard objects representing puzzle positions
        augment_data (bool): Whether to augment the puzzle data (not implemented)
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the DoubleThirdStudy.
        
        Args:
            *args: Variable length argument list passed to parent class
            **kwargs: Arbitrary keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)

    def load_handcrafted_puzzles(self, puzzles):
        """Load handcrafted chess puzzles for double third analysis.
        
        Takes a pandas DataFrame containing chess puzzles and converts them
        to LeelaBoard objects for analysis. Currently does not support data
        augmentation for double third studies.
        
        Args:
            puzzles (pd.DataFrame): DataFrame where each row contains puzzle data
                                  with FEN notation that can be processed by LeelaBoard.from_puzzle()
        
        Raises:
            NotImplementedError: If augment_data is True, as augmentation is not
                               implemented for double third studies
        
        Side Effects:
            Sets self.boards to a list of LeelaBoard objects created from the puzzles
        """

        boards = []

        if self.augment_data:
            raise NotImplementedError("Augmentation not implemented for double third study")

        for _, fen in puzzles.iterrows():
            board = LeelaBoard.from_puzzle(fen)
            boards.append(board)

        self.boards = boards