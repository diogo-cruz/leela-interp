"""Double attention study module for analyzing chess position attention patterns.

This module extends the FifthMoveStudy class to analyze double attention patterns
in chess positions. It provides functionality to load and visualize attention
heads data for different chess puzzles and piece movement patterns.
"""

from leela_interp.core.fifth_move_study import FifthMoveStudy
import os
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from leela_interp.tools import figure_helpers as fh
import numpy as np
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)


class DoubleAttentionStudy(FifthMoveStudy):
    """Study class for analyzing double attention patterns in chess positions.
    
    This class extends FifthMoveStudy to provide specialized functionality for
    analyzing attention patterns across different chess puzzle sets and piece
    movement types.
    
    Attributes:
        double_attention_sets (dict): Dictionary containing loaded double attention
            data organized by tag, possibility, and order.
    """
    
    def __init__(self, *args, load_all=True, **kwargs):
        """Initialize the DoubleAttentionStudy.
        
        Args:
            *args: Variable length argument list passed to parent class.
            load_all (bool): Whether to load all data sets. Defaults to True.
            **kwargs: Arbitrary keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs, load_all=load_all)
        self.load_puzzle_sets()
        self.load_effect_sets()
        self.load_attention_sets()
        self.load_double_attention_sets()

    def load_double_attention_sets(self):
        """Load double attention sets from saved result files.
        
        Scans the results/global_patching directory for attention head result files
        and loads them into the double_attention_sets dictionary organized by tag,
        possibility, and order.
        """
        self.double_attention_sets = {}
        for filename in os.listdir("results/global_patching"):
            match = re.search(rf'interesting_puzzles{self.puzzlename}_([a-zA-Z])_(\d+)_(\d+)_attention_head_results\.pt$', filename)
            if match:
                tag, possibility, order = match.group(1), match.group(2), match.group(3)
                if tag not in self.double_attention_sets:
                    self.double_attention_sets[tag] = {}
                with open(f"results/global_patching/{filename}", "rb") as f:
                    if possibility not in self.double_attention_sets[tag]:
                        self.double_attention_sets[tag][possibility] = {}
                    self.double_attention_sets[tag][possibility][order] = torch.load(f)

    def plot_attention(self, tag, possibility, vmax=0.5, filename=None, apply_abs=False, mask=None):
        """Plot attention heatmap for a specific tag and possibility.
        
        Creates a heatmap visualization of attention patterns across layers and heads,
        with annotations for different piece types (Knight, Bishop, Rook).
        
        Args:
            tag (str): Tag identifier for the attention set.
            possibility (str): Possibility identifier within the tag.
            vmax (float): Maximum value for the colormap scale. Defaults to 0.5.
            filename (str, optional): If provided, saves the figure to this filename.
            apply_abs (bool): Whether to apply absolute value to effects. Defaults to False.
            mask (optional): Mask to apply to the attention data.
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
        if mask is not None:
            effects = self.attention_sets[tag][possibility][mask]
        else:
            effects = self.attention_sets[tag][possibility]

        # Calculate the effects for the single position
        if apply_abs:
            position_effects = torch.abs(effects).mean(dim=0).cpu().numpy()  # Use [0] to get the first (and only) item
        else:
            position_effects = -effects.mean(dim=0).cpu().numpy()  # Use [0] to get the first (and only) item

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
        """Plot a grid of attention heatmaps for multiple possibilities.
        
        Creates a grid layout of attention heatmaps, each showing attention patterns
        for different possibilities within the same tag. Each subplot includes
        piece type annotations.
        
        Args:
            tag (str): Tag identifier for the attention set.
            possibilities (list): List of possibility identifiers to plot.
            n_cols (int): Number of columns in the grid. Defaults to 4.
            vmax (float): Maximum value for the colormap scale. Defaults to 0.5.
            filename (str, optional): If provided, saves the figure to this filename.
        """

        n_plots = len(possibilities)
        n_rows = (n_plots + n_cols - 1) // n_cols  # Equivalent to math.ceil without import

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
        
        
        
        
        
