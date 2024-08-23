
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
from leela_interp.tools import figure_helpers as fh
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)
from datetime import datetime
from leela_interp.tools.attention import attention_attribution, top_k_attributions
from leela_interp.tools.patching import activation_patch

class DoubleMateStudy:
    
    def __init__(self, model, fens, augment_data=True, load_all=True, save_plots=False, min_prob=0.1, limit=4):
        self.model = model
        # Bad puzzle
        #min_prob = [0.45, 0.2, 0.2, 0.05]
        # Good puzzle
        #min_prob = [0.3, 0.3, 0.2, 0.1]
        # Simple puzzle
        #min_prob = [0.3, 0.3, 0.1, 0.1, 0.1, 0.05]
        self.min_prob = min_prob
        self.limit = limit
        self.augment_data = augment_data
        self.save_plots = save_plots
        self.filetime = datetime.strftime(datetime.now(), '%YY%mM%dD%Hh%Mm%Ss')
        fh.set()
        self.load_handcrafted_puzzles(fens)
        if load_all:
            self.load_moveset()
            self.load_game_trees()
            self.get_attributions()
        if save_plots:
            self.save_boards()
            self.save_trees()
            self.save_attributions()

    def save_boards(self):
        for i, board in enumerate(self.boards):
            board.plot(show_lastmove=False).render(filename='double_mate/' + self.filetime + '_board_' + str(i) + '.svg')
        
        # Convert saved SVGs to PDFs
        svg_files = Path('double_mate').glob(f'{self.filetime}_board_*.svg')
        for svg_file in svg_files:
            pdf_file = svg_file.with_suffix('.pdf')
            os.system(f'inkscape --export-filename={pdf_file} {svg_file}')
            os.remove(svg_file)


    def save_trees(self):
        for i, tree in enumerate(self.game_trees):
            tree.savefig('double_mate/' + self.filetime + '_tree_' + str(i) + '.pdf', bbox_inches='tight')

    def save_attributions(self):
        for i, board in enumerate(self.attribution_boards):
            board.render(filename='double_mate/' + self.filetime + '_attribution_' + str(i) + '.svg')
        
        svg_files = Path('double_mate').glob(f'{self.filetime}_attribution_*.svg')
        for svg_file in svg_files:
            pdf_file = svg_file.with_suffix('.pdf')
            os.system(f'inkscape --export-filename={pdf_file} {svg_file}')
            os.remove(svg_file)

    def load_game_trees(self):
        self.game_trees = []
        for moveset in self.movesets:
            self.game_trees.append(self.tree_figure(moveset))

    @staticmethod
    def tree_figure(moveset, width=6, height=4):
        graph = create_tree_graph(moveset)
        pos = hierarchy_pos(graph)
        
        # Find all paths from root to leaves
        root = 'init'
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        # Find paths of length 3 (4 nodes including root and leaf)
        mate_nodes = set()
        for leaf in leaves:
            for path in nx.all_simple_paths(graph, root, leaf):
                if len(path) == 4:  # Path of length 3 (4 nodes)
                    mate_nodes.update(path)
        #print(root, leaves, mate_nodes)

        node_colors = ['lightgreen' if node in mate_nodes else 'lightblue' for node in graph.nodes()]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(width, height))

        # Draw the graph
        nx.draw(graph, pos, with_labels=False, node_size=1200, node_color=node_colors, 
                font_size=12, font_weight='bold', arrows=True, ax=ax)

        # Add labels to nodes
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_labels(graph, pos, labels, font_size=12, ax=ax)

        # Save the figure
        ax.axis('off')

        return fig

    def load_moveset(self):
        self.movesets = []
        for board in self.boards:
            moveset = get_top_moves(self.model, board, limit=self.limit, min_prob=self.min_prob)
            total_moveset = {'init': {'prob': 1.0} | moveset}
            self.movesets.append(total_moveset)
    
    def load_handcrafted_puzzles(self, fens):

        fen_list = fens.copy()

        boards = []

        if self.augment_data:
            fen_list = self._augment_fen_list(fen_list)

        for fen in fen_list:
            board = LeelaBoard.from_fen(fen)
            boards.append(board)

        #print(fen_list)

        fen_list = self._delete_duplicates(fen_list)

        self.boards = boards

    def _delete_duplicates(self, fen_list):
        fen_list = list(set(fen_list))
        return fen_list

    def _augment_fen_list(self, fen_list, mirror=True, swap_colors=True, add_noise=False):
        if add_noise:
            raise NotImplementedError("Add noise not implemented")
            fen_list = self._add_noise(fen_list)
        if mirror:
            fen_list = self._mirror(fen_list)
        if swap_colors:
            fen_list = self._swap_colors(fen_list)
        return fen_list
    
    def _add_noise(self, fen_list):
        fen_list += [self._add_noise_fen(fen) for fen in fen_list]
        return fen_list

    def _add_noise_fen(self, fen):
        pass

    def _mirror(self, fen_list):
        fen_list += [self._mirror_fen(fen) for fen in fen_list]
        return fen_list

    def _mirror_fen(self, fen):
        # Split the FEN string into its components
        board, turn, castling, en_passant, halfmove, fullmove = fen.split()

        # Mirror the board horizontally
        rows = board.split('/')
        mirrored_rows = [row[::-1] for row in rows]
        mirrored_board = '/'.join(mirrored_rows)

        # Mirror castling rights
        mirrored_castling = castling.translate(str.maketrans('KQkq', 'QKqk'))

        # Mirror en passant square if it exists
        mirrored_en_passant = en_passant
        if en_passant != '-':
            file = chr(ord('h') - (ord(en_passant[0]) - ord('a')))
            mirrored_en_passant = file + en_passant[1]

        # Combine the components back into a FEN string
        mirrored_fen = f"{mirrored_board} {turn} {mirrored_castling} {mirrored_en_passant} {halfmove} {fullmove}"

        return mirrored_fen

    def _swap_colors(self, fen_list):
        fen_list += [self._swap_colors_fen(fen) for fen in fen_list]
        return fen_list
    
    def _swap_colors_fen(self, fen):
        # Split the FEN string into its components
        board, turn, castling, en_passant, halfmove, fullmove = fen.split()

        # Swap the pieces' colors
        new_board = board.swapcase()

        # Swap the turn
        new_turn = 'b' if turn == 'w' else 'w'

        # Swap the castling rights
        new_castling = castling.swapcase() if castling != '-' else '-'

        # Swap the en passant square if it exists
        new_en_passant = en_passant
        if en_passant != '-':
            rank = '3' if en_passant[1] == '6' else '6'
            new_en_passant = en_passant[0] + rank

        # Combine the components back into a FEN string
        new_fen = f"{new_board} {new_turn} {new_castling} {new_en_passant} {halfmove} {fullmove}"

        return new_fen

    def get_attributions(self):
        self.attribution_boards = []
        self.attribution_values = []
        for board in self.boards:
            attribution = attention_attribution(
                [board], layer=12, head=12, model=self.model, return_pt=True
            )[0]
            values, colors = top_k_attributions(attribution, board, k=64*64)
            colors = {k: v for k, v in reversed(list(colors.items())[:4])}
            self.attribution_values.append(values)
            self.attribution_boards.append(board.plot(arrows=colors, show_lastmove=False))

        if self.save_plots:
            self.save_values()

    def save_values(self):
        
        for i, values in enumerate(self.attribution_values):
            fig, ax = plt.subplots(figsize=(5, 3))

            # Plot the data
            x = np.arange(1, len(values)+1)
            y = np.abs(list(values.values()))
            ax.plot(x, y, marker='o', linestyle='-', color='#1f77b4', markersize=5, linewidth=1)

            # Set scales and labels
            ax.set_xscale('log')
            ax.set_xlim(0.9, 100)
            ax.set_yscale('log')
            ax.set_ylim(y[100], y[0]*1.1)
            ax.set_xlabel('Rank')
            ax.set_ylabel('Absolute move attribution value')

            fig.savefig('double_mate/' + self.filetime + '_values_' + str(i) + '.pdf', bbox_inches='tight')


