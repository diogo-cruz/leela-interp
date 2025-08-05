"""Alternative moves analysis for Leela chess engine.

This module provides functionality to analyze alternative moves and create game trees
for chess positions using the Leela Chess Zero model. It includes functions to:
- Get top moves from a position with recursive depth analysis
- Check for double-branching game scenarios
- Visualize game trees using NetworkX and matplotlib
"""

from matplotlib import pyplot as plt
import torch
import networkx as nx
from leela_interp import Lc0Model, LeelaBoard

def get_top_moves(
    model: Lc0Model,
    board: LeelaBoard,
    limit = 3,
    min_prob: float | list[float] = 0.1,
) -> dict[str, dict[str, float]]:
    """Get top moves from a chess position recursively.
    
    Analyzes the given board position and returns the top moves with their probabilities,
    recursively exploring move trees up to a specified depth limit.
    
    Args:
        model: The Leela Chess Zero model to use for move prediction
        board: The chess board position to analyze
        limit: Maximum depth to explore (default: 3)
        min_prob: Minimum probability threshold for including moves.
                 Can be a float or list of floats for different depths
    
    Returns:
        Dict mapping move strings to dictionaries containing 'prob' and nested moves
        
    Example:
        {'e4': {'prob': 0.3, 'e5': {'prob': 0.4}}, 'Nf3': {'prob': 0.25}}
    """
    
    if limit == 0:
        return {}

    if not isinstance(min_prob, float):
        tol = min_prob[limit-1]
    else:
        tol = min_prob

    policy, _, _ = model.batch_play([board], return_probs=True)
    not_nan = ~torch.isnan(policy).any(-1)
    num_not_nan = not_nan.sum().item()
    assert isinstance(num_not_nan, int)  # make the type checker happy
    assert torch.allclose(
        policy[not_nan].sum(-1),
        torch.ones(num_not_nan, device=policy.device),
    ), policy.sum(-1)

    try:
        top_moves = model.top_moves(board, policy[0], top_k=None)
    except AssertionError:
        #print(board)
        #print(policy)
        top_moves = {}

    # Remove all entries where val is not above 0.1
    top_moves = {k: v for k, v in top_moves.items() if v > tol}
    model_moves = list(top_moves.keys())

    top_dict = {}
    for model_move in model_moves:
        new_board = board.copy()
        new_board.push_uci(model_move)
        top_dict_partial = get_top_moves(model, new_board, limit-1, min_prob=min_prob)
        top_dict[model_move] = {'prob': top_moves[model_move]}
        top_dict[model_move] = top_dict[model_move] | top_dict_partial

    return top_dict

def get_double_branch_moves(
    model: Lc0Model,
    board: LeelaBoard,
    limit: int = 0,
    end: int = 3,
    min_prob: float | list[float] = 0.1,
) -> dict[str, dict[str, float]] | None:
    """Get moves for double-branching game scenarios.
    
    Analyzes positions where the game tree branches into exactly two viable paths
    at each level. Returns None if the position doesn't satisfy double-branching criteria.
    
    Args:
        model: The Leela Chess Zero model to use for move prediction
        board: The chess board position to analyze
        limit: Current depth level (default: 0)
        end: Maximum depth to explore (default: 3)
        min_prob: Minimum probability threshold for including moves.
                 Can be a float or list of floats for different depths
    
    Returns:
        Dict mapping move strings to nested move dictionaries, or None if criteria not met
        
    Note:
        At depth 0, expects exactly 2 moves above threshold.
        At other depths, expects exactly 1 move above threshold.
    """
    
    if limit == end:
        return {}

    if not isinstance(min_prob, float):
        tol = min_prob[limit]
    else:
        tol = min_prob

    policy, _, _ = model.batch_play([board], return_probs=True)
    not_nan = ~torch.isnan(policy).any(-1)
    num_not_nan = not_nan.sum().item()
    assert isinstance(num_not_nan, int)  # make the type checker happy
    assert torch.allclose(
        policy[not_nan].sum(-1),
        torch.ones(num_not_nan, device=policy.device),
    ), policy.sum(-1)

    try:
        top_moves = model.top_moves(board, policy[0], top_k=None)
    except AssertionError:
        return None

    # Remove all entries where val is not above 0.1
    top_moves = {k: v for k, v in top_moves.items() if v > tol}
    if limit == 0:
        if len(top_moves) != 2:
            return None
    else:
        if len(top_moves) != 1:
            return None
    model_moves = list(top_moves.keys())

    top_dict = {}
    for model_move in model_moves:
        new_board = board.copy()
        new_board.push_uci(model_move)
        top_dict_partial = get_double_branch_moves(model, new_board, limit+1, end, min_prob=min_prob)
        if top_dict_partial is None:
            return None
        top_dict[model_move] = {'prob': top_moves[model_move]}
        top_dict[model_move] = top_dict[model_move] | top_dict_partial

    return top_dict

def check_if_double_game(model: Lc0Model, puzzle):
    """Check if a puzzle represents a double-branching game scenario.
    
    Analyzes a chess puzzle to determine if it follows a double-branching pattern
    where the game tree consistently branches into exactly 2-3 viable moves at each level.
    
    Args:
        model: The Leela Chess Zero model to use for move prediction
        puzzle: Chess puzzle object containing FEN and correct moves
    
    Returns:
        False if puzzle doesn't match double-game criteria, or
        Tuple of (correct_moves, total_moveset) if it does match
        
    Note:
        Uses the slower but more thorough get_top_moves function.
    """
    #display(puzzle)
    board = LeelaBoard.from_puzzle(puzzle)
    correct_moves = puzzle.Moves.split()
    correct_moves[0] = 'init'
    total_moveset = {correct_moves[0]: {'prob': 1.0} | get_top_moves(model, board, limit=3, min_prob=0.1)}
    
    zeroth_move = list(total_moveset)[0]
    first_round_moves = total_moveset[zeroth_move]
    if len(first_round_moves) != 3:
        return False
    
    for first_move, second_round_moves in first_round_moves.items():
        if first_move == 'prob':
            continue
        if len(second_round_moves) != 2:
            return False
        for second_move, third_round_moves in second_round_moves.items():
            if second_move == 'prob':
                continue
            if len(third_round_moves) != 2:
                return False
    
    return (correct_moves, total_moveset)

def check_if_double_game_fast(model: Lc0Model, puzzle, end: int = 3, min_prob: float | list[float] = 0.1):
    """Fast check for double-branching game scenarios.
    
    Optimized version of check_if_double_game that uses get_double_branch_moves
    for more efficient analysis of double-branching patterns.
    
    Args:
        model: The Leela Chess Zero model to use for move prediction
        puzzle: Chess puzzle object containing FEN and correct moves
        end: Maximum depth to explore (default: 3)
        min_prob: Minimum probability threshold for including moves.
                 Can be a float or list of floats for different depths
    
    Returns:
        False if puzzle doesn't match double-game criteria, or
        Tuple of (correct_moves, total_moveset) if it does match
        
    Note:
        More efficient than check_if_double_game but with stricter branching criteria.
    """
    #display(puzzle)
    board = LeelaBoard.from_puzzle(puzzle)
    correct_moves = puzzle.Moves.split()
    correct_moves[0] = 'init'
    top_moves = get_double_branch_moves(model, board, limit=0, end=end, min_prob=min_prob)
    if top_moves == {} or top_moves is None:
        return False

    #print(top_moves)

    total_moveset = {correct_moves[0]: {'prob': 1.0} | top_moves}
    
    # zeroth_move = list(total_moveset)[0]
    # first_round_moves = total_moveset[zeroth_move]
    # if len(first_round_moves) != 3:
    #     return False
    
    # for first_move, second_round_moves in first_round_moves.items():
    #     if first_move == 'prob':
    #         continue
    #     if len(second_round_moves) != 2:
    #         return False
    #     for second_move, third_round_moves in second_round_moves.items():
    #         if second_move == 'prob':
    #             continue
    #         if len(third_round_moves) != 2:
    #             return False
    
    return (correct_moves, total_moveset)


def create_tree_graph(tree, parent_path=None, graph=None):
    """Create a NetworkX directed graph from a move tree.
    
    Recursively converts a nested move dictionary into a NetworkX DiGraph
    suitable for visualization.
    
    Args:
        tree: Nested dictionary representing the move tree
        parent_path: Path string of the parent node (for recursion)
        graph: Existing NetworkX DiGraph to add to (created if None)
    
    Returns:
        NetworkX DiGraph with nodes representing moves and edges representing sequences
        
    Note:
        Node labels include move notation and probability values.
        Paths are constructed by concatenating parent_path with move strings.
    """
    if graph is None:
        graph = nx.DiGraph()
    
    for move, data in tree.items():
        prob = data['prob']
        path = f"{parent_path}_{move}" if parent_path else move
        label = f"{move}\n{prob:.2f}"
        graph.add_node(path, label=label)
        
        if parent_path:
            graph.add_edge(parent_path, path)
        
        subtree = {k: v for k, v in data.items() if k != 'prob'}
        if subtree:
            create_tree_graph(subtree, path, graph)
    
    return graph

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    """Calculate hierarchical positions for tree graph visualization.
    
    Computes node positions for a tree-like layout where nodes are arranged
    in levels with proper spacing and centering.
    
    Args:
        G: NetworkX graph to position
        root: Root node to start from (auto-detected if None)
        width: Total width available for positioning (default: 1.0)
        vert_gap: Vertical gap between levels (default: 0.2)
        vert_loc: Starting vertical location (default: 0)
        xcenter: Horizontal center point (default: 0.5)
    
    Returns:
        Dictionary mapping node names to (x, y) coordinate tuples
        
    Note:
        Uses recursive helper function to calculate positions level by level.
    """
    if root is None:
        roots = [n for n,d in G.in_degree() if d==0]
        if len(roots) > 1:
            root = roots[0]
        else:
            root = roots[0]

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def plot_game_trees(puzzle_movesets, filename=None, width=6, height=4):
    """Plot game trees for multiple puzzle movesets.
    
    Creates matplotlib visualizations of game trees, highlighting correct moves
    in green and alternative moves in blue.
    
    Args:
        puzzle_movesets: List of tuples containing (correct_moves, moveset) pairs
        filename: Optional filename to save plots (default: None, display only)
        width: Figure width in inches (default: 6)
        height: Figure height in inches (default: 4)
    
    Note:
        - Correct move paths are colored light green
        - Alternative move paths are colored light blue
        - Each puzzle generates a separate plot
        - Plots are displayed and optionally saved to file
    """
    for correct_moves, moveset in puzzle_movesets:
        graph = create_tree_graph(moveset)
        pos = hierarchy_pos(graph)
        node_colors = ['lightblue' if node.split('_') != correct_moves[:len(node.split('_'))] else 'lightgreen' for node in graph.nodes()]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(width, height))

        # Draw the graph
        nx.draw(graph, pos, with_labels=False, node_size=1200, node_color=node_colors, 
                font_size=12, font_weight='bold', arrows=True)

        # Add labels to nodes
        labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_labels(graph, pos, labels, font_size=12)

        # Save the figure
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        if filename:
            fig.savefig(filename, bbox_inches='tight')