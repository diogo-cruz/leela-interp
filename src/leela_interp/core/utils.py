"""Utility functions for chess board position conversions and manipulations.

This module provides functions to convert between chess square notation (e.g., 'a1', 'h8')
and array indices, handling both white and black perspectives on the board.

Constants:
    _LETTERS: Dictionary mapping chess file letters (a-h) to indices (0-7)
    _SQUARES: List of all chess squares in algebraic notation
    _IDX2SQ: Pre-computed mappings from indices to squares for both turns
"""

_LETTERS = {letter: i for i, letter in enumerate("abcdefgh")}


def sq2idx(sq: str, turn: bool):
    """Convert chess square notation to array index.
    
    Args:
        sq: Chess square in algebraic notation (e.g., 'a1', 'h8')
        turn: True for white's perspective, False for black's perspective
        
    Returns:
        int: Array index (0-63) corresponding to the square
        
    Examples:
        >>> sq2idx('a1', True)  # White's perspective
        0
        >>> sq2idx('a1', False)  # Black's perspective  
        56
    """
    file, row = sq
    file = _LETTERS[file]
    if not turn:
        # Black's turn
        row = 9 - int(row)
    return (int(row) - 1) * 8 + file


_SQUARES = [f"{file}{rank}" for file in _LETTERS for rank in range(1, 9)]

_IDX2SQ = {
    True: {sq2idx(sq, True): sq for sq in _SQUARES},
    False: {sq2idx(sq, False): sq for sq in _SQUARES},
}


def idx2sq(idx: int, turn: bool):
    """Convert array index to chess square notation.
    
    Args:
        idx: Array index (0-63) representing a board position
        turn: True for white's perspective, False for black's perspective
        
    Returns:
        str: Chess square in algebraic notation (e.g., 'a1', 'h8')
        
    Examples:
        >>> idx2sq(0, True)  # White's perspective
        'a1'
        >>> idx2sq(56, False)  # Black's perspective
        'a1'
    """
    return _IDX2SQ[turn][idx]
