#!/usr/bin/env python3
"""
Analyze 30 chess puzzles from CSV, evaluating both branches with Stockfish.
"""

import chess
import chess.engine
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json


def parse_branch_string(branch_str: str) -> List[str]:
    """Parse branch string from CSV format to list of moves."""
    # Remove brackets and quotes, then split
    branch_str = branch_str.strip("[]'\"")
    moves = [m.strip(" '\"") for m in branch_str.split(',')]
    return moves


def play_moves_from_position(board: chess.Board, moves: List[str]) -> Tuple[chess.Board, bool, List[str]]:
    """
    Play a sequence of moves from a given position.
    Returns: (final_board, success, moves_played)
    """
    current_board = board.copy()
    moves_played = []
    
    for move_str in moves:
        try:
            # Try parsing as SAN first
            try:
                move = current_board.parse_san(move_str)
            except:
                # Try UCI format
                move = chess.Move.from_uci(move_str)
            
            if move in current_board.legal_moves:
                current_board.push(move)
                moves_played.append(move.uci())
            else:
                return current_board, False, moves_played
                
        except Exception as e:
            return current_board, False, moves_played
    
    return current_board, True, moves_played


def evaluate_position(board: chess.Board, engine: chess.engine.SimpleEngine, 
                      time_limit: float = 1.0, depth: int = 20) -> Dict[str, Any]:
    """Evaluate a position with Stockfish."""
    info = engine.analyse(board, chess.engine.Limit(time=time_limit, depth=depth))
    score = info['score']
    
    # Get score from White's perspective
    if score.is_mate():
        mate_score = score.white().mate()
        cp_score = 30000 if mate_score and mate_score > 0 else -30000
        mate_in = abs(mate_score) if mate_score else None
    else:
        cp_score = score.white().cp
        cp_score = cp_score if cp_score is not None else 0
        mate_in = None
    
    return {
        'cp': cp_score,
        'mate_in': mate_in,
        'fen': board.fen()
    }


def analyze_puzzle(puzzle_row: pd.Series, engine: chess.engine.SimpleEngine, 
                   time_limit: float = 1.0) -> Dict[str, Any]:
    """Analyze a single puzzle with both branches."""
    
    puzzle_id = puzzle_row['PuzzleId']
    fen = puzzle_row['FEN']
    moves_str = puzzle_row['Moves']
    
    # Parse branches
    branch_1 = parse_branch_string(puzzle_row['branch_1'])
    branch_2 = parse_branch_string(puzzle_row['branch_2'])
    principal_var = parse_branch_string(puzzle_row['principal_variation'])
    
    # Initial board
    board = chess.Board(fen)
    side_to_move = "White" if board.turn else "Black"
    
    # Parse the complete solution moves
    all_moves = moves_str.split()
    first_move = all_moves[0] if all_moves else None
    
    # Evaluate initial position
    initial_eval = evaluate_position(board, engine, time_limit)
    
    # Play first move
    board_after_first = board.copy()
    if first_move:
        try:
            move = board_after_first.parse_san(first_move)
            board_after_first.push(move)
        except:
            pass
    
    # Evaluate Branch 1 (playing from position after first move)
    board_b1, success_b1, moves_b1 = play_moves_from_position(board_after_first, branch_1)
    eval_b1 = evaluate_position(board_b1, engine, time_limit) if success_b1 else None
    
    # Evaluate Branch 2
    board_b2, success_b2, moves_b2 = play_moves_from_position(board_after_first, branch_2)
    eval_b2 = evaluate_position(board_b2, engine, time_limit) if success_b2 else None
    
    # Determine which branch matches principal variation
    is_branch1_pv = (branch_1 == principal_var)
    
    # Determine which branch Stockfish prefers
    stockfish_prefers = None
    if eval_b1 and eval_b2:
        if side_to_move == "White":
            stockfish_prefers = 1 if eval_b1['cp'] > eval_b2['cp'] else 2
        else:  # Black
            stockfish_prefers = 1 if eval_b1['cp'] < eval_b2['cp'] else 2
    
    return {
        'puzzle_id': puzzle_id,
        'side_to_move': side_to_move,
        'initial_cp': initial_eval['cp'],
        'first_move': first_move,
        'branch_1_success': success_b1,
        'branch_1_cp': eval_b1['cp'] if eval_b1 else None,
        'branch_2_success': success_b2,
        'branch_2_cp': eval_b2['cp'] if eval_b2 else None,
        'pv_is_branch': 1 if is_branch1_pv else 2,
        'stockfish_prefers_branch': stockfish_prefers,
        'agrees_with_pv': stockfish_prefers == (1 if is_branch1_pv else 2) if stockfish_prefers else None
    }


def main():
    # Load puzzle data
    df = pd.read_csv('study_puzzles_head30.csv')
    print(f"Loaded {len(df)} puzzles")
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
        engine.configure({"Threads": 4, "Hash": 1024})
        print("Stockfish started successfully\n")
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    results = []
    
    try:
        for idx, row in df.iterrows():
            print(f"Analyzing puzzle {idx+1}/{len(df)}: {row['PuzzleId']}", end=' ')
            
            result = analyze_puzzle(row, engine, time_limit=1.0)
            results.append(result)
            
            # Quick summary
            if result['agrees_with_pv'] is not None:
                if result['agrees_with_pv']:
                    print("✓ Stockfish agrees with PV")
                else:
                    print(f"✗ Stockfish prefers branch {result['stockfish_prefers_branch']}, PV is branch {result['pv_is_branch']}")
            else:
                print("? Could not evaluate")
    
    finally:
        engine.quit()
        print("\nStockfish closed")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('stockfish_analysis_30puzzles.csv', index=False)
    print(f"\nResults saved to stockfish_analysis_30puzzles.csv")
    
    # Analysis summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Success rate
    total = len(results_df)
    both_success = results_df[(results_df['branch_1_success']) & (results_df['branch_2_success'])].shape[0]
    print(f"\nEvaluation success: {both_success}/{total} puzzles ({100*both_success/total:.1f}%)")
    
    # Agreement with PV
    has_preference = results_df[results_df['stockfish_prefers_branch'].notna()]
    agrees = has_preference[has_preference['agrees_with_pv'] == True].shape[0]
    disagrees = has_preference[has_preference['agrees_with_pv'] == False].shape[0]
    
    print(f"\nStockfish vs Principal Variation:")
    print(f"  Agrees: {agrees}/{len(has_preference)} ({100*agrees/len(has_preference):.1f}%)")
    print(f"  Disagrees: {disagrees}/{len(has_preference)} ({100*disagrees/len(has_preference):.1f}%)")
    
    # Show disagreements
    if disagrees > 0:
        print(f"\nPuzzles where Stockfish disagrees with PV:")
        disagreements = results_df[results_df['agrees_with_pv'] == False]
        for _, row in disagreements.iterrows():
            print(f"  {row['puzzle_id']}: SF prefers branch {row['stockfish_prefers_branch']}, PV is branch {row['pv_is_branch']}")
            print(f"    Branch 1: {row['branch_1_cp']} cp, Branch 2: {row['branch_2_cp']} cp")
    
    # Average evaluation differences
    valid_evals = results_df[(results_df['branch_1_cp'].notna()) & (results_df['branch_2_cp'].notna())]
    if len(valid_evals) > 0:
        valid_evals['cp_diff'] = abs(valid_evals['branch_1_cp'] - valid_evals['branch_2_cp'])
        print(f"\nAverage CP difference between branches: {valid_evals['cp_diff'].mean():.1f} cp")
        print(f"Median CP difference: {valid_evals['cp_diff'].median():.1f} cp")
        print(f"Max CP difference: {valid_evals['cp_diff'].max():.1f} cp")


if __name__ == "__main__":
    main()