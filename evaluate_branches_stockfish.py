#!/usr/bin/env python3
"""
Evaluate chess puzzle branches using Stockfish.
"""

import chess
import chess.engine
import pandas as pd
import pickle
from pathlib import Path
import argparse
from typing import List, Dict, Any
import json


def evaluate_position_sequence(board: chess.Board, moves: List[str], engine: chess.engine.SimpleEngine, time_limit: float = 0.1) -> List[Dict[str, Any]]:
    """
    Evaluate a sequence of moves from a given position.
    
    Args:
        board: Initial chess board position
        moves: List of moves in UCI format
        engine: Stockfish engine instance
        time_limit: Time limit per position evaluation in seconds
    
    Returns:
        List of evaluation results for each position in the sequence
    """
    evaluations = []
    current_board = board.copy()
    
    for i, move_str in enumerate(moves):
        # Parse and make the move
        try:
            # First try as UCI format
            try:
                move = chess.Move.from_uci(move_str)
                if move not in current_board.legal_moves:
                    raise ValueError("Not a legal move")
            except:
                # Try to parse as SAN format
                try:
                    move = current_board.parse_san(move_str)
                except:
                    # Last attempt: try the move from opponent's perspective
                    # This handles cases where moves are recorded from wrong side
                    print(f"Warning: Trying to interpret {move_str} from opponent's perspective")
                    # Make a null move to switch sides
                    test_board = current_board.copy()
                    test_board.push(chess.Move.null())
                    move = test_board.parse_san(move_str)
                    # But apply to the correct board
                    if move not in current_board.legal_moves:
                        # The move was meant for the other side, find equivalent
                        # Check if any legal move matches the target square
                        from_sq = chess.parse_square(move_str[:2])
                        to_sq = chess.parse_square(move_str[2:4])
                        for legal_move in current_board.legal_moves:
                            if legal_move.to_square == to_sq:
                                move = legal_move
                                print(f"  Corrected to: {move}")
                                break
                        else:
                            raise ValueError(f"Could not find legal move for {move_str}")
            
            current_board.push(move)
            
            # Evaluate the position after this move
            info = engine.analyse(current_board, chess.engine.Limit(time=time_limit))
            
            score = info['score']
            if score.is_mate():
                # Convert mate score to centipawns (large value)
                cp_score = 10000 if score.mate() > 0 else -10000
                mate_in = score.mate()
            else:
                cp_score = score.relative.cp
                mate_in = None
            
            evaluations.append({
                'move_num': i + 1,
                'move': move_str,
                'cp_score': cp_score,
                'mate_in': mate_in,
                'depth': info.get('depth', None),
                'fen_after': current_board.fen()
            })
            
        except Exception as e:
            print(f"Error processing move {move_str} at position {i}: {e}")
            evaluations.append({
                'move_num': i + 1,
                'move': move_str,
                'cp_score': None,
                'mate_in': None,
                'depth': None,
                'fen_after': None,
                'error': str(e)
            })
            break
    
    return evaluations


def evaluate_puzzle_branches(puzzle_data: pd.DataFrame, stockfish_path: str = "stockfish", 
                            time_limit: float = 0.1, depth: int = 20) -> pd.DataFrame:
    """
    Evaluate both branches for each puzzle using Stockfish.
    
    Args:
        puzzle_data: DataFrame containing puzzle information
        stockfish_path: Path to Stockfish executable
        time_limit: Time limit per position evaluation
        depth: Maximum search depth
    
    Returns:
        DataFrame with evaluation results added
    """
    results = []
    
    # Start Stockfish engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        engine.configure({"Threads": 4, "Hash": 512})
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        print("Please ensure Stockfish is installed and provide the correct path.")
        print("You can install it with: sudo apt-get install stockfish (on Ubuntu)")
        print("Or download from: https://stockfishchess.org/download/")
        return puzzle_data
    
    try:
        for idx, row in puzzle_data.iterrows():
            print(f"\nEvaluating puzzle {idx + 1}/{len(puzzle_data)}: {row['PuzzleId']}")
            
            # Create board from FEN
            board = chess.Board(row['FEN'])
            
            # Evaluate branch 1
            print(f"  Branch 1: {row['branch_1']}")
            branch1_evals = evaluate_position_sequence(board, row['branch_1'], engine, time_limit)
            
            # Evaluate branch 2
            print(f"  Branch 2: {row['branch_2']}")
            branch2_evals = evaluate_position_sequence(board, row['branch_2'], engine, time_limit)
            
            # Store results
            result = {
                'PuzzleId': row['PuzzleId'],
                'FEN': row['FEN'],
                'branch_1': row['branch_1'],
                'branch_2': row['branch_2'],
                'branch_1_evaluations': branch1_evals,
                'branch_2_evaluations': branch2_evals,
                'branch_1_final_cp': branch1_evals[-1]['cp_score'] if branch1_evals else None,
                'branch_2_final_cp': branch2_evals[-1]['cp_score'] if branch2_evals else None,
            }
            
            # Determine which branch is better (from the perspective of the side to move)
            if result['branch_1_final_cp'] is not None and result['branch_2_final_cp'] is not None:
                # Higher score is better for white, lower for black
                if board.turn == chess.WHITE:
                    result['better_branch'] = 1 if result['branch_1_final_cp'] > result['branch_2_final_cp'] else 2
                else:
                    result['better_branch'] = 1 if result['branch_1_final_cp'] < result['branch_2_final_cp'] else 2
                result['cp_difference'] = abs(result['branch_1_final_cp'] - result['branch_2_final_cp'])
            else:
                result['better_branch'] = None
                result['cp_difference'] = None
            
            results.append(result)
            
            # Print summary
            print(f"  Branch 1 final evaluation: {result['branch_1_final_cp']} cp")
            print(f"  Branch 2 final evaluation: {result['branch_2_final_cp']} cp")
            if result['better_branch']:
                print(f"  Better branch: {result['better_branch']} (difference: {result['cp_difference']} cp)")
    
    finally:
        engine.quit()
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate chess puzzle branches with Stockfish')
    parser.add_argument('--input', type=str, help='Input pickle file with puzzle data')
    parser.add_argument('--output', type=str, default='stockfish_evaluations.pkl', 
                       help='Output pickle file for results')
    parser.add_argument('--stockfish-path', type=str, default='stockfish',
                       help='Path to Stockfish executable')
    parser.add_argument('--time-limit', type=float, default=0.1,
                       help='Time limit per position in seconds')
    parser.add_argument('--depth', type=int, default=20,
                       help='Maximum search depth')
    parser.add_argument('--from-notebook', action='store_true',
                       help='Use the sample data from the notebook')
    
    args = parser.parse_args()
    
    if args.from_notebook:
        # Create sample data from the notebook
        data = {
            'PuzzleId': ['0hgJT', '1zJDp', '2JetC'],
            'FEN': [
                '8/b7/P4k2/1N2p3/8/3r3P/R4p1K/8 w - - 0 50',
                '2k2r1r/2p5/5pp1/2pppn2/Q7/4PqB1/PP3P1P/1K1R2R1 b - - 1 27',
                'r5k1/5p1p/6p1/1B3q2/P2R1n1P/2B2P2/KP3P2/4R3 b - - 0 33'
            ],
            'branch_1': [
                ['d3g3', 'g2g3', 'f2f1q'],
                ['g1g3', 'f3e4', 'a4e4'],
                ['d4d8', 'a8d8', 'a4b5']
            ],
            'branch_2': [
                ['d3d1', 'a2f2', 'a7f2'],
                ['a4a8', 'c8d7', 'd1d5'],
                ['d4f4', 'a8a4', 'f4a4']
            ]
        }
        puzzle_df = pd.DataFrame(data)
    elif args.input:
        # Load puzzle data from pickle file
        with open(args.input, 'rb') as f:
            puzzle_df = pickle.load(f)
    else:
        print("Please provide --input file or use --from-notebook flag")
        return
    
    # Evaluate branches
    results_df = evaluate_puzzle_branches(
        puzzle_df, 
        stockfish_path=args.stockfish_path,
        time_limit=args.time_limit,
        depth=args.depth
    )
    
    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump(results_df, f)
    
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n=== Summary ===")
    for _, row in results_df.iterrows():
        print(f"\nPuzzle {row['PuzzleId']}:")
        print(f"  Branch 1 final: {row['branch_1_final_cp']} cp")
        print(f"  Branch 2 final: {row['branch_2_final_cp']} cp")
        if row['better_branch']:
            print(f"  Better: Branch {row['better_branch']} (by {row['cp_difference']} cp)")


if __name__ == "__main__":
    main()