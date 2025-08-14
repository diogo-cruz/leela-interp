#!/usr/bin/env python3
"""
Evaluate chess puzzle branches using Stockfish.
Handles puzzles where the move sequences represent the opponent's responses.
"""

import chess
import chess.engine
import pandas as pd
import pickle
from pathlib import Path
import argparse
from typing import List, Dict, Any, Tuple
import json


def find_best_move_to_target(board: chess.Board, target_move_str: str) -> chess.Move:
    """
    Find the best legal move that would allow the opponent to make the target move.
    
    Args:
        board: Current board position
        target_move_str: The opponent's expected response move
    
    Returns:
        The best move to play that enables the target response
    """
    best_move = None
    
    # Try all legal moves
    for move in board.legal_moves:
        test_board = board.copy()
        test_board.push(move)
        
        # Check if opponent can make the target move
        try:
            # Try parsing target move in different formats
            try:
                opponent_move = chess.Move.from_uci(target_move_str)
            except:
                try:
                    opponent_move = test_board.parse_san(target_move_str)
                except:
                    continue
            
            if opponent_move in test_board.legal_moves:
                # This move allows the opponent to make their response
                best_move = move
                break
        except:
            continue
    
    return best_move


def evaluate_puzzle_sequence(board: chess.Board, opponent_moves: List[str], 
                            engine: chess.engine.SimpleEngine, time_limit: float = 0.1) -> Tuple[List[Dict], List[str]]:
    """
    Evaluate a puzzle sequence where we need to find moves that lead to specific opponent responses.
    
    Args:
        board: Initial chess board position  
        opponent_moves: List of opponent's response moves
        engine: Stockfish engine instance
        time_limit: Time limit per position evaluation
    
    Returns:
        Tuple of (evaluations list, our moves list)
    """
    evaluations = []
    our_moves = []
    current_board = board.copy()
    
    # Initial position evaluation
    info = engine.analyse(current_board, chess.engine.Limit(time=time_limit))
    score = info['score']
    
    if score.is_mate():
        mate_score = score.white().mate() if board.turn == chess.WHITE else score.black().mate()
        initial_cp = 10000 if mate_score and mate_score > 0 else -10000
    else:
        initial_cp = score.white().cp if board.turn == chess.WHITE else score.black().cp
        if initial_cp is None:
            initial_cp = 0
    
    evaluations.append({
        'move_num': 0,
        'move': 'initial',
        'cp_score': initial_cp,
        'fen_after': current_board.fen()
    })
    
    for i, opponent_move_str in enumerate(opponent_moves):
        # First, we need to make a move that allows this opponent response
        
        # Get Stockfish's best move for us
        result = engine.play(current_board, chess.engine.Limit(time=time_limit))
        our_move = result.move
        
        # Make our move
        current_board.push(our_move)
        our_moves.append(str(our_move))
        
        # Evaluate after our move
        info = engine.analyse(current_board, chess.engine.Limit(time=time_limit))
        score = info['score']
        
        if score.is_mate():
            mate_score = score.white().mate() if current_board.turn == chess.WHITE else score.black().mate()
            cp_score = 10000 if mate_score and mate_score > 0 else -10000
        else:
            cp_score = score.white().cp if current_board.turn == chess.WHITE else score.black().cp
            if cp_score is None:
                cp_score = 0
        
        evaluations.append({
            'move_num': i * 2 + 1,
            'move': f"our: {our_move}",
            'cp_score': cp_score,
            'fen_after': current_board.fen()
        })
        
        # Now make the opponent's response
        try:
            # Try different formats
            try:
                opp_move = chess.Move.from_uci(opponent_move_str)
                if opp_move not in current_board.legal_moves:
                    raise ValueError("Not legal")
            except:
                opp_move = current_board.parse_san(opponent_move_str)
            
            current_board.push(opp_move)
            
            # Evaluate after opponent's move
            info = engine.analyse(current_board, chess.engine.Limit(time=time_limit))
            score = info['score']
            
            if score.is_mate():
                cp_score = 10000 if score.mate() > 0 else -10000
            else:
                cp_score = score.relative.cp if score.relative else 0
            
            evaluations.append({
                'move_num': i * 2 + 2,
                'move': f"opp: {opponent_move_str}",
                'cp_score': cp_score,
                'fen_after': current_board.fen()
            })
            
        except Exception as e:
            print(f"  Error with opponent move {opponent_move_str}: {e}")
            evaluations.append({
                'move_num': i * 2 + 2,
                'move': f"opp: {opponent_move_str}",
                'cp_score': None,
                'error': str(e)
            })
            break
    
    return evaluations, our_moves


def evaluate_puzzle_branches(puzzle_data: pd.DataFrame, stockfish_path: str = "stockfish",
                            time_limit: float = 0.5, depth: int = 20) -> pd.DataFrame:
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
        return puzzle_data
    
    try:
        for idx, row in puzzle_data.iterrows():
            print(f"\nEvaluating puzzle {idx + 1}/{len(puzzle_data)}: {row['PuzzleId']}")
            
            # Create board from FEN
            board = chess.Board(row['FEN'])
            print(f"  Initial position: {'White' if board.turn else 'Black'} to move")
            
            # Evaluate branch 1 (these are opponent's moves)
            print(f"  Branch 1 (opponent responses): {row['branch_1']}")
            branch1_evals, our_moves_1 = evaluate_puzzle_sequence(board, row['branch_1'], engine, time_limit)
            
            # Evaluate branch 2
            print(f"  Branch 2 (opponent responses): {row['branch_2']}")
            branch2_evals, our_moves_2 = evaluate_puzzle_sequence(board, row['branch_2'], engine, time_limit)
            
            # Get final evaluations
            branch1_final = branch1_evals[-1]['cp_score'] if branch1_evals else None
            branch2_final = branch2_evals[-1]['cp_score'] if branch2_evals else None
            
            # Store results
            result = {
                'PuzzleId': row['PuzzleId'],
                'FEN': row['FEN'],
                'side_to_move': 'White' if board.turn else 'Black',
                'branch_1_opponent': row['branch_1'],
                'branch_1_our_moves': our_moves_1,
                'branch_2_opponent': row['branch_2'],
                'branch_2_our_moves': our_moves_2,
                'branch_1_evaluations': branch1_evals,
                'branch_2_evaluations': branch2_evals,
                'branch_1_final_cp': branch1_final,
                'branch_2_final_cp': branch2_final,
            }
            
            # Determine which branch is better
            if branch1_final is not None and branch2_final is not None:
                # Higher is better for white, lower for black
                if board.turn == chess.WHITE:
                    result['better_branch'] = 1 if branch1_final > branch2_final else 2
                else:
                    result['better_branch'] = 1 if branch1_final < branch2_final else 2
                result['cp_difference'] = abs(branch1_final - branch2_final)
            else:
                result['better_branch'] = None
                result['cp_difference'] = None
            
            results.append(result)
            
            # Print summary
            print(f"  Branch 1 final evaluation: {branch1_final} cp")
            print(f"  Branch 2 final evaluation: {branch2_final} cp")
            if result['better_branch']:
                print(f"  Better branch: {result['better_branch']} (difference: {result['cp_difference']} cp)")
    
    finally:
        engine.quit()
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate chess puzzle branches with Stockfish')
    parser.add_argument('--input', type=str, help='Input pickle file with puzzle data')
    parser.add_argument('--output', type=str, default='stockfish_evaluations_v2.pkl',
                       help='Output pickle file for results')
    parser.add_argument('--stockfish-path', type=str, default='/usr/games/stockfish',
                       help='Path to Stockfish executable')
    parser.add_argument('--time-limit', type=float, default=0.5,
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
    
    # Print detailed summary
    print("\n=== Detailed Summary ===")
    for _, row in results_df.iterrows():
        print(f"\nPuzzle {row['PuzzleId']} ({row['side_to_move']} to move):")
        print(f"  Branch 1:")
        print(f"    Our moves: {row['branch_1_our_moves']}")
        print(f"    Opponent responses: {row['branch_1_opponent']}")
        print(f"    Final eval: {row['branch_1_final_cp']} cp")
        print(f"  Branch 2:")
        print(f"    Our moves: {row['branch_2_our_moves']}")
        print(f"    Opponent responses: {row['branch_2_opponent']}")
        print(f"    Final eval: {row['branch_2_final_cp']} cp")
        if row['better_branch']:
            print(f"  Stockfish prefers: Branch {row['better_branch']} (by {row['cp_difference']} cp)")


if __name__ == "__main__":
    main()