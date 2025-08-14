#!/usr/bin/env python3
"""
Direct evaluation of chess puzzle branches using Stockfish.
This version directly plays out the given move sequences and evaluates the resulting positions.
"""

import chess
import chess.engine
import pandas as pd
import pickle
import argparse
from typing import List, Dict, Any


def play_and_evaluate_sequence(initial_fen: str, moves: List[str], engine: chess.engine.SimpleEngine, 
                               time_limit: float = 0.5) -> Dict[str, Any]:
    """
    Play out a sequence of moves and evaluate positions.
    
    Args:
        initial_fen: Starting position in FEN notation
        moves: List of moves to play (will try multiple formats)
        engine: Stockfish engine instance
        time_limit: Time limit for evaluation
    
    Returns:
        Dictionary with evaluation results
    """
    board = chess.Board(initial_fen)
    evaluations = []
    successful_moves = []
    
    # Evaluate initial position
    info = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = info['score']
    
    if score.is_mate():
        mate_score = score.white().mate() if board.turn == chess.WHITE else score.black().mate()
        cp = 30000 if mate_score and mate_score > 0 else -30000
    else:
        cp = score.white().cp if board.turn == chess.WHITE else score.black().cp
        cp = cp if cp is not None else 0
    
    evaluations.append({
        'position': 0,
        'fen': board.fen(),
        'cp_score': cp,
        'side_to_move': 'white' if board.turn else 'black'
    })
    
    # Play through the moves
    for i, move_str in enumerate(moves):
        try:
            move = None
            
            # Try different move formats
            # 1. Try as UCI
            try:
                move = chess.Move.from_uci(move_str)
                if move not in board.legal_moves:
                    move = None
            except:
                pass
            
            # 2. Try as SAN
            if move is None:
                try:
                    move = board.parse_san(move_str)
                except:
                    pass
            
            # 3. Try interpreting as coordinates with piece inference
            if move is None and len(move_str) >= 4:
                try:
                    from_sq = chess.parse_square(move_str[:2])
                    to_sq = chess.parse_square(move_str[2:4])
                    
                    # Find any legal move to that square
                    for legal_move in board.legal_moves:
                        if legal_move.to_square == to_sq:
                            # Check if a piece could move from near the from_sq
                            piece = board.piece_at(legal_move.from_square)
                            if piece:
                                move = legal_move
                                print(f"    Interpreted {move_str} as {move}")
                                break
                except:
                    pass
            
            if move is None:
                print(f"    Could not parse move {move_str} at position {i+1}")
                break
            
            # Make the move
            board.push(move)
            successful_moves.append(str(move))
            
            # Evaluate the position
            info = engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = info['score']
            
            if score.is_mate():
                mate_score = score.white().mate() if board.turn == chess.WHITE else score.black().mate()
                cp = 30000 if mate_score and mate_score > 0 else -30000
            else:
                cp = score.white().cp if board.turn == chess.WHITE else score.black().cp
                cp = cp if cp is not None else 0
            
            evaluations.append({
                'position': i + 1,
                'move': move_str,
                'actual_move': str(move),
                'fen': board.fen(),
                'cp_score': cp,
                'side_to_move': 'white' if board.turn else 'black'
            })
            
        except Exception as e:
            print(f"    Error processing move {move_str}: {e}")
            break
    
    return {
        'successful_moves': successful_moves,
        'evaluations': evaluations,
        'final_fen': board.fen(),
        'final_cp': evaluations[-1]['cp_score'] if evaluations else None,
        'moves_completed': len(successful_moves)
    }


def main():
    parser = argparse.ArgumentParser(description='Directly evaluate chess puzzle branches with Stockfish')
    parser.add_argument('--stockfish-path', type=str, default='/usr/games/stockfish',
                       help='Path to Stockfish executable')
    parser.add_argument('--time-limit', type=float, default=0.5,
                       help='Time limit per position in seconds')
    parser.add_argument('--output', type=str, default='stockfish_direct_eval.pkl',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Sample puzzle data from notebook
    puzzles = [
        {
            'PuzzleId': '0hgJT',
            'FEN': '8/b7/P4k2/1N2p3/8/3r3P/R4p1K/8 w - - 0 50',
            'branch_1': ['d3g3', 'g2g3', 'f2f1q'],
            'branch_2': ['d3d1', 'a2f2', 'a7f2'],
            'side_to_move': 'White'
        },
        {
            'PuzzleId': '1zJDp',
            'FEN': '2k2r1r/2p5/5pp1/2pppn2/Q7/4PqB1/PP3P1P/1K1R2R1 b - - 1 27',
            'branch_1': ['g1g3', 'f3e4', 'a4e4'],
            'branch_2': ['a4a8', 'c8d7', 'd1d5'],
            'side_to_move': 'Black'
        },
        {
            'PuzzleId': '2JetC',
            'FEN': 'r5k1/5p1p/6p1/1B3q2/P2R1n1P/2B2P2/KP3P2/4R3 b - - 0 33',
            'branch_1': ['d4d8', 'a8d8', 'a4b5'],
            'branch_2': ['d4f4', 'a8a4', 'f4a4'],
            'side_to_move': 'Black'
        }
    ]
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        engine.configure({"Threads": 4, "Hash": 512})
        print("Stockfish started successfully\n")
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        print("Please ensure Stockfish is installed")
        return
    
    results = []
    
    try:
        for puzzle in puzzles:
            print(f"Evaluating puzzle {puzzle['PuzzleId']} ({puzzle['side_to_move']} to move)")
            print(f"  Initial FEN: {puzzle['FEN']}")
            
            # Evaluate branch 1
            print(f"  Branch 1: {puzzle['branch_1']}")
            branch1_result = play_and_evaluate_sequence(
                puzzle['FEN'], puzzle['branch_1'], engine, args.time_limit
            )
            print(f"    Completed {branch1_result['moves_completed']}/{len(puzzle['branch_1'])} moves")
            print(f"    Final evaluation: {branch1_result['final_cp']} cp")
            
            # Evaluate branch 2
            print(f"  Branch 2: {puzzle['branch_2']}")
            branch2_result = play_and_evaluate_sequence(
                puzzle['FEN'], puzzle['branch_2'], engine, args.time_limit
            )
            print(f"    Completed {branch2_result['moves_completed']}/{len(puzzle['branch_2'])} moves")
            print(f"    Final evaluation: {branch2_result['final_cp']} cp")
            
            # Determine better branch
            better_branch = None
            if branch1_result['final_cp'] is not None and branch2_result['final_cp'] is not None:
                # From perspective of side to move in initial position
                if puzzle['side_to_move'] == 'White':
                    better_branch = 1 if branch1_result['final_cp'] > branch2_result['final_cp'] else 2
                else:
                    better_branch = 1 if branch1_result['final_cp'] < branch2_result['final_cp'] else 2
            
            result = {
                'PuzzleId': puzzle['PuzzleId'],
                'FEN': puzzle['FEN'],
                'side_to_move': puzzle['side_to_move'],
                'branch_1': puzzle['branch_1'],
                'branch_1_result': branch1_result,
                'branch_2': puzzle['branch_2'],
                'branch_2_result': branch2_result,
                'better_branch': better_branch,
                'cp_difference': abs(branch1_result['final_cp'] - branch2_result['final_cp']) 
                                if branch1_result['final_cp'] and branch2_result['final_cp'] else None
            }
            
            results.append(result)
            
            if better_branch:
                print(f"  --> Branch {better_branch} is better by {result['cp_difference']} cp\n")
            else:
                print(f"  --> Could not determine better branch\n")
    
    finally:
        engine.quit()
        print("Stockfish closed")
    
    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for result in results:
        print(f"\nPuzzle {result['PuzzleId']}:")
        print(f"  Branch 1: {result['branch_1_result']['final_cp']} cp after {result['branch_1_result']['moves_completed']} moves")
        print(f"  Branch 2: {result['branch_2_result']['final_cp']} cp after {result['branch_2_result']['moves_completed']} moves")
        if result['better_branch']:
            print(f"  Better: Branch {result['better_branch']} (by {result['cp_difference']} cp)")


if __name__ == "__main__":
    main()