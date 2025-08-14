#!/usr/bin/env python3
"""
Evaluate chess puzzle branches using Stockfish.
This version correctly interprets the puzzle structure where:
- The initial FEN is the starting position
- Branch moves are continuations that alternate between players
- The first move in 'Moves' is the puzzle's intended first move
"""

import chess
import chess.engine
import pandas as pd
import pickle
import argparse
from typing import List, Dict, Any, Tuple


def play_variation(board: chess.Board, moves: List[str], first_move: str = None) -> Tuple[chess.Board, List[str], bool]:
    """
    Play a variation starting with an optional first move.
    
    Args:
        board: Initial board position
        moves: List of moves to play (in SAN or UCI format)
        first_move: Optional first move to play before the variation
    
    Returns:
        Tuple of (final board, moves played in UCI format, success flag)
    """
    current_board = board.copy()
    moves_played = []
    
    # Play the first move if provided
    if first_move:
        try:
            move = current_board.parse_san(first_move)
            current_board.push(move)
            moves_played.append(move.uci())
        except:
            try:
                move = chess.Move.from_uci(first_move)
                if move in current_board.legal_moves:
                    current_board.push(move)
                    moves_played.append(move.uci())
            except:
                print(f"    Could not play first move: {first_move}")
                return current_board, moves_played, False
    
    # Play the rest of the variation
    for i, move_str in enumerate(moves):
        try:
            # Try SAN first
            try:
                move = current_board.parse_san(move_str)
            except:
                # Try UCI
                move = chess.Move.from_uci(move_str)
            
            if move in current_board.legal_moves:
                current_board.push(move)
                moves_played.append(move.uci())
            else:
                print(f"    Move {i+1}: Illegal move {move_str}")
                return current_board, moves_played, False
                
        except Exception as e:
            print(f"    Move {i+1}: Error with {move_str}: {e}")
            return current_board, moves_played, False
    
    return current_board, moves_played, True


def evaluate_position(board: chess.Board, engine: chess.engine.SimpleEngine, 
                      time_limit: float = 1.0, depth: int = 20) -> Dict[str, Any]:
    """
    Evaluate a chess position using Stockfish.
    
    Returns:
        Dictionary with evaluation details
    """
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
        'depth': info.get('depth', depth),
        'fen': board.fen()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate chess puzzle branches with Stockfish')
    parser.add_argument('--stockfish-path', type=str, default='/usr/games/stockfish',
                       help='Path to Stockfish executable')
    parser.add_argument('--time-limit', type=float, default=2.0,
                       help='Time limit per position in seconds')
    parser.add_argument('--depth', type=int, default=25,
                       help='Maximum search depth')
    parser.add_argument('--output', type=str, default='stockfish_branch_analysis.pkl',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Your puzzle data with the Moves field included
    puzzles = [
        {
            'PuzzleId': '0hgJT',
            'FEN': '8/b7/P4k2/1N2p3/8/3r3P/R4p1K/8 w - - 0 50',
            'Moves': 'h2g2 d3g3 g2g3 f2f1q',  # Complete puzzle solution
            'branch_1': ['d3g3', 'g2g3', 'f2f1q'],  # Continuation after first move
            'branch_2': ['d3d1', 'a2f2', 'a7f2']    # Alternative continuation
        },
        {
            'PuzzleId': '1zJDp',
            'FEN': '2k2r1r/2p5/5pp1/2pppn2/Q7/4PqB1/PP3P1P/1K1R2R1 b - - 1 27',
            'Moves': 'f5g3 a4a8 c8d7 d1d5',
            'branch_1': ['g1g3', 'f3e4', 'a4e4'],  # These might be from a different perspective
            'branch_2': ['a4a8', 'c8d7', 'd1d5']
        },
        {
            'PuzzleId': '2JetC',
            'FEN': 'r5k1/5p1p/6p1/1B3q2/P2R1n1P/2B2P2/KP3P2/4R3 b - - 0 33',
            'Moves': 'f5b5 d4d8 a8d8 a4b5',
            'branch_1': ['d4d8', 'a8d8', 'a4b5'],
            'branch_2': ['d4f4', 'a8a4', 'f4a4']
        }
    ]
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        engine.configure({"Threads": 4, "Hash": 512})
        print("Stockfish started successfully")
        print("="*70)
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    results = []
    
    try:
        for puzzle in puzzles:
            board = chess.Board(puzzle['FEN'])
            side_to_move = "White" if board.turn else "Black"
            
            print(f"\nPuzzle {puzzle['PuzzleId']}")
            print(f"Initial position: {side_to_move} to move")
            
            # Get the first move from the Moves field
            all_moves = puzzle['Moves'].split()
            first_move = all_moves[0] if all_moves else None
            
            print(f"First move: {first_move}")
            
            # Evaluate initial position
            initial_eval = evaluate_position(board, engine, args.time_limit, args.depth)
            print(f"Initial evaluation: {initial_eval['cp']} cp")
            
            # Play and evaluate branch 1
            # Branch 1 represents the continuation after the first move
            print(f"\nBranch 1: {first_move} followed by {puzzle['branch_1']}")
            board1, moves1, success1 = play_variation(board, puzzle['branch_1'], first_move)
            
            if success1:
                eval1 = evaluate_position(board1, engine, args.time_limit, args.depth)
                print(f"  Final position: {eval1['cp']} cp")
                print(f"  Change from initial: {eval1['cp'] - initial_eval['cp']:+d} cp")
                if eval1['mate_in']:
                    print(f"  Mate in {eval1['mate_in']}")
            else:
                eval1 = initial_eval
                print(f"  Could not complete variation")
            
            # Play and evaluate branch 2
            print(f"\nBranch 2: {first_move} followed by {puzzle['branch_2']}")
            board2, moves2, success2 = play_variation(board, puzzle['branch_2'], first_move)
            
            if success2:
                eval2 = evaluate_position(board2, engine, args.time_limit, args.depth)
                print(f"  Final position: {eval2['cp']} cp")
                print(f"  Change from initial: {eval2['cp'] - initial_eval['cp']:+d} cp")
                if eval2['mate_in']:
                    print(f"  Mate in {eval2['mate_in']}")
            else:
                eval2 = initial_eval
                print(f"  Could not complete variation")
            
            # Determine which branch is better
            if success1 and success2:
                # From the perspective of the side to move
                if side_to_move == "White":
                    better = 1 if eval1['cp'] > eval2['cp'] else 2
                    advantage = abs(eval1['cp'] - eval2['cp'])
                else:
                    better = 1 if eval1['cp'] < eval2['cp'] else 2
                    advantage = abs(eval1['cp'] - eval2['cp'])
                
                print(f"\n→ Branch {better} is better by {advantage} cp for {side_to_move}")
            else:
                better = None
                advantage = None
                print(f"\n→ Could not compare branches due to errors")
            
            result_data = {
                'PuzzleId': puzzle['PuzzleId'],
                'FEN': puzzle['FEN'],
                'side_to_move': side_to_move,
                'first_move': first_move,
                'branch_1': puzzle['branch_1'],
                'branch_1_success': success1,
                'branch_1_eval': eval1 if success1 else None,
                'branch_2': puzzle['branch_2'],
                'branch_2_success': success2,
                'branch_2_eval': eval2 if success2 else None,
                'better_branch': better,
                'advantage_cp': advantage,
                'initial_eval': initial_eval
            }
            
            results.append(result_data)
            print("-"*70)
    
    finally:
        engine.quit()
        print("\nStockfish closed")
    
    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {args.output}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\nPuzzle {result['PuzzleId']} ({result['side_to_move']} to move):")
        print(f"  Initial: {result['initial_eval']['cp']:+6d} cp")
        
        if result['branch_1_success']:
            b1_cp = result['branch_1_eval']['cp']
            print(f"  Branch 1: {b1_cp:+6d} cp (change: {b1_cp - result['initial_eval']['cp']:+d})")
        else:
            print(f"  Branch 1: Failed to evaluate")
        
        if result['branch_2_success']:
            b2_cp = result['branch_2_eval']['cp']
            print(f"  Branch 2: {b2_cp:+6d} cp (change: {b2_cp - result['initial_eval']['cp']:+d})")
        else:
            print(f"  Branch 2: Failed to evaluate")
        
        if result['better_branch']:
            print(f"  → Branch {result['better_branch']} is better by {result['advantage_cp']} cp")


if __name__ == "__main__":
    main()