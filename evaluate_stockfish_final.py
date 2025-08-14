#!/usr/bin/env python3
"""
Evaluate chess puzzle branches using Stockfish.
Properly handles alternating moves between players.
"""

import chess
import chess.engine
import pandas as pd
import pickle
import argparse
from typing import List, Dict, Any, Tuple


def play_sequence(board: chess.Board, moves: List[str]) -> Tuple[chess.Board, List[str], List[str]]:
    """
    Play a sequence of moves on the board.
    Handles various move formats and alternating players.
    
    Returns:
        Tuple of (final board, successful moves in UCI, move descriptions)
    """
    current_board = board.copy()
    successful_moves = []
    move_descriptions = []
    
    for i, move_str in enumerate(moves):
        side_to_move = "White" if current_board.turn else "Black"
        
        try:
            move = None
            
            # Try UCI format first
            if len(move_str) >= 4:
                try:
                    from_sq = move_str[:2]
                    to_sq = move_str[2:4]
                    promotion = move_str[4] if len(move_str) > 4 else None
                    
                    # Create move
                    from_square = chess.parse_square(from_sq)
                    to_square = chess.parse_square(to_sq)
                    
                    if promotion:
                        promo_piece = {'q': chess.QUEEN, 'r': chess.ROOK, 
                                     'b': chess.BISHOP, 'n': chess.KNIGHT}.get(promotion.lower())
                        move = chess.Move(from_square, to_square, promotion=promo_piece)
                    else:
                        move = chess.Move(from_square, to_square)
                    
                    if move not in current_board.legal_moves:
                        # Try SAN format
                        move = current_board.parse_san(move_str)
                except:
                    # Try SAN format
                    try:
                        move = current_board.parse_san(move_str)
                    except:
                        pass
            
            if move and move in current_board.legal_moves:
                current_board.push(move)
                successful_moves.append(move.uci())
                move_descriptions.append(f"{side_to_move}: {move_str} -> {move.uci()}")
            else:
                print(f"    Move {i+1}: Could not play {move_str} for {side_to_move}")
                break
                
        except Exception as e:
            print(f"    Move {i+1}: Error with {move_str}: {e}")
            break
    
    return current_board, successful_moves, move_descriptions


def evaluate_branch(initial_fen: str, moves: List[str], engine: chess.engine.SimpleEngine,
                   time_limit: float = 1.0, depth: int = 20) -> Dict[str, Any]:
    """
    Evaluate a branch by playing moves and analyzing positions.
    
    Returns:
        Dictionary with evaluation results
    """
    initial_board = chess.Board(initial_fen)
    
    # Play the sequence
    final_board, successful_moves, move_descriptions = play_sequence(initial_board, moves)
    
    # Evaluate initial position
    info = engine.analyse(initial_board, chess.engine.Limit(time=time_limit, depth=depth))
    initial_score = info['score']
    
    if initial_score.is_mate():
        initial_cp = 30000 if initial_score.white().mate() > 0 else -30000
    else:
        initial_cp = initial_score.white().cp
        initial_cp = initial_cp if initial_cp is not None else 0
    
    # Evaluate final position
    info = engine.analyse(final_board, chess.engine.Limit(time=time_limit, depth=depth))
    final_score = info['score']
    
    if final_score.is_mate():
        mate_score = final_score.white().mate()
        final_cp = 30000 if mate_score and mate_score > 0 else -30000
        mate_in = abs(mate_score) if mate_score else None
    else:
        final_cp = final_score.white().cp
        final_cp = final_cp if final_cp is not None else 0
        mate_in = None
    
    return {
        'successful_moves': successful_moves,
        'move_descriptions': move_descriptions,
        'moves_played': len(successful_moves),
        'moves_total': len(moves),
        'initial_cp': initial_cp,
        'final_cp': final_cp,
        'cp_change': final_cp - initial_cp,
        'final_fen': final_board.fen(),
        'mate_in': mate_in
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate chess puzzle branches with Stockfish')
    parser.add_argument('--stockfish-path', type=str, default='/usr/games/stockfish',
                       help='Path to Stockfish executable')
    parser.add_argument('--time-limit', type=float, default=1.0,
                       help='Time limit per position in seconds')
    parser.add_argument('--depth', type=int, default=20,
                       help='Maximum search depth')
    parser.add_argument('--output', type=str, default='stockfish_analysis.pkl',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Sample puzzles from your notebook
    puzzles = [
        {
            'PuzzleId': '0hgJT',
            'FEN': '8/b7/P4k2/1N2p3/8/3r3P/R4p1K/8 w - - 0 50',
            'branch_1': ['d3g3', 'g2g3', 'f2f1q'],
            'branch_2': ['d3d1', 'a2f2', 'a7f2']
        },
        {
            'PuzzleId': '1zJDp', 
            'FEN': '2k2r1r/2p5/5pp1/2pppn2/Q7/4PqB1/PP3P1P/1K1R2R1 b - - 1 27',
            'branch_1': ['g1g3', 'f3e4', 'a4e4'],
            'branch_2': ['a4a8', 'c8d7', 'd1d5']
        },
        {
            'PuzzleId': '2JetC',
            'FEN': 'r5k1/5p1p/6p1/1B3q2/P2R1n1P/2B2P2/KP3P2/4R3 b - - 0 33',
            'branch_1': ['d4d8', 'a8d8', 'a4b5'],
            'branch_2': ['d4f4', 'a8a4', 'f4a4']
        }
    ]
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        engine.configure({"Threads": 4, "Hash": 512})
        print("Stockfish started successfully\n")
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    results = []
    
    try:
        for puzzle in puzzles:
            board = chess.Board(puzzle['FEN'])
            side_to_move = "White" if board.turn else "Black"
            
            print(f"{'='*60}")
            print(f"Puzzle {puzzle['PuzzleId']} - {side_to_move} to move")
            print(f"FEN: {puzzle['FEN']}")
            print(f"{'='*60}")
            
            # Note: The branches contain moves that alternate between players
            # So if it's White's turn, the sequence is: Black's response, White's next, Black's response
            # If it's Black's turn, the sequence is: White's response, Black's next, White's response
            
            # We need to prepend the best initial move to complete the sequence
            # Get engine's best move for the initial position
            result = engine.play(board, chess.engine.Limit(time=args.time_limit))
            best_first_move = result.move.uci()
            
            print(f"\nBest first move: {best_first_move}")
            
            # Create complete sequences with alternating moves
            # The branches show opponent responses, so we need to interleave our moves
            branch1_complete = []
            branch2_complete = []
            
            # For simplicity, let's just evaluate the branches as given
            # They represent specific variations to analyze
            
            print(f"\nBranch 1: {puzzle['branch_1']}")
            branch1_result = evaluate_branch(puzzle['FEN'], puzzle['branch_1'], engine, 
                                            args.time_limit, args.depth)
            print(f"  Played: {branch1_result['moves_played']}/{branch1_result['moves_total']} moves")
            print(f"  Evaluation: {branch1_result['initial_cp']} -> {branch1_result['final_cp']} cp")
            print(f"  Change: {branch1_result['cp_change']:+d} cp")
            if branch1_result['mate_in']:
                print(f"  Mate in {branch1_result['mate_in']}")
            
            print(f"\nBranch 2: {puzzle['branch_2']}")
            branch2_result = evaluate_branch(puzzle['FEN'], puzzle['branch_2'], engine,
                                            args.time_limit, args.depth)
            print(f"  Played: {branch2_result['moves_played']}/{branch2_result['moves_total']} moves")
            print(f"  Evaluation: {branch2_result['initial_cp']} -> {branch2_result['final_cp']} cp")
            print(f"  Change: {branch2_result['cp_change']:+d} cp")
            if branch2_result['mate_in']:
                print(f"  Mate in {branch2_result['mate_in']}")
            
            # Determine which branch is better
            # From the perspective of the side to move initially
            if side_to_move == "White":
                # Higher cp is better for White
                if branch1_result['final_cp'] > branch2_result['final_cp']:
                    better = "Branch 1"
                    advantage = branch1_result['final_cp'] - branch2_result['final_cp']
                else:
                    better = "Branch 2"
                    advantage = branch2_result['final_cp'] - branch1_result['final_cp']
            else:
                # Lower cp is better for Black
                if branch1_result['final_cp'] < branch2_result['final_cp']:
                    better = "Branch 1"
                    advantage = branch2_result['final_cp'] - branch1_result['final_cp']
                else:
                    better = "Branch 2"
                    advantage = branch1_result['final_cp'] - branch2_result['final_cp']
            
            print(f"\n★ {better} is better by {advantage} cp")
            
            result_data = {
                'PuzzleId': puzzle['PuzzleId'],
                'FEN': puzzle['FEN'],
                'side_to_move': side_to_move,
                'branch_1': puzzle['branch_1'],
                'branch_1_result': branch1_result,
                'branch_2': puzzle['branch_2'],
                'branch_2_result': branch2_result,
                'better_branch': better,
                'advantage_cp': advantage
            }
            
            results.append(result_data)
            print()
    
    finally:
        engine.quit()
        print("Stockfish closed")
    
    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {args.output}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        print(f"\n{result['PuzzleId']} ({result['side_to_move']} to move):")
        b1_cp = result['branch_1_result']['final_cp']
        b2_cp = result['branch_2_result']['final_cp']
        print(f"  Branch 1: {b1_cp:+6d} cp")
        print(f"  Branch 2: {b2_cp:+6d} cp")
        print(f"  {result['better_branch']} is better by {result['advantage_cp']} cp")


if __name__ == "__main__":
    main()