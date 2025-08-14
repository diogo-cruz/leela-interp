#!/usr/bin/env python3
"""
Investigate why Stockfish disagrees with first moves but agrees with subsequent moves.
"""

import chess
import chess.engine
import pandas as pd


def parse_pv_string(pv_str: str) -> list:
    """Parse principal variation string from CSV format to list of moves."""
    pv_str = pv_str.strip("[]'\"")
    moves = [m.strip(" '\"") for m in pv_str.split(',')]
    return moves


def detailed_analysis(puzzle_row: pd.Series, engine: chess.engine.SimpleEngine) -> dict:
    """
    Detailed analysis of a puzzle to understand the pattern.
    """
    puzzle_id = puzzle_row['PuzzleId']
    fen = puzzle_row['FEN']
    moves_str = puzzle_row['Moves']
    pv = parse_pv_string(puzzle_row['principal_variation'])
    
    board = chess.Board(fen)
    side_to_move = "White" if board.turn else "Black"
    
    all_moves = moves_str.split()
    
    print(f"\n{'='*60}")
    print(f"Puzzle {puzzle_id} - {side_to_move} to move")
    print(f"FEN: {fen}")
    print(f"Solution: {' '.join(all_moves)}")
    print(f"Principal Variation: {pv}")
    
    # Check if PV is just moves 2-4 of the solution
    if len(all_moves) >= 4 and len(pv) >= 3:
        solution_234 = all_moves[1:4]
        print(f"\nSolution moves 2-4: {solution_234}")
        print(f"PV first 3 moves: {pv[:3]}")
        print(f"Are they the same? {solution_234 == pv[:3]}")
    
    # Analyze from initial position
    print(f"\n1. From initial position:")
    info = engine.analyse(board, chess.engine.Limit(time=2.0, depth=25))
    score = info['score'].white().score(mate_score=30000)
    
    result = engine.play(board, chess.engine.Limit(time=2.0, depth=25))
    sf_best = result.move
    
    print(f"   Stockfish evaluation: {score} cp")
    print(f"   Stockfish best move: {board.san(sf_best)}")
    print(f"   Puzzle first move: {all_moves[0]}")
    
    # Make puzzle's first move
    try:
        puzzle_move = board.parse_san(all_moves[0])
        board_after_puzzle = board.copy()
        board_after_puzzle.push(puzzle_move)
        
        # Evaluate after puzzle move
        info_after = engine.analyse(board_after_puzzle, chess.engine.Limit(time=2.0, depth=25))
        score_after = info_after['score'].white().score(mate_score=30000)
        
        print(f"\n2. After puzzle's first move ({all_moves[0]}):")
        print(f"   Position evaluation: {score_after} cp")
        print(f"   Change: {score_after - score:+d} cp")
        
        # What would Stockfish play here?
        result2 = engine.play(board_after_puzzle, chess.engine.Limit(time=2.0, depth=25))
        sf_response = result2.move
        print(f"   Stockfish would respond: {board_after_puzzle.san(sf_response)}")
        if len(pv) > 0:
            print(f"   PV first move (opponent's response): {pv[0]}")
            try:
                pv_move = board_after_puzzle.parse_san(pv[0])
                if pv_move == sf_response:
                    print(f"   ✓ Stockfish agrees with PV response!")
                else:
                    print(f"   ✗ Different from PV")
            except:
                pass
    except Exception as e:
        print(f"   Error: {e}")
    
    # Make Stockfish's preferred first move instead
    print(f"\n3. After Stockfish's preferred move ({board.san(sf_best)}):")
    board_after_sf = board.copy()
    board_after_sf.push(sf_best)
    
    info_sf = engine.analyse(board_after_sf, chess.engine.Limit(time=2.0, depth=25))
    score_sf = info_sf['score'].white().score(mate_score=30000)
    
    print(f"   Position evaluation: {score_sf} cp")
    print(f"   Change: {score_sf - score:+d} cp")
    
    # Compare the two approaches
    print(f"\n4. Comparison:")
    print(f"   Puzzle move leads to: {score_after} cp")
    print(f"   Stockfish move leads to: {score_sf} cp")
    
    if side_to_move == "White":
        if score_sf > score_after:
            print(f"   → Stockfish's move is better by {score_sf - score_after} cp")
        else:
            print(f"   → Puzzle move is better by {score_after - score_sf} cp (!)")
    else:
        if score_sf < score_after:
            print(f"   → Stockfish's move is better by {score_after - score_sf} cp")
        else:
            print(f"   → Puzzle move is better by {score_sf - score_after} cp (!)")


def main():
    # Load puzzle data
    df = pd.read_csv('study_puzzles_head30_normal.csv')
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
        engine.configure({"Threads": 4, "Hash": 1024})
        print("Stockfish started successfully")
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    try:
        # Analyze first 5 puzzles in detail
        for i in range(min(5, len(df))):
            detailed_analysis(df.iloc[i], engine)
        
        print("\n" + "="*60)
        print("KEY INSIGHT")
        print("="*60)
        print("\nThe principal variation in this dataset appears to be:")
        print("- NOT the best moves from the initial position")
        print("- But rather the OPPONENT'S RESPONSES after the puzzle's first move")
        print("\nThis explains why:")
        print("1. Stockfish disagrees with the 'first move' of the PV (it's actually the opponent's response)")
        print("2. But agrees with subsequent moves (the tactical sequence)")
        print("\nThe puzzle format is: [our_move, opponent_response, our_move, opponent_response]")
        print("The PV contains: [opponent_response, our_move, opponent_response] (moves 2-4)")
        
    finally:
        engine.quit()
        print("\nStockfish closed")


if __name__ == "__main__":
    main()