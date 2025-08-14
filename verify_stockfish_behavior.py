#!/usr/bin/env python3
"""
Verify Stockfish behavior:
1. Check if score signs are correct
2. See what move Stockfish actually suggests vs the principal variation
"""

import chess
import chess.engine
import pandas as pd


def parse_branch_string(branch_str: str) -> list:
    """Parse branch string from CSV format to list of moves."""
    branch_str = branch_str.strip("[]'\"")
    moves = [m.strip(" '\"") for m in branch_str.split(',')]
    return moves


def main():
    # Load puzzle data
    df = pd.read_csv('study_puzzles_head30.csv')
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
        engine.configure({"Threads": 4, "Hash": 1024})
        print("Stockfish started successfully")
        print("="*70)
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    try:
        # Test first 5 puzzles in detail
        for idx in range(min(5, len(df))):
            row = df.iloc[idx]
            puzzle_id = row['PuzzleId']
            fen = row['FEN']
            moves_str = row['Moves']
            pv = parse_branch_string(row['principal_variation'])
            branch_1 = parse_branch_string(row['branch_1'])
            branch_2 = parse_branch_string(row['branch_2'])
            
            board = chess.Board(fen)
            side = "White" if board.turn else "Black"
            
            print(f"\nPuzzle {puzzle_id} - {side} to move")
            print(f"FEN: {fen}")
            
            # Get Stockfish's evaluation and best move
            info = engine.analyse(board, chess.engine.Limit(time=3.0, depth=25))
            score = info['score']
            
            # Check score interpretation
            print(f"\nScore object: {score}")
            print(f"  Is mate: {score.is_mate()}")
            if not score.is_mate():
                print(f"  White perspective: {score.white()}")
                print(f"  Black perspective: {score.black()}")
                print(f"  Relative (from side to move): {score.relative}")
            
            # Get Stockfish's best move
            result = engine.play(board, chess.engine.Limit(time=3.0, depth=25))
            sf_best_move = result.move
            
            # Get puzzle's first move
            puzzle_moves = moves_str.split()
            puzzle_first = puzzle_moves[0] if puzzle_moves else None
            
            print(f"\nFirst moves comparison:")
            print(f"  Puzzle solution: {puzzle_first}")
            if puzzle_first:
                try:
                    puzzle_move = board.parse_san(puzzle_first)
                    print(f"  Puzzle move UCI: {puzzle_move.uci()}")
                except:
                    print(f"  Could not parse puzzle move")
                    puzzle_move = None
            else:
                puzzle_move = None
            
            print(f"  Stockfish best: {sf_best_move.uci()} ({board.san(sf_best_move)})")
            
            if puzzle_move and puzzle_move == sf_best_move:
                print(f"  ✓ Stockfish agrees with puzzle's first move!")
            else:
                print(f"  ✗ Stockfish suggests a different move")
            
            # Now check what happens after the puzzle's first move
            if puzzle_first:
                try:
                    board_after = board.copy()
                    move = board_after.parse_san(puzzle_first)
                    board_after.push(move)
                    
                    print(f"\nAfter puzzle's first move ({puzzle_first}):")
                    
                    # Get Stockfish's response
                    result2 = engine.play(board_after, chess.engine.Limit(time=2.0, depth=20))
                    sf_response = result2.move
                    print(f"  Stockfish would respond: {sf_response.uci()} ({board_after.san(sf_response)})")
                    
                    # Check if this matches either branch
                    if branch_1 and branch_1[0]:
                        try:
                            branch1_move = board_after.parse_san(branch_1[0])
                            if branch1_move == sf_response:
                                print(f"  → Matches Branch 1 first move!")
                        except:
                            pass
                    
                    if branch_2 and branch_2[0]:
                        try:
                            branch2_move = board_after.parse_san(branch_2[0])
                            if branch2_move == sf_response:
                                print(f"  → Matches Branch 2 first move!")
                        except:
                            pass
                    
                    # Check which branch is the principal variation
                    is_pv_branch1 = (branch_1 == pv)
                    print(f"\n  Principal variation is: Branch {'1' if is_pv_branch1 else '2'}")
                    print(f"  Branch 1: {branch_1[:3] if len(branch_1) >= 3 else branch_1}")
                    print(f"  Branch 2: {branch_2[:3] if len(branch_2) >= 3 else branch_2}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
            
            print("-"*70)
        
        # Now do a broader check on all puzzles
        print("\n" + "="*70)
        print("CHECKING ALL PUZZLES: Does Stockfish play the puzzle's first move?")
        print("="*70)
        
        agrees_count = 0
        for idx, row in df.iterrows():
            board = chess.Board(row['FEN'])
            moves = row['Moves'].split()
            
            if moves:
                puzzle_first = moves[0]
                try:
                    puzzle_move = board.parse_san(puzzle_first)
                    result = engine.play(board, chess.engine.Limit(time=1.0, depth=20))
                    sf_move = result.move
                    
                    if puzzle_move == sf_move:
                        agrees_count += 1
                        print(f"✓ {row['PuzzleId']}: Stockfish plays {puzzle_first}")
                    else:
                        print(f"✗ {row['PuzzleId']}: Puzzle={puzzle_first}, SF={board.san(sf_move)}")
                except Exception as e:
                    print(f"? {row['PuzzleId']}: Error - {e}")
        
        print(f"\nSummary: Stockfish agrees with puzzle's first move in {agrees_count}/{len(df)} cases ({100*agrees_count/len(df):.1f}%)")
        
    finally:
        engine.quit()
        print("\nStockfish closed")


if __name__ == "__main__":
    main()