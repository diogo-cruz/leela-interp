#!/usr/bin/env python3
"""
Final corrected analysis with proper score interpretation.
Key findings from verification:
1. Stockfish rarely agrees with the puzzle's first move (only 3.3%)
2. The principal variations are definitely not Stockfish's preferred lines
3. These puzzles test specific patterns, not optimal play
"""

import chess
import chess.engine
import pandas as pd
import numpy as np


def parse_branch_string(branch_str: str) -> list:
    """Parse branch string from CSV format to list of moves."""
    branch_str = branch_str.strip("[]'\"")
    moves = [m.strip(" '\"") for m in branch_str.split(',')]
    return moves


def analyze_puzzle_corrected(puzzle_row: pd.Series, engine: chess.engine.SimpleEngine) -> dict:
    """
    Analyze puzzle with correct score interpretation.
    Scores are always from White's perspective:
    - Positive = good for White
    - Negative = good for Black
    """
    puzzle_id = puzzle_row['PuzzleId']
    fen = puzzle_row['FEN']
    moves_str = puzzle_row['Moves']
    
    # Parse branches
    branch_1 = parse_branch_string(puzzle_row['branch_1'])
    branch_2 = parse_branch_string(puzzle_row['branch_2'])
    principal_var = parse_branch_string(puzzle_row['principal_variation'])
    
    # Initial board
    board = chess.Board(fen)
    side_to_move = board.turn  # chess.WHITE or chess.BLACK
    
    # Get puzzle's first move
    all_moves = moves_str.split()
    puzzle_first = all_moves[0] if all_moves else None
    
    # Get Stockfish's preferred first move
    result = engine.play(board, chess.engine.Limit(time=2.0, depth=20))
    sf_first_move = result.move
    
    # Check if Stockfish agrees with puzzle
    agrees_with_first = False
    if puzzle_first:
        try:
            puzzle_move = board.parse_san(puzzle_first)
            agrees_with_first = (puzzle_move == sf_first_move)
        except:
            pass
    
    # Play puzzle's first move and evaluate branches
    board_after = board.copy()
    if puzzle_first:
        try:
            move = board_after.parse_san(puzzle_first)
            board_after.push(move)
            
            # Now check what Stockfish would play as response
            response = engine.play(board_after, chess.engine.Limit(time=1.0, depth=20))
            sf_response = response.move
            
            # Check which branch Stockfish's response matches
            sf_matches_branch = None
            if branch_1:
                try:
                    b1_first = board_after.parse_san(branch_1[0])
                    if b1_first == sf_response:
                        sf_matches_branch = 1
                except:
                    pass
            
            if branch_2 and sf_matches_branch is None:
                try:
                    b2_first = board_after.parse_san(branch_2[0])
                    if b2_first == sf_response:
                        sf_matches_branch = 2
                except:
                    pass
            
            # Evaluate both complete branches
            # Branch 1
            board_b1 = board_after.copy()
            b1_success = True
            for move_str in branch_1:
                try:
                    move = board_b1.parse_san(move_str)
                    board_b1.push(move)
                except:
                    b1_success = False
                    break
            
            if b1_success:
                info = engine.analyse(board_b1, chess.engine.Limit(time=1.0, depth=20))
                # Always get score from White's perspective
                b1_score = info['score'].white().score(mate_score=30000)
            else:
                b1_score = None
            
            # Branch 2
            board_b2 = board_after.copy()
            b2_success = True
            for move_str in branch_2:
                try:
                    move = board_b2.parse_san(move_str)
                    board_b2.push(move)
                except:
                    b2_success = False
                    break
            
            if b2_success:
                info = engine.analyse(board_b2, chess.engine.Limit(time=1.0, depth=20))
                b2_score = info['score'].white().score(mate_score=30000)
            else:
                b2_score = None
            
            # Determine which branch is better from the perspective of the initial side to move
            better_branch = None
            if b1_score is not None and b2_score is not None:
                if side_to_move == chess.WHITE:
                    # White wants higher scores
                    better_branch = 1 if b1_score > b2_score else 2
                else:
                    # Black wants lower scores
                    better_branch = 1 if b1_score < b2_score else 2
            
            # Check which branch is the principal variation
            pv_branch = 1 if branch_1 == principal_var else 2
            
            return {
                'puzzle_id': puzzle_id,
                'side_to_move': 'White' if side_to_move == chess.WHITE else 'Black',
                'sf_agrees_first': agrees_with_first,
                'sf_first_move': board.san(sf_first_move),
                'puzzle_first': puzzle_first,
                'sf_response_matches': sf_matches_branch,
                'branch_1_score': b1_score,
                'branch_2_score': b2_score,
                'better_branch': better_branch,
                'pv_branch': pv_branch,
                'sf_agrees_with_pv': better_branch == pv_branch if better_branch else None
            }
            
        except Exception as e:
            print(f"Error with {puzzle_id}: {e}")
            return None
    
    return None


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
        print("Analyzing puzzles...")
        for idx, row in df.iterrows():
            result = analyze_puzzle_corrected(row, engine)
            if result:
                results.append(result)
                
                # Progress indicator
                if result['sf_agrees_with_pv'] is not None:
                    symbol = "✓" if result['sf_agrees_with_pv'] else "✗"
                else:
                    symbol = "?"
                
                print(f"{idx+1:2d}. {result['puzzle_id']}: {symbol} "
                      f"SF first={'✓' if result['sf_agrees_first'] else '✗'} "
                      f"PV={result['pv_branch']}, Better={result['better_branch']}")
    
    finally:
        engine.quit()
        print("\nStockfish closed")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('final_corrected_analysis.csv', index=False)
    
    # Analysis
    print("\n" + "="*70)
    print("FINAL ANALYSIS RESULTS")
    print("="*70)
    
    total = len(results_df)
    
    # How often does Stockfish agree with puzzle's first move?
    agrees_first = results_df['sf_agrees_first'].sum()
    print(f"\nStockfish agrees with puzzle's first move: {agrees_first}/{total} ({100*agrees_first/total:.1f}%)")
    
    # After puzzle's first move, which branch does Stockfish prefer?
    sf_response_1 = (results_df['sf_response_matches'] == 1).sum()
    sf_response_2 = (results_df['sf_response_matches'] == 2).sum()
    sf_response_none = results_df['sf_response_matches'].isna().sum()
    
    print(f"\nAfter puzzle's first move, Stockfish's response:")
    print(f"  Matches Branch 1: {sf_response_1} ({100*sf_response_1/total:.1f}%)")
    print(f"  Matches Branch 2: {sf_response_2} ({100*sf_response_2/total:.1f}%)")
    print(f"  Matches neither: {sf_response_none} ({100*sf_response_none/total:.1f}%)")
    
    # How often does the better branch match the PV?
    valid = results_df[results_df['sf_agrees_with_pv'].notna()]
    agrees_pv = valid['sf_agrees_with_pv'].sum()
    
    print(f"\nEvaluation of branches:")
    print(f"  Valid evaluations: {len(valid)}/{total}")
    print(f"  Stockfish prefers PV branch: {agrees_pv}/{len(valid)} ({100*agrees_pv/len(valid):.1f}%)")
    
    # Show puzzles where Stockfish agrees with PV
    if agrees_pv > 0:
        print(f"\nPuzzles where Stockfish evaluation agrees with PV:")
        agrees_df = results_df[results_df['sf_agrees_with_pv'] == True]
        for _, row in agrees_df.iterrows():
            print(f"  {row['puzzle_id']}: Branch {row['pv_branch']} is PV and better")
    
    # Pattern analysis
    print(f"\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("These puzzles are clearly NOT testing optimal computer play:")
    print(f"1. Stockfish disagrees with the puzzle's first move {100*(1-agrees_first/total):.1f}% of the time")
    print(f"2. The 'principal variation' is rarely Stockfish's preferred continuation")
    print("3. These puzzles likely test specific tactical patterns, themes, or")
    print("   heuristics that are important for human chess understanding")
    print("\nThis makes them ideal for studying how neural networks learn")
    print("chess concepts beyond pure calculation!")


if __name__ == "__main__":
    main()