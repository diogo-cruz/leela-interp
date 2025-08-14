#!/usr/bin/env python3
"""
Analyze the normal puzzle dataset to see if Stockfish plays the principal variation.
"""

import chess
import chess.engine
import pandas as pd
import json


def parse_pv_string(pv_str: str) -> list:
    """Parse principal variation string from CSV format to list of moves."""
    pv_str = pv_str.strip("[]'\"")
    moves = [m.strip(" '\"") for m in pv_str.split(',')]
    return moves


def analyze_puzzle(puzzle_row: pd.Series, engine: chess.engine.SimpleEngine, depth: int = 20) -> dict:
    """
    Analyze a single puzzle to see if Stockfish plays the principal variation.
    """
    puzzle_id = puzzle_row['PuzzleId']
    fen = puzzle_row['FEN']
    moves_str = puzzle_row['Moves']
    pv = parse_pv_string(puzzle_row['principal_variation'])
    
    # Initial board
    board = chess.Board(fen)
    side_to_move = "White" if board.turn else "Black"
    
    # Parse the complete solution moves
    all_moves = moves_str.split()
    
    # Track if Stockfish agrees with each move
    sf_agrees = []
    sf_moves = []
    pv_moves = []
    
    # Start with the initial position
    current_board = board.copy()
    
    # First, check the puzzle's first move
    if all_moves:
        puzzle_first = all_moves[0]
        
        # Get Stockfish's best move from initial position
        result = engine.play(current_board, chess.engine.Limit(time=1.0, depth=depth))
        sf_best = result.move
        
        try:
            puzzle_move = current_board.parse_san(puzzle_first)
            sf_agrees.append(puzzle_move == sf_best)
            sf_moves.append(current_board.san(sf_best))
            pv_moves.append(puzzle_first)
            
            # Make the puzzle's first move
            current_board.push(puzzle_move)
            
        except Exception as e:
            return {
                'puzzle_id': puzzle_id,
                'side_to_move': side_to_move,
                'error': f"Could not parse first move: {e}"
            }
    
    # Now check the principal variation moves
    for i, move_str in enumerate(pv):
        # Get Stockfish's best move
        result = engine.play(current_board, chess.engine.Limit(time=0.5, depth=depth))
        sf_best = result.move
        
        try:
            # Parse the PV move
            pv_move = current_board.parse_san(move_str)
            
            # Check if Stockfish agrees
            agrees = (pv_move == sf_best)
            sf_agrees.append(agrees)
            sf_moves.append(current_board.san(sf_best))
            pv_moves.append(move_str)
            
            # Make the PV move to continue
            current_board.push(pv_move)
            
        except Exception as e:
            # Can't continue if we can't parse the move
            break
    
    # Calculate agreement statistics
    total_moves = len(sf_agrees)
    agreed_moves = sum(sf_agrees)
    
    return {
        'puzzle_id': puzzle_id,
        'side_to_move': side_to_move,
        'total_moves': total_moves,
        'agreed_moves': agreed_moves,
        'agreement_rate': agreed_moves / total_moves if total_moves > 0 else 0,
        'first_move_agrees': sf_agrees[0] if sf_agrees else None,
        'all_moves_agree': all(sf_agrees) if sf_agrees else False,
        'sf_moves': sf_moves[:3],  # First 3 Stockfish moves
        'pv_moves': pv_moves[:3],  # First 3 PV moves
        'move_by_move': list(zip(pv_moves[:5], sf_moves[:5], sf_agrees[:5]))
    }


def main():
    # Load puzzle data
    df = pd.read_csv('study_puzzles_head30_normal.csv')
    print(f"Loaded {len(df)} puzzles from normal dataset")
    print("="*70)
    
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
        print("Analyzing puzzles (showing first move agreement):")
        print("-"*70)
        
        for idx, row in df.iterrows():
            result = analyze_puzzle(row, engine, depth=20)
            
            if 'error' not in result:
                results.append(result)
                
                # Print progress
                symbol = "✓" if result['first_move_agrees'] else "✗"
                all_symbol = "✓✓" if result['all_moves_agree'] else ""
                
                print(f"{idx+1:2d}. {result['puzzle_id']}: {symbol} First move, "
                      f"{result['agreed_moves']}/{result['total_moves']} total "
                      f"({100*result['agreement_rate']:.0f}%) {all_symbol}")
                
                # Show details for disagreements
                if not result['first_move_agrees'] and result['pv_moves']:
                    print(f"    PV: {result['pv_moves'][0]}, SF: {result['sf_moves'][0]}")
            else:
                print(f"{idx+1:2d}. {result['puzzle_id']}: Error - {result['error']}")
    
    finally:
        engine.quit()
        print("\nStockfish closed")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('normal_puzzles_analysis.csv', index=False)
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY - NORMAL PUZZLE DATASET")
    print("="*70)
    
    total = len(results_df)
    
    # First move agreement
    first_agrees = results_df['first_move_agrees'].sum()
    print(f"\nFirst Move Agreement:")
    print(f"  Stockfish plays puzzle's first move: {first_agrees}/{total} ({100*first_agrees/total:.1f}%)")
    
    # Full agreement
    full_agrees = results_df['all_moves_agree'].sum()
    print(f"\nFull Sequence Agreement:")
    print(f"  Stockfish plays entire sequence: {full_agrees}/{total} ({100*full_agrees/total:.1f}%)")
    
    # Average agreement
    avg_agreement = results_df['agreement_rate'].mean()
    print(f"\nAverage Agreement Rate: {100*avg_agreement:.1f}%")
    
    # Show puzzles with full agreement
    if full_agrees > 0:
        print(f"\nPuzzles where Stockfish plays the entire principal variation:")
        full_agreement_df = results_df[results_df['all_moves_agree']]
        for _, row in full_agreement_df.iterrows():
            print(f"  {row['puzzle_id']}: {row['total_moves']} moves")
    
    # Show puzzles with high agreement
    high_agreement = results_df[results_df['agreement_rate'] >= 0.75]
    if len(high_agreement) > len(full_agreement_df) if full_agrees > 0 else len(high_agreement) > 0:
        print(f"\nPuzzles with ≥75% move agreement:")
        for _, row in high_agreement.iterrows():
            if not row['all_moves_agree']:
                print(f"  {row['puzzle_id']}: {row['agreed_moves']}/{row['total_moves']} moves ({100*row['agreement_rate']:.0f}%)")
    
    # Detailed breakdown for a few examples
    print("\n" + "="*70)
    print("DETAILED EXAMPLES (First 5 puzzles)")
    print("="*70)
    
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\nPuzzle {row['puzzle_id']}:")
        print(f"  Side to move: {row['side_to_move']}")
        print(f"  Agreement: {row['agreed_moves']}/{row['total_moves']} moves")
        
        if row['move_by_move']:
            print("  Move-by-move comparison:")
            for j, (pv_move, sf_move, agrees) in enumerate(row['move_by_move'][:3]):
                symbol = "✓" if agrees else "✗"
                print(f"    Move {j+1}: PV={pv_move:6s} SF={sf_move:6s} {symbol}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if first_agrees / total < 0.5:
        print("Even in the 'normal' dataset, Stockfish frequently disagrees with")
        print("the puzzle solutions, confirming these are testing specific patterns")
        print("or tactical themes rather than optimal computer play.")
    else:
        print("The 'normal' dataset shows better agreement with Stockfish,")
        print("suggesting these may be more aligned with computer evaluation.")
    
    print(f"\nKey finding: Stockfish agrees with the first move only {100*first_agrees/total:.1f}% of the time")
    print(f"and plays the full principal variation only {100*full_agrees/total:.1f}% of the time.")


if __name__ == "__main__":
    main()