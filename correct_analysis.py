#!/usr/bin/env python3
"""
Correct analysis understanding that:
1. FEN is the position BEFORE the first move
2. First move in "Moves" gets us to the actual puzzle position
3. Principal variation/branches apply from THAT position
"""

import chess
import chess.engine
import pandas as pd


def parse_string_list(list_str: str) -> list:
    """Parse string representation of list from CSV."""
    list_str = list_str.strip("[]'\"")
    moves = [m.strip(" '\"") for m in list_str.split(',')]
    return moves


def analyze_puzzle_correctly(puzzle_row: pd.Series, engine: chess.engine.SimpleEngine, 
                            puzzle_type: str = "normal") -> dict:
    """
    Correctly analyze a puzzle understanding the FEN offset.
    """
    puzzle_id = puzzle_row['PuzzleId']
    fen = puzzle_row['FEN']
    moves_str = puzzle_row['Moves']
    pv = parse_string_list(puzzle_row['principal_variation'])
    
    # Parse complete moves
    all_moves = moves_str.split()
    
    # Initial board (BEFORE the setup move)
    board_before = chess.Board(fen)
    
    # Make the first move to get to the actual puzzle position
    setup_move = all_moves[0] if all_moves else None
    board_puzzle = board_before.copy()
    
    if setup_move:
        try:
            move = board_puzzle.parse_san(setup_move)
            board_puzzle.push(move)
        except Exception as e:
            return {'puzzle_id': puzzle_id, 'error': f"Cannot parse setup move: {e}"}
    
    # NOW we're at the actual puzzle position
    side_to_move = "White" if board_puzzle.turn else "Black"
    
    # Get Stockfish's evaluation and best move from the PUZZLE position
    info = engine.analyse(board_puzzle, chess.engine.Limit(time=2.0, depth=25))
    eval_puzzle = info['score'].white().score(mate_score=30000)
    
    result = engine.play(board_puzzle, chess.engine.Limit(time=2.0, depth=25))
    sf_best = result.move
    
    # The first move of the PV should be compared to Stockfish's best move
    pv_first = None
    if pv:
        try:
            pv_first = board_puzzle.parse_san(pv[0])
        except:
            pass
    
    agrees_first = (pv_first == sf_best) if pv_first else False
    
    # For double-branch puzzles, also check branches
    if puzzle_type == "double":
        branch_1 = parse_string_list(puzzle_row['branch_1']) if 'branch_1' in puzzle_row else []
        branch_2 = parse_string_list(puzzle_row['branch_2']) if 'branch_2' in puzzle_row else []
        
        # Evaluate branch 1
        board_b1 = board_puzzle.copy()
        b1_valid = True
        for move_str in branch_1:
            try:
                move = board_b1.parse_san(move_str)
                board_b1.push(move)
            except:
                b1_valid = False
                break
        
        if b1_valid:
            info_b1 = engine.analyse(board_b1, chess.engine.Limit(time=1.0, depth=20))
            eval_b1 = info_b1['score'].white().score(mate_score=30000)
        else:
            eval_b1 = None
        
        # Evaluate branch 2
        board_b2 = board_puzzle.copy()
        b2_valid = True
        for move_str in branch_2:
            try:
                move = board_b2.parse_san(move_str)
                board_b2.push(move)
            except:
                b2_valid = False
                break
        
        if b2_valid:
            info_b2 = engine.analyse(board_b2, chess.engine.Limit(time=1.0, depth=20))
            eval_b2 = info_b2['score'].white().score(mate_score=30000)
        else:
            eval_b2 = None
        
        # Which branch is PV?
        pv_is_branch = 1 if branch_1 == pv else (2 if branch_2 == pv else None)
        
        # Which branch does Stockfish prefer?
        if eval_b1 is not None and eval_b2 is not None:
            if board_puzzle.turn == chess.WHITE:
                sf_prefers = 1 if eval_b1 > eval_b2 else 2
            else:
                sf_prefers = 1 if eval_b1 < eval_b2 else 2
        else:
            sf_prefers = None
        
        return {
            'puzzle_id': puzzle_id,
            'setup_move': setup_move,
            'side_to_move': side_to_move,
            'eval_at_puzzle': eval_puzzle,
            'sf_best': board_puzzle.san(sf_best),
            'pv_first': pv[0] if pv else None,
            'sf_agrees_pv': agrees_first,
            'branch_1_eval': eval_b1,
            'branch_2_eval': eval_b2,
            'pv_is_branch': pv_is_branch,
            'sf_prefers_branch': sf_prefers,
            'sf_agrees_branch': (sf_prefers == pv_is_branch) if sf_prefers and pv_is_branch else None
        }
    else:
        # Normal puzzle - just check if SF plays the PV
        return {
            'puzzle_id': puzzle_id,
            'setup_move': setup_move,
            'side_to_move': side_to_move,
            'eval_at_puzzle': eval_puzzle,
            'sf_best': board_puzzle.san(sf_best),
            'pv_first': pv[0] if pv else None,
            'sf_agrees_pv': agrees_first
        }


def main():
    print("="*70)
    print("CORRECT ANALYSIS - Understanding FEN is position BEFORE first move")
    print("="*70)
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
        engine.configure({"Threads": 4, "Hash": 1024})
        print("Stockfish started successfully\n")
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    try:
        # Analyze normal puzzles
        print("\nNORMAL PUZZLES (study_puzzles_head30_normal.csv)")
        print("-"*70)
        
        df_normal = pd.read_csv('study_puzzles_head30_normal.csv')
        normal_results = []
        
        for idx in range(min(10, len(df_normal))):
            result = analyze_puzzle_correctly(df_normal.iloc[idx], engine, "normal")
            if 'error' not in result:
                normal_results.append(result)
                symbol = "✓" if result['sf_agrees_pv'] else "✗"
                print(f"{idx+1:2d}. {result['puzzle_id']}: {symbol} "
                      f"SF={result['sf_best']}, PV={result['pv_first']}")
        
        normal_agrees = sum(1 for r in normal_results if r['sf_agrees_pv'])
        print(f"\nNormal puzzles: Stockfish agrees with PV in {normal_agrees}/{len(normal_results)} "
              f"({100*normal_agrees/len(normal_results):.1f}%)")
        
        # Analyze double-branch puzzles
        print("\n\nDOUBLE-BRANCH PUZZLES (study_puzzles_head30.csv)")
        print("-"*70)
        
        df_double = pd.read_csv('study_puzzles_head30.csv')
        double_results = []
        
        for idx in range(min(10, len(df_double))):
            result = analyze_puzzle_correctly(df_double.iloc[idx], engine, "double")
            if 'error' not in result:
                double_results.append(result)
                pv_symbol = "✓" if result['sf_agrees_pv'] else "✗"
                branch_symbol = "✓" if result['sf_agrees_branch'] else "✗"
                print(f"{idx+1:2d}. {result['puzzle_id']}: "
                      f"PV first {pv_symbol}, "
                      f"PV branch={result['pv_is_branch']}, "
                      f"SF prefers={result['sf_prefers_branch']} {branch_symbol}")
        
        double_pv_agrees = sum(1 for r in double_results if r['sf_agrees_pv'])
        double_branch_agrees = sum(1 for r in double_results if r['sf_agrees_branch'])
        
        print(f"\nDouble-branch puzzles:")
        print(f"  SF agrees with PV first move: {double_pv_agrees}/{len(double_results)} "
              f"({100*double_pv_agrees/len(double_results):.1f}%)")
        print(f"  SF prefers PV branch: {double_branch_agrees}/{len(double_results)} "
              f"({100*double_branch_agrees/len(double_results):.1f}%)")
        
        # Show some detailed examples
        print("\n" + "="*70)
        print("DETAILED EXAMPLE")
        print("="*70)
        
        # Pick first puzzle for detailed analysis
        row = df_normal.iloc[0]
        puzzle_id = row['PuzzleId']
        fen = row['FEN']
        moves = row['Moves'].split()
        pv = parse_string_list(row['principal_variation'])
        
        print(f"\nPuzzle {puzzle_id}")
        print(f"FEN (before setup): {fen}")
        print(f"All moves: {moves}")
        print(f"Principal variation: {pv}")
        
        board = chess.Board(fen)
        print(f"\n1. Initial position (from FEN): {board.fen()}")
        print(f"   Side to move: {'White' if board.turn else 'Black'}")
        
        # Make setup move
        setup_move = moves[0]
        move = board.parse_san(setup_move)
        board.push(move)
        
        print(f"\n2. After setup move ({setup_move}): {board.fen()}")
        print(f"   Side to move: {'White' if board.turn else 'Black'}")
        print(f"   This is the ACTUAL PUZZLE POSITION")
        
        # Evaluate
        info = engine.analyse(board, chess.engine.Limit(time=2.0, depth=25))
        eval_pos = info['score'].white().score(mate_score=30000)
        
        result = engine.play(board, chess.engine.Limit(time=2.0, depth=25))
        sf_best = result.move
        
        print(f"\n3. From puzzle position:")
        print(f"   Evaluation: {eval_pos} cp")
        print(f"   Stockfish best: {board.san(sf_best)}")
        print(f"   PV first move: {pv[0] if pv else 'none'}")
        
        if pv:
            try:
                pv_move = board.parse_san(pv[0])
                if pv_move == sf_best:
                    print(f"   ✓ Stockfish agrees with principal variation!")
                else:
                    print(f"   ✗ Stockfish prefers a different move")
            except:
                print(f"   Error parsing PV move")
        
    finally:
        engine.quit()
        print("\nStockfish closed")


if __name__ == "__main__":
    main()