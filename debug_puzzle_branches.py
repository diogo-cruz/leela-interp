#!/usr/bin/env python3
"""
Debug script to understand puzzle branch evaluations.
"""

import chess
import chess.engine
import chess.pgn


def analyze_puzzle_in_detail(fen, moves_str, branch_1, branch_2, engine, puzzle_id):
    """Detailed analysis of a single puzzle."""
    
    print(f"\n{'='*70}")
    print(f"PUZZLE {puzzle_id}")
    print(f"{'='*70}")
    
    board = chess.Board(fen)
    side_to_move = "White" if board.turn else "Black"
    
    print(f"Initial FEN: {fen}")
    print(f"Side to move: {side_to_move}")
    
    # Parse the complete solution
    moves = moves_str.split()
    print(f"\nComplete solution: {moves}")
    
    # Evaluate initial position
    info = engine.analyse(board, chess.engine.Limit(time=2.0, depth=25))
    initial_score = info['score'].white().cp
    if initial_score is None:
        if info['score'].is_mate():
            initial_score = 30000 if info['score'].white().mate() > 0 else -30000
        else:
            initial_score = 0
    
    print(f"Initial evaluation: {initial_score} cp")
    
    # Play first move
    first_move = moves[0]
    print(f"\nFirst move: {first_move}")
    
    try:
        move = board.parse_san(first_move)
        board_after_first = board.copy()
        board_after_first.push(move)
        
        # Evaluate after first move
        info = engine.analyse(board_after_first, chess.engine.Limit(time=2.0, depth=25))
        after_first_score = info['score'].white().cp
        if after_first_score is None:
            if info['score'].is_mate():
                after_first_score = 30000 if info['score'].white().mate() > 0 else -30000
            else:
                after_first_score = 0
        
        print(f"After first move: {after_first_score} cp")
        
        # Now analyze both branches
        print(f"\n--- BRANCH 1 (Principal Variation) ---")
        print(f"Moves: {branch_1}")
        
        board_b1 = board_after_first.copy()
        for i, move_str in enumerate(branch_1):
            try:
                move = board_b1.parse_san(move_str)
                board_b1.push(move)
                side = "White" if (i % 2 == 1) else "Black"  # Alternating after first move
                print(f"  Move {i+1}: {side} plays {move_str}")
            except Exception as e:
                print(f"  Error at move {i+1} ({move_str}): {e}")
                break
        
        # Evaluate branch 1 final position
        info = engine.analyse(board_b1, chess.engine.Limit(time=2.0, depth=25))
        b1_score = info['score'].white().cp
        if b1_score is None:
            if info['score'].is_mate():
                b1_score = 30000 if info['score'].white().mate() > 0 else -30000
            else:
                b1_score = 0
        
        print(f"Branch 1 final evaluation: {b1_score} cp")
        print(f"Branch 1 change from initial: {b1_score - initial_score:+d} cp")
        
        print(f"\n--- BRANCH 2 (Alternative) ---")
        print(f"Moves: {branch_2}")
        
        board_b2 = board_after_first.copy()
        for i, move_str in enumerate(branch_2):
            try:
                move = board_b2.parse_san(move_str)
                board_b2.push(move)
                side = "White" if (i % 2 == 1) else "Black"  # Alternating after first move
                print(f"  Move {i+1}: {side} plays {move_str}")
            except Exception as e:
                print(f"  Error at move {i+1} ({move_str}): {e}")
                break
        
        # Evaluate branch 2 final position
        info = engine.analyse(board_b2, chess.engine.Limit(time=2.0, depth=25))
        b2_score = info['score'].white().cp
        if b2_score is None:
            if info['score'].is_mate():
                b2_score = 30000 if info['score'].white().mate() > 0 else -30000
            else:
                b2_score = 0
        
        print(f"Branch 2 final evaluation: {b2_score} cp")
        print(f"Branch 2 change from initial: {b2_score - initial_score:+d} cp")
        
        # Comparison
        print(f"\n--- COMPARISON ---")
        print(f"Branch 1 (PV): {b1_score} cp")
        print(f"Branch 2 (Alt): {b2_score} cp")
        
        if side_to_move == "White":
            if b1_score > b2_score:
                print(f"✓ Branch 1 (PV) is better for White by {b1_score - b2_score} cp")
            else:
                print(f"⚠️ Branch 2 is better for White by {b2_score - b1_score} cp")
                print(f"   But Branch 1 is the principal variation!")
        else:  # Black
            if b1_score < b2_score:
                print(f"✓ Branch 1 (PV) is better for Black by {b2_score - b1_score} cp")
            else:
                print(f"⚠️ Branch 2 is better for Black by {b1_score - b2_score} cp")
                print(f"   But Branch 1 is the principal variation!")
        
        # Let's also check if these are tactical puzzles with forced sequences
        print(f"\n--- DEEPER ANALYSIS ---")
        
        # Get the best move according to Stockfish from initial position
        result = engine.play(board, chess.engine.Limit(time=2.0, depth=25))
        stockfish_best = result.move.uci()
        print(f"Stockfish's best move from initial: {stockfish_best}")
        print(f"Puzzle's first move in UCI: {board.parse_san(first_move).uci()}")
        
        if stockfish_best == board.parse_san(first_move).uci():
            print(f"✓ Stockfish agrees with puzzle's first move")
        else:
            print(f"⚠️ Stockfish prefers a different first move")
            
    except Exception as e:
        print(f"Error analyzing puzzle: {e}")


def main():
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
        engine.configure({"Threads": 4, "Hash": 512})
        print("Stockfish started successfully")
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    try:
        # Puzzle 1
        analyze_puzzle_in_detail(
            fen='8/b7/P4k2/1N2p3/8/3r3P/R4p1K/8 w - - 0 50',
            moves_str='h2g2 d3g3 g2g3 f2f1q',
            branch_1=['d3g3', 'g2g3', 'f2f1q'],
            branch_2=['d3d1', 'a2f2', 'a7f2'],
            engine=engine,
            puzzle_id='0hgJT'
        )
        
        # Puzzle 2
        analyze_puzzle_in_detail(
            fen='2k2r1r/2p5/5pp1/2pppn2/Q7/4PqB1/PP3P1P/1K1R2R1 b - - 1 27',
            moves_str='f5g3 a4a8 c8d7 d1d5',
            branch_1=['g1g3', 'f3e4', 'a4e4'],
            branch_2=['a4a8', 'c8d7', 'd1d5'],
            engine=engine,
            puzzle_id='1zJDp'
        )
        
        # Puzzle 3
        analyze_puzzle_in_detail(
            fen='r5k1/5p1p/6p1/1B3q2/P2R1n1P/2B2P2/KP3P2/4R3 b - - 0 33',
            moves_str='f5b5 d4d8 a8d8 a4b5',
            branch_1=['d4d8', 'a8d8', 'a4b5'],
            branch_2=['d4f4', 'a8a4', 'f4a4'],
            engine=engine,
            puzzle_id='2JetC'
        )
        
    finally:
        engine.quit()
        print("\nStockfish closed")


if __name__ == "__main__":
    main()