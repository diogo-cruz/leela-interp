#!/usr/bin/env python3
"""
Convert Stockfish centipawn evaluations to move probabilities.
Compare with the probabilities already in the dataset.
"""

import chess
import chess.engine
import pandas as pd
import numpy as np
import math


def parse_list_string(list_str: str) -> list:
    """Parse string representation of list from CSV."""
    list_str = list_str.strip("[]'\"")
    items = [m.strip(" '\"") for m in list_str.split(',')]
    return items


def parse_prob_string(prob_str: str) -> list:
    """Parse probability list from string."""
    prob_str = prob_str.strip("[]")
    probs = [float(p.strip()) for p in prob_str.split(',')]
    return probs


def cp_to_win_probability(cp: int) -> float:
    """
    Convert centipawn score to win probability using a sigmoid function.
    This is a common approximation used in chess engines.
    
    Formula: P(win) = 1 / (1 + exp(-k * cp))
    where k is a scaling factor (typically around 0.004 for centipawns)
    """
    # Common scaling factors
    k = 0.004  # This gives ~73% win prob at +100cp, ~88% at +200cp
    
    # Handle mate scores
    if abs(cp) >= 10000:
        return 1.0 if cp > 0 else 0.0
    
    # Sigmoid transformation
    win_prob = 1.0 / (1.0 + math.exp(-k * cp))
    return win_prob


def evaluate_moves_with_probabilities(board: chess.Board, moves: list, engine: chess.engine.SimpleEngine,
                                     time_limit: float = 2.0, depth: int = 25) -> dict:
    """
    Evaluate multiple moves and convert to probabilities.
    """
    evaluations = {}
    
    for move_str in moves:
        try:
            # Parse move
            try:
                move = board.parse_san(move_str)
            except:
                move = chess.Move.from_uci(move_str)
            
            if move in board.legal_moves:
                # Make move and evaluate
                board_after = board.copy()
                board_after.push(move)
                
                info = engine.analyse(board_after, chess.engine.Limit(time=time_limit, depth=depth))
                score = info['score']
                
                # Get score from the perspective of the side that just moved
                if score.is_mate():
                    mate_score = score.white().mate()
                    cp = 30000 if mate_score and mate_score > 0 else -30000
                else:
                    cp = score.white().score(mate_score=30000)
                
                # Adjust for side to move (negate if black moved)
                if board.turn == chess.BLACK:
                    cp = -cp
                
                evaluations[move_str] = cp
        except Exception as e:
            print(f"Error evaluating {move_str}: {e}")
            evaluations[move_str] = None
    
    return evaluations


def softmax_from_scores(scores: dict, temperature: float = 1.0) -> dict:
    """
    Convert scores to probabilities using softmax.
    This is often used to convert evaluations to move probabilities.
    
    Temperature controls the sharpness:
    - Low temperature (< 1): More peaked distribution (best move gets higher prob)
    - High temperature (> 1): More uniform distribution
    """
    # Filter out None values
    valid_scores = {k: v for k, v in scores.items() if v is not None}
    
    if not valid_scores:
        return {}
    
    # Convert to numpy array for easier computation
    moves = list(valid_scores.keys())
    scores_array = np.array(list(valid_scores.values()))
    
    # Scale scores (centipawns to pawns for reasonable range)
    scores_array = scores_array / 100.0
    
    # Apply temperature
    scores_array = scores_array / temperature
    
    # Softmax
    exp_scores = np.exp(scores_array - np.max(scores_array))  # Subtract max for numerical stability
    probabilities = exp_scores / np.sum(exp_scores)
    
    return dict(zip(moves, probabilities))


def analyze_puzzle_probabilities(puzzle_row: pd.Series, engine: chess.engine.SimpleEngine) -> dict:
    """
    Analyze a puzzle and compare Stockfish-derived probabilities with dataset probabilities.
    """
    puzzle_id = puzzle_row['PuzzleId']
    fen = puzzle_row['FEN']
    moves_str = puzzle_row['Moves']
    
    # Parse data
    all_moves = moves_str.split()
    branch_1 = parse_list_string(puzzle_row['branch_1'])
    branch_2 = parse_list_string(puzzle_row['branch_2'])
    branch_1_probs = parse_prob_string(puzzle_row['branch_1_probs'])
    branch_2_probs = parse_prob_string(puzzle_row['branch_2_probs'])
    
    # Setup board (FEN + first move)
    board = chess.Board(fen)
    if all_moves:
        try:
            setup_move = board.parse_san(all_moves[0])
            board.push(setup_move)
        except:
            return {'puzzle_id': puzzle_id, 'error': 'Cannot parse setup move'}
    
    # Now we're at the puzzle position
    results = {
        'puzzle_id': puzzle_id,
        'side_to_move': 'White' if board.turn else 'Black'
    }
    
    # Evaluate the first moves of each branch
    first_moves = []
    if branch_1:
        first_moves.append(('branch_1', branch_1[0]))
    if branch_2:
        first_moves.append(('branch_2', branch_2[0]))
    
    # Get Stockfish evaluations
    move_evals = {}
    for branch_name, move_str in first_moves:
        try:
            move = board.parse_san(move_str)
            board_after = board.copy()
            board_after.push(move)
            
            info = engine.analyse(board_after, chess.engine.Limit(time=2.0, depth=25))
            score = info['score']
            
            # Get score in centipawns
            if score.is_mate():
                cp = 30000 if score.white().mate() > 0 else -30000
            else:
                cp = score.white().score(mate_score=30000)
            
            move_evals[branch_name] = cp
            results[f'{branch_name}_cp'] = cp
            
        except Exception as e:
            results[f'{branch_name}_cp'] = None
    
    # Convert to probabilities using different methods
    if len(move_evals) == 2 and all(v is not None for v in move_evals.values()):
        cp1 = move_evals['branch_1']
        cp2 = move_evals['branch_2']
        
        # Method 1: Direct sigmoid on score difference
        if board.turn == chess.WHITE:
            score_diff = cp1 - cp2  # Positive if branch_1 is better for white
        else:
            score_diff = cp2 - cp1  # Positive if branch_1 is better for black (lower cp)
        
        # Probability that branch_1 is better
        prob_branch1_sigmoid = 1.0 / (1.0 + math.exp(-0.002 * score_diff))
        
        # Method 2: Softmax with temperature
        scores = {'branch_1': cp1 if board.turn == chess.WHITE else -cp1,
                 'branch_2': cp2 if board.turn == chess.WHITE else -cp2}
        
        probs_t1 = softmax_from_scores(scores, temperature=1.0)
        probs_t05 = softmax_from_scores(scores, temperature=0.5)
        probs_t2 = softmax_from_scores(scores, temperature=2.0)
        
        results['sf_prob_b1_sigmoid'] = prob_branch1_sigmoid
        results['sf_prob_b1_softmax_t1'] = probs_t1.get('branch_1', 0)
        results['sf_prob_b1_softmax_t05'] = probs_t05.get('branch_1', 0)
        results['sf_prob_b1_softmax_t2'] = probs_t2.get('branch_1', 0)
        
        # Dataset probability (first move only)
        results['data_prob_b1'] = branch_1_probs[0] if branch_1_probs else None
        results['data_prob_b2'] = branch_2_probs[0] if branch_2_probs else None
        
    return results


def main():
    print("="*70)
    print("STOCKFISH PROBABILITIES ANALYSIS")
    print("="*70)
    print("\nConverting Stockfish evaluations to probabilities and comparing")
    print("with the probabilities in the dataset.\n")
    
    # Load data
    df = pd.read_csv('study_puzzles_head30.csv')
    
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
        # Analyze first 10 puzzles
        for idx in range(min(10, len(df))):
            row = df.iloc[idx]
            result = analyze_puzzle_probabilities(row, engine)
            
            if 'error' not in result:
                results.append(result)
                
                print(f"{idx+1}. {result['puzzle_id']} ({result['side_to_move']}):")
                print(f"   Branch 1: {result.get('branch_1_cp', 'N/A')} cp")
                print(f"   Branch 2: {result.get('branch_2_cp', 'N/A')} cp")
                
                if 'sf_prob_b1_sigmoid' in result:
                    print(f"   Stockfish prob(B1): {result['sf_prob_b1_sigmoid']:.3f} (sigmoid)")
                    print(f"                       {result['sf_prob_b1_softmax_t1']:.3f} (softmax T=1.0)")
                    print(f"                       {result['sf_prob_b1_softmax_t05']:.3f} (softmax T=0.5)")
                    print(f"   Dataset prob(B1):   {result['data_prob_b1']:.3f}")
                    print(f"   Dataset prob(B2):   {result['data_prob_b2']:.3f}")
    
    finally:
        engine.quit()
        print("\nStockfish closed")
    
    # Analysis
    if results:
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("PROBABILITY COMPARISON")
        print("="*70)
        
        # Compare Stockfish probabilities with dataset probabilities
        valid = results_df.dropna(subset=['sf_prob_b1_sigmoid', 'data_prob_b1'])
        
        if len(valid) > 0:
            # Calculate correlation
            from scipy.stats import pearsonr
            
            corr_sigmoid, _ = pearsonr(valid['sf_prob_b1_sigmoid'], valid['data_prob_b1'])
            corr_soft1, _ = pearsonr(valid['sf_prob_b1_softmax_t1'], valid['data_prob_b1'])
            corr_soft05, _ = pearsonr(valid['sf_prob_b1_softmax_t05'], valid['data_prob_b1'])
            
            print(f"\nCorrelation between Stockfish and dataset probabilities:")
            print(f"  Sigmoid method:        {corr_sigmoid:.3f}")
            print(f"  Softmax (T=1.0):      {corr_soft1:.3f}")
            print(f"  Softmax (T=0.5):      {corr_soft05:.3f}")
            
            # Mean absolute difference
            mae_sigmoid = abs(valid['sf_prob_b1_sigmoid'] - valid['data_prob_b1']).mean()
            mae_soft1 = abs(valid['sf_prob_b1_softmax_t1'] - valid['data_prob_b1']).mean()
            mae_soft05 = abs(valid['sf_prob_b1_softmax_t05'] - valid['data_prob_b1']).mean()
            
            print(f"\nMean absolute error:")
            print(f"  Sigmoid method:        {mae_sigmoid:.3f}")
            print(f"  Softmax (T=1.0):      {mae_soft1:.3f}")
            print(f"  Softmax (T=0.5):      {mae_soft05:.3f}")
            
            print("\n" + "="*70)
            print("INTERPRETATION")
            print("="*70)
            print("\n1. Stockfish evaluations CAN be converted to probabilities")
            print("2. Common methods include:")
            print("   - Sigmoid on score difference (for binary choice)")
            print("   - Softmax on scores (for multiple moves)")
            print("3. The temperature parameter in softmax controls sharpness:")
            print("   - Lower T = more confident (closer to 0/1)")
            print("   - Higher T = more uncertain (closer to 0.5)")
            print("\n4. The dataset probabilities likely come from a neural network")
            print("   (like Leela) which naturally outputs probabilities via softmax")
            print("\n5. Correlation shows if the models agree on relative move strength")
            print("   even if absolute probabilities differ")


if __name__ == "__main__":
    main()