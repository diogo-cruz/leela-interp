#!/usr/bin/env python3
"""
Create scatter plot of Stockfish centipawn evaluation vs branch probability.
Uses the alternate dataset from study_puzzles_head_all_alternate.csv
"""

import chess
import chess.engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import seaborn as sns


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


def evaluate_branches(puzzle_row: pd.Series, engine: chess.engine.SimpleEngine) -> dict:
    """
    Evaluate both branches for a puzzle and get Stockfish score deltas.
    """
    try:
        puzzle_id = puzzle_row['PuzzleId']
        fen = puzzle_row['FEN']
        moves_str = puzzle_row['Moves']
        
        # Parse data
        all_moves = moves_str.split()
        branch_1 = parse_list_string(puzzle_row['branch_1'])
        branch_2 = parse_list_string(puzzle_row['branch_2'])
        branch_1_probs = parse_prob_string(puzzle_row['branch_1_probs'])
        branch_2_probs = parse_prob_string(puzzle_row['branch_2_probs'])
        
        # Parse principal variation
        pv = parse_list_string(puzzle_row['principal_variation'])
        
        # Setup board (FEN + first move to get actual puzzle position)
        board = chess.Board(fen)
        if all_moves:
            setup_move = board.parse_san(all_moves[0])
            board.push(setup_move)
        
        # EVALUATE BASELINE POSITION (before any branch move)
        baseline_info = engine.analyse(board, chess.engine.Limit(time=0.3, depth=18))
        baseline_score = baseline_info['score']
        
        if baseline_score.is_mate():
            baseline_cp = 30000 if baseline_score.white().mate() > 0 else -30000
        else:
            baseline_cp = baseline_score.white().score(mate_score=30000)
        
        # Adjust baseline for perspective
        if board.turn == chess.BLACK:
            baseline_cp = -baseline_cp
        
        # Determine which branch is PV (compare first moves)
        branch_1_is_pv = False
        branch_2_is_pv = False
        if pv and branch_1 and branch_1[0] == pv[0]:
            branch_1_is_pv = True
        elif pv and branch_2 and branch_2[0] == pv[0]:
            branch_2_is_pv = True
        
        # Evaluate first move of each branch
        results = {
            'puzzle_id': puzzle_id,
            'side_to_move': board.turn,
            'baseline_cp': baseline_cp
        }
        
        # Branch 1
        if branch_1 and branch_1_probs:
            try:
                move = board.parse_san(branch_1[0])
                board_after = board.copy()
                board_after.push(move)
                
                info = engine.analyse(board_after, chess.engine.Limit(time=0.3, depth=18))
                score = info['score']
                
                if score.is_mate():
                    cp_after = 30000 if score.white().mate() > 0 else -30000
                else:
                    cp_after = score.white().score(mate_score=30000)
                
                # Adjust for perspective (positive = good for side to move)
                if board.turn == chess.BLACK:
                    cp_after = -cp_after
                
                # Calculate DELTA from baseline
                cp_delta = cp_after - baseline_cp
                
                results['branch_1_cp'] = cp_delta  # Now storing delta, not absolute
                results['branch_1_prob'] = branch_1_probs[0]
                results['branch_1_is_pv'] = branch_1_is_pv
                
            except:
                pass
        
        # Branch 2
        if branch_2 and branch_2_probs:
            try:
                move = board.parse_san(branch_2[0])
                board_after = board.copy()
                board_after.push(move)
                
                info = engine.analyse(board_after, chess.engine.Limit(time=0.3, depth=18))
                score = info['score']
                
                if score.is_mate():
                    cp_after = 30000 if score.white().mate() > 0 else -30000
                else:
                    cp_after = score.white().score(mate_score=30000)
                
                # Adjust for perspective
                if board.turn == chess.BLACK:
                    cp_after = -cp_after
                
                # Calculate DELTA from baseline
                cp_delta = cp_after - baseline_cp
                
                results['branch_2_cp'] = cp_delta  # Now storing delta, not absolute
                results['branch_2_prob'] = branch_2_probs[0]
                results['branch_2_is_pv'] = branch_2_is_pv
                
            except:
                pass
        
        return results
    
    except Exception as e:
        print(f"Error processing {puzzle_row['PuzzleId']}: {e}")
        return None


def main():
    print("="*70)
    print("STOCKFISH CENTIPAWN vs BRANCH PROBABILITY ANALYSIS (ALTERNATE)")
    print("="*70)
    
    # Load data
    df = pd.read_csv('study_puzzles_head_all_alternate.csv')
    print(f"Total puzzles in alternate dataset: {len(df)}")
    
    # Limit to first 100 puzzles for reasonable runtime
    df = df.head(100)
    print(f"Processing first {len(df)} puzzles")
    
    # Start Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci('/usr/games/stockfish')
        engine.configure({"Threads": 4, "Hash": 1024})
        print("Stockfish started successfully")
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    # Collect data
    all_data = []
    
    try:
        print("\nEvaluating puzzles...")
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{len(df)}")
            
            result = evaluate_branches(row, engine)
            if result:
                # Add both branches as separate data points
                if 'branch_1_cp' in result and 'branch_1_prob' in result:
                    all_data.append({
                        'cp': result['branch_1_cp'],
                        'prob': result['branch_1_prob'],
                        'branch': 1,
                        'is_pv': result.get('branch_1_is_pv', False),
                        'puzzle_id': result['puzzle_id']
                    })
                
                if 'branch_2_cp' in result and 'branch_2_prob' in result:
                    all_data.append({
                        'cp': result['branch_2_cp'],
                        'prob': result['branch_2_prob'],
                        'branch': 2,
                        'is_pv': result.get('branch_2_is_pv', False),
                        'puzzle_id': result['puzzle_id']
                    })
    
    finally:
        engine.quit()
        print("\nStockfish closed")
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(all_data)
    print(f"\nCollected {len(plot_df)} data points")
    
    # Filter out extreme values for better visualization
    plot_df = plot_df[(plot_df['cp'] > -10000) & (plot_df['cp'] < 10000)]
    print(f"After filtering extreme values: {len(plot_df)} data points")
    
    # Save data
    plot_df.to_csv('stockfish_vs_probabilities_alternate_data.csv', index=False)
    
    # Create scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Main scatter plot with different markers for PV/non-PV
    ax = axes[0, 0]
    
    # Separate PV and non-PV points
    pv_data = plot_df[plot_df['is_pv'] == True]
    non_pv_data = plot_df[plot_df['is_pv'] == False]
    
    # Plot non-PV points as circles
    if len(non_pv_data) > 0:
        ax.scatter(non_pv_data['cp'], non_pv_data['prob'], 
                  alpha=0.4, s=30, marker='o', c='red', label='Non-PV branch')
    
    # Plot PV points as triangles
    if len(pv_data) > 0:
        ax.scatter(pv_data['cp'], pv_data['prob'], 
                  alpha=0.6, s=50, marker='^', c='blue', label='PV branch')
    
    ax.set_xlabel('Stockfish Evaluation Delta (centipawns)', fontsize=12)
    ax.set_ylabel('Branch Probability (from dataset)', fontsize=12)
    ax.set_title('Stockfish Evaluation Change vs Branch Probability (Alternate)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10)
    
    # Calculate and display correlation
    valid_data = plot_df.dropna(subset=['cp', 'prob'])
    if len(valid_data) > 0:
        pearson_corr, pearson_p = pearsonr(valid_data['cp'], valid_data['prob'])
        spearman_corr, spearman_p = spearmanr(valid_data['cp'], valid_data['prob'])
        
        # Separate correlations for PV and non-PV
        pv_valid = pv_data.dropna(subset=['cp', 'prob'])
        non_pv_valid = non_pv_data.dropna(subset=['cp', 'prob'])
        
        stats_text = f'Overall: r = {pearson_corr:.3f} (p={pearson_p:.3e})\n'
        
        if len(pv_valid) > 0:
            pv_pearson, pv_p = pearsonr(pv_valid['cp'], pv_valid['prob'])
            stats_text += f'PV branches: r = {pv_pearson:.3f} (n={len(pv_valid)})\n'
        
        if len(non_pv_valid) > 0:
            non_pv_pearson, non_pv_p = pearsonr(non_pv_valid['cp'], non_pv_valid['prob'])
            stats_text += f'Non-PV branches: r = {non_pv_pearson:.3f} (n={len(non_pv_valid)})'
        
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hexbin plot for density
    ax = axes[0, 1]
    hexbin = ax.hexbin(plot_df['cp'], plot_df['prob'], gridsize=30, cmap='YlOrRd')
    ax.set_xlabel('Stockfish Evaluation Delta (centipawns)', fontsize=12)
    ax.set_ylabel('Branch Probability', fontsize=12)
    ax.set_title('Density Plot (Hexbin)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(hexbin, ax=ax, label='Count')
    
    # Histogram of centipawn scores
    ax = axes[1, 0]
    ax.hist(plot_df['cp'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Stockfish Evaluation Delta (centipawns)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Evaluation Changes', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No change')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Binned analysis
    ax = axes[1, 1]
    
    # Create bins for centipawn scores
    bins = [-10000, -1000, -500, -200, -50, 50, 200, 500, 1000, 10000]
    labels = ['<-1000', '-1000 to -500', '-500 to -200', '-200 to -50', 
              '-50 to 50', '50 to 200', '200 to 500', '500 to 1000', '>1000']
    
    plot_df['cp_bin'] = pd.cut(plot_df['cp'], bins=bins, labels=labels)
    
    # Calculate mean probability for each bin
    bin_stats = plot_df.groupby('cp_bin', observed=False)['prob'].agg(['mean', 'std', 'count'])
    bin_stats = bin_stats[bin_stats['count'] > 0]
    
    x_pos = np.arange(len(bin_stats))
    ax.bar(x_pos, bin_stats['mean'], yerr=bin_stats['std'], 
           capsize=5, alpha=0.7, color='darkgreen', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats.index, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Evaluation Delta Range (cp)', fontsize=12)
    ax.set_ylabel('Mean Branch Probability', fontsize=12)
    ax.set_title('Mean Probability by Evaluation Range', fontsize=14, fontweight='bold')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% probability')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    # Add sample sizes
    for i, count in enumerate(bin_stats['count']):
        ax.text(i, bin_stats['mean'].iloc[i] + bin_stats['std'].iloc[i] + 0.02,
                f'n={count}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('stockfish_vs_probabilities_alternate_plot.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to stockfish_vs_probabilities_alternate_plot.png")
    
    # Statistical summary
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    print(f"\nTotal data points: {len(plot_df)}")
    print(f"CP delta range: [{plot_df['cp'].min():.0f}, {plot_df['cp'].max():.0f}]")
    print(f"Probability range: [{plot_df['prob'].min():.3f}, {plot_df['prob'].max():.3f}]")
    
    print(f"\nCorrelation Analysis:")
    print(f"  Overall Pearson correlation: {pearson_corr:.3f} (p={pearson_p:.3e})")
    print(f"  Overall Spearman correlation: {spearman_corr:.3f} (p={spearman_p:.3e})")
    
    # Separate statistics for PV and non-PV
    pv_count = len(plot_df[plot_df['is_pv'] == True])
    non_pv_count = len(plot_df[plot_df['is_pv'] == False])
    print(f"\n  PV branches: {pv_count} data points")
    print(f"  Non-PV branches: {non_pv_count} data points")
    
    if len(pv_valid) > 0:
        print(f"  PV correlation: {pv_pearson:.3f}")
    if len(non_pv_valid) > 0:
        print(f"  Non-PV correlation: {non_pv_pearson:.3f}")
    
    if abs(pearson_corr) < 0.3:
        print("\n  → Weak overall correlation between Stockfish evaluation and branch probability")
    elif abs(pearson_corr) < 0.7:
        print("\n  → Moderate overall correlation between Stockfish evaluation and branch probability")
    else:
        print("\n  → Strong overall correlation between Stockfish evaluation and branch probability")
    
    print("\nInterpretation:")
    print("- Positive CP deltas indicate moves that improve the position")
    print("- Negative CP deltas indicate moves that worsen the position")
    print("- Higher probabilities should correspond to moves with better deltas")
    print("- The correlation shows how well the neural network probabilities")
    print("  align with Stockfish's evaluation of move quality")
    
    # Show bins with highest/lowest mean probabilities
    print(f"\nHighest mean probability bin: {bin_stats['mean'].idxmax()} "
          f"({bin_stats['mean'].max():.3f})")
    print(f"Lowest mean probability bin: {bin_stats['mean'].idxmin()} "
          f"({bin_stats['mean'].min():.3f})")
    
    # Compare with original dataset if available
    print("\n" + "="*70)
    print("DATASET COMPARISON")
    print("="*70)
    print("\nThis alternate dataset may have different characteristics than")
    print("the original, such as different puzzle difficulty levels or")
    print("different types of positions (tactical vs positional).")


if __name__ == "__main__":
    main()