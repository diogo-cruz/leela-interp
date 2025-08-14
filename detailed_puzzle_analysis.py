#!/usr/bin/env python3
"""
Detailed analysis of puzzle branches to understand why Stockfish disagrees with PV.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Load the results
    df = pd.read_csv('stockfish_analysis_30puzzles.csv')
    
    print("="*70)
    print("DETAILED ANALYSIS OF STOCKFISH VS PRINCIPAL VARIATION DISAGREEMENT")
    print("="*70)
    
    # Overall statistics
    total = len(df)
    agrees = df[df['agrees_with_pv'] == True].shape[0]
    disagrees = df[df['agrees_with_pv'] == False].shape[0]
    
    print(f"\nOverall Agreement:")
    print(f"  Total puzzles: {total}")
    print(f"  Stockfish agrees with PV: {agrees} ({100*agrees/total:.1f}%)")
    print(f"  Stockfish disagrees with PV: {disagrees} ({100*disagrees/total:.1f}%)")
    
    # Pattern analysis: Which branch is typically the PV?
    pv_branch_1 = df[df['pv_is_branch'] == 1].shape[0]
    pv_branch_2 = df[df['pv_is_branch'] == 2].shape[0]
    
    print(f"\nPrincipal Variation Distribution:")
    print(f"  PV is Branch 1: {pv_branch_1} puzzles ({100*pv_branch_1/total:.1f}%)")
    print(f"  PV is Branch 2: {pv_branch_2} puzzles ({100*pv_branch_2/total:.1f}%)")
    
    # Which branch does Stockfish prefer?
    sf_prefers_1 = df[df['stockfish_prefers_branch'] == 1].shape[0]
    sf_prefers_2 = df[df['stockfish_prefers_branch'] == 2].shape[0]
    
    print(f"\nStockfish Preference:")
    print(f"  Prefers Branch 1: {sf_prefers_1} puzzles ({100*sf_prefers_1/total:.1f}%)")
    print(f"  Prefers Branch 2: {sf_prefers_2} puzzles ({100*sf_prefers_2/total:.1f}%)")
    
    # Analyze evaluation differences
    df['cp_diff'] = abs(df['branch_1_cp'] - df['branch_2_cp'])
    df['pv_cp'] = df.apply(lambda x: x['branch_1_cp'] if x['pv_is_branch'] == 1 else x['branch_2_cp'], axis=1)
    df['alt_cp'] = df.apply(lambda x: x['branch_2_cp'] if x['pv_is_branch'] == 1 else x['branch_1_cp'], axis=1)
    df['sf_preferred_cp'] = df.apply(lambda x: x['branch_1_cp'] if x['stockfish_prefers_branch'] == 1 else x['branch_2_cp'], axis=1)
    
    # Calculate advantage for each side
    df['pv_advantage'] = df.apply(
        lambda x: x['pv_cp'] if x['side_to_move'] == 'White' else -x['pv_cp'], 
        axis=1
    )
    df['sf_advantage'] = df.apply(
        lambda x: x['sf_preferred_cp'] if x['side_to_move'] == 'White' else -x['sf_preferred_cp'], 
        axis=1
    )
    
    print(f"\nEvaluation Statistics:")
    print(f"  Average difference between branches: {df['cp_diff'].mean():.1f} cp")
    print(f"  Median difference: {df['cp_diff'].median():.1f} cp")
    print(f"  Max difference: {df['cp_diff'].max():.1f} cp")
    
    disagreements = df[df['agrees_with_pv'] == False]
    if len(disagreements) > 0:
        print(f"\nFor puzzles where Stockfish disagrees:")
        print(f"  Average PV evaluation: {disagreements['pv_cp'].mean():.1f} cp")
        print(f"  Average alternative evaluation: {disagreements['alt_cp'].mean():.1f} cp")
        print(f"  Average Stockfish preferred: {disagreements['sf_preferred_cp'].mean():.1f} cp")
        
        # Check if PV tends to be worse
        pv_worse = disagreements.apply(
            lambda x: (x['pv_cp'] < x['alt_cp'] if x['side_to_move'] == 'White' 
                      else x['pv_cp'] > x['alt_cp']), 
            axis=1
        ).sum()
        
        print(f"\n  PV is objectively worse in {pv_worse}/{len(disagreements)} cases ({100*pv_worse/len(disagreements):.1f}%)")
    
    # Pattern: Are PVs testing specific tactical themes?
    print(f"\n" + "="*70)
    print("HYPOTHESIS: Principal Variations may be testing tactical patterns")
    print("rather than optimal play")
    print("="*70)
    
    # Look at large evaluation differences
    large_diff = df[df['cp_diff'] > 500]
    print(f"\nPuzzles with >500 cp difference between branches: {len(large_diff)}")
    
    if len(large_diff) > 0:
        print("These puzzles likely test specific tactical patterns:")
        for _, row in large_diff.head(10).iterrows():
            pv_branch = "Branch 1" if row['pv_is_branch'] == 1 else "Branch 2"
            sf_branch = "Branch 1" if row['stockfish_prefers_branch'] == 1 else "Branch 2"
            print(f"  {row['puzzle_id']}: PV={pv_branch} ({row['pv_cp']:.0f} cp), SF={sf_branch} ({row['sf_preferred_cp']:.0f} cp), diff={row['cp_diff']:.0f} cp")
    
    # Save detailed analysis
    df.to_csv('detailed_puzzle_analysis.csv', index=False)
    print(f"\nDetailed analysis saved to detailed_puzzle_analysis.csv")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Distribution of CP differences
    axes[0, 0].hist(df['cp_diff'], bins=20, edgecolor='black')
    axes[0, 0].set_xlabel('CP Difference between branches')
    axes[0, 0].set_ylabel('Number of puzzles')
    axes[0, 0].set_title('Distribution of Evaluation Differences')
    
    # Plot 2: PV vs Alternative evaluations
    disagreements = df[df['agrees_with_pv'] == False]
    if len(disagreements) > 0:
        axes[0, 1].scatter(disagreements['pv_cp'], disagreements['alt_cp'], alpha=0.6)
        axes[0, 1].plot([-3000, 3000], [-3000, 3000], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('PV Evaluation (cp)')
        axes[0, 1].set_ylabel('Alternative Evaluation (cp)')
        axes[0, 1].set_title('PV vs Alternative Branch Evaluations')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Agreement by side to move
    white_puzzles = df[df['side_to_move'] == 'White']
    black_puzzles = df[df['side_to_move'] == 'Black']
    
    white_agree = white_puzzles[white_puzzles['agrees_with_pv'] == True].shape[0]
    black_agree = black_puzzles[black_puzzles['agrees_with_pv'] == True].shape[0]
    
    axes[1, 0].bar(['White to move', 'Black to move'], 
                   [100*white_agree/len(white_puzzles) if len(white_puzzles) > 0 else 0,
                    100*black_agree/len(black_puzzles) if len(black_puzzles) > 0 else 0])
    axes[1, 0].set_ylabel('Agreement with PV (%)')
    axes[1, 0].set_title('Agreement by Side to Move')
    
    # Plot 4: Evaluation advantage comparison
    axes[1, 1].scatter(df['pv_advantage'], df['sf_advantage'], alpha=0.6)
    axes[1, 1].plot([-3000, 3000], [-3000, 3000], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('PV Advantage (from mover perspective)')
    axes[1, 1].set_ylabel('Stockfish Preferred Advantage')
    axes[1, 1].set_title('Tactical Advantage Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('puzzle_analysis_plots.png', dpi=150)
    print("\nVisualization saved to puzzle_analysis_plots.png")
    
    # Final insight
    print(f"\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("The systematic disagreement between Stockfish and the principal variations")
    print("suggests these puzzles are designed to test specific chess patterns or")
    print("heuristics rather than pure tactical calculation. This makes them ideal")
    print("for studying how neural networks learn chess concepts beyond brute-force")
    print("evaluation.")


if __name__ == "__main__":
    main()