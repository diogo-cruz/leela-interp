import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from leela_interp import Lc0Model, get_lc0_pv_probabilities
from leela_interp.core.double_branch import DoubleBranchStudy


def main(args):
    base_dir = Path(args.base_dir)

    puzzles = pd.read_csv("lichess_db_puzzle.csv", nrows=args.n_puzzles)
    
    print("Number of puzzles:", len(puzzles))

    puzzles["principal_variation"] = pd.Series(
        [p.Moves.split(" ")[1:] for p in puzzles.itertuples()], index=puzzles.index
    )
    puzzles = puzzles[puzzles.principal_variation.apply(lambda x:len(x) == 3)]
    print("Number of puzzles after filtering for PV length 3:", len(puzzles))

    #puzzles = puzzles[puzzles.Themes.apply(lambda x: "mateIn2" not in x)]
    #print("Number of puzzles after filtering for no mateIn2:", len(puzzles))

    puzzles = puzzles[puzzles.principal_variation.apply(lambda x: x[0][2:4] != x[2][2:4])]
    print("Number of puzzles after filtering for different 1st and 3rd targets:", len(puzzles))

    #puzzles = puzzles[puzzles.principal_variation.apply(lambda x: x[0][2:4] == x[1][2:4])]
    #print("Number of puzzles after filtering for same 1st and 2nd targets:", len(puzzles))

    big_model = Lc0Model(base_dir / "lc0.onnx", device=args.device)
    small_model = Lc0Model(base_dir / "LD2.onnx", device=args.device)

    (
        puzzles["sparring_full_pv_probs"],
        puzzles["sparring_full_model_moves"],
        puzzles["sparring_wdl"],
    ) = get_lc0_pv_probabilities(small_model, puzzles, batch_size=args.batch_size)

    sparring_first_prob = puzzles["sparring_full_pv_probs"].apply(lambda x: x[0])
    sparring_second_prob = puzzles["sparring_full_pv_probs"].apply(lambda x: x[1])

    hard = sparring_first_prob < 0.3
    forcing = sparring_second_prob > 0.5
    interesting = hard & forcing
    puzzles = puzzles[interesting]
    print(f"Number of puzzles after filtering for hard and forcing second move: {len(puzzles)}")

    puzzles = DoubleBranchStudy.check_if_double_branch(big_model, puzzles, end=3, min_prob=[0.3, 0.7, 0.7])
    puzzles["full_model_moves"] = puzzles["branch_1"]
    # Use branch_1_probs for full_pv_probs if puzzles["branch_1"][0] == puzzles["principal_variation"][0], and branch_2_probs otherwise
    puzzles["full_pv_probs"] = puzzles.apply(lambda row: 
        row["branch_1_probs"] if row["branch_1"][0] == row["principal_variation"][0] 
        else row["branch_2_probs"], axis=1)
    
    # puzzles["full_pv_probs"], puzzles["full_model_moves"], puzzles["full_wdl"] = (
    #     get_lc0_pv_probabilities(big_model, puzzles, batch_size=batch_size)
    # )
    print(f"Number of puzzles after filtering for double branch: {len(puzzles)}")

    with open(base_dir / "puzzles/interesting_puzzles_nomate3_without_corruptions.pkl", "wb") as f:
        pickle.dump(puzzles, f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--n_puzzles", default=4_062_423, type=int)
    parser.add_argument("--hardness_threshold", default=2.0, type=float)
    parser.add_argument("--correctness_threshold", default=-1.0, type=float)
    parser.add_argument("--forcing_threshold", default=-1.0, type=float)
    args = parser.parse_args()
    main(args)
