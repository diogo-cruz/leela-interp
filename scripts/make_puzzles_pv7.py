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
    puzzles = puzzles[puzzles.principal_variation.apply(lambda x:len(x) == 7)]
    print("Number of puzzles after filtering for PV length 7:", len(puzzles)) #229775

    #puzzles = puzzles[puzzles.Themes.apply(lambda x: "mateIn2" not in x)]
    #print("Number of puzzles after filtering for no mateIn2:", len(puzzles))

    puzzles = puzzles[puzzles.principal_variation.apply(lambda x: x[0][2:4] != x[6][2:4])]
    print("Number of puzzles after filtering for different 1st and 7th targets:", len(puzzles)) #217795

    puzzles = puzzles[puzzles.principal_variation.apply(lambda x: x[2][2:4] != x[6][2:4])]
    print("Number of puzzles after filtering for different 3rd and 7th targets:", len(puzzles)) #205366

    puzzles = puzzles[puzzles.principal_variation.apply(lambda x: x[4][2:4] != x[6][2:4])]
    print("Number of puzzles after filtering for different 5th and 7th targets:", len(puzzles)) #179483

    #puzzles = puzzles[puzzles.principal_variation.apply(lambda x: x[0][2:4] == x[1][2:4])]
    #print("Number of puzzles after filtering for same 1st and 2nd targets:", len(puzzles))

    big_model = Lc0Model(base_dir / "lc0.onnx", device=args.device)
    small_model = Lc0Model(base_dir / "LD2.onnx", device=args.device)

    batch_size = args.batch_size
    (
        puzzles["sparring_full_pv_probs"],
        puzzles["sparring_full_model_moves"],
        puzzles["sparring_wdl"],
    ) = get_lc0_pv_probabilities(small_model, puzzles, batch_size=batch_size)
    sparring_first_prob = puzzles["sparring_full_pv_probs"].apply(lambda x: x[0])
    sparring_second_prob = puzzles["sparring_full_pv_probs"].apply(lambda x: x[1])
    #hard = sparring_first_prob < 0.05
    #forcing = sparring_second_prob > 0.7
    hard = sparring_first_prob < 0.2
    forcing = sparring_second_prob > -1.0
    interesting = hard & forcing
    puzzles = puzzles[interesting]
    print(f"Number of puzzles after filtering for hard and forcing second move: {len(puzzles)}") #pv7:10370, pv7_v2:69886

    puzzles["full_pv_probs"], puzzles["full_model_moves"], puzzles["full_wdl"] = (
        get_lc0_pv_probabilities(big_model, puzzles, batch_size=batch_size)
    )
    # Smallest probability the full model puts on any of the player's moves:
    player_min_prob = puzzles.apply(lambda row: row.full_pv_probs[::2], axis=1).apply(
        np.min
    )
    correct = player_min_prob > 0.5
    puzzles = puzzles[correct]
    print(f"Number of puzzles after filtering for correctness: {len(puzzles)}") #pv7:2496, pv7_v2:24492

    with open(base_dir / "puzzles/interesting_puzzles_pv7_v2_without_corruptions.pkl", "wb") as f:
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
