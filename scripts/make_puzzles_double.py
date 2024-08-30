import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from leela_interp import Lc0Model, get_lc0_pv_probabilities


def main(args):
    base_dir = Path(args.base_dir)

    if args.generate:
        if (base_dir / "unfiltered_puzzles_double.pkl").exists():
            raise FileExistsError(
                "Unfiltered puzzles already exist, run without --generate option."
            )

        try:
            puzzles = pd.read_csv("lichess_db_puzzle.csv", nrows=args.n_puzzles)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Lichess puzzle database not found. "
                "Please download `lichess_db_puzzle.csv.zst` "
                "from https://database.lichess.org/#puzzles. "
                "Then decompress using `zstd -d lichess_db_puzzle.csv.zst`."
            )
        print("Number of puzzles:", len(puzzles))
        puzzles = puzzles[puzzles.Themes.apply(lambda x: "mateIn2" in x)]
        print("Number of puzzles after filtering for mateIn2:", len(puzzles))

        puzzles["principal_variation"] = pd.Series(
            [p.Moves.split(" ")[1:] for p in puzzles.itertuples()], index=puzzles.index
        )
        puzzles = puzzles[puzzles.principal_variation.apply(lambda x: x[0][2:4] != x[2][2:4])]
        print("Number of puzzles after filtering for different targets:", len(puzzles))

        big_model = Lc0Model(base_dir / "lc0.onnx", device=args.device)
        small_model = Lc0Model(base_dir / "LD2.onnx", device=args.device)
        
        batch_size = args.batch_size
        puzzles["full_pv_probs"], puzzles["full_model_moves"], puzzles["full_wdl"] = (
            get_lc0_pv_probabilities(big_model, puzzles, batch_size=batch_size)
        )

        (
            puzzles["sparring_full_pv_probs"],
            puzzles["sparring_full_model_moves"],
            puzzles["sparring_wdl"],
        ) = get_lc0_pv_probabilities(small_model, puzzles, batch_size=batch_size)

        with open(base_dir / "unfiltered_puzzles_double.pkl", "wb") as f:
            pickle.dump(puzzles, f)

    else:
        try:
            with open(base_dir / "unfiltered_puzzles_double.pkl", "rb") as f:
                puzzles = pickle.load(f)
        except FileNotFoundError:
            raise ValueError("Unfiltered puzzles not found, run with --generate")

        if args.n_puzzles:
            puzzles = puzzles.iloc[: args.n_puzzles]

    n_before = len(puzzles)
    pv_length = puzzles["principal_variation"].apply(len)
    puzzles = puzzles[pv_length == 3]
    n_after = len(puzzles)
    print(
        f"Filtered out {(n_before - n_after) / n_before:.2%} of puzzles with PV length != 3"
    )

    interesting_1 = puzzles["Themes"].apply(lambda x: "mateIn2" in x)
    #interesting_2 = puzzles["full_pv_probs"].apply(lambda x: 0.3 <= x[0] <= 0.5) # double
    interesting_2 = puzzles["full_pv_probs"].apply(lambda x: x[0] <= 0.3) # doublelow
    interesting_3 = puzzles["principal_variation"].apply(lambda x: x[0][2:4] != x[2][2:4])
    interesting = interesting_1 & interesting_2 & interesting_3

    print(
        f"Found {np.sum(interesting)} interesting puzzles ({np.mean(interesting):.2%})"
    )

    # Filter out the uninteresting puzzles:
    puzzles = puzzles[interesting]

    # Whether 1st, 2nd, and 3rd move target squares are all different:
    puzzles["different_targets"] = puzzles.apply(
        lambda x: len({move[2:4] for move in x.principal_variation[:3]}) == 3, axis=1
    )
    print(
        f"Out of those, {np.sum(puzzles['different_targets'])} "
        f"({np.mean(puzzles['different_targets']):.2%}) have different targets"
    )

    with open(base_dir / "interesting_puzzles_double_without_corruptions.pkl", "wb") as f:
        pickle.dump(puzzles, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--n_puzzles", default=4_062_423, type=int)
    parser.add_argument("--hardness_threshold", default=2.0, type=float)
    parser.add_argument("--correctness_threshold", default=-1.0, type=float)
    parser.add_argument("--forcing_threshold", default=-1.0, type=float)
    args = parser.parse_args()
    main(args)
