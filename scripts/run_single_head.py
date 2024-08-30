import argparse
import itertools
import pickle
from pathlib import Path
import re

import torch
import tqdm
from einops import rearrange
from leela_interp import Lc0sight, LeelaBoard
from leela_interp.tools import patching


def squarewise_patching(model, puzzles, args):
    return -patching.activation_patch(
        module_func=lambda layer: model.headwise_attention_output(layer),
        # Layer 12, Head 12, all query squares:
        locations=[(args.layer, args.head, i) for i in range(64)],
        model=model,
        puzzles=puzzles,
        batch_size=args.batch_size,
    )


def zero_ablate(location, model, batch_indices):
    layer, head, query, key = location
    model.attention_scores(layer).output[:, head, query, key] = 0


def single_weight_ablation(model, boards, args):
    effects = -patching.patch(
        patching_func=zero_ablate,
        locations=list(
            itertools.product([args.layer], [args.head], range(64), range(64))
        ),
        model=model,
        boards=boards,
        batch_size=args.batch_size,
    )
    return rearrange(effects, "batch (query key) -> batch query key", key=64, query=64)


def third_to_first_vs_other_ablations(model, boards, puzzles, args):
    first_target_squares = puzzles.principal_variation.apply(lambda x: x[0][2:4])
    third_target_squares = puzzles.principal_variation.apply(lambda x: x[2][2:4])
    first_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, first_target_squares)]
    )
    third_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, third_target_squares)]
    )

    def _third_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            third_target_indices[batch_indices],
        ] = 0

    def _other_ablate(location, model, batch_indices):
        layer, head = location
        other_mask = torch.ones(
            len(batch_indices), 64, 64, dtype=torch.bool, device=args.device
        )
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            third_target_indices[batch_indices],
        ] = False
        model.attention_scores(layer).output[:, head][other_mask] = 0

    third_to_first_effects = -patching.patch(
        patching_func=_third_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    other_effects = -patching.patch(
        patching_func=_other_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )

    return third_to_first_effects.squeeze(-1), other_effects.squeeze(-1)

def third_to_first_vs_other_double_branch_ablations(model, boards, puzzles, args):
    first_target_squares_1 = puzzles.branch_1.apply(lambda x: x[0][2:4])
    third_target_squares_1 = puzzles.branch_1.apply(lambda x: x[2][2:4])
    first_target_squares_2 = puzzles.branch_2.apply(lambda x: x[0][2:4])
    third_target_squares_2 = puzzles.branch_2.apply(lambda x: x[2][2:4])
    first_target_indices_1 = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, first_target_squares_1)]
    )
    third_target_indices_1 = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, third_target_squares_1)]
    )
    first_target_indices_2 = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, first_target_squares_2)]
    )
    third_target_indices_2 = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, third_target_squares_2)]
    )

    def _third_1_to_first_1_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices_1[batch_indices],
            third_target_indices_1[batch_indices],
        ] = 0

    def _third_2_to_first_2_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices_2[batch_indices],
            third_target_indices_2[batch_indices],
        ] = 0

    def _third_1_to_first_2_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices_1[batch_indices],
            third_target_indices_2[batch_indices],
        ] = 0

    def _third_2_to_first_1_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices_2[batch_indices],
            third_target_indices_1[batch_indices],
        ] = 0

    def _other_ablate(location, model, batch_indices):
        layer, head = location
        other_mask = torch.ones(
            len(batch_indices), 64, 64, dtype=torch.bool, device=args.device
        )
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices_1[batch_indices],
            third_target_indices_1[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices_1[batch_indices],
            third_target_indices_2[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices_2[batch_indices],
            third_target_indices_1[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices_2[batch_indices],
            third_target_indices_2[batch_indices],
        ] = False
        model.attention_scores(layer).output[:, head][other_mask] = 0

    third_1_to_first_1_effects = -patching.patch(
        patching_func=_third_1_to_first_1_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    third_2_to_first_2_effects = -patching.patch(
        patching_func=_third_2_to_first_2_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    third_1_to_first_2_effects = -patching.patch(
        patching_func=_third_1_to_first_2_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    third_2_to_first_1_effects = -patching.patch(
        patching_func=_third_2_to_first_1_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    other_effects = -patching.patch(
        patching_func=_other_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )

    return third_1_to_first_1_effects.squeeze(-1), third_2_to_first_2_effects.squeeze(-1), third_1_to_first_2_effects.squeeze(-1), third_2_to_first_1_effects.squeeze(-1), other_effects.squeeze(-1)

def case_ABC_vs_other_ablations(model, boards, puzzles, args):
    first_target_squares = puzzles.principal_variation.apply(lambda x: x[0][2:4])
    third_target_squares = puzzles.principal_variation.apply(lambda x: x[2][2:4])
    fifth_target_squares = puzzles.principal_variation.apply(lambda x: x[4][2:4])
    first_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, first_target_squares)]
    )
    third_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, third_target_squares)]
    )
    fifth_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, fifth_target_squares)]
    )

    def _third_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            third_target_indices[batch_indices],
        ] = 0

    def _fifth_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = 0

    def _fifth_to_third_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            third_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = 0

    def _other_ablate(location, model, batch_indices):
        layer, head = location
        other_mask = torch.ones(
            len(batch_indices), 64, 64, dtype=torch.bool, device=args.device
        )
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            third_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            third_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = False
        model.attention_scores(layer).output[:, head][other_mask] = 0

    third_to_first_effects = -patching.patch(
        patching_func=_third_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    fifth_to_first_effects = -patching.patch(
        patching_func=_fifth_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    fifth_to_third_effects = -patching.patch(
        patching_func=_fifth_to_third_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    other_effects = -patching.patch(
        patching_func=_other_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )

    return third_to_first_effects.squeeze(-1), fifth_to_first_effects.squeeze(-1), fifth_to_third_effects.squeeze(-1), other_effects.squeeze(-1)


def case_AAC_vs_other_ablations(model, boards, puzzles, args):
    first_target_squares = puzzles.principal_variation.apply(lambda x: x[0][2:4])
    fifth_target_squares = puzzles.principal_variation.apply(lambda x: x[4][2:4])
    first_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, first_target_squares)]
    )
    fifth_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, fifth_target_squares)]
    )

    def _fifth_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = 0

    def _other_ablate(location, model, batch_indices):
        layer, head = location
        other_mask = torch.ones(
            len(batch_indices), 64, 64, dtype=torch.bool, device=args.device
        )
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = False
        model.attention_scores(layer).output[:, head][other_mask] = 0

    fifth_to_first_effects = -patching.patch(
        patching_func=_fifth_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    other_effects = -patching.patch(
        patching_func=_other_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )

    return fifth_to_first_effects.squeeze(-1), other_effects.squeeze(-1)

def case_ABCD_vs_other_ablations(model, boards, puzzles, args):
    first_target_squares = puzzles.principal_variation.apply(lambda x: x[0][2:4])
    third_target_squares = puzzles.principal_variation.apply(lambda x: x[2][2:4])
    fifth_target_squares = puzzles.principal_variation.apply(lambda x: x[4][2:4])
    seventh_target_squares = puzzles.principal_variation.apply(lambda x: x[6][2:4])
    first_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, first_target_squares)]
    )
    third_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, third_target_squares)]
    )
    fifth_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, fifth_target_squares)]
    )
    seventh_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, seventh_target_squares)]
    )

    def _third_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            third_target_indices[batch_indices],
        ] = 0

    def _fifth_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = 0

    def _seventh_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = 0

    def _fifth_to_third_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            third_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = 0

    def _seventh_to_third_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            third_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = 0

    def _seventh_to_fifth_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            fifth_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = 0

    def _other_ablate(location, model, batch_indices):
        layer, head = location
        other_mask = torch.ones(
            len(batch_indices), 64, 64, dtype=torch.bool, device=args.device
        )
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            third_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            third_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            third_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            fifth_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = False
        model.attention_scores(layer).output[:, head][other_mask] = 0

    third_to_first_effects = -patching.patch(
        patching_func=_third_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    fifth_to_first_effects = -patching.patch(
        patching_func=_fifth_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    fifth_to_third_effects = -patching.patch(
        patching_func=_fifth_to_third_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    seventh_to_first_effects = -patching.patch(
        patching_func=_seventh_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    seventh_to_third_effects = -patching.patch(
        patching_func=_seventh_to_third_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    seventh_to_fifth_effects = -patching.patch(
        patching_func=_seventh_to_fifth_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    other_effects = -patching.patch(
        patching_func=_other_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )

    return third_to_first_effects.squeeze(-1), fifth_to_first_effects.squeeze(-1), fifth_to_third_effects.squeeze(-1), seventh_to_first_effects.squeeze(-1), seventh_to_third_effects.squeeze(-1), seventh_to_fifth_effects.squeeze(-1), other_effects.squeeze(-1)

def case_AACD_vs_other_ablations(model, boards, puzzles, args):
    first_target_squares = puzzles.principal_variation.apply(lambda x: x[0][2:4])
    fifth_target_squares = puzzles.principal_variation.apply(lambda x: x[4][2:4])
    seventh_target_squares = puzzles.principal_variation.apply(lambda x: x[6][2:4])
    first_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, first_target_squares)]
    )
    fifth_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, fifth_target_squares)]
    )
    seventh_target_indices = torch.tensor(
        [board.sq2idx(sq) for board, sq in zip(boards, seventh_target_squares)]
    )

    def _fifth_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = 0

    def _seventh_to_first_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            first_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = 0

    def _seventh_to_fifth_ablate(location, model, batch_indices):
        layer, head = location
        model.attention_scores(layer).output[
            torch.arange(len(batch_indices)),
            head,
            fifth_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = 0

    def _other_ablate(location, model, batch_indices):
        layer, head = location
        other_mask = torch.ones(
            len(batch_indices), 64, 64, dtype=torch.bool, device=args.device
        )
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            fifth_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            first_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = False
        other_mask[
            torch.arange(len(batch_indices)),
            fifth_target_indices[batch_indices],
            seventh_target_indices[batch_indices],
        ] = False
        model.attention_scores(layer).output[:, head][other_mask] = 0

    fifth_to_first_effects = -patching.patch(
        patching_func=_fifth_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    seventh_to_first_effects = -patching.patch(
        patching_func=_seventh_to_first_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    seventh_to_fifth_effects = -patching.patch(
        patching_func=_seventh_to_fifth_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )
    other_effects = -patching.patch(
        patching_func=_other_ablate,
        locations=[(args.layer, args.head)],
        model=model,
        boards=boards,
        batch_size=args.batch_size,
        pbar="batch",
    )

    return fifth_to_first_effects.squeeze(-1), seventh_to_first_effects.squeeze(-1), seventh_to_fifth_effects.squeeze(-1), other_effects.squeeze(-1)

def attention_pattern(model, boards, args):
    pattern = torch.zeros(len(boards), 64, 64, device=args.device)
    scores = torch.zeros(len(boards), 64, 64, device=args.device)
    qk_only = torch.zeros(len(boards), 64, 64, device=args.device)
    for i in tqdm.trange(0, len(boards), args.batch_size):
        with model.trace(boards[i : i + args.batch_size]):
            new_pattern = model.attention_scores(args.layer).output[:, args.head].save()
            new_scores = (
                model.attention_scores(args.layer, pre_softmax=True)
                .output[:, args.head]
                .save()
            )
            new_qk_only = (
                model.attention_scores(args.layer, QK_only=True)
                .output[:, args.head]
                .save()
            )

        new_pattern = new_pattern.value
        new_scores = new_scores.value
        new_qk_only = new_qk_only.value
        pattern[i : i + args.batch_size] = new_pattern
        scores[i : i + args.batch_size] = new_scores
        qk_only[i : i + args.batch_size] = new_qk_only

    return pattern, scores, qk_only


def main(args):
    torch.set_num_threads(args.num_threads)

    base_dir = Path(args.base_dir)
    model = Lc0sight(base_dir / "lc0.onnx", device=args.device)

    match = re.search(r'\d+$', args.filename)
    case_number = args.filename.split("_")[-1]
    case_size = len(case_number)
    common_str = args.filename[20:-len(case_number)-1]

    if match and (case_size in [5, 7, 9]):
        if case_size == 5:
            # Check if the first and third digits of case_number are the same
            if case_number[0] == case_number[2]:
                case_type = "AAC"
            else:
                if case_number[0] == case_number[4]:
                    case_type = "ABA"
                elif case_number[2] == case_number[4]:
                    case_type = "ABB"
                else:
                    case_type = "ABC"
        elif case_size == 7:
            if case_number[0] == case_number[2]:
                if len(set(list(case_number[::2]))) == 3:
                    case_type = "AACD"
                else:
                    raise NotImplementedError
            else:
                if len(set(list(case_number[::2]))) == 4:
                    case_type = "ABCD"
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        if case_number[0] == case_number[2]:
            case_type = "AB"
        else:
            raise NotImplementedError

    # if case_type == "AB":
    #     save_dir = base_dir / f"results/L{args.layer}H{args.head}"
    # else:
    save_dir = base_dir / f"results/L{args.layer}H{args.head}_{common_str}_{case_number}"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(base_dir / ("puzzles/" + args.filename + ".pkl"), "rb") as f:
            puzzles = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Corrupted puzzles not found, run make_corruptions.py first")

    if args.n_puzzles:
        puzzles = puzzles.iloc[: args.n_puzzles]

    boards = [LeelaBoard.from_puzzle(puzzle) for _, puzzle in puzzles.iterrows()]

    if args.main:
        if case_type == "AB" or case_type == "ABA" or case_type == "ABB":
            if args.double_game:
                third_1_to_first_1_effects, third_2_to_first_2_effects, third_1_to_first_2_effects, third_2_to_first_1_effects, other_effects = third_to_first_vs_other_double_branch_ablations(
                    model, boards, puzzles, args
                )
                torch.save(third_1_to_first_1_effects, save_dir / "third_1_to_first_1_ablation.pt")
                torch.save(third_2_to_first_2_effects, save_dir / "third_2_to_first_2_ablation.pt")
                torch.save(third_1_to_first_2_effects, save_dir / "third_1_to_first_2_ablation.pt")
                torch.save(third_2_to_first_1_effects, save_dir / "third_2_to_first_1_ablation.pt")
                torch.save(other_effects, save_dir / "other_ablation.pt")
            else:
                third_to_first_effects, other_effects = third_to_first_vs_other_ablations(
                    model, boards, puzzles, args
                )
                torch.save(third_to_first_effects, save_dir / "third_to_first_ablation.pt")
                torch.save(other_effects, save_dir / "other_ablation.pt")
        elif case_type == "AAC":
            fifth_to_first_effects, other_effects = case_AAC_vs_other_ablations(
                model, boards, puzzles, args
            )
            torch.save(fifth_to_first_effects, save_dir / "fifth_to_first_ablation.pt")
            torch.save(other_effects, save_dir / "other_ablation.pt")
        elif case_type == "ABC":
            third_to_first_effects, fifth_to_first_effects, fifth_to_third_effects, other_effects = case_ABC_vs_other_ablations(
                model, boards, puzzles, args
            )
            torch.save(third_to_first_effects, save_dir / "third_to_first_ablation.pt")
            torch.save(fifth_to_first_effects, save_dir / "fifth_to_first_ablation.pt")
            torch.save(fifth_to_third_effects, save_dir / "fifth_to_third_ablation.pt")
            torch.save(other_effects, save_dir / "other_ablation.pt")
        elif case_type == "ABCD":
            third_to_first_effects, fifth_to_first_effects, fifth_to_third_effects, seventh_to_first_effects, seventh_to_third_effects, seventh_to_fifth_effects, other_effects = case_ABCD_vs_other_ablations(
                model, boards, puzzles, args
            )
            torch.save(third_to_first_effects, save_dir / "third_to_first_ablation.pt")
            torch.save(fifth_to_first_effects, save_dir / "fifth_to_first_ablation.pt")
            torch.save(fifth_to_third_effects, save_dir / "fifth_to_third_ablation.pt")
            torch.save(seventh_to_first_effects, save_dir / "seventh_to_first_ablation.pt")
            torch.save(seventh_to_third_effects, save_dir / "seventh_to_third_ablation.pt")
            torch.save(seventh_to_fifth_effects, save_dir / "seventh_to_fifth_ablation.pt")
            torch.save(other_effects, save_dir / "other_ablation.pt")
        elif case_type == "AACD":
            fifth_to_first_effects, seventh_to_first_effects, seventh_to_fifth_effects, other_effects = case_AACD_vs_other_ablations(
                model, boards, puzzles, args
            )
            torch.save(fifth_to_first_effects, save_dir / "fifth_to_first_ablation.pt")
            torch.save(seventh_to_first_effects, save_dir / "seventh_to_first_ablation.pt")
            torch.save(seventh_to_fifth_effects, save_dir / "seventh_to_fifth_ablation.pt")
            torch.save(other_effects, save_dir / "other_ablation.pt")

    if args.squarewise:
        effects = squarewise_patching(model, puzzles, args)
        torch.save(effects, save_dir / "squarewise_patching.pt")

    if args.single_weight:
        effects = single_weight_ablation(model, boards, args)
        torch.save(effects, save_dir / "single_weight_ablation.pt")

    if args.attention_pattern:
        pattern, scores, qk_only = attention_pattern(model, boards, args)
        torch.save(pattern, save_dir / "attention_pattern_post_softmax.pt")
        torch.save(scores, save_dir / "attention_scores_pre_softmax.pt")
        torch.save(qk_only, save_dir / "attention_scores_qk_only.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--filename", default="interesting_puzzles", type=str)
    parser.add_argument("--layer", default=12, type=int)
    parser.add_argument("--head", default=12, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--double_game", action="store_true")
    parser.add_argument("--squarewise", action="store_true")
    parser.add_argument("--single_weight", action="store_true")
    parser.add_argument("--attention_pattern", action="store_true")
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--n_puzzles", default=0, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    args = parser.parse_args()
    main(args)
