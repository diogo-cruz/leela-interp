import argparse
import itertools
import pickle
from pathlib import Path

import torch
from einops import rearrange
from leela_interp import Lc0sight
from leela_interp.tools import patching


def main(args):
    if not (args.residual_stream or args.attention or args.double_attention[0] != -1):
        raise ValueError(
            "At least one of --residual_stream or --attention must be specified"
        )
    torch.set_num_threads(args.num_threads)

    base_dir = Path(args.base_dir)
    model = Lc0sight(base_dir / "lc0.onnx", device=args.device)

    save_dir = base_dir / "results/global_patching"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(base_dir / ("puzzles/" + args.filename + ".pkl"), "rb") as f:
            puzzles = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Corrupted puzzles not found, run make_corruptions.py first")

    if args.n_puzzles:
        puzzles = puzzles.iloc[: args.n_puzzles]

    if args.residual_stream:
        # Ablate one square in the residual stream at a time
        effects = patching.residual_stream_activation_patch(
            model=model,
            # The puzzles we loaded already specify corrupted board positions
            puzzles=puzzles,
            batch_size=args.batch_size,
        )
        torch.save(effects, save_dir / (args.filename + "_residual_stream_results.pt"))

    if args.attention:
        # Ablate one attention head at a time
        locations = list(itertools.product(range(15), range(24)))
        effects = patching.activation_patch(
            model=model,
            module_func=model.headwise_attention_output,
            locations=locations,
            puzzles=puzzles,
            batch_size=args.batch_size,
        )
        effects = rearrange(
            effects, "batch (layer head) -> batch layer head", layer=15, head=24
        )
        torch.save(effects, save_dir / (args.filename + "_attention_head_results.pt"))

    if args.double_attention[0] != -1:
        # Create n-tuples of attention heads
        n_layers = 15
        n_heads = 24
        margin = args.double_attention[0]
        n = args.double_attention[1]
        
        # Create a tensor to store effects
        # The shape is adjusted to accommodate n heads
        effects_shape = [len(puzzles)] + [n_layers, n_heads] + [margin + 1, n_heads] * (n-1)
        effects_array = torch.zeros(effects_shape)
        
        def generate_combinations(n_layers, n_heads, n, margin):
            # Generate all possible (layer, head) tuples
            all_tuples = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
            
            # Filter combinations to ensure all layers are within the margin
            valid_combinations = []
            for combo in itertools.combinations(all_tuples, n):
                layers = [layer for layer, head in combo]
                if max(layers) - min(layers) <= margin:
                    valid_combinations.append(tuple(sorted(combo, key=lambda x: x[0])))
            
            return valid_combinations
        
        locations = generate_combinations(n_layers, n_heads, n, margin)
        
        effects = patching.activation_patch(
            model=model,
            module_func=model.headwise_attention_output,
            locations=locations,
            puzzles=puzzles,
            batch_size=args.batch_size,
        )
        
        # Fill the effects_array with the computed effects
        for i, location in enumerate(locations):
            index = [slice(None), location[0][0], location[0][1]]  # This is for the batch dimension and first layer, head
            for j in range(1, n):
                layer_offset = location[j][0] - location[0][0]
                index.extend([layer_offset, location[j][1]])
            effects_array[tuple(index)] = effects[:, i]
        
        # Save results
        torch.save(effects_array, save_dir / (args.filename + f"_{n}_attention_head_results.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--filename", default="interesting_puzzles", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--n_puzzles", default=0, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    parser.add_argument("--residual_stream", action="store_true")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--double_attention", default=(-1, 2), nargs=2, type=int)
    args = parser.parse_args()
    main(args)
