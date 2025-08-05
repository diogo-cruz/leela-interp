# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch-based research codebase for analyzing neural network interpretability in chess-playing models, specifically the Leela Chess Zero (LC0) engine. The project investigates evidence of learned look-ahead behavior in chess neural networks, with extensions to analyze longer-term planning capabilities and alternative move considerations.

## Setup and Installation

Install the package in development mode:
```bash
pip install -e .
```

Required external files (download from https://figshare.com/s/adc80845c00b67c8fce5):
- `lc0.onnx` - Main Leela Chess Zero model (required for most experiments)
- `interesting_puzzles.pkl` - Preprocessed puzzle dataset (required for most experiments)
- `lc0-random.onnx` - Random initialization baseline (optional, for probing baseline only)
- `LD2.onnx` - Alternative model (optional, for custom puzzle filtering)
- `unfiltered_puzzles.pkl` - Raw puzzle data (optional, for custom puzzle filtering)

## Key Commands

### Data Generation
```bash
# Generate puzzle dataset from scratch (requires lichess_db_puzzle.csv.zst)
python scripts/make_puzzles.py --generate

# Add corruptions to puzzles
python scripts/make_corruptions.py

# Generate puzzles for specific analyses
python scripts/make_puzzles_pv7.py  # 7-move look-ahead
python scripts/make_puzzles_alternate.py  # Alternative move analysis
python scripts/make_puzzles_double.py  # Double-branch analysis
```

### Analysis Scripts
```bash
# Probing analysis
python scripts/probing.py --main --random_model --n_seeds 5

# Global patching experiments
python scripts/run_global_patching.py --residual_stream --attention

# Single head analysis (L12H12 results)
python scripts/run_single_head.py --main

# Piece movement head analysis
python scripts/piece_movement_heads.py
```

### Common Script Options
- `--n_puzzles N`: Limit number of puzzles to process
- `--batch_size N`: Set batch size for processing
- `--device DEVICE`: Specify device (cuda/cpu/mps, defaults to cuda)
- `--layer L --head H`: Analyze specific attention head
- `--squarewise`: Enable square-wise ablations
- `--single_weight`: Enable single weight ablations
- `--attention_pattern`: Cache attention patterns

## Code Architecture

### Core Package (`src/leela_interp/core/`)
- `lc0.py`: Main Lc0Model class wrapping the ONNX model with PyTorch
- `leela_board.py`: LeelaBoard wrapper around python-chess for Leela-formatted inputs
- `nnsight.py`: NNsight integration for activation analysis
- `forward_pass_implementation.py`: Custom forward pass implementation
- `ablation_study.py`: Framework for ablation experiments
- `probing_study.py`: Framework for probing experiments
- Study modules for specific analyses: `alternative_moves.py`, `checkmate_study.py`, `double_*.py`, `effect_study.py`, `fifth_move_study.py`

### Tools Package (`src/leela_interp/tools/`)
- `patching.py`: Activation patching utilities and effect functions
- `probing.py`: Probing utilities and probe training
- `activations.py`: Activation caching and management
- `attention.py`: Attention analysis utilities
- `figure_helpers.py`: Visualization helpers
- `piece_movement_heads.py`: Piece movement analysis
- `play.py`: Game playing utilities

### Data Organization
- `puzzles/`: Generated puzzle datasets with various configurations
- `results/`: Cached results from analysis scripts organized by experiment
- `figures/`: Generated figures and visualizations
- `notebooks/`: Jupyter notebooks for analysis and figure generation

## Environment Configuration

The package supports these environment variables:
- `LC0_MODEL_PATH`: Path to the main LC0 ONNX model
- `DEVICE`: Default device for computations (cuda/cpu/mps)

## Development Notes

### Jupyter Notebooks
Set working directory in VS Code:
```json
{
    "jupyter.notebookFileRoot": "/path/to/leela-interp"
}
```

Key notebooks:
- `demo.ipynb`: Introduction to codebase features
- `other_figures.ipynb`: Main figure generation
- `paper_figures.ipynb`: Paper-specific visualizations
- `notebooks/act_patching.ipynb`, `notebooks/puzzle_example.ipynb`, `notebooks/figure_1.ipynb`: Specific figure creation

### Known Issues
- NaN outputs observed on MPS device (use CPU or CUDA instead)
- Memory requirements: ~70GB RAM for full probing analysis (can be reduced with `--n_puzzles`)

### Code Quality
- Uses ruff for linting with rules: "E", "W", "F", "I"
- Python 3.10+ required
- Results are cached in `results/` directory to avoid recomputation