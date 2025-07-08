# !/usr/bin/env ppg_project
"""
Plot ensemble PPG waves stratified by age groups
================================================

This script loads a preprocessed `data_dict_*.pt` (PyTorch) file containing
ensemble PPG waveforms and associated subject metadata, bins subjects into
defined age groups, and plots all ensemble waves in each bin for visual
comparison.

Usage:
    python plot_ensemble_by_age.py \
        --dict_path path/to/data_dict.pt \
        --output_dir out/figures_age_bins

Dependencies:
    torch, numpy, matplotlib
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from initialize import PREPROCESSED_AURORA_DATA_PATH

DEFAULT_DICT_PATH = PREPROCESSED_AURORA_DATA_PATH + "/data_dict_osc_auc_with_derivatives_with_classes.pt"
OUT_DIR   = Path(__file__).resolve().with_name("out")
FIG_DIR   = OUT_DIR / "figures_age_bins"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Define age bins and labels
def get_age_group(age: float) -> str:
    if age < 30:
        return '<30 years'
    elif age < 40:
        return '30 to 39 years'
    elif age < 50:
        return '40 to 49 years'
    else:
        return '>=50 years'


def main():
    parser = argparse.ArgumentParser("Plot ensemble waves by age group")
    parser.add_argument(
        '--dict_path', '-d', type=Path,
        default=Path(DEFAULT_DICT_PATH),
        help='Path to the preprocessed data_dict .pt file'
    )
    parser.add_argument(
        '--output_dir', '-o', type=Path,
        default=Path(FIG_DIR),
        help='Directory where figures will be saved'
    )
    parser.add_argument(
        '--wave_key', type=str, default='ensemble_wave',
        help='Key in each subject dict for the ensemble waveform'
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data_dict
    print(f"Loading data_dict from {args.dict_path}")
    data_dict = torch.load(args.dict_path, weights_only=False)

    # Bin ensemble waves by age group
    bins = {
        '<30 years': [],
        '30 to 39 years': [],
        '40 to 49 years': [],
        '>=50 years': []
    }
    for pid, entry in data_dict.items():
        age = entry.get('age', None)
        wave = entry.get(args.wave_key, None)
        if age is None or wave is None:
            continue
        # convert to NumPy array if tensor
        wave = wave.cpu().numpy() if hasattr(wave, 'cpu') else np.array(wave)
        group = get_age_group(age)
        bins[group].append(wave)

    # Plotting: one panel per age bin
    n_bins = len(bins)
    fig, axes = plt.subplots(1, n_bins, figsize=(4*n_bins, 4), sharey=True)
    if n_bins == 1:
        axes = [axes]

    # x-axis: percent of waveform (0-100)
    sample_length = next(iter(bins.values()))[0].shape[0] if bins and any(bins.values()) else 0
    x = np.linspace(0, 100, sample_length)

    for ax, (label, waves) in zip(axes, bins.items()):
        ax.set_title(label)
        for w in waves:
            ax.plot(x, w, lw=0.8, alpha=0.6)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Normalized time [%]')
    axes[0].set_ylabel('PPG Amplitude')
    fig.tight_layout()

    # Save figure
    out_png = args.output_dir / 'ensemble_waves_by_age.png'
    out_pdf = args.output_dir / 'ensemble_waves_by_age.pdf'
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"Saved plots to {out_png} and {out_pdf}")


if __name__ == '__main__':
    main()
