#!/usr/bin/env python
"""
Detect onsets with NeuroKit2 and save a NumPy file with all PW durations
for a MAUS device (finger or wrist).

Example:
    python preprocessing/detect_neurokit.py --device finger \
        --raw_path  "/…/MAUS/raw/Raw_data" \
        --out_dir   "/…/MAUS/preprocessed"
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import neurokit2 as nk

from initialize import (  
    RAW_MAUS_DATA_PATH,
    PREPROCESSED_MAUS_DATA_PATH,
)


def durations_one_file(fpath: Path, fs: int, recording_site: str) -> list[float]:
    """Return beat-to-beat durations (ms) for a single CSV."""
    if recording_site == "finger": 
        df = pd.read_csv(fpath, index_col=0)
        sig = df["Resting_PPG"]
    elif recording_site == "wrist":
        df = pd.read_csv(fpath, index_col=None)
        print("Got this file: \n", df.head())
        sig = df["Resting"]
    else:
        raise ValueError(f"Unknown recording site: {recording_site}")

    cleaned = nk.ppg_clean(sig, sampling_rate=fs)
    ppg_peaks = nk.ppg_findpeaks(cleaned, sampling_rate=fs, show=True)
    peaks = ppg_peaks["PPG_Peaks"]
    # difference → samples → milliseconds
    return np.diff(peaks) / fs * 1_000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["finger", "wrist"], required=True)
    p.add_argument("--raw_path", type=Path, default=Path(RAW_MAUS_DATA_PATH))
    p.add_argument("--out_dir",  type=Path, default=Path(PREPROCESSED_MAUS_DATA_PATH))
    args = p.parse_args()

    glob_pat  = "fPPG_*back.csv" if args.device == "finger" else "wPPG_*back.csv"
    fs        = 256 if args.device == "finger" else 100 # Hz 
    durations = []

    print("[INFO] Processing files for device:", args.device)
    print("[INFO] Raw data path:", args.raw_path)
    for subj in sorted(args.raw_path.glob("*")):

        csv_file = subj / ("pixart_resting.csv" if args.device == "wrist" else "inf_resting.csv") 
        print(f"[INFO] Processing subject: {subj.name}  →  {csv_file}")
        durations.extend(durations_one_file(csv_file, fs, recording_site=args.device))

    # save durations as NumPy array
    out_folder = args.out_dir / "comparison_algos"
    out_folder.mkdir(parents=True, exist_ok=True)
    out_file = out_folder / f"durations_neurokit_{args.device}.npy"
    np.save(out_file, np.asarray(durations, dtype=float))
    print(f"[INFO] {args.device}: {len(durations)} beats  →  {out_file}")


if __name__ == "__main__":
    main()
