#!/usr/bin/env python
"""
Detect onsets with *vanilla* pyPPG and save a NumPy file with all
pulse-width (beat-to-beat) durations for a MAUS device (finger or wrist).

Example
-------
python detect_pyppg_vanilla.py --device finger \
    --raw_path  "/…/MAUS/raw/Raw_data" \
    --out_dir   "/…/MAUS/preprocessed"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotmap import DotMap
from pyPPG import PPG
import pyPPG.preproc as PP
import pyPPG.fiducials as FP

from initialize import RAW_MAUS_DATA_PATH, PREPROCESSED_MAUS_DATA_PATH


# … (imports & CLI left unchanged) … -------------------------------------

def durations_one_file(fpath: Path, fs: int, recording_site: str) -> list[float]:
    """Return beat-to-beat durations (ms) for one MAUS CSV."""
    # ── 1.  read correct PPG column ----------------------------------------
    if recording_site == "finger":          # inf_resting.csv
        sig = pd.read_csv(fpath, index_col=0)["Resting_PPG"].values
    else:                                   # wrist → pixart_resting.csv
        sig = pd.read_csv(fpath)["Resting"].values

    # ── 2.  run pyPPG’s filter & derivatives -------------------------------
    from dotmap import DotMap
    s = DotMap(v=sig, fs=fs, start=0, end=len(sig),
               name=fpath.stem, filtering=True)

    ppg, vpg, apg, jpg = PP.Preprocess().get_signals(s)

    # ── 3.  build a *complete* PPG object with all derivatives -------------
    s.filt_sig, s.filt_d1, s.filt_d2, s.filt_d3 = ppg, vpg, apg, jpg
    ppg_obj = PPG(s)
    ppg_obj.ppg, ppg_obj.vpg, ppg_obj.apg, ppg_obj.jpg = ppg, vpg, apg, jpg

    # pyPPG’s fiducial extractor insists on a “correction” dataframe
    s.correction = pd.DataFrame([[True]*6],
                                columns=['on', 'dn', 'dp', 'v', 'w', 'f'])

    # ── 4.  detect onsets and convert to ms --------------------------------
    on = FP.FpCollection(ppg_obj).get_fiducials(ppg_obj)["on"]
    return np.diff(on) / fs * 1_000


# -----------------------------------------------------------------------------#
#                                   main CLI                                   #
# -----------------------------------------------------------------------------#
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["finger", "wrist"], required=True)
    p.add_argument("--raw_path", type=Path, default=Path(RAW_MAUS_DATA_PATH))
    p.add_argument("--out_dir",  type=Path, default=Path(PREPROCESSED_MAUS_DATA_PATH))
    args = p.parse_args()

    # Sampling rate & filename pattern ----------------------------------------
    fs = 256 if args.device == "finger" else 100  # Hz
    csv_name = "inf_resting.csv" if args.device == "finger" else "pixart_resting.csv"

    durations: list[float] = []

    print(f"[INFO] Processing device = {args.device}")
    print(f"[INFO] Raw path         = {args.raw_path}")

    # Each subject sits directly in RAW_MAUS_DATA_PATH/Raw_data/00X/ ----------
    for subj in sorted(args.raw_path.glob("*")):           # 001/, 002/, …
        csv_file = subj / csv_name
        if not csv_file.exists():
            print(f"[WARN] Missing file for {subj.name}: {csv_file.name}")
            continue

        print(f"[INFO] {subj.name} → {csv_file.name}")
        try:
            durations.extend(
                durations_one_file(csv_file, fs, recording_site=args.device)
            )
        except Exception as exc:
            print(f"[WARN] {csv_file.name}: {exc}")

    # --------------------------------------------------------------------- #
    # Save all durations as a NumPy array                                   #
    # --------------------------------------------------------------------- #
    out_folder = args.out_dir / "comparison_algos"
    out_folder.mkdir(parents=True, exist_ok=True)

    out_file = out_folder / f"durations_pyppg_{args.device}.npy"
    np.save(out_file, np.asarray(durations, dtype=float))

    print(
        f"[INFO] {args.device}: {len(durations)} beats saved → {out_file}"
    )


if __name__ == "__main__":
    main()
