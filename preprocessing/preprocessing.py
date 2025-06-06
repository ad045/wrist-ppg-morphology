#!/usr/bin/env python
"""
Combined preprocessing script for both oscillometric and auscultatory datasets.
It loads metadata, filters participants, reads raw optical waveforms,
computes sampling frequency, processes the raw signal with pyPPG to compute derivatives,
splits the signal into individual waves, computes an ensemble wave, average HR,
and finally computes derivatives for both the ensemble and individual waves.
"""



# Creates this data: 
# "...\\aurora_data\\preprocessed\\data_dict_aurora_final_{dataset_type}_2.pt"

# run in "02_clean_ppg" folder
# python -m preprocessing.preprocessing # m ist wichtig!! 

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import warnings
from scipy import signal
from scipy.signal import find_peaks

from initialize import DATA_PATH, AURORA_DATA_PATH, RAW_AURORA_DATA_PATH, PREPROCESSED_AURORA_DATA_PATH


# Suppress FutureWarnings from pandas (e.g. chained assignment warnings)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- pyPPG imports ---
from pyPPG import PPG, Fiducials
import pyPPG.preproc as PP
import pyPPG.fiducials as FP

# --- Utility Functions ---

def process_ppg_wave(wave):
    """
    Remove linear trend (so that first and last samples become 0)
    and normalize the detrended wave to [0,1].
    """
    n = len(wave)
    trend = np.linspace(wave[0], wave[-1], n)
    detrended = wave - trend
    mn = detrended.min()
    mx = detrended.max()
    if abs(mx - mn) < 1e-9:
        return np.zeros_like(detrended)
    normalized = (detrended - mn) / (mx - mn)
    return normalized

def compute_derivatives_pyppg(ensemble_wave, fs=1000):
    """
    Repeat the (processed) ensemble wave 20 times,
    run pyPPG.Preprocess to compute derivatives,
    and then extract the 11th wave (index 10).
    """
    repeated = np.tile(ensemble_wave, 20)
    from dotmap import DotMap
    s = DotMap()
    s.start = 0
    s.end = len(repeated)
    s.v = repeated
    s.fs = fs
    s.name = "temp_signal"
    try:
        ppg, vpg, apg, jpg = PP.Preprocess().get_signals(s) # , filtering=True)
    except Exception as e:
        print("Error in PP.Preprocess:", e)
        # Return zeros if processing fails
        zeros = np.zeros(len(ensemble_wave))
        return zeros, zeros, zeros, zeros
    wave_len = len(ensemble_wave)
    start_idx = 10 * wave_len
    end_idx = 11 * wave_len
    ppg_11 = ppg[start_idx:end_idx]
    vpg_11 = vpg[start_idx:end_idx]
    apg_11 = apg[start_idx:end_idx]
    jpg_11 = jpg[start_idx:end_idx]
    return ppg_11, vpg_11, apg_11, jpg_11


def process_with_pyPPG_for_subject(subj_data, pid):
    """
    Process a single subject's raw optical signal with pyPPG.
    Returns a dictionary with ppg, vpg, apg, jpg derivatives, plus fiducials.
    """
    from dotmap import DotMap
    s = DotMap()
    s.start = 0
    s.end = len(subj_data["raw_optical"])
    s.v = subj_data["raw_optical"]
    s.fs = subj_data["sampling_rate"]
    s.name = str(pid)
    s.filtering = True                # required by the new Preprocess class

    # ── 1.  run the new pre-processing code you pasted ────────────────────────
    ppg, vpg, apg, jpg = PP.Preprocess().get_signals(s)

    # stash the derivatives on DotMap (rest of the pipeline uses these names)
    s.filt_sig = ppg
    s.filt_d1  = vpg
    s.filt_d2  = apg
    s.filt_d3  = jpg
    dt = 1.0 / s.fs
    # print("s.filt_sig", s.fs, s.filt_sig.shape)

    spg = np.gradient(s.filt_d3, dt)

    # ── 2.  make sure the PPG object exposes the attrs FpCollection expects ── 

    # Check if the signal is long enough for processing -> if not, continue with next subject and save information in csv file. Otherwise, pyPPG will throw an error.
    if len(s.filt_sig) <= s.fs*15: 
        print(f"Skipping subject {pid} due to insufficient signal length.")
        # log into a CSV file 
        with open(PREPROCESSED_AURORA_DATA_PATH + "/skipped_subjects.csv", "a") as f:
            f.write(f"{pid}, {len(s.filt_sig)}\n")
        return None

    ppg_class = PPG(s)
    ppg_class.ppg = s.filt_sig        # 0-th derivative
    ppg_class.vpg = s.filt_d1         # 1-st derivative
    ppg_class.apg = s.filt_d2         # 2-nd derivative
    ppg_class.jpg = s.filt_d3         # 3-rd derivative
            
    # pyPPG’s fiducial extractor needs a “correction” DataFrame on s
    corr_cols = ['on', 'dn', 'dp', 'v', 'w', 'f']
    s.correction = pd.DataFrame([[True]*len(corr_cols)], columns=corr_cols)

    # ── 3.  extract fiducials ────────────────────────────────────────────────
    fiducials = FP.FpCollection(ppg_class).get_fiducials(ppg_class)

    return {
        "ppg": s.filt_sig,
        "vpg": s.filt_d1,
        "apg": s.filt_d2,
        "jpg": s.filt_d3,
        "spg": spg,
        "fiducials": fiducials
    }


def assign_bin_edges(value, bin_edges):
    """
    Return the index of the bin for 'value' given bin_edges.
    """
    if value is None or np.isnan(value):
        return None
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= value < bin_edges[i + 1]:
            return i
    return None


def load_data(raw_path, dataset_type="oscillometric"):
    """
    Load participants, features, and measurements metadata.
    For 'oscillometric', use measurements_oscillometric.tsv and waveform folder "raw/initial_supine_data".
    For 'auscultatory', use measurements_auscultatory.tsv and folder "raw/measurements_auscultatory/measurements_auscultatory".
    Returns the filtered DataFrame and the appropriate waveform_base_path.
    """
    ppt_df = pd.read_csv(raw_path / "participants.tsv", delimiter='\t')
    feat_df = pd.read_csv(raw_path / "features.tsv", delimiter='\t')
    if dataset_type == "oscillometric":
        meas_file = raw_path / "measurements_oscillometric.tsv"
        waveform_base_path = raw_path / "measurements_oscillometric/measurements_oscillometric" # initial_supine_data"
        feat_df = feat_df[(feat_df["phase"] == "initial") & (feat_df["measurement"].str.startswith("Supine"))]
        meas_df = pd.read_csv(meas_file, delimiter='\t')
        meas_df = meas_df[(meas_df["phase"] == "initial") & (meas_df["measurement"].str.startswith("Supine"))]
    elif dataset_type == "auscultatory":
        meas_file = raw_path / "measurements_auscultatory.tsv"
        waveform_base_path = raw_path / "measurements_auscultatory/measurements_auscultatory"
        feat_df = feat_df[(feat_df["phase"] == "initial") & (feat_df["measurement"].str.startswith("Static challenge start 1"))]
        meas_df = pd.read_csv(meas_file, delimiter='\t')
        meas_df = meas_df[(meas_df["phase"] == "initial") & (meas_df["measurement"].str.startswith("Static challenge start 1"))]
    else:
        raise ValueError("Unknown dataset type.")
    
    comb_df = ppt_df.merge(feat_df, how='inner', on='pid')
    comb_df = comb_df.merge(meas_df, how='inner', on='pid', suffixes=('_feat', '_meas'))
    # Filter by signal quality:
    comb_df = comb_df[comb_df["quality_pressure"] > 0.65]
    comb_df = comb_df[comb_df["quality_optical"] > 0.8]
    # Remove participants with major comorbidities:
    no_comorb_df = comb_df[
        (comb_df["coronary_artery_disease"] == 0) &
        (comb_df["diabetes"] == 0) &
        (comb_df["arrythmia"] == 0) &
        (comb_df["previous_heart_attack"] == 0) &
        (comb_df["previous_stroke"] == 0) &
        (comb_df["heart_failure"] == 0) &
        (comb_df["aortic_stenosis"] == 0) &
        (comb_df["valvular_heart_disease"] == 0) &
        (comb_df["other_cv_diseases"] == 0)
    ]
    return no_comorb_df, waveform_base_path

def create_data_dict(no_comorb_df, waveform_base_path):
    """
    For each participant (row) in the filtered DataFrame, read the raw optical waveform
    from the appropriate folder and store relevant metadata.
    """
    data_dict = {}
    for idx, row in no_comorb_df.iterrows():
        pid = row['pid']
        data_dict[pid] = {}
        # Save participant-level info
        data_dict[pid]["age"] = row.get('age', None)
        data_dict[pid]["gender"] = row.get('gender', None)
        data_dict[pid]["high_bp"] = row.get('high_blood_pressure', None)
        data_dict[pid]["baseline_sbp"] = row.get('sbp_meas', None)
        data_dict[pid]["baseline_dbp"] = row.get('dbp_meas', None)
        data_dict[pid]["height"] = row.get('height', None)
        data_dict[pid]["weight"] = row.get('weight', None)
        data_dict[pid]["cvd_meds"] = row.get('cvd_meds', None)
        data_dict[pid]["fitzpatrick_scale"] = row.get('fitzpatrick_scale', None)
        data_dict[pid]["pressure_quality"] = row.get('pressure_quality', None)
        data_dict[pid]["optical_quality"] = row.get('optical_quality', None)
        
        # Construct full path to waveform file (assuming folder structure: waveform_base_path / pid / filename)
        raw_file_name = os.path.basename(row["waveform_file_path"])
        full_wave_path = os.path.join(waveform_base_path, str(pid), raw_file_name)
        try:
            wave_df = pd.read_csv(full_wave_path, sep='\t')
        except Exception as e:
            print(f"Error loading {full_wave_path}: {e}")
            continue
        optical = wave_df["optical"].to_numpy()
        time_array = wave_df["t"].to_numpy()
        if len(time_array) > 1:
            delta_t = time_array[1] - time_array[0]
        else:
            delta_t = 1/500.0
        sampling_rate = 1.0 / delta_t
        data_dict[pid]["raw_optical"] = optical
        data_dict[pid]["delta_t"] = delta_t
        data_dict[pid]["sampling_rate"] = sampling_rate
    return data_dict

# --- Processing Steps ---

def process_all_subjects(data_dict):
    """
    For each subject in the dictionary, process the raw optical signal
    with pyPPG to compute initial derivatives.
    """
    for pid, subj_data in data_dict.items():
        if "raw_optical" not in subj_data or "sampling_rate" not in subj_data:
            continue
        out = process_with_pyPPG_for_subject(subj_data, pid)
        # If processing fails (e.g. due to short signal), skip this subject
        if out is None:
            continue

        subj_data["ppg"] = out["ppg"]
        subj_data["vpg"] = out["vpg"]
        subj_data["apg"] = out["apg"]
        subj_data["jpg"] = out["jpg"]
        subj_data["fiducials_pyPPG"] = out["fiducials"]
    return data_dict

def detect_onsets_ppg_v2(ppg_signal, fs, expected_period_sec=1.0, min_prominence=0.02):
    """
    Detect local minima (onsets) in the PPG signal using find_peaks on the inverted signal.
    """
    inverted_ppg = -ppg_signal
    expected_period_samples = int(round(expected_period_sec * fs))
    min_distance = int(max(1, 0.5 * expected_period_samples))
    peaks, _ = find_peaks(inverted_ppg, distance=min_distance, prominence=min_prominence)
    return peaks

def split_ppg_into_waves(ppg_signal, onsets, fs, resample_length=1000, min_wave_sec=0.4, max_wave_sec=1.6, lower_threshold=-0.1):
    """
    Given onsets (indices of local minima), split the PPG signal into individual waves.
    Each wave is resampled to resample_length samples, detrended, and normalized to [-1, 1].
    Removes waves that have low values in the middle 30-60% of the wave (lower_threshold)
    """
    processed_waves = []
    onsets_kept = []
    durations_waves = []
    for i in range(len(onsets) - 1):
        start_i = onsets[i]
        end_i = onsets[i+1]
        wave = ppg_signal[start_i:end_i]
        wave_samples = len(wave)
        wave_duration_sec = wave_samples / fs
        if wave_duration_sec < min_wave_sec or wave_duration_sec > max_wave_sec:
            continue
        if wave_samples < 2:
            continue
        x_old = np.linspace(0, 1, wave_samples)
        x_new = np.linspace(0, 1, resample_length)
        wave_resampled = np.interp(x_new, x_old, wave)
        line = np.linspace(wave_resampled[0], wave_resampled[-1], resample_length)
        wave_detrended = wave_resampled - line
        mn = np.min(wave_detrended)
        mx = np.max(wave_detrended)
        if (mx - mn) < 1e-9:
            wave_norm = wave_detrended
        else:
            wave_norm = 2 * (wave_detrended - mn) / (mx - mn) - 1.0
        
        # Logic to remove all waves that have entries in the middle 30-60% that are below lower_threshold (defaults to -0.1)
        if np.any(wave_norm[int(0.3 * resample_length):int(0.6 * resample_length)] < lower_threshold):
            # print(f"Wave from {start_i} to {end_i} has low values in the middle 30-60% and is skipped.")
            continue

        processed_waves.append(wave_norm)
        onsets_kept.append(start_i)
        durations_waves.append(wave_duration_sec)
    return processed_waves, onsets_kept, durations_waves


def step4_improved(data_dict, expected_period_sec=1.0, min_prominence=0.02, resample_length=1000, tolerance=0.6, lower_threshold=-0.1):
    """
    For each subject, detect onsets, split into individual waves,
    compute the average wave (ensemble), average heart rate,
    and then compute derivatives for both the ensemble wave and each individual wave.
    """
    for pid, subj_data in data_dict.items():
        if "ppg" not in subj_data or "sampling_rate" not in subj_data:
            continue
        ppg_signal = subj_data["ppg"]
        fs = subj_data["sampling_rate"]
        onsets = detect_onsets_ppg_v2(ppg_signal, fs, expected_period_sec, min_prominence)
        min_wave_sec = expected_period_sec * (1 - tolerance)
        max_wave_sec = expected_period_sec * (1 + tolerance)
        waves, onsets_kept, durations_waves = split_ppg_into_waves(ppg_signal, onsets, fs, resample_length, min_wave_sec, max_wave_sec, lower_threshold=lower_threshold)
        subj_data["individual_waves"] = waves
        subj_data["fid_on_ppg"] = np.array(onsets_kept, dtype=int)
        subj_data["wave_durations"] = durations_waves
        
        # Get average HR 
        if len(durations_waves) > 0:
            avg_period_sec = np.mean(durations_waves)
            avg_hr_bpm = 60.0 / avg_period_sec
        else:
            avg_hr_bpm = None
        subj_data["average_hr"] = avg_hr_bpm
        
        # Get rise time (time from onset to peak) and fall time (time from peak to next onset)
        rise_times_norm = []
        rise_times_ms = []
        
        # get rime times for each wave
        for (wave, duration) in zip(waves, durations_waves):
            peak_idx = np.argmax(wave)
            # print(len(wave), peak_idx, duration)
            # get rise time, also considering duration (time from onset to peak)
            rise_time_norm = peak_idx / len(wave)
            rise_times_norm.append(rise_time_norm)
            rise_time_ms = peak_idx / len(wave) * duration
            rise_times_ms.append(rise_time_ms)    
            # print("rise time norm", rise_time_norm)
            # print("rise time ms", rise_time_ms)
        subj_data["rise_times_norm"] = rise_times_norm
        subj_data["rise_times_ms"] = rise_times_ms
        
        # Get average (median) rise time per subject 
        avg_rise_time_norm = np.median(rise_times_norm)
        avg_rise_time_ms = np.median(rise_times_ms)
        subj_data["average_rise_time_norm"] = avg_rise_time_norm
        subj_data["average_rise_time_ms"] = avg_rise_time_ms    
                
        # Print per subject to check 
        # print(f"Subject {pid}: {len(waves)} waves, avg HR = {avg_hr_bpm:.2f} bpm")
        # print(f"Subject {pid}: avg rise time norm = {avg_rise_time_norm:.2f}, avg rise time ms = {avg_rise_time_ms:.2f}")

        
        # Compute ensemble wave
        if len(waves) > 0:
            arr = np.array(waves)
            ensemble = arr.mean(axis=0)
        else:
            ensemble = np.zeros(resample_length)
        subj_data["ensemble_wave"] = ensemble
        processed_ensemble = process_ppg_wave(ensemble)
        try:
            ppg_11, vpg_11, apg_11, jpg_11 = compute_derivatives_pyppg(processed_ensemble, fs=1000)
        except Exception as e:
            print(f"Error computing ensemble derivatives for {pid}: {e}")
            ppg_11 = vpg_11 = apg_11 = jpg_11 = np.zeros(resample_length)
            
        subj_data.update({
            "ensemble_ppg": ppg_11,
            "ensemble_vpg": vpg_11,
            "ensemble_apg": apg_11,
            "ensemble_jpg": jpg_11
        })
        # Compute derivatives for each individual wave
        # individual_waves_derivatives = []
        ippg_arr = []
        ivpg_arr = []
        iapg_arr = []
        ijpg_arr = []
        
        for wave in waves:
            processed_wave = process_ppg_wave(wave)
            try:
                ippg, ivpg, iapg, ijpg = compute_derivatives_pyppg(processed_wave, fs=1000)
            except Exception as e:
                print(f"Error computing individual wave derivatives for {pid}: {e}")
                ippg = ivpg = iapg = ijpg = np.zeros(resample_length)
            
            ippg_arr.append(ippg)
            ivpg_arr.append(ivpg)
            iapg_arr.append(iapg)
            ijpg_arr.append(ijpg)
            
        #     derivative_dict = {
        #         "ensemble_ppg": ippg,
        #         "ensemble_vpg": ivpg,
        #         "ensemble_apg": iapg,
        #         "ensemble_jpg": ijpg
        #     }
        #     individual_waves_derivatives.append(derivative_dict)
        # subj_data["individual_waves_derivatives"] = individual_waves_derivatives
        
        subj_data["individual_wave_derivs_ppg_arr"] = ippg_arr
        subj_data["individual_wave_derivs_vpg_arr"] = ivpg_arr
        subj_data["individual_wave_derivs_apg_arr"] = iapg_arr
        subj_data["individual_wave_derivs_jpg_arr"] = ijpg_arr
        
        # Get each one ensemble average 
        subj_data["ensemble_ppg_avg"] = np.mean(ippg_arr, axis=0)
        subj_data["ensemble_vpg_avg"] = np.mean(ivpg_arr, axis=0)   
        subj_data["ensemble_apg_avg"] = np.mean(iapg_arr, axis=0)
        subj_data["ensemble_jpg_avg"] = np.mean(ijpg_arr, axis=0)
        
        print("Duration ensemble derivative wave:", len(subj_data["ensemble_ppg_avg"]))

    return data_dict


def save_final_dict(data_dict, output_file):
    torch.save(data_dict, output_file)
    print("Final dictionary saved as", output_file)

# --- Optional: Merge datasets function (if you wish to merge oscillometric and auscultatory) ---
def merge_datasets_and_add_regressors(data_path):
    data_osc = torch.load(data_path / "data_dict_oscillometric.pt", weights_only=False)
    data_auc = torch.load(data_path / "data_dict_auscultatory.pt", weights_only=False)
    
    data_dict = {**data_osc, **data_auc}
    for pid, entry in data_dict.items():
        ht_in = entry.get("height", None)
        wt_lbs = entry.get("weight", None)
        if ht_in is not None and wt_lbs is not None and ht_in > 0 and wt_lbs > 0:
            ht_m = ht_in * 0.0254
            wt_kg = wt_lbs * 0.453592
            bmi = wt_kg / (ht_m**2)
            entry["bmi"] = bmi
            entry["height_m"] = ht_m
            entry["weight_kg"] = wt_kg
        else:
            entry["bmi"] = None
            entry["height_m"] = None
            entry["weight_kg"] = None
    out_path = data_path / "data_dict_osc_auc_with_derivatives.pt"
    torch.save(data_dict, out_path)
    print(f"Saved merged data_dict with derivatives to: {out_path}")
    return data_dict


def pipeline(dataset_type, raw_path, preprocessed_path):
    print(f"Processing {dataset_type} data.")
    
    print("Step 1: Load metadata and get waveform folder based on dataset type")
    no_comorb_df, waveform_base_path = load_data(raw_path=raw_path, dataset_type=dataset_type)
    
    print("Step 2: Create dictionary for each participant")
    data_dict = create_data_dict(no_comorb_df, waveform_base_path)
    
    print("Step 3: Process each subject's raw optical signal with pyPPG (initial derivatives)")
    data_dict = process_all_subjects(data_dict)
    
    print("Step 4: Split into individual waves, compute ensemble wave, average HR, and compute derivatives")
    data_dict = step4_improved(data_dict, expected_period_sec=1.0, min_prominence=0.02, resample_length=1000, tolerance=0.6, lower_threshold=-0.1)
    
    print("Step 5: Save final dictionary")
    final_file = preprocessed_path / f"data_dict_{dataset_type}.pt"
    save_final_dict(data_dict, final_file)

    return "completed - saved as " + str(final_file)


# --- Main Execution ---

def main():
    
    raw_path = Path(RAW_AURORA_DATA_PATH)
    preprocessed_path = Path(PREPROCESSED_AURORA_DATA_PATH)

    # base_path = Path("D:\\00_ppg_project\\aurora_data")
    # output_pt = base_path / "preprocessed"
    
    # Choose dataset type: "oscillometric" or "auscultatory"
    print(pipeline(dataset_type="oscillometric", 
                    raw_path=raw_path,
                    preprocessed_path = preprocessed_path))
    
    print(pipeline(dataset_type="auscultatory", 
                    raw_path=raw_path,
                    preprocessed_path = preprocessed_path))

    print("Merging datasets and adding regressors.")
    merge_datasets_and_add_regressors(preprocessed_path)


if __name__ == "__main__":
    main()
