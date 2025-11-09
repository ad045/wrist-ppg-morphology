# This script is to compare the durations of waves. The durations that are extracted here are the ones that were detected with our custom pipeline. 
# Get durations_custom_finger.npy and durations_custom_wrist.npy

import torch 
import numpy as np
from pathlib import Path

from initialize import RAW_MAUS_DATA_PATH, PREPROCESSED_MAUS_DATA_PATH

def extract_custom_wave_durations(preprocessed_path, file_name):
    """
    Extracts wave durations from the custom data dictionary and saves them as a numpy array.
    Args:
        preprocessed_path (str): Path to the custom data dictionary file.
    """
    # Load the custom data dictionary
    preprocessed_path = Path(preprocessed_path)
    custom_data_dict = torch.load(preprocessed_path / file_name, weights_only=False) 

    # Extract wave durations
    durations_custom_dict = []
    for entry in custom_data_dict: 
        durations_custom_dict.append(np.array(custom_data_dict[entry]["wave_durations"]) * 1000)  # Convert to milliseconds
    durations_custom_dict = np.concatenate(durations_custom_dict)
    print(len(durations_custom_dict))

    # make comparison_algos directory if it does not exist
    (preprocessed_path / "comparison_algos").mkdir(parents=True, exist_ok=True)

    # save durations_custom_dict similar to durations_neurokit
    if "finger" in file_name:
        np.save(preprocessed_path / "comparison_algos/durations_custom_finger.npy", durations_custom_dict)
    elif "wrist" in file_name:
        np.save(preprocessed_path / "comparison_algos/durations_custom_wrist.npy", durations_custom_dict)
    else:
        raise ValueError("Data path must contain 'finger' or 'wrist'.")


extract_custom_wave_durations(preprocessed_path=PREPROCESSED_MAUS_DATA_PATH, file_name="data_dict_maus_finger_filtered.pt")
extract_custom_wave_durations(preprocessed_path=PREPROCESSED_MAUS_DATA_PATH, file_name="data_dict_maus_wrist_filtered.pt")
