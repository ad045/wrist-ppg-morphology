from pathlib import Path

DATA_PATH = "/Users/adrian/Documents/01_projects/00_ppg_project/data" # change later to 02_clean_ppg/data

AURORA_DATA_PATH = f"{DATA_PATH}/AURORA"
RAW_AURORA_DATA_PATH = f"{AURORA_DATA_PATH}/raw"
# PREPROCESSED_AURORA_DATA_PATH = f"{AURORA_DATA_PATH}/preprocessed"
PREPROCESSED_AURORA_DATA_PATH = "/Users/adrian/Documents/01_projects/02_clean_ppg/data/AURORA/preprocessed"

MAUS_DATA_PATH = f"{DATA_PATH}/MAUS"
RAW_MAUS_DATA_PATH = f"{MAUS_DATA_PATH}/raw/Raw_data"
# PREPROCESSED_MAUS_DATA_PATH = f"{MAUS_DATA_PATH}/preprocessed"
PREPROCESSED_MAUS_DATA_PATH = "/Users/adrian/Documents/01_projects/02_clean_ppg/data/MAUS/preprocessed"

# Create directories if they do not exist
import os
os.makedirs(RAW_AURORA_DATA_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_AURORA_DATA_PATH, exist_ok=True)
os.makedirs(RAW_MAUS_DATA_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_MAUS_DATA_PATH, exist_ok=True) 
