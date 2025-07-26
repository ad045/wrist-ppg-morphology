#!/usr/bin/env ppg_project


# OUTPUT FORMATS 

TABLE_FORMATS = ["txt"] #, "tex"]
IMAGE_FORMATS = ["svg"] #, "png"] #, "pdf"]  # for plots


# PATHS

from pathlib import Path
import os

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
os.makedirs(RAW_AURORA_DATA_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_AURORA_DATA_PATH, exist_ok=True)
os.makedirs(RAW_MAUS_DATA_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_MAUS_DATA_PATH, exist_ok=True) 


# Output paths for comparison algorithms
PREPROCESSED_MAUS_COMPARISON_ALGOS_PATH = f"{PREPROCESSED_MAUS_DATA_PATH}/comparison_algos"
os.makedirs(PREPROCESSED_MAUS_COMPARISON_ALGOS_PATH, exist_ok=True)


# Output path 
OUTPUT_PATH = "/Users/adrian/Documents/01_projects/02_clean_ppg/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Output path for comparison of algorithms 
OUTPUT_COMPARISON_ALGOS_PATH = f"{OUTPUT_PATH}/comparison_algos"
os.makedirs(OUTPUT_COMPARISON_ALGOS_PATH, exist_ok=True)

OUTPUT_REGRESSION_PATH = f"{OUTPUT_PATH}/regression"
os.makedirs(OUTPUT_REGRESSION_PATH, exist_ok=True)