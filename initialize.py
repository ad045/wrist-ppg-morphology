#!/usr/bin/env ppg_project
from pathlib import Path
import os

# PATHS
DATA_PATH = "<your_path>" # User's TODO: Change this

AURORA_DATA_PATH = f"{DATA_PATH}/AURORA"
RAW_AURORA_DATA_PATH = f"{AURORA_DATA_PATH}/raw"
PREPROCESSED_AURORA_DATA_PATH = f"{AURORA_DATA_PATH}/preprocessed"

MAUS_DATA_PATH = f"{DATA_PATH}/MAUS"
RAW_MAUS_DATA_PATH = f"{MAUS_DATA_PATH}/raw/Raw_data"
PREPROCESSED_MAUS_DATA_PATH = f"{MAUS_DATA_PATH}/preprocessed"

# Create directories if they do not exist
os.makedirs(RAW_AURORA_DATA_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_AURORA_DATA_PATH, exist_ok=True)
os.makedirs(RAW_MAUS_DATA_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_MAUS_DATA_PATH, exist_ok=True) 

# Output paths for comparison algorithms
PREPROCESSED_MAUS_COMPARISON_ALGOS_PATH = f"{PREPROCESSED_MAUS_DATA_PATH}/comparison_algos"
os.makedirs(PREPROCESSED_MAUS_COMPARISON_ALGOS_PATH, exist_ok=True)

# Output path for all results
OUTPUT_PATH = "<a_second_path_of_your_choice>/output" # User's TODO: Change this
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Output path for comparison of algorithms 
OUTPUT_COMPARISON_ALGOS_PATH = f"{OUTPUT_PATH}/comparison_algos"
os.makedirs(OUTPUT_COMPARISON_ALGOS_PATH, exist_ok=True)

OUTPUT_REGRESSION_PATH = f"{OUTPUT_PATH}/regression"
os.makedirs(OUTPUT_REGRESSION_PATH, exist_ok=True)

# OUTPUT FORMATS 
TABLE_FORMATS = ["txt"] 
IMAGE_FORMATS = ["svg"]
