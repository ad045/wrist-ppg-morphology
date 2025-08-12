# Clean PPG Project

A toolkit for preprocessing, analysing, and comparing photoplethysmography (PPG) pulse waves recorded at the wrist. It combines a custom preprocessing pipeline with the [pyPPG](pyPPG) library to extract fiducials, derivatives, and morphological features from datasets such as AURORA‑BP and MAUS.

## Repository layout

```
preprocessing/        Pre‑ and post‑processing pipeline
analysis/             Wave classification and statistical analysis
notebooks/            Jupyter notebooks for exploration and algorithm comparison
pyPPG/                Embedded copy of the pyPPG library
initialize.py         User configurable paths and output directories
```

## Installation

1. Create a Python 3.10+ environment and activate it.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   # .\.venv\Scripts\activate for Windows PowerShell
   ```
2. Install the required packages. The project relies on common scientific‑Python
   libraries such as `numpy`, `pandas`, `scipy`, `torch`, `matplotlib`, `seaborn`,
   `statsmodels`, `scikit-learn`, `neurokit2`, and `dotmap`.
   ```bash
   pip install numpy pandas scipy torch matplotlib seaborn statsmodels scikit-learn neurokit2 dotmap
   ```
   *(A requirements file is not yet available.)*

## Preparing data

All project‑specific paths live in [`initialize.py`](initialize.py). Update these
constants to point to your copies of the datasets and the desired output
locations before running any scripts. Missing directories are created
automatically.

## Preprocessing

The [`preprocessing`](preprocessing) module handles both raw‑data
pre‑processing and optional post‑processing (threshold‑based filtering and
recomputation of derivatives).

- **Full pipeline**
  ```bash
  python -m preprocessing --lower_threshold -0.15
  ```
- **Post‑process an existing dictionary only**
  ```bash
  python -m preprocessing --lower_threshold -0.1
  ```

Each run produces `data_dict_*.pt` files containing individual and ensemble
pulse waves alongside their derivatives.

## Analysis

Scripts in [`analysis`](analysis) operate on the preprocessed dictionaries:

- [`analysis/aurora_analysis.py`](analysis/aurora_analysis.py) classifies each
  pulse wave using a five‑class decision tree and performs regression analyses.
  It generates figures and tables in `output/regression/` and writes the
  updated dictionary with wave‑shape classes.
- [`analysis/compare_durations/`](analysis/compare_durations) provides utilities
  to benchmark processing times for different algorithms.
- [`analysis/plot_age_stratified_ensemble_waves.py`](analysis/plot_age_stratified_ensemble_waves.py)
  visualises ensemble waveforms across age groups.

## Notebooks

The `notebooks/` directory contains exploratory material:

- `00_preprocessing.ipynb` – preprocess the AURORA‑BP dataset.
- `01_comparison_different_algorithms.ipynb` – compare pyPPG, NeuroKit2, and the
  custom pipeline.
- `01_5_multivariate_analysis.ipynb` – multivariate analysis examples.
- `02_*` – additional derivation and plotting experiments.

## pyPPG library

The `pyPPG/` directory is an embedded copy of the pyPPG library. The project
uses its routines for fiducial extraction, derivative computation, and
biomarker generation; the library can also be used independently.

## Contributing

Issues and pull requests are welcome. The repository does not yet define coding
standards or automated checks, but please run `pytest` to ensure that existing
code behaves as expected before submitting changes.

## License

No explicit licence is provided. Contact the author for usage permissions.

## Acknowledgements

- AURORA‑BP and MAUS datasets
- The pyPPG community
- Contributors to open‑source scientific‑Python libraries

