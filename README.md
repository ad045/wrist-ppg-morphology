# Clean PPG Project

A toolkit for preprocessing, analysing, and comparing photoplethysmography (PPG) pulse waves recorded at the wrist. It combines a custom pipeline wrapped around the [pyPPG](pyPPG) library to extract fiducials, derivatives, and morphological features from datasets such as [AURORA‑BP](https://ieeexplore.ieee.org/document/9721156) and [MAUS](https://ieee-dataport.org/open-access/maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor).

-----

## 1. Installation

1. Create a conda environment from the given environment.yaml file:
   ```bash
   conda env create -f environment.yaml
   ```
2. Install pyPPG into the main folder (i.e. replace the current placeholder folder ['pyPPG'](pyPPG)). Then go into ['pyPPG/fiducials.py'](pyPPG/fiducials.py), and replace all 'np.NaN' with 'np.nan', as numpy > 2.0 does not further support 'np.NaN'.

-----

## 2. Getting started: Setup 

Update path definitions in [`initialize.py`](initialize.py) to point to your copies of the datasets, and modify the paths to point to your output folders before running any scripts. 
Missing directories are created automatically.

-----

## 3. First run: Notebooks

The `notebooks/` directory provides jupyter notebooks that conveniently lead through the functionalities of this pipeline. We recommend to run them in the following order: 

- `00_preprocessing.ipynb` – preprocess the AURORA‑BP dataset.
- `01_comparison_different_algorithms.ipynb` – compare pyPPG, NeuroKit2, and this custom wrapper.
- `02_multivariate_analysis.ipynb` – multivariate analysis examples.
- `03_plotting_derivations_AURORA.ipynb` – additional derivation and plotting experiments.
- `04_comparing_fPPG_and_wPPG_PWs_for_MAUS` - comparison of wrist and finger pulse wave morphologies.

-----

## 4. Repository layout

```
notebooks/            The entry point into the pipelines. Run individual scripts to conveniently process and analyze the data
initialize.py         User configurable paths and output directories
----
preprocessing/        Pre‑ and post‑processing pipeline
analysis/             Wave classification and statistical analysis
pyPPG/                Embedded copy of the pyPPG library
```