import argparse
from pathlib import Path

import torch
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# /opt/miniconda3/envs/ppg_project/bin/python /Users/adrian/Documents/01_projects/02_clean_ppg/analysis/regression/rise_time_regression.py --path /Users/adrian/Documents/01_projects/02_clean_ppg/data/AURORA/preprocessed/data_dict_osc_auc_with_derivatives.pt

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "baseline_sbp",
    "baseline_dbp",
    "gender",
    "height",
    "weight",
    "oscillo_or_auscul",
    "cvd_meds",
    "fitzpatrick_scale",
]

CATEGORICAL_COLS = ["gender", "oscillo_or_auscul", "cvd_meds", "fitzpatrick_scale"]


def load_dataset(path: Path) -> dict:
    """Load the preprocessed data dictionary."""
    return torch.load(path, weights_only=False)


def make_dataframe(data: dict) -> pd.DataFrame:
    """Flatten waves into a long table with rise time per wave."""
    rows = []
    for entry in data.values():
        rise_times = entry.get("rise_times_ms")
        if rise_times is None:
            continue
        base = {key: entry.get(key) for key in FEATURE_COLUMNS}
        for rt in rise_times:
            row = base.copy()
            row["rise_time_ms"] = rt
            rows.append(row)
    return pd.DataFrame(rows)


def run_univariate(df: pd.DataFrame) -> dict:
    """Perform univariate linear regression for each feature."""
    results = {}
    y = df["rise_time_ms"]
    for col in FEATURE_COLUMNS:
        X = pd.get_dummies(df[[col]], drop_first=True)
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        results[col] = {"r2": r2, "coef": model.coef_, "intercept": model.intercept_}
    return results


def run_multivariate(df: pd.DataFrame):
    y = df["rise_time_ms"]
    X = df[FEATURE_COLUMNS]
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_COLS)],
        remainder="passthrough",
    )
    model = Pipeline(steps=[("preprocess", preprocessor), ("regressor", LinearRegression())])
    model.fit(X, y)
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    return model, r2


def main():
    parser = argparse.ArgumentParser(description="Run rise time regression")
    parser.add_argument("--path", type=Path, help="Path to data_dict_osc_auc_with_derivatives.pt")
    args = parser.parse_args()

    data = load_dataset(args.path)
    df = make_dataframe(data)

    if df.empty:
        print("No rise time data found in the dataset.")
        return

    print("Univariate regression results:")
    uni = run_univariate(df)
    for feat, res in uni.items():
        print(f"{feat:20s} R^2={res['r2']:.3f} coef={res['coef']} intercept={res['intercept']:.3f}")

    model, r2 = run_multivariate(df)
    print(f"\nMultivariate regression R^2: {r2:.3f}")


if __name__ == "__main__":
    main()