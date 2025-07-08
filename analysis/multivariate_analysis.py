# file: multivariate_regression.py
"""
Rise-time regression for AURORA PPG data
=======================================

Run with
    python multivariate_regression.py                # full report in ./reports/
    python multivariate_regression.py --help         # CLI flags

Requires: pandas, statsmodels, matplotlib, seaborn, torch
"""

from __future__ import annotations
import argparse, pathlib, re, sys, textwrap
from itertools import chain
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

REPORT_DIR = pathlib.Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------#
#  Helpers                                                                     #
# -----------------------------------------------------------------------------#
def _flatten_dict_to_rows(d: dict[str, dict],
                          wave_level: bool = True) -> pd.DataFrame:
    """
    Convert the big data_dict into a tidy DataFrame.

    Parameters
    ----------
    wave_level : bool
        • True  → one row per *individual* wave (uses rise_times_ms).
        • False → one row per subject (uses average_rise_time_ms).

    Returns
    -------
    DataFrame
    """
    rows: list[dict] = []

    for pid, entry in d.items():
        # -- explanatory variables (identical for every wave of the subject) --#
        base = dict(
            pid               = pid,
            age               = entry.get("age"),
            bmi               = entry.get("bmi"),
            sbp               = entry.get("baseline_sbp"),
            dbp               = entry.get("baseline_dbp"),
            gender            = entry.get("gender"),          # 0/1 or M/F ?
            height            = entry.get("height_m"),
            weight            = entry.get("weight_kg"),
            oscillo_or_auscul = entry.get("oscillo_or_auscul"),
            cvd_meds          = entry.get("cvd_meds"),        # 0/1
            fitzpatrick       = entry.get("fitzpatrick_scale"),
            heart_rate        = entry.get("average_hr"),
        )

        if wave_level:
            for rt in entry.get("rise_times_ms", []):
                rows.append({**base, "rise_time_ms": rt})
        else:
            rt = entry.get("average_rise_time_ms")
            if rt is not None:
                rows.append({**base, "rise_time_ms": rt})

    df = pd.DataFrame(rows)
    return df


def _prep_dataframe(df: pd.DataFrame,
                    cat_to_dummy: Iterable[str] = ("gender",
                                                    "oscillo_or_auscul",
                                                    "cvd_meds")) -> pd.DataFrame:
    """Numeric-ise categorical variables, drop obviously missing rows."""
    # gender might be 'm'/'f' or 0/1 – make sure it is 0/1
    if df["gender"].dtype == object:
        df["gender"] = df["gender"].str.lower().map({"m": 0, "f": 1})
    # minimal cleaning
    df = df.dropna(subset=["rise_time_ms"])          # outcome must exist
    # dummy-encode remaining categoricals
    df = pd.get_dummies(df, columns=list(cat_to_dummy), drop_first=True)
    return df


def _univariate_ols(df: pd.DataFrame, y: str, xcols: list[str]) -> pd.DataFrame:
    records = []
    for x in xcols:
        model = sm.OLS(df[y], sm.add_constant(df[[x]])).fit()
        beta  = model.params[x]
        se    = model.bse[x]
        pval  = model.pvalues[x]
        r2    = model.rsquared
        records.append(dict(predictor=x, beta=beta, se=se, p=pval, r2=r2))
    return pd.DataFrame(records).sort_values("p")


def _multivariate_ols(df: pd.DataFrame, y: str, xcols: list[str]):
    X = sm.add_constant(df[xcols])
    model = sm.OLS(df[y], X).fit()      # add  cov_type="cluster" …  for robust SEs
    return model


def _save_latex_table(model, fname: pathlib.Path):
    latex = model.summary().as_latex()
    fname.write_text(latex)
    print(f"[INFO] LaTeX written → {fname}")


def _plot_univariate(df: pd.DataFrame,
                     y: str,
                     x: str,
                     out: pathlib.Path):
    fig, ax = plt.subplots()
    sns.regplot(x=x, y=y, data=df, ax=ax)
    ax.set_title(f"{y} vs {x}")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def _correlation_heatmap(df: pd.DataFrame,
                         cols: list[str],
                         out: pathlib.Path):
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=True,
                cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlations between variables")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)

# -----------------------------------------------------------------------------#
#  Main analysis                                                               #
# -----------------------------------------------------------------------------#
def run_analysis(pt_file: pathlib.Path, wave_level: bool, out_folder: pathlib.Path):
    level = "wave" if wave_level else "ensemble"
    print(f"[INFO] Loading {pt_file}")
    data_dict = torch.load(pt_file, weights_only=False)

    df = _prep_dataframe(_flatten_dict_to_rows(data_dict, wave_level=wave_level))

    # --------------------------- design matrix ------------------------------
    outcome = "rise_time_ms"
    # drop pid and outcome for predictors
    candidate_x = df.columns.difference(["pid", outcome]).tolist()

    # --------------------------- correlation map ---------------------------
    _correlation_heatmap(df, candidate_x + [outcome],
                         out_folder / f"corr_{level}.png")

    # --------------------------- univariate --------------------------------
    uni = _univariate_ols(df, outcome, candidate_x)
    uni.to_csv(out_folder / f"univariate_{level}.csv", index=False)
    print(f"[INFO] Univariate stats saved → univariate_{level}.csv")

    # individual scatter plots (optional – can be heavy with many waves)
    max_plots = 9                               # stop spamming
    for i, row in uni.head(max_plots).iterrows():
        _plot_univariate(df, outcome, row["predictor"],
                         out_folder / f"scatter_{row['predictor']}_{level}.png")

    # --------------------------- multivariate ------------------------------
    multi = _multivariate_ols(df, outcome, candidate_x)
    _save_latex_table(multi,
                      out_folder / f"multivariate_{level}.tex")

    # also dump a concise CSV with betas & p-values
    coefs = (multi.params.rename("beta")
             .to_frame()
             .join(multi.pvalues.rename("p"))
             .drop("const"))
    coefs.to_csv(out_folder / f"coefficients_{level}.csv")
    print(f"[INFO] Multivariate coefficients saved → coefficients_{level}.csv")

    # quick prediction plot
    df["pred"] = multi.predict(sm.add_constant(df[candidate_x]))
    fig, ax = plt.subplots()
    ax.scatter(df[outcome], df["pred"], s=10, alpha=0.6)
    lims = [df[outcome].min(), df[outcome].max()]
    ax.plot(lims, lims, ls="--", c="r")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted ({level})")
    fig.tight_layout()
    fig.savefig(out_folder / f"actual_vs_pred_{level}.png", dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------#
#  CLI                                                                         #
# -----------------------------------------------------------------------------#
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
        Linear regression of PPG rise-time.

        Example
        -------
        python multivariate_regression.py \\
               --data_pt  /path/to/data_dict_osc_auc_with_derivatives.pt
        """))
    p.add_argument("--data_pt", type=pathlib.Path, required=True,
                   help="*.pt file produced by your preprocessing pipeline.")
    p.add_argument("--skip_wave_level", action="store_true",
                   help="Only run analysis on ensemble beats.")
    p.add_argument("--skip_ensemble_level", action="store_true",
                   help="Only run analysis on individual waves.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.skip_wave_level and args.skip_ensemble_level:
        sys.exit("[ERROR] chose to skip everything – nothing to do.")

    out = REPORT_DIR
    out.mkdir(exist_ok=True)

    if not args.skip_wave_level:
        run_analysis(args.data_pt, wave_level=True, out_folder=out)

    if not args.skip_ensemble_level:
        run_analysis(args.data_pt, wave_level=False, out_folder=out)


if __name__ == "__main__":
    main()
