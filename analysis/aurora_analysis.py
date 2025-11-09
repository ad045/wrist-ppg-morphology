#!/usr/bin/env ppg_project
"""
AURORA | MAUS – wave-class labelling & regression analysis
=========================================================

**What this script does**
-------------------------
1. **Classify every pulse wave** according to the five-class decision tree (adds *individual_waves_classes* & *ensemble_class* to the dict).
2. **Run statistics & create figures**
   * univariate scatter/OLS line plots
   * correlation heat-map
   * rise-time histogram
   * multivariate *and* statsmodels OLS regression (z-normalised inputs)
3. **Save outputs**
   * figures → `output/regression/figures/*.png` & `*.pdf`
   * coefficient CSV (`output/regression/tables/coefficients.csv`)
   * pretty LaTeX table of coefficients (`output/regression/tables/coefficients.tex`)
   * full statsmodels summary as LaTeX (`output/regression/tables/ols_summary.tex`) *and* as txt file (`output/regression/tables/ols_summary.txt`)
   * updated data_dict with classes → `*_with_classes.pt`
"""

from __future__ import annotations

import argparse
import logging
import sys
import pathlib
from pathlib import Path
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm



# make project root importable no matter how we launch the script
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from initialize import (
    PREPROCESSED_AURORA_DATA_PATH,
    OUTPUT_REGRESSION_PATH,
    TABLE_FORMATS,
    IMAGE_FORMATS,
)

# ───────────────────────────── configuration ──────────────────────────────
DEFAULT_DICT_PATH = Path(PREPROCESSED_AURORA_DATA_PATH) / "data_dict_osc_auc_with_derivatives.pt"

FIG_DIR   = Path(OUTPUT_REGRESSION_PATH) / "figures"
TAB_DIR   = Path(OUTPUT_REGRESSION_PATH) / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

PREDICTORS_CONT = [
    "age", "bmi", "height_m", 
    "average_hr", "baseline_sbp", "baseline_dbp"
]

PREDICTORS_CAT  = [
    "gender", "cvd_meds", "fitzpatrick_scale",
    "pressure_quality", "optical_quality", "oscillo_or_auscul",
]

# ───────────────── predictor-set presets ────────────────────────────

PREDICTOR_SETS: dict[str, dict[str, list[str]]] = { # Include all variable sets to use in the analysis here 
    "all": { # all variables, including categorical
        "cont": PREDICTORS_CONT,
        "cat":  PREDICTORS_CAT,
    },

    "core": { # the six core variables, no categorical
        "cont": ["age", "bmi", "height_m",
                 "average_hr", "baseline_sbp", "baseline_dbp"],
        "cat":  [],           # no categorical variables
    },

        "core_plus_fitzpatrick": { # all variables, including categorical
        "cont": ["age", "bmi", "height_m",
                 "average_hr", "baseline_sbp", "baseline_dbp", "fitzpatrick_scale"],
        "cat":  [],
    },
}


TARGETS = ["rise_time_ms", "rise_time_norm"]

# Configuration parameters for area under curve (AUC) calculation
AUC_START, AUC_END = 200, 800 # sample range used for APG area (inclusive)

# ─────────────────────── wave-shape classification helpers ─────────────────

def _smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Savitzky–Golay smoothing with fallback for short waves."""
    return y if len(y) < window else signal.savgol_filter(y, window, poly)


def _auc_apg(apg_wave: np.ndarray,
             start: int = AUC_START,
             end:   int = AUC_END) -> float:
    """Unsigned APG area between *start* and *end* samples (NaN if too short)."""
    mask = apg_wave > 0
    wave = apg_wave*mask
    return float(np.trapezoid(wave[start:end+1]))


def _inflection_before_after(y: np.ndarray, peak_idx: int, threshold=1e-3) -> Tuple[bool, bool]:
    """Detect whether *a* zero-crossing of the 2nd derivative exists **before**
    and/or **after** the main peak.  A small hysteresis avoids noise."""
    d2 = np.gradient(np.gradient(y))
    # hysteresis: ignore crossings where |d2| > thr on either side
    thr = threshold
    zc = np.where(np.diff(np.sign(d2)))[0]
    # introduce area in which inflection points "count".
    left_boundary = 200 
    right_boundary = 600 
    zc = [i for i in zc if left_boundary < i < right_boundary]
    # determine if there is an inflection point before or after the peak
    before = any((abs(d2[i]) < thr or abs(d2[i+1]) < thr) and i < peak_idx for i in zc)
    after  = any((abs(d2[i]) < thr or abs(d2[i+1]) < thr) and i > peak_idx for i in zc)
    return before, after


def classify_wave(wave: np.ndarray, threshold) -> int:
    """Return class **1-5** (0 = invalid) following the decision tree."""
    if wave.size == 0 or np.all(np.isnan(wave)):
        return 0

    y = _smooth(wave)
    peaks, _ = signal.find_peaks(y, distance=len(y)//10)

    # ── branch: two local maxima ──────────────────────────────────────────
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(y[peaks])][-2:]
        top2.sort()
        first, second = top2
        return 1 if y[first] > y[second] else 5

    # ── branch: one local maximum ─────────────────────────────────────────
    peak = int(peaks[0]) if len(peaks) else int(np.argmax(y))
    before, after = _inflection_before_after(y, peak, threshold=threshold)
    if before:
        return 4  # inflection before max → Class 4
    if after:
        return 2  # inflection after max  → Class 2
    return 3      # no inflection         → Class 3


def label_entry(entry: Dict[str, Any], threshold) -> None:
    if "ensemble_class" in entry:
        return
    waves  = entry.get("individual_wave_derivs_ppg_arr", [])
    apg_waves = entry.get("individual_wave_derivs_apg_arr", [])

    classes = np.array([classify_wave(w, threshold) for w in waves], dtype=np.int8)

    entry["individual_waves_classes"] = classes
    entry["ensemble_class"] = int(classify_wave(entry.get("ensemble_wave", np.empty(0)), 
                                                threshold=threshold))

    # ───────────────── APG area (unsigned & signed) ──────────────────
    auc_waves = np.array([_auc_apg(w) for w in apg_waves])
    sign_wave_factor  = np.where(np.isin(classes, [1, 2]), -1, 1)
    entry["area_under_the_curve_unsign_wave"] = auc_waves
    entry["area_under_the_curve_sign_wave"]     = auc_waves * sign_wave_factor

    ens_auc = _auc_apg(entry.get("ensemble_apg_avg", np.empty(0)))
    ens_sign = -1 if entry["ensemble_class"] in (1, 2) else 1
    entry["area_under_the_curve_unsign"] = ens_auc
    entry["area_under_the_curve_sign"]    = ens_sign * ens_auc


# ────────────────────────── dataframe & plotting helpers ───────────────────

def _make_dataframe(data_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pid, d in data_dict.items():
        if "average_rise_time_ms" not in d:
            continue
        rows.append({
            "pid": pid,
            **{k: d.get(k, np.nan) for k in PREDICTORS_CONT + PREDICTORS_CAT},
            "rise_time_ms": d.get("average_rise_time_ms", np.nan),
            "rise_time_norm": d.get("average_rise_time_norm", np.nan),
            "ensemble_class": d.get("ensemble_class", np.nan),
            "area_under_the_curve_unsign": d.get("area_under_the_curve_unsign", np.nan), # so the AUC for the ensemble wave
            "area_under_the_curve_sign":     d.get("area_under_the_curve_sign",     np.nan),
        })
    df = pd.DataFrame(rows)
    for c in PREDICTORS_CAT:
        df[c] = df[c].astype("category")
    return df

# ---------- univariate scatter ----------

def _scatter(ax, x, y, xl):
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 20}, line_kws={"lw": 1})
    if x.notna().any() and y.notna().any():
        r, p = stats.pearsonr(x, y)
        ax.set_title(f"r={r:.2f}\np={p:.1e}")
    ax.set_xlabel(xl)
    ax.set_ylabel("rise-time [ms]")


def univariate_plots(
        df: pd.DataFrame, 
        variable_to_predict: str = "rise_time_ms",
        tag: str | None = None):
    """Create univariate scatter plots for each continuous predictor."""
    cols = PREDICTORS_CONT
    rows = int(np.ceil(len(cols)/3))
    fig, axs = plt.subplots(rows, 3, figsize=(14, 4*rows))
    axs = axs.ravel()
    for ax, col in zip(axs, cols):
        _scatter(ax, df[col], df[variable_to_predict], col)
    for ax in axs[len(cols):]:
        ax.axis("off")

    fig.suptitle(f"Univariate scatter plots for {_get_variable_name_alias(variable_to_predict)}", fontsize=16)

    fig.tight_layout()
    for ext in IMAGE_FORMATS:
        fig.savefig(FIG_DIR / f"univariate_corr{f'_{tag}' if tag else ''}.{ext}")
    plt.close(fig)

# ---------- corr heat-map ----------

def correlation_heatmap(
        df: pd.DataFrame, 
        tag: str | None = None):
    
    corr = df[PREDICTORS_CONT + ["rise_time_ms", "area_under_the_curve_sign"]].corr()
    mask = np.triu(np.ones_like(corr, bool), k=1) # remove k=1 to remove the diagnoal values 
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

    # draw a double separator between ‘bmi’ (col-6) and ‘rise_time_ms’ (col-7)
    sep = corr.columns.get_loc("rise_time_ms")       # position *between* the two vars
    ax.axvline(sep, color="black", lw=1)  # vertical lines
    ax.axhline(sep, color="black", lw=1)  # horizontal lines

    # relabel all ticks with dictionary
    tick_labels = {
        "age": "Age\n[years]",
        "bmi": "BMI\n[kg/m²]",
        "height_m": "Height\n[m]",
        "weight_kg": "Weight\n[kg]",
        "average_hr": "Average HR\n[bpm]",
        "baseline_sbp": "SBP\n[mmHg]",
        "baseline_dbp": "DBP\n[mmHg]",
        "rise_time_ms": "Rise-time\n[ms]",
        "rise_time_norm": "Rise-time\n(norm.)",
        "area_under_the_curve_sign": "AUC\n[signed]",
    }
    
    ax.set_xticklabels([tick_labels[label._text] for label in ax.get_xticklabels()])
    ax.set_yticklabels([tick_labels[label._text] for label in ax.get_yticklabels()])

    fig.tight_layout()
    for ext in IMAGE_FORMATS:
        fig.savefig(FIG_DIR / f"corr_heatmap{f'_{tag}' if tag else ''}.{ext}")
    plt.close(fig)

# ---------- histogram ----------

def rise_time_hist(
        df: pd.DataFrame,
        tag: str | None = None):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.histplot(df["rise_time_ms"].dropna(), bins=30, ax=ax)
    ax.set_xlabel("rise-time [ms]")
    fig.tight_layout()
    for ext in IMAGE_FORMATS:
        fig.savefig(FIG_DIR / f"rise_time_hist{f'_{tag}' if tag else ''}.{ext}")
    plt.close(fig)

# ---------- linear regression (sklearn + statsmodels) ----------
def multivariate_regression(
    df: pd.DataFrame,
    cont_vars: list[str],
    cat_vars:  list[str],
    variable_to_predict: str = "rise_time_ms",
    tag: str | None = None, # appended to filenames
) -> None:
    # --- design matrix ----------------------------------------------------
    X_num = df[cont_vars].values
    if cat_vars:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = ohe.fit_transform(df[cat_vars])
        X = np.hstack([X_num, X_cat])
        feature_names = cont_vars + ohe.get_feature_names_out(cat_vars).tolist()
    else:
        X = X_num
        feature_names = cont_vars

    # --- remove nans from X and y ----------------------------- (only important due to fitzpatrick_scale)
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(df[variable_to_predict])
    X = X[mask]
    df = df[mask]
    print(f"There were {np.any(mask)} rows with NaNs in X or y, which were removed.")

    # --- scale ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df[variable_to_predict].values

    # sklearn (for coefficient CSV) ---------------------------------------
    lin = LinearRegression().fit(X_scaled, y)
    r2 = lin.score(X_scaled, y)

    coef_df = pd.DataFrame({"feature": feature_names, "coef": lin.coef_})
    coef_df["abs"] = coef_df["coef"].abs()
    coef_df.sort_values("abs", ascending=False, inplace=True)
    coef_df.drop("abs", axis=1, inplace=True)

    # for saving (everywhere in this function):all file outputs now include the optional tag
    suffix = f"_{tag}" if tag else ""
    coef_df.to_csv(TAB_DIR / f"coefficients_for_{variable_to_predict}{suffix}.csv", index=False)

    with open(TAB_DIR / f"coefficients_for_{variable_to_predict}.tex", "w") as f:
        f.write(coef_df.to_latex(index=False, float_format="%.3f"))

    # scatter actual vs pred ----------------------------------------------
    y_pred = lin.predict(X_scaled)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y, y_pred, s=15, alpha=0.7)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)

    variable_name_alias = _get_variable_name_alias(variable_to_predict)
    ax.set_xlabel("Actual " + variable_name_alias)
    ax.set_ylabel("Predicted " + variable_name_alias)
    ax.set_title(f"Linear model  R²={r2:.3f}")
    fig.tight_layout()
    for ext in IMAGE_FORMATS:
        fig.savefig(FIG_DIR / f"actual_vs_pred_{variable_to_predict}{suffix}.{ext}")
    plt.close(fig)

    # statsmodels OLS (for pretty summary) ---------------------------------
    X_sm = pd.DataFrame(X_scaled, columns=feature_names) # give columns names
    X_sm = sm.add_constant(X_sm, prepend=True)  # add intercept term

    sm_mod = sm.OLS(y, X_sm).fit()

    # save tables
    for fmt in TABLE_FORMATS:
        if fmt == "txt":
            summary = sm_mod.summary().as_text()
            (TAB_DIR / f"ols_summary_{variable_to_predict}{suffix}.txt").write_text(summary)
        elif fmt == "tex":
            summary = sm_mod.summary().as_latex()
            (TAB_DIR / f"ols_summary_{variable_to_predict}{suffix}.tex").write_text(summary)

    logging.info("Multivariate regression finished(R² = %.3f) for variable '%s'", r2, variable_to_predict)


def _get_variable_name_alias(variable_to_predict):
    """Return a human-readable alias for the variable to predict."""
    if variable_to_predict == "rise_time_ms":
        variable_name_alias = "rise-time [ms]"
    elif variable_to_predict == "rise_time_norm":
        variable_name_alias = "rise-time (norm.)"
    elif variable_to_predict == "area_under_the_curve_unsign":
        variable_name_alias = "APG area (unsigned)"
    elif variable_to_predict == "area_under_the_curve_sign":
        variable_name_alias = "APG area (signed)"
    else:
        variable_name_alias = variable_to_predict.replace("_", " ").title()
    return variable_name_alias


# ─────────────────────────────── CLI helpers ───────────────────────────────

def build_parser():
    p = argparse.ArgumentParser("Wave-class labelling & analysis")
    p.add_argument("--dict_path", type=Path, default=DEFAULT_DICT_PATH,
                    help="path to data_dict .pt (default: %(default)s)")
    p.add_argument("--classify", default=True, action="store_true", help="add class labels only")
    p.add_argument("--plot_auc", default=True, action="store_true", # set default=False, later. 
                    help="plot a few waves with their AUC value")
    p.add_argument("--analyse", default=True, action="store_true", help="run statistics/plots")
    p.add_argument("--subset", type=float, default=1.0,
                     help="use random subset of subjects (0<subset≤1)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--inflect_thr", type=float, default=1e-2, # 3,
                     help="2nd-derivative threshold for inflection detection")
    p.add_argument("--reg_set", choices=PREDICTOR_SETS.keys(), nargs="+", default=["core"], # , "all", "core_plus_fitzpatrick"],
                     help="Which predefined predictor set(s) to use. May be given multiple times to run multiple versions, e.g.  --reg_set all core")


    return p

# ────────────────────────────────── main ───────────────────────────────────

def main(argv: List[str] | None = None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)

    # ── load data_dict ───────────────────────────────────────────────
    logging.info("Loading → %s", args.dict_path)
    data_dict: Dict[str, Dict[str, Any]] = torch.load(args.dict_path, weights_only=False)

    if args.subset < 1.0:
        keys = np.random.choice(list(data_dict.keys()),
                                size=int(len(data_dict)*args.subset),
                                replace=False)
        data_dict = {k: data_dict[k] for k in keys}
        logging.info("Subset mode — %d subjects", len(data_dict))

    INFLECTION_THR = args.inflect_thr      # global so helper can read it

    # ── classification ───────────────────────────────────────────────
    if args.classify: 
        logging.info("Classifying pulse waves …")
        need_classes = any("ensemble_class" not in e for e in data_dict.values())
        if need_classes:
            logging.info("Labelling pulse waves …")
            for e in data_dict.values():
                label_entry(e, threshold=INFLECTION_THR)
            out_pt = args.dict_path.with_name(args.dict_path.stem + "_with_classes.pt")
            torch.save(data_dict, out_pt)
            logging.info("Classes written to %s", out_pt)
        elif args.classify and not args.analyse:
            logging.info("Classes already present – nothing to do.")

    # ── analysis ───────────────────────────────────────────────
    if args.analyse:
        logging.info("Running analysis on %d entries", len(data_dict))

        df = _make_dataframe(data_dict)
        logging.info("DataFrame created with %d rows", len(df))

        for set_name in args.reg_set:
            univariate_plots(df, variable_to_predict="rise_time_ms", tag=set_name)
            univariate_plots(df, variable_to_predict="area_under_the_curve_sign", tag=set_name)

            correlation_heatmap(df, tag=set_name)
            rise_time_hist(df, tag=set_name)

            pred_cont = PREDICTOR_SETS[set_name]["cont"]
            pred_cat  = PREDICTOR_SETS[set_name]["cat"]

            multivariate_regression(df, pred_cont, pred_cat,
                                    variable_to_predict="rise_time_ms",
                                    tag=set_name)

            multivariate_regression(df, pred_cont, pred_cat,
                                    variable_to_predict="area_under_the_curve_sign",
                                    tag=set_name)

        logging.info("Figures → %s | Tables → %s", FIG_DIR, TAB_DIR)


if __name__ == "__main__":
    sys.exit(main())
