# #!/usr/bin/env python
# """
# AURORA | MAUS – wave-class labelling & regression analysis
# =========================================================

# This self-contained CLI script extends the existing preprocessing pipeline.
# It can
#   1. **label every individual PW** (plus the ensemble beat) with one of the
#      five morphology classes from Piltz *et al.* (decision-tree attached in
#      the thesis draft),
#   2. **run the full statistical analysis** bundle (univariate plots,
#      correlation matrix, multivariate linear regression, rise-time histogram),
#   3. produce manuscript-ready figures (PNG & PDF) and a LaTeX table of
#      regression coefficients.

# The script is *read-only* w.r.t. the heavy *.pt* files – results are stored as
# adjacent *.npy/CSV/PNG/PDF/tex* files so that it can be re-run without touching
# raw data.

# Usage examples
# --------------
# $ python aurora_analysis.py                                   # full pipeline (default paths)
# $ python aurora_analysis.py --classify                       # only add wave classes
# $ python aurora_analysis.py --analyse --subset 0.25          # quick 25 % sample run
# $ python aurora_analysis.py --dict_path /data/data_dict.pt   # custom dict

# Dependencies: numpy, scipy, pandas, matplotlib, seaborn, scikit-learn,
#               statsmodels, torch.
# """
# from __future__ import annotations

# import argparse
# import logging
# import sys
# from pathlib import Path
# from typing import Any, Dict, List, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from scipy import signal, stats
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# import matplotlib.pyplot as plt
# import seaborn as sns


# # ────────────────────────────────────────────────────────────────────────────
# # Configuration constants – tweak here if your folder layout differs
# # ────────────────────────────────────────────────────────────────────────────
# # /Users/adrian/Documents/01_projects/02_clean_ppg/data/AURORA/preprocessed/data_dict_osc_auc_with_derivatives.pt
# DEFAULT_DICT_PATH = "/Users/adrian/Documents/01_projects/02_clean_ppg/data/AURORA/preprocessed/data_dict_osc_auc_with_derivatives.pt" # Path(__file__).resolve().with_name("data_dict_osc_auc_with_derivatives.pt")
# OUT_DIR = Path(__file__).resolve().with_name("out")
# FIG_DIR = OUT_DIR / "figures"
# TAB_DIR = OUT_DIR / "tables"
# FIG_DIR.mkdir(parents=True, exist_ok=True)
# TAB_DIR.mkdir(parents=True, exist_ok=True)

# # Predictors to use (order preserved for tables)
# PREDICTORS_CONT = [
#     "age", "baseline_sbp", "baseline_dbp", "height_m", "weight_kg",
#     "average_hr", "bmi",
# ]
# PREDICTORS_CAT = ["gender", "cvd_meds", "fitzpatrick_scale",
#                   "pressure_quality", "optical_quality", "oscillo_or_auscul"]
# TARGETS = ["rise_time_ms", "rise_time_norm"]  # ensemble_class handled separately

# # ────────────────────────────────────────────────────────────────────────────
# # Wave-shape classification helpers
# # ────────────────────────────────────────────────────────────────────────────

# def _smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
#     """Savitzky–Golay smoothing (window & poly can be overridden).
#     Falls back to original *y* if window > len(y)."""
#     if len(y) < window:
#         return y
#     return signal.savgol_filter(y, window_length=window, polyorder=poly)


# def _first_and_second_derivatives(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     d1 = np.gradient(y)
#     d2 = np.gradient(d1)
#     return d1, d2


# def _inflection_indices(d2: np.ndarray, *, threshold: float = 1e-4) -> List[int]:
#     """Return indices where 2nd-derivative crosses zero (±threshold)."""
#     signs = np.sign(d2)
#     crossings = np.where(np.diff(signs))[0]
#     # keep only those where magnitude around zero is small
#     return [i for i in crossings if abs(d2[i]) < threshold or abs(d2[i+1]) < threshold]


# def classify_wave(wave: np.ndarray) -> int:
#     """Implements the decision tree from the attached diagram.  Returns class 1-5."""
#     if np.all(np.isnan(wave)):
#         return 0  # 0 = unknown / invalid

#     y = _smooth(wave)
#     # Local maxima (strict)
#     peaks, _ = signal.find_peaks(y, distance=len(y)//10)  # avoid spurious close peaks
#     if len(peaks) >= 2:
#         # Keep two highest peaks (by amplitude). Sort by time (index) afterwards.
#         top2 = peaks[np.argsort(y[peaks])][-2:]
#         top2 = top2[np.argsort(top2)]
#         first, second = top2
#         return 1 if y[first] > y[second] else 5

#     # ==> len(peaks) == 1 (or 0)  – use inflection logic
#     peak = int(peaks[0]) if len(peaks) else int(np.argmax(y))
#     _, d2 = _first_and_second_derivatives(y)
#     infl = _inflection_indices(d2)
#     if not infl:
#         return 3  # no inflection ⇒ class 3 per flow-chart
#     before = [i for i in infl if i < peak]
#     after  = [i for i in infl if i > peak]
#     if before:
#         return 4
#     if after:
#         return 2
#     return 3  # fallback


# def label_entry(entry: Dict[str, Any]) -> None:
#     """Add individual_waves_classes and ensemble_class fields in-place."""
#     if "individual_waves_classes" in entry and "ensemble_class" in entry:
#         return  # already done – skip

#     waves = entry.get("individual_waves", [])
#     classes = np.array([classify_wave(w) for w in waves], dtype=int)
#     entry["individual_waves_classes"] = classes
#     entry["ensemble_class"] = int(classify_wave(entry.get("ensemble_wave", np.empty(0))))

# # ────────────────────────────────────────────────────────────────────────────
# # Regression & plotting helpers
# # ────────────────────────────────────────────────────────────────────────────

# def _make_dataframe(data_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
#     rows = []
#     for pid, d in data_dict.items():
#         if not d.get("rise_times_ms"):
#             continue  # skip subjects without waves
#         rows.append({
#             "pid": pid,
#             **{k: d.get(k, np.nan) for k in PREDICTORS_CONT + PREDICTORS_CAT},
#             "rise_time_ms": d.get("average_rise_time_ms", np.nan),
#             "rise_time_norm": d.get("average_rise_time_norm", np.nan),
#             "ensemble_class": d.get("ensemble_class", np.nan),
#         })
#     df = pd.DataFrame(rows)
#     # gender etc. may be non-numeric, force categorical dtype
#     for c in PREDICTORS_CAT:
#         df[c] = df[c].astype("category")
#     return df


# def _scatter_and_corr(ax, x, y, xlabel, ylabel):
#     sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 20}, line_kws={"lw": 1})
#     r, p = stats.pearsonr(x.dropna(), y.dropna()) if x.notna().any() else (np.nan, np.nan)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_title(f"r = {r:.2f}, p = {p:.3e}")


# def univariate_plots(df: pd.DataFrame):
#     fig, axs = plt.subplots(2, 3, figsize=(13, 8))
#     axs = axs.ravel()
#     for ax, col in zip(axs, PREDICTORS_CONT[:6]):
#         _scatter_and_corr(ax, df[col], df["rise_time_ms"], col, "rise-time [ms]")
#     fig.tight_layout()
#     fig.suptitle("Univariate correlations (first 6 predictors)", y=1.02)
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"univariate_corr1.{ext}", bbox_inches="tight")
#     plt.close(fig)

#     # second panel (remaining predictors)
#     rest = PREDICTORS_CONT[6:]
#     if rest:
#         fig, axs = plt.subplots(1, len(rest), figsize=(5*len(rest), 4))
#         if len(rest) == 1:
#             axs = [axs]
#         for ax, col in zip(axs, rest):
#             _scatter_and_corr(ax, df[col], df["rise_time_ms"], col, "rise-time [ms]")
#         fig.tight_layout()
#         fig.suptitle("Univariate correlations (remaining)", y=1.05)
#         for ext in ("png", "pdf"):
#             fig.savefig(FIG_DIR / f"univariate_corr2.{ext}", bbox_inches="tight")
#         plt.close(fig)


# def correlation_heatmap(df: pd.DataFrame):
#     corr = df[PREDICTORS_CONT + ["rise_time_ms", "rise_time_norm"]].corr()
#     mask = np.triu(np.ones_like(corr, dtype=bool))
#     fig, ax = plt.subplots(figsize=(8, 8))
#     sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#     fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"corr_heatmap.{ext}")
#     plt.close(fig)


# def rise_time_hist(df: pd.DataFrame):
#     fig, ax = plt.subplots(figsize=(5, 4))
#     sns.histplot(df["rise_time_ms"].dropna(), bins=30, ax=ax, kde=False)
#     ax.set_xlabel("rise-time [ms]")
#     ax.set_ylabel("count")
#     fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"rise_time_hist.{ext}")
#     plt.close(fig)


# def multivariate_regression(df: pd.DataFrame):
#     X_num = df[PREDICTORS_CONT].values
#     # one-hot encode categoricals
#     ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#     X_cat = ohe.fit_transform(df[PREDICTORS_CAT])
#     X = np.hstack([X_num, X_cat])

#     # store column names (continuous + expanded cats)
#     cat_cols = ohe.get_feature_names_out(PREDICTORS_CAT)
#     feature_names = PREDICTORS_CONT + cat_cols.tolist()

#     y = df["rise_time_ms"].values

#     model = Pipeline([
#         ("scale", StandardScaler()),
#         ("linreg", LinearRegression())
#     ])
#     model.fit(X, y)
#     y_pred = model.predict(X)
#     r2 = model.score(X, y)

#     # Coeff table
#     coefs = model.named_steps["linreg"].coef_
#     coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
#     coef_df["abs"] = coef_df["coef"].abs()
#     coef_df.sort_values("abs", ascending=False, inplace=True)
#     coef_df.drop("abs", axis=1, inplace=True)

#     coef_csv = TAB_DIR / "coefficients.csv"
#     coef_tex = TAB_DIR / "coefficients.tex"
#     coef_df.to_csv(coef_csv, index=False)
#     with open(coef_tex, "w") as f:
#         f.write(coef_df.to_latex(index=False, float_format="%.3f"))

#     # Prediction plot
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.scatter(y, y_pred, s=15, alpha=0.7)
#     lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
#     ax.plot(lims, lims, "r--", lw=1)
#     ax.set_xlabel("Actual rise-time [ms]")
#     ax.set_ylabel("Predicted rise-time [ms]")
#     ax.set_title(f"Linear model  R² = {r2:.3f}")
#     fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"actual_vs_pred.{ext}")
#     plt.close(fig)

#     logging.info("Multivariate regression done – R² = %.3f", r2)

# # ────────────────────────────────────────────────────────────────────────────
# # CLI & entry point
# # ────────────────────────────────────────────────────────────────────────────

# def build_arg_parser():
#     p = argparse.ArgumentParser("Wave-class labelling & analysis")
#     p.add_argument("--dict_path", type=Path, default=DEFAULT_DICT_PATH,
#                    help="path to the data_dict *.pt file (default: %(default)s)")
#     p.add_argument("--classify", action="store_true", help="label waves only and exit")
#     p.add_argument("--analyse", action="store_true", help="run full statistics/plots")
#     p.add_argument("--subset", type=float, default=1.0,
#                    help="use only a random subset of subjects (0<subset≤1)")
#     p.add_argument("--verbose", action="store_true")
#     return p


# def main(argv: List[str] | None = None):
#     args = build_arg_parser().parse_args(argv)
#     logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
#                         format="%(levelname)s:%(message)s")

#     logging.info("Loading dictionary → %s", args.dict_path)
#     data_dict: Dict[str, Dict[str, Any]] = torch.load(args.dict_path, weights_only=False)

#     if args.subset < 1.0:
#         keys = list(data_dict.keys())
#         keep = int(len(keys) * args.subset)
#         sel = set(np.random.choice(keys, keep, replace=False))
#         data_dict = {k: v for k, v in data_dict.items() if k in sel}
#         logging.info("Subset mode → kept %d/%d subjects", keep, len(keys))

#     # ------------------------------------------------------------------
#     # A.  Wave classification
#     # ------------------------------------------------------------------
#     work_needed = any("ensemble_class" not in e for e in data_dict.values())
#     if work_needed:
#         logging.info("Classifying waves ...")
#         for entry in data_dict.values():
#             label_entry(entry)
#         # write classes next to original dict (no overwrite)
#         out = args.dict_path.with_name(args.dict_path.stem + "_with_classes.pt")
#         torch.save(data_dict, out)
#         logging.info("Classes saved into %s", out)
#     else:
#         logging.info("All entries already have classes – skipping classification step")

#     if args.classify and not args.analyse:
#         return  # user only wanted classification

#     # ------------------------------------------------------------------
#     # B.  Regression & plots
#     # ------------------------------------------------------------------
#     logging.info("Building DataFrame for analysis ...")
#     df = _make_dataframe(data_dict)
#     logging.info("%d subjects with all required fields", len(df))

#     logging.info("Generating univariate plots ...")
#     univariate_plots(df)

#     logging.info("Correlation heat-map ...")
#     correlation_heatmap(df)

#     logging.info("Rise-time histogram ...")
#     rise_time_hist(df)

#     logging.info("Multivariate regression ...")
#     multivariate_regression(df)

#     logging.info("All done – figures → %s, tables → %s", FIG_DIR, TAB_DIR)


# if __name__ == "__main__":
#     sys.exit(main())


#!/usr/bin/env python
"""
AURORA | MAUS – wave-class labelling & regression analysis
=========================================================

**What this script does**
-------------------------
1. **Classify every pulse wave** according to the five-class decision tree in the
   thesis (adds *individual_waves_classes* & *ensemble_class* to the dict).
2. **Run statistics & create figures**
   * univariate scatter/OLS line plots
   * correlation heat-map
   * rise-time histogram
   * multivariate *and* statsmodels OLS regression (z-normalised inputs)
3. **Write outputs**
   * figures → `out/figures/*.png` & `*.pdf`
   * coefficient CSV (`out/tables/coefficients.csv`)
   * pretty LaTeX table of coefficients (`out/tables/coefficients.tex`)
   * full statsmodels summary as LaTeX (`out/tables/ols_summary.tex`) *and* text
   * updated data_dict with classes → `*_with_classes.pt`

Quick start
-----------
```bash
python aurora_analysis.py                 # full pipeline (detect dict automatically)
python aurora_analysis.py --classify      # only label waves
python aurora_analysis.py --analyse       # only figures/stats (dict already has classes)
```

Dependencies: numpy, scipy, pandas, matplotlib, seaborn, scikit-learn,
statsmodels, torch.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
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

# ───────────────────────────── configuration ──────────────────────────────
# DEFAULT_DICT_PATH = Path(__file__).resolve().with_name("data_dict_osc_auc_with_derivatives.pt")
DEFAULT_DICT_PATH = "/Users/adrian/Documents/01_projects/02_clean_ppg/data/AURORA/preprocessed/data_dict_osc_auc_with_derivatives.pt"
OUT_DIR   = Path(__file__).resolve().with_name("out")
FIG_DIR   = OUT_DIR / "figures"
TAB_DIR   = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

PREDICTORS_CONT = [
    "age", "baseline_sbp", "baseline_dbp", "height_m",
    "weight_kg", "average_hr", "bmi",
]
PREDICTORS_CAT  = [
    "gender", "cvd_meds", "fitzpatrick_scale",
    "pressure_quality", "optical_quality", "oscillo_or_auscul",
]
TARGETS = ["rise_time_ms", "rise_time_norm"]

# ─────────────────────── wave-shape classification helpers ─────────────────

def _smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """Savitzky–Golay smoothing with fallback for short waves."""
    return y if len(y) < window else signal.savgol_filter(y, window, poly)


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
    waves  = entry.get("individual_waves", [])
    classes = np.array([classify_wave(w, threshold) for w in waves], dtype=np.int8)
    entry["individual_waves_classes"] = classes
    entry["ensemble_class"] = int(classify_wave(entry.get("ensemble_wave", np.empty(0)), threshold=threshold))

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


def univariate_plots(df: pd.DataFrame):
    cols = PREDICTORS_CONT
    rows = int(np.ceil(len(cols)/3))
    fig, axs = plt.subplots(rows, 3, figsize=(14, 4*rows))
    axs = axs.ravel()
    for ax, col in zip(axs, cols):
        _scatter(ax, df[col], df["rise_time_ms"], col)
    for ax in axs[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"univariate_corr.{ext}")
    plt.close(fig)

# ---------- corr heat-map ----------

def correlation_heatmap(df):
    corr = df[PREDICTORS_CONT + ["rise_time_ms", "rise_time_norm"]].corr()
    mask = np.triu(np.ones_like(corr, bool))
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"corr_heatmap.{ext}")
    plt.close(fig)

# ---------- histogram ----------

def rise_time_hist(df):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.histplot(df["rise_time_ms"].dropna(), bins=30, ax=ax)
    ax.set_xlabel("rise-time [ms]")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"rise_time_hist.{ext}")
    plt.close(fig)

# ---------- linear regression (sklearn + statsmodels) ----------

def multivariate_regression(df: pd.DataFrame):
    # --- design matrix ----------------------------------------------------
    X_num = df[PREDICTORS_CONT].values
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = ohe.fit_transform(df[PREDICTORS_CAT])
    X = np.hstack([X_num, X_cat])
    feature_names = PREDICTORS_CONT + ohe.get_feature_names_out(PREDICTORS_CAT).tolist()

    # --- scale ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df["rise_time_ms"].values

    # sklearn (for coefficient CSV) ---------------------------------------
    lin = LinearRegression().fit(X_scaled, y)
    r2 = lin.score(X_scaled, y)

    coef_df = pd.DataFrame({"feature": feature_names, "coef": lin.coef_})
    coef_df["abs"] = coef_df["coef"].abs()
    coef_df.sort_values("abs", ascending=False, inplace=True)
    coef_df.drop("abs", axis=1, inplace=True)
    coef_df.to_csv(TAB_DIR / "coefficients.csv", index=False)
    with open(TAB_DIR / "coefficients.tex", "w") as f:
        f.write(coef_df.to_latex(index=False, float_format="%.3f"))

    # scatter actual vs pred ----------------------------------------------
    y_pred = lin.predict(X_scaled)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(y, y_pred, s=15, alpha=0.7)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual rise-time [ms]")
    ax.set_ylabel("Predicted rise-time [ms]")
    ax.set_title(f"Linear model  R²={r2:.3f}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"actual_vs_pred.{ext}")
    plt.close(fig)

    # statsmodels OLS (for pretty summary) ---------------------------------
    X_sm = sm.add_constant(X_scaled)
    sm_mod = sm.OLS(y, X_sm).fit()
    summary_txt = sm_mod.summary().as_text()
    summary_tex = sm_mod.summary().as_latex()
    (TAB_DIR / "ols_summary.txt").write_text(summary_txt)
    (TAB_DIR / "ols_summary.tex").write_text(summary_tex)

    logging.info("Multivariate regression finished (R² = %.3f).", r2)

# def _inflection_before_after(y: np.ndarray, peak_idx: int,
#                              threshold: float) -> tuple[bool, bool]:
#     d2 = np.gradient(np.gradient(y))
#     zc = np.where(np.diff(np.sign(d2)))[0]
#     before = any(abs(d2[i]) < threshold and i <  peak_idx for i in zc)
#     after  = any(abs(d2[i]) < threshold and i >  peak_idx for i in zc)
#     return before, after

# FIX CLASS DISTRIBUTIONS; GET PLOTS AND LATEX TABLES!

# ─────────────────────────────── CLI helpers ───────────────────────────────

def build_parser():
    p = argparse.ArgumentParser("Wave-class labelling & analysis")
    p.add_argument("--dict_path", type=Path, default=DEFAULT_DICT_PATH,
                   help="path to data_dict .pt (default: %(default)s)")
    p.add_argument("--classify", action="store_true", help="add class labels only")
    p.add_argument("--analyse", action="store_true", help="run statistics/plots")
    p.add_argument("--subset", type=float, default=1.0,
                   help="use random subset of subjects (0<subset≤1)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--inflect_thr", type=float, default=1e-2, # 3,
               help="2nd-derivative threshold for inflection detection")

    return p

# ────────────────────────────────── main ───────────────────────────────────

def main(argv: List[str] | None = None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)

    logging.info("Loading → %s", args.dict_path)
    data_dict: Dict[str, Dict[str, Any]] = torch.load(args.dict_path, weights_only=False)

    if args.subset < 1.0:
        keys = np.random.choice(list(data_dict.keys()),
                                size=int(len(data_dict)*args.subset),
                                replace=False)
        data_dict = {k: data_dict[k] for k in keys}
        logging.info("Subset mode — %d subjects", len(data_dict))

    INFLECTION_THR = args.inflect_thr      # global so helper can read it

    # ── classification ──
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
        return

    if not args.analyse:
        return  # user wanted classification only

    # ── analysis ──
    df = _make_dataframe(data_dict)
    logging.info("DataFrame created with %d rows", len(df))

    univariate_plots(df)
    correlation_heatmap(df)
    rise_time_hist(df)
    multivariate_regression(df)

    logging.info("Figures → %s | Tables → %s", FIG_DIR, TAB_DIR)


if __name__ == "__main__":
    sys.exit(main())
