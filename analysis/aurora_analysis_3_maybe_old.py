#!/usr/bin/env python
"""
AURORA | MAUS – wave-class labelling & regression analysis
=========================================================

**What this script does**
-------------------------
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
DEFAULT_DICT_PATH = "/Users/adrian/Documents/01_projects/02_clean_ppg/data/AURORA/preprocessed/data_dict_osc_auc_with_derivatives_with_classes.pt" # Path(__file__).resolve().with_name("data_dict_osc_auc_with_derivatives.pt")
OUT_DIR   = Path(__file__).resolve().with_name("out")
FIG_DIR   = OUT_DIR / "figures_antik"
TAB_DIR   = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

PREDICTORS_CONT = [
    "age", "baseline_sbp", "baseline_dbp", "height_m",
    # "weight_kg",
    "average_hr", "bmi",  # "fitzpatrick_scale", "pressure_quality", "optical_quality",
]
PREDICTORS_CAT  = [
    # "gender", "cvd_meds", "oscillo_or_auscul",
]
TARGETS = ["rise_time_ms", "rise_time_norm"]

# human-friendly labels for plotting
ALIASES = {
    "age": "age [years]",
    "baseline_sbp": "SBP [mmHg]",
    "baseline_dbp": "DBP [mmHg]",
    "height_m": "height [m]",
    "weight_kg": "weight [kg]",
    "average_hr": "heart rate [bpm]",
    "bmi": "BMI [kg/m^2]",
    "rise_time_ms": "rise-time [ms]",
    "rise_time_norm": "rise-time (norm)",
    # add more as needed
}

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

def _scatter(ax, x, y, xl):
    # perform linear regression
    mask = x.notna() & y.notna()
    slope, intercept, r_val, p_val, _ = stats.linregress(x[mask], y[mask])
    # scatter and regression line with legend
    ax.scatter(x, y, s=20, label='Data points')
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope*xs + intercept, 'r-', label=f'Line: y={slope:.4f}x+{intercept:.4f}')
    ax.legend(loc='upper right', fontsize='small')
    # annotate Pearson r & p-value
    signif = 'significant' if p_val < 0.05 else 'not significant'
    txt = f"Pearson's r: {r_val:.2f} \nP-value: {p_val:.2e}\n({signif})"
    ax.text(0.95, 0.05, txt, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray'))
    ax.set_xlabel(xl)
    ax.set_ylabel("rise-time [ms]")



def univariate_plots(df: pd.DataFrame):
    cols = PREDICTORS_CONT
    rows = int(np.ceil(len(cols)/3))
    fig, axs = plt.subplots(rows, 3, figsize=(14, 4*rows))
    axs = axs.ravel()
    for ax, col in zip(axs, cols):
        # _scatter(ax, df[col], df["rise_time_ms"], col)
        _scatter(ax, df[col], df["rise_time_ms"], ALIASES.get(col, col))
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
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

    # Aliases for labels
    ax.set_xticklabels([ALIASES.get(c, c) for c in corr.columns], rotation=45, ha='right')
    ax.set_yticklabels([ALIASES.get(c, c) for c in corr.index], rotation=0)

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
    y = df["rise_time_ms"] # .values

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
    FONTSIZE = 16
    ax.set_xlabel("Actual rise-time [ms]", fontsize=FONTSIZE)
    ax.set_ylabel("Predicted rise-time [ms]", fontsize=FONTSIZE)
    ax.set_title(f"Linear model  R²={r2:.3f}", fontsize=FONTSIZE)
    # Set tick labels to be larger
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE-2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"actual_vs_pred.{ext}")
    plt.close(fig)

    # statsmodels OLS (for pretty summary) ---------------------------------
    # X_sm = sm.add_constant(X_scaled)
    # sm_mod = sm.OLS(y, X_sm).fit()
    # summary_txt = sm_mod.summary().as_text()
    # summary_tex = sm_mod.summary().as_latex()
    # (TAB_DIR / "ols_summary.txt").write_text(summary_txt)
    # (TAB_DIR / "ols_summary.tex").write_text(summary_tex)

    # logging.info("Multivariate regression finished (R² = %.3f).", r2)

    # --- statsmodels OLS with **named columns** -------------------------------
    X_df = pd.DataFrame(X_scaled, columns=feature_names)  # attach names
    X_df = sm.add_constant(X_df, prepend=True)            # adds ‘const’ column
    sm_mod = sm.OLS(y, X_df).fit()

    summary_txt = sm_mod.summary().as_text()
    summary_tex = sm_mod.summary().as_latex()
    (TAB_DIR / "ols_summary.txt").write_text(summary_txt)
    (TAB_DIR / "ols_summary.tex").write_text(summary_tex)


# ─────────────────────────────── CLI helpers ───────────────────────────────

def build_parser():
    p = argparse.ArgumentParser("Wave-class labelling & analysis")
    p.add_argument("--dict_path", type=Path, default=DEFAULT_DICT_PATH,
                   help="path to data_dict .pt (default: %(default)s)")
    p.add_argument("--analyse", action="store_true", help="run statistics/plots")
    p.add_argument("--subset", type=float, default=1.0,
                   help="use random subset of subjects (0<subset≤1)")
    p.add_argument("--verbose", action="store_true")
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




# #########


# #!/usr/bin/env python
# """
# AURORA | MAUS – wave‑class labelling & regression analysis
# =========================================================

# **What this script does**
# -------------------------
# 1. **Classify every pulse wave** according to the five‑class decision tree in the
#    thesis (adds *individual_waves_classes* & *ensemble_class* to the dict).
# 2. **Run statistics & create figures**
#    * univariate scatter/OLS line plots
#    * correlation heat‑map
#    * rise‑time histogram
#    * multivariate *and* statsmodels OLS regression (z‑normalised inputs)
# 3. **Write outputs**
#    * figures → `out/figures/*.png` & `*.pdf`
#    * coefficient CSV (`out/tables/coefficients.csv`)
#    * pretty LaTeX table of coefficients (`out/tables/coefficients.tex`)
#    * full statsmodels summary as LaTeX (`out/tables/ols_summary.tex`) *and* text
#    * updated data_dict with classes → `*_with_classes.pt`

# Quick start
# -----------
# ```bash
# python aurora_analysis.py                 # full pipeline (detect dict automatically)
# python aurora_analysis.py --classify      # only label waves
# python aurora_analysis.py --analyse       # only figures/stats (dict already has classes)
# ```

# Dependencies: numpy, scipy, pandas, matplotlib, seaborn, scikit‑learn,
# statsmodels, torch.
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
# import statsmodels.api as sm

# # ───────────────────────────── configuration ──────────────────────────────
# DEFAULT_DICT_PATH = Path(__file__).resolve().with_name("data_dict_osc_auc_with_derivatives.pt")
# OUT_DIR   = Path(__file__).resolve().with_name("out")
# FIG_DIR   = OUT_DIR / "figures"
# TAB_DIR   = OUT_DIR / "tables"
# FIG_DIR.mkdir(parents=True, exist_ok=True)
# TAB_DIR.mkdir(parents=True, exist_ok=True)

# PREDICTORS_CONT = [
#     "age", "baseline_sbp", "baseline_dbp", "height_m",
#     "weight_kg", "average_hr", "bmi",
# ]

# # human-friendly labels for plotting
# ALIASES = {
#     "age": "age [years]",
#     "baseline_sbp": "SBP [mmHg]",
#     "baseline_dbp": "DBP [mmHg]",
#     "height_m": "height [m]",
#     "weight_kg": "weight [kg]",
#     "average_hr": "heart rate [bpm]",
#     "bmi": "BMI [kg/m^2]",
#     "rise_time_ms": "rise-time [ms]",
#     "rise_time_norm": "rise-time (norm)",
#     # add more as needed
# ]
# PREDICTORS_CAT  = [
#     "gender", "cvd_meds", "fitzpatrick_scale",
#     "pressure_quality", "optical_quality", "oscillo_or_auscul",
# ]
# TARGETS = ["rise_time_ms", "rise_time_norm"]

# # ─────────────────────── wave‑shape classification helpers ─────────────────

# def _smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
#     """Savitzky–Golay smoothing with fallback for short waves."""
#     return y if len(y) < window else signal.savgol_filter(y, window, poly)


# def _inflection_before_after(y: np.ndarray, peak_idx: int) -> Tuple[bool, bool]:
#     """Detect whether *a* zero‑crossing of the 2nd derivative exists **before**
#     and/or **after** the main peak.  A small hysteresis avoids noise."""
#     d2 = np.gradient(np.gradient(y))
#     # hysteresis: ignore crossings where |d2| > thr on either side
#     thr = 1e-3
#     zc = np.where(np.diff(np.sign(d2)))[0]
#     before = any((abs(d2[i]) < thr or abs(d2[i+1]) < thr) and i < peak_idx for i in zc)
#     after  = any((abs(d2[i]) < thr or abs(d2[i+1]) < thr) and i > peak_idx for i in zc)
#     return before, after


# def classify_wave(wave: np.ndarray) -> int:
#     """Return class **1‑5** (0 = invalid) following the decision tree."""
#     if wave.size == 0 or np.all(np.isnan(wave)):
#         return 0

#     y = _smooth(wave)
#     peaks, _ = signal.find_peaks(y, distance=len(y)//10)

#     # ── branch: two local maxima ──────────────────────────────────────────
#     if len(peaks) >= 2:
#         top2 = peaks[np.argsort(y[peaks])][-2:]
#         top2.sort()
#         first, second = top2
#         return 1 if y[first] > y[second] else 5

#     # ── branch: one local maximum ─────────────────────────────────────────
#     peak = int(peaks[0]) if len(peaks) else int(np.argmax(y))
#     before, after = _inflection_before_after(y, peak)
#     if before:
#         return 4  # inflection before max → Class 4
#     if after:
#         return 2  # inflection after max  → Class 2
#     return 3      # no inflection         → Class 3


# def label_entry(entry: Dict[str, Any]) -> None:
#     if "ensemble_class" in entry:
#         return
#     waves  = entry.get("individual_waves", [])
#     classes = np.array([classify_wave(w) for w in waves], dtype=np.int8)
#     entry["individual_waves_classes"] = classes
#     entry["ensemble_class"] = int(classify_wave(entry.get("ensemble_wave", np.empty(0))))

# # ────────────────────────── dataframe & plotting helpers ───────────────────

# def _make_dataframe(data_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
#     rows = []
#     for pid, d in data_dict.items():
#         if "average_rise_time_ms" not in d:
#             continue
#         rows.append({
#             "pid": pid,
#             **{k: d.get(k, np.nan) for k in PREDICTORS_CONT + PREDICTORS_CAT},
#             "rise_time_ms": d.get("average_rise_time_ms", np.nan),
#             "rise_time_norm": d.get("average_rise_time_norm", np.nan),
#             "ensemble_class": d.get("ensemble_class", np.nan),
#         })
#     df = pd.DataFrame(rows)
#     for c in PREDICTORS_CAT:
#         df[c] = df[c].astype("category")
#     return df

# # ---------- univariate scatter ----------

# def _scatter(ax, x, y, xl):
#     sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 20}, line_kws={"lw": 1})
#     if x.notna().any() and y.notna().any():
#         r, p = stats.pearsonr(x, y)
#         ax.set_title(f"r={r:.2f}\np={p:.1e}")
#     ax.set_xlabel(xl)
#     ax.set_ylabel("rise‑time [ms]")


# def univariate_plots(df: pd.DataFrame):
#     cols = PREDICTORS_CONT
#     rows = int(np.ceil(len(cols)/3))
#     fig, axs = plt.subplots(rows, 3, figsize=(14, 4*rows))
#     axs = axs.ravel()
#     for ax, col in zip(axs, cols):
#         _scatter(ax, df[col], df["rise_time_ms"], ALIASES.get(col, col))
#     for ax in axs[len(cols):]:
#         ax.axis("off")
#     fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"univariate_corr.{ext}")
#     plt.close(fig)

# # ---------- corr heat‑map ----------

# def correlation_heatmap(df):
#     cols = PREDICTORS_CONT + ["rise_time_ms", "rise_time_norm"]
#     corr = df[cols].corr()
#     mask = np.triu(np.ones_like(corr, bool))
#     fig, ax = plt.subplots(figsize=(8,8))
#     sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
#     # rename tick labels
#     ax.set_xticklabels([ALIASES.get(c, c) for c in corr.columns], rotation=45, ha='right')
#     ax.set_yticklabels([ALIASES.get(c, c) for c in corr.index], rotation=0)
#     fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"corr_heatmap.{ext}")
#     plt.close(fig)

# # ---------- histogram ----------

# def rise_time_hist(df):
#     fig, ax = plt.subplots(figsize=(5,4))
#     sns.histplot(df["rise_time_ms"].dropna(), bins=30, ax=ax)
#     ax.set_xlabel("rise‑time [ms]")
#     fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"rise_time_hist.{ext}")
#     plt.close(fig)

# # ---------- linear regression (sklearn + statsmodels) ----------

# def multivariate_regression(df: pd.DataFrame):
#     # --- design matrix ----------------------------------------------------
#     X_num = df[PREDICTORS_CONT].values
#     ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#     X_cat = ohe.fit_transform(df[PREDICTORS_CAT])
#     X = np.hstack([X_num, X_cat])
#     feature_names = PREDICTORS_CONT + ohe.get_feature_names_out(PREDICTORS_CAT).tolist()

#     # --- scale ------------------------------------------------------------
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     y = df["rise_time_ms"].values

#     # sklearn (for coefficient CSV) ---------------------------------------
#     lin = LinearRegression().fit(X_scaled, y)
#     r2 = lin.score(X_scaled, y)

#     coef_df = pd.DataFrame({"feature": feature_names, "coef": lin.coef_})
#     coef_df["abs"] = coef_df["coef"].abs()
#     coef_df.sort_values("abs", ascending=False, inplace=True)
#     coef_df.drop("abs", axis=1, inplace=True)
#     coef_df.to_csv(TAB_DIR / "coefficients.csv", index=False)
#     with open(TAB_DIR / "coefficients.tex", "w") as f:
#         f.write(coef_df.to_latex(index=False, float_format="%.3f"))

#     # scatter actual vs pred ----------------------------------------------
#     y_pred = lin.predict(X_scaled)
#     fig, ax = plt.subplots(figsize=(5,5))
#     ax.scatter(y, y_pred, s=15, alpha=0.7)
#     lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
#     ax.plot(lims, lims, "r--", lw=1)
#     ax.set_xlabel("Actual rise‑time [ms]")
#     ax.set_ylabel("Predicted rise‑time [ms]")
#     ax.set_title(f"Linear model  R²={r2:.3f}")
#     fig.tight_layout()
#     for ext in ("png", "pdf"):
#         fig.savefig(FIG_DIR / f"actual_vs_pred.{ext}")
#     plt.close(fig)

#     # statsmodels OLS (for pretty summary) ---------------------------------
#     X_sm = sm.add_constant(X_scaled)
#     sm_mod = sm.OLS(y, X_sm).fit()
#     summary_txt = sm_mod.summary().as_text()
#     summary_tex = sm_mod.summary().as_latex()
#     (TAB_DIR / "ols_summary.txt").write_text(summary_txt)
#     (TAB_DIR / "ols_summary.tex").write_text(summary_tex)

#     logging.info("Multivariate regression finished (R² = %.3f).", r2)

# # ─────────────────────────────── CLI helpers ───────────────────────────────

# def build_parser():
#     p = argparse.ArgumentParser("Wave‑class labelling & analysis")
#     p.add_argument("--dict_path", type=Path, default=DEFAULT_DICT_PATH,
#                    help="path to data_dict .pt (default: %(default)s)")
#     p.add_argument("--classify", action="store_true", help="add class labels only")
#     p.add_argument("--analyse", action="store_true", help="run statistics/plots")
#     p.add_argument("--subset", type=float, default=1.0,
#                    help="use random subset of subjects (0<subset≤1)")
#     p.add_argument("--verbose", action="store_true")
#     return p

# # ────────────────────────────────── main ───────────────────────────────────

# def main(argv: List[str] | None = None):
#     args = build_parser().parse_args(argv)
#     logging.basicConfig(format="%(levelname)s: %(message)s",
#                         level=logging.DEBUG if args.verbose else logging.INFO)

#     logging.info("Loading → %s", args.dict_path)
#     data_dict: Dict[str, Dict[str, Any]] = torch.load(args.dict_path, weights_only=False)

#     if args.subset < 1.0:
#         keys = np.random.choice(list(data_dict.keys()),
#                                 size=int(len(data_dict)*args.subset),
#                                 replace=False)
#         data_dict = {k: data_dict[k] for k in keys}
#         logging.info("Subset mode — %d subjects", len(data_dict))

#     # ── classification ──
#     need_classes = any("ensemble_class" not in e for e in data_dict.values())
#     if need_classes:
#         logging.info("Labelling pulse waves …")
#         for e in data_dict.values():
#             label_entry(e)
#         out_pt = args.dict_path.with_name(args.dict_path.stem + "_with_classes.pt")
#         torch.save(data_dict, out_pt)
#         logging.info("Classes written to %s", out_pt)
#     elif args.classify and not args.analyse:
#         logging.info("Classes already present – nothing to do.")
#         return

#     if not args.analyse:
#         return  # user wanted classification only

#     # ── analysis ──
#     df = _make_dataframe(data_dict)
#     logging.info("DataFrame created with %d rows", len(df))

#     univariate_plots(df)
#     correlation_heatmap(df)
#     rise_time_hist(df)
#     multivariate_regression(df)

#     logging.info("Figures → %s | Tables → %s", FIG_DIR, TAB_DIR)


# if __name__ == "__main__":
#     sys.exit(main())



# NEWWWWWW ############
# #!/usr/bin/env python
# """
# AURORA | MAUS – wave‑class labelling & regression analysis
# =========================================================

# **What this script does**
# -------------------------
# 1. **Classify every pulse wave** according to the five‑class decision tree in the
#    thesis (adds *individual_waves_classes* & *ensemble_class* to the dict).
# 2. **Run statistics & create figures**
#    * univariate scatter/OLS line plots
#    * correlation heat‑map
#    * rise‑time histogram
#    * multivariate *and* statsmodels OLS regression (z‑normalised inputs)
# 3. **Write outputs**
#    * figures → `out/figures/*.png` & `*.pdf`
#    * coefficient CSV (`out/tables/coefficients.csv`)
#    * pretty LaTeX table of coefficients (`out/tables/coefficients.tex`)
#    * full statsmodels summary as LaTeX (`out/tables/ols_summary.tex`) *and* text
#    * updated data_dict with classes → `*_with_classes.pt`

# Quick start
# -----------
# ```bash
# python aurora_analysis.py                 # full pipeline (detect dict automatically)
# python aurora_analysis.py --classify      # only label waves
# python aurora_analysis.py --analyse       # only figures/stats (dict already has classes)
# ```

# Dependencies: numpy, scipy, pandas, matplotlib, seaborn, scikit‑learn,
# statsmodels, torch.
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
# import statsmodels.api as sm

# # ───────────────────────────── configuration ──────────────────────────────
# DEFAULT_DICT_PATH = Path(__file__).resolve().with_name("data_dict_osc_auc_with_derivatives.pt")
# OUT_DIR   = Path(__file__).resolve().with_name("out")
# FIG_DIR   = OUT_DIR / "figures"
# TAB_DIR   = OUT_DIR / "tables"
# FIG_DIR.mkdir(parents=True, exist_ok=True)
# TAB_DIR.mkdir(parents=True, exist_ok=True)

# PREDICTORS_CONT = [
#     "age", "baseline_sbp", "baseline_dbp", "height_m",
#     "weight_kg", "average_hr", "bmi",
# ]

# # human-friendly labels for plotting
# ALIASES = {
#     "age": "age [years]",
#     "baseline_sbp": "SBP [mmHg]",
#     "baseline_dbp": "DBP [mmHg]",
#     "height_m": "height [m]",
#     "weight_kg": "weight [kg]",
#     "average_hr": "heart rate [bpm]",
#     "bmi": "BMI [kg/m^2]",
#     "rise_time_ms": "rise-time [ms]",
#     "rise_time_norm": "rise-time (norm)",
#     # add more as needed
# }
# PREDICTORS_CAT  = [
#     "gender", "cvd_meds", "fitzpatrick_scale",
#     "pressure_quality", "optical_quality", "oscillo_or_auscul",
# ]
# TARGETS = ["rise_time_ms", "rise_time_norm"]

# # ─────────────────────── wave‑shape classification helpers ─────────────────

# def _smooth(y: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
#     """Savitzky–Golay smoothing with fallback for short waves."""
#     return y if len(y) < window else signal.savgol_filter(y, window, poly)


# def _inflection_before_after(y: np.ndarray, peak_idx: int) -> Tuple[bool, bool]:
#     """Detect whether *a* zero‑crossing of the 2nd derivative exists **before**
#     and/or **after** the main peak.  A small hysteresis avoids noise."""
#     d2 = np.gradient(np.gradient(y))
#     # hysteresis: ignore crossings where |d2| > thr on either side
#     thr = 1e-3
#     zc = np.where(np.diff(np.sign(d2)))[0]
#     before = any((abs(d2[i]) < thr or abs(d2[i+1]) < thr) and i < peak_idx for i in zc)
#     after  = any((abs(d2[i]) < thr or abs(d2[i+1]) < thr) and i > peak_idx for i in zc)
#     return before, after


# def classify_wave(wave: np.ndarray) -> int:
#     """Return class **1‑5** (0 = invalid) following the decision tree."""
#     if wave.size == 0 or np.all(np.isnan(wave)):
#         return 0

#     y = _smooth(wave)
#     peaks, _ = signal.find_peaks(y, distance=len(y)//10)

#     # ── branch: two local maxima ──────────────────────────────────────────
#     if len(peaks) >= 2:
#         top2 = peaks[np.argsort(y[peaks])][-2:]
#         top2.sort()
#         first, second = top2
#         return 1 if y[first] > y[second] else 5

#     # ── branch: one local maximum ─────────────────────────────────────────
#     peak = int(peaks[0]) if len(peaks) else int(np.argmax(y))
#     before, after = _inflection_before_after(y, peak)
#     if before:
#         return 4  # inflection before max → Class 4
#     if after:
#         return 2  # inflection after max  → Class 2
#     return 3      # no inflection         → Class 3


# def label_entry(entry: Dict[str, Any]) -> None:
#     if "ensemble_class" in entry:
#         return
#     waves  = entry.get("individual_waves", [])
#     classes = np.array([classify_wave(w) for w in waves], dtype=np.int8)
#     entry["individual_waves_classes"] = classes
#     entry["ensemble_class"] = int(classify_wave(entry.get("ensemble_wave", np.empty(0))))

# # ────────────────────────── dataframe & plotting helpers ───────────────────

# def _make_dataframe(data_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
#     rows = []
#     for pid, d in data_dict.items():
#         if "average_rise_time_ms" not in d:
#             continue
#         rows.append({
#             "pid": pid,
#             **{k: d.get(k, np.nan) for k in PREDICTORS_CONT + PREDICTORS_CAT},
#             "rise_time_ms": d.get("average_rise_time_ms", np.nan),
#             "rise_time_norm": d.get("average_rise_time_norm", np.nan),
#             "ensemble_class": d.get("ensemble_class", np.nan),
#         })
#     df = pd.DataFrame(rows)
#     for c in PREDICTORS_CAT:
#         df[c] = df[c].astype("category")
#     return df

# # ---------- univariate scatter ----------

