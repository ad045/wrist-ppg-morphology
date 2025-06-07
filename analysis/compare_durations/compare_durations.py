#!/usr/bin/env python
"""
Plot histograms & compute distribution-similarity metrics for

   • the custom pipeline  (durations_custom_*.npy) 
   • vanilla pyPPG  (durations_pyppg_*.npy)
   • NeuroKit2      (durations_neurokit_*.npy)

Metrics shown:
   – Kolmogorov-Smirnov statistic  (two-sample)
   – Wasserstein distance
   – Jensen–Shannon divergence
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance, entropy

from initialize import PREPROCESSED_MAUS_DATA_PATH
BASE = Path(PREPROCESSED_MAUS_DATA_PATH) / "comparison_algos"

def js_div(p, q):
    """Jensen–Shannon divergence for empirical samples."""
    hist_p, _ = np.histogram(p, bins=100, range=(200, 1800), density=True)
    hist_q, _ = np.histogram(q, bins=100, range=(200, 1800), density=True)
    m = 0.5 * (hist_p + hist_q)
    return 0.5 * (entropy(hist_p, m) + entropy(hist_q, m))

algos   = ["custom", "pyppg", "neurokit"]
devices = ["finger", "wrist"]
colors  = {"custom": "tab:green", "pyppg": "tab:blue",
           "neurokit": "tab:orange"}



fig, axes = plt.subplots(len(algos), 2, figsize=(10, 9), sharex=True, sharey=True)

# Determine common bin edges over the full expected range:
bin_edges = np.linspace(200, 1800, 31)  # 30 equal-width bins from 200 to 1800 ms

all_metrics = [] 

for r, algo in enumerate(algos):
    data = {}

    # plots
    for device in devices:
        # load data
        f = BASE / f"durations_{algo}_{device}.npy"
        data[device] = np.load(f)
        ax = axes[r, 0 if device == 'finger' else 1]

        # Plot histogram for each device
        ax = axes[r, 0 if device == 'finger' else 1]
        # Use the shared bin_edges for equal width
        ax.hist(data[device], bins=bin_edges,
                alpha=0.75, color=colors[algo],
                label=f"{device.capitalize()} - {algo}")
        ax.set_title(f"{device.capitalize()} · {algo}")
        if device == 'finger':
            ax.set_ylabel("Count")

    # similarity metrics  (finger vs wrist)
    ks      = ks_2samp(data["finger"], data["wrist"]).statistic
    w_dist  = wasserstein_distance(data["finger"], data["wrist"])
    js      = js_div(data["finger"], data["wrist"])
    all_metrics.append((algo, ks, w_dist, js))

axes[-1, 0].set_xlabel("PW duration [ms]")
axes[-1, 1].set_xlabel("PW duration [ms]")
plt.tight_layout()
plt.show()

print("Finger ↔ Wrist similarity per algorithm")
print("Algorithm        KS-stat   Wasserstein   JS-div")
for a, ks, wd, js in all_metrics:
    print(f"{a:<12}  {ks:8.3f}    {wd:10.2f}   {js:7.3f}")
