# #!/usr/bin/env python
# """
# Plot histograms & compute distribution-similarity metrics for

#    • the custom pipeline  (durations_custom_*.npy) 
#    • vanilla pyPPG  (durations_pyppg_*.npy)
#    • NeuroKit2      (durations_neurokit_*.npy)

# Metrics shown:
#    – Kolmogorov-Smirnov statistic  (two-sample)
#    – Wasserstein distance
#    – Jensen–Shannon divergence
# """
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import ks_2samp, wasserstein_distance, entropy

# from initialize import PREPROCESSED_MAUS_DATA_PATH
# BASE = Path(PREPROCESSED_MAUS_DATA_PATH) / "comparison_algos"

# def js_div(p, q):
#     """Jensen–Shannon divergence for empirical samples."""
#     hist_p, _ = np.histogram(p, bins=100, range=(200, 1800), density=True)
#     hist_q, _ = np.histogram(q, bins=100, range=(200, 1800), density=True)
#     m = 0.5 * (hist_p + hist_q)
#     return 0.5 * (entropy(hist_p, m) + entropy(hist_q, m))

# algos   = ["custom", "pyppg", "neurokit"]
# devices = ["finger", "wrist"]
# colors  = {"custom": "tab:green", "pyppg": "tab:blue",
#            "neurokit": "tab:orange"}



# fig, axes = plt.subplots(len(algos), 2, figsize=(10, 9), sharex=True, sharey=True)

# # Determine common bin edges over the full expected range:
# bin_edges = np.linspace(200, 1800, 31)  # 30 equal-width bins from 200 to 1800 ms

# all_metrics = [] 

# for r, algo in enumerate(algos):
#     data = {}

#     # plots
#     for device in devices:
#         # load data
#         f = BASE / f"durations_{algo}_{device}.npy"
#         data[device] = np.load(f)
#         ax = axes[r, 0 if device == 'finger' else 1]

#         # Plot histogram for each device
#         ax = axes[r, 0 if device == 'finger' else 1]
#         # Use the shared bin_edges for equal width
#         ax.hist(data[device], bins=bin_edges,
#                 alpha=0.75, color=colors[algo],
#                 label=f"{device.capitalize()} - {algo}")
#         ax.set_title(f"{device.capitalize()} · {algo}")
#         if device == 'finger':
#             ax.set_ylabel("Count")

#     # similarity metrics  (finger vs wrist)
#     ks      = ks_2samp(data["finger"], data["wrist"]).statistic
#     w_dist  = wasserstein_distance(data["finger"], data["wrist"])
#     js      = js_div(data["finger"], data["wrist"])
#     all_metrics.append((algo, ks, w_dist, js))

# axes[-1, 0].set_xlabel("PW duration [ms]")
# axes[-1, 1].set_xlabel("PW duration [ms]")
# plt.tight_layout()
# plt.show()

# print("Finger ↔ Wrist similarity per algorithm")
# print("Algorithm        KS-stat   Wasserstein   JS-div")
# for a, ks, wd, js in all_metrics:
#     print(f"{a:<12}  {ks:8.3f}    {wd:10.2f}   {js:7.3f}")


#!/usr/bin/env python
"""
Visual comparison of finger- vs- wrist PW-duration distributions
captured by three onset-detection algorithms.

Figures created
---------------
1. Upper-triangle heat-map of pairwise Jensen–Shannon divergence.
2. Vertical bar chart: divergence of every *wrist* distribution from
   a “finger‐reference” (PyPPG-finger ∪ NeuroKit-finger).  Lower = better.
3. Overlay of the *three* distributions that the bar chart suggests
   are closest (finger-PyPPG, finger-NeuroKit, wrist-Custom).
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
import seaborn as sns            

from initialize import PREPROCESSED_MAUS_DATA_PATH, OUTPUT_COMPARISON_ALGOS_PATH
BASE = Path(PREPROCESSED_MAUS_DATA_PATH) / "comparison_algos"
OUTPUT_COMPARISON_ALGOS_PATH = Path(OUTPUT_COMPARISON_ALGOS_PATH) 

# ─────────────────────────────────────────────────────────────────────────────
# Helper – Jensen–Shannon divergence
# ─────────────────────────────────────────────────────────────────────────────
def js_div(a: np.ndarray, b: np.ndarray,
           bins: int = 100, lo: int = 200, hi: int = 1800) -> float:
    p, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

algos   = ["custom", "pyppg", "neurokit"]
devices = ["finger", "wrist"]

# ─────────────────────────────────────────────────────────────────────────────
# 1)  load data
# ─────────────────────────────────────────────────────────────────────────────
samples = {}
for algo in algos:
    for dev in devices:
        key = f"{algo}_{dev}"
        samples[key] = np.load(BASE / f"durations_{algo}_{dev}.npy")
        print(f"Loaded {key} with {len(samples[key])} samples.")

# finger reference  = PyPPG-finger ∪ NeuroKit-finger
finger_ref = np.concatenate([samples["pyppg_finger"],
                             samples["neurokit_finger"]])

# ─────────────────────────────────────────────────────────────────────────────
# 2)  upper-triangle heat-map
# ─────────────────────────────────────────────────────────────────────────────
labels = [f"{a}_{d}"      
          for a in algos for d in devices]
print("Labels:", labels)
n = len(labels)
mat = np.zeros((n, n))
for i, li in enumerate(labels):
    for j, lj in enumerate(labels):
        mat[i, j] = js_div(samples[li.replace('-', '_')],
                           samples[lj.replace('-', '_')])

# mask lower-triangle incl. diagonal
# mask = np.tril(np.ones_like(mat, dtype=bool))
mask = np.triu(np.ones_like(mat, dtype=bool))

fig, ax = plt.subplots(figsize=(7, 5.8))
sns.heatmap(mat,
            mask=mask,
            cmap="RdYlGn_r",           # green = low, red = high
            annot=True, fmt=".3f",
            xticklabels=labels, yticklabels=labels,
            linewidths=.5, linecolor='white',
            cbar_kws=dict(label="JS divergence", shrink=.8), 
            ax=ax)

# Add circles 
from matplotlib.patches import Circle

# finger and finger: black 
idx_x = 2 # x direction, starts with 0 
idx_y = 6-2 # y direction, is reversed, so starts with 5
circ = Circle((idx_x + 0.5, idx_y+ 0.5), 0.4, edgecolor='black',
              facecolor='none', lw=2)
ax.add_patch(circ)

# wrist and wrist: black
idx_x = 1 # x direction, starts with 0 
idx_y = 6-2 # y direction, is reversed, so starts with 5
circ = Circle((idx_x + 0.5, idx_y+ 0.5), 0.4, edgecolor='darkgreen',
              facecolor='none', lw=2)
ax.add_patch(circ)
idx_x = 1 # x direction, starts with 0 
idx_y = 6-4 # y direction, is reversed, so starts with 5
circ = Circle((idx_x + 0.5, idx_y+ 0.5), 0.4, edgecolor='darkgreen',
              facecolor='none', lw=2)
ax.add_patch(circ)


print_labels = [label.replace('_', '\n') for label in labels]
plt.xticks(ticks=[i+1/2 for i in range(len(print_labels))], labels=print_labels, rotation=0)
plt.yticks(ticks=[i+1/2 for i in range(len(print_labels))], labels=print_labels, rotation=0)
plt.title("Pairwise JS divergence (↓ better)")
plt.tight_layout()
plt.savefig(OUTPUT_COMPARISON_ALGOS_PATH / "js_divergence_heatmap.png", dpi=300)
plt.savefig(OUTPUT_COMPARISON_ALGOS_PATH / "js_divergence_heatmap.pdf")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 3)  vertical bar chart – wrist vs finger-reference
# ─────────────────────────────────────────────────────────────────────────────
wrist_keys = [f"{a}_wrist" for a in algos]
divergences = [js_div(samples[k], finger_ref) for k in wrist_keys]

plt.figure(figsize=(5, 4))
colors = ["green", "white", "white"]
edges  = ["black" if c == "white" else "green" for c in colors]
bars = plt.bar(wrist_keys, divergences, color=colors, edgecolor=edges, linewidth=1.5)
plt.ylabel("JS divergence to finger reference  (↓ better)")
plt.title("Performance of wrist PW detection (↓ better)") # Finger-like similarity of wrist distributions")

plt.xticks(ticks=range(len(wrist_keys)), labels=["Wrist \n" + k.split("_")[0] for k in wrist_keys])
# add numerical labels on top
for bar, val in zip(bars, divergences):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.002,
             f"{val:.3f}", ha="center", va="bottom")

plt.ylim(0, max(divergences)*1.15)
plt.tight_layout()
plt.savefig(OUTPUT_COMPARISON_ALGOS_PATH / "wrist_vs_finger_reference.png", dpi=300)
plt.savefig(OUTPUT_COMPARISON_ALGOS_PATH / "wrist_vs_finger_reference.pdf")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 4)  Overlay plot of the three closest distributions
#     (finger-PyPPG, finger-NeuroKit, wrist-Custom)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4))

# targets = [("pyppg_finger",   "PyPPG-finger"),
#            ("neurokit_finger","NeuroKit-finger"),
#            ("custom_wrist",   "Custom-wrist")]

# targets = [("pyppg_finger",   "PyPPG-finger"),
#            ("neurokit_finger","NeuroKit-finger"),
#            ("custom_wrist",   "Custom-wrist"), 

#            ("custom_finger",  "Custom-finger"), 
#            ("pyppg_wrist",    "PyPPG-wrist"),
#            ("neurokit_wrist", "NeuroKit-wrist")]


targets = [("pyppg_finger",   f"Finger: PyPPG (N={len(samples['pyppg_finger'])})"),
           ("neurokit_finger",f"Finger: NeuroKit2 (N={len(samples['neurokit_finger'])})"),
           ("custom_wrist",   f"Wrist: Custom (N={len(samples['custom_wrist'])})"),
           ("custom_finger",  f"Finger: Custom (N={len(samples['custom_finger'])})"),
           ("pyppg_wrist",    f"Wrist: PyPPG (N={len(samples['pyppg_wrist'])})"),
           ("neurokit_wrist", f"Wrist: NeuroKit2 (N={len(samples['neurokit_wrist'])})")]

# targets = [("pyppg_finger",   "Finger: PyPPG"),
        #    ("neurokit_finger","Finger: NeuroKit2"),
        #    ("custom_finger",  "Finger: Custom"), 
        #    ("pyppg_wrist",    "Wrist: PyPPG"),
        #    ("neurokit_wrist", "Wrist: NeuroKit2"), 
        #    ("custom_wrist",   "Wrist: Custom"), ]

x_grid = np.linspace(200, 1800, 800)

for key, label in targets:
    kde = gaussian_kde(samples[key])
    y   = kde(x_grid)
    if "Wrist: Custom" in label:
        ax.fill_between(x_grid, y, alpha=.35, color="green")
        ax.plot(x_grid, y, color="green", linewidth=1, label=label)
    elif "Finger: Custom" in label:
        # MAYBE CHANGE! 
        continue
    elif "Wrist" in label:
        ax.plot(x_grid, y, color="gray", linewidth=1, label=label) # linestyle="dotted", 
    else:
        ax.plot(x_grid, y, color="black", linewidth=1, label=label) # linestyle="--", 

ax.set_xlabel("PW duration [ms]")
ax.set_ylabel("Density")
ax.set_title("Distributions of the wave durations detected")
ax.legend(frameon=False)
ax.set_xlim(400, 1400)          # zoom into the informative region
plt.tight_layout()
plt.savefig(OUTPUT_COMPARISON_ALGOS_PATH / "overlay_closest_distributions.png", dpi=300)
plt.savefig(OUTPUT_COMPARISON_ALGOS_PATH / "overlay_closest_distributions.pdf")
plt.show()