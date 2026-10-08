# M4 ROC figure from results_dp_measure.json (roc_thr_fpr_tpr), both spike versions.
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "results_dp_measure.json")))["M4"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ver, ls in (("excl_spikes", "-"), ("incl_spikes", "--")):
    roc = np.array(r[ver]["roc_thr_fpr_tpr"])
    axes[0].plot(roc[:, 1], roc[:, 2], ls, label=f"{ver} (AUROC {r[ver]['auroc']:.3f})")
    axes[1].plot(roc[:, 0], roc[:, 1], ls, label=f"{ver}: false-alarm rate (folded, BFS<=1)")
    axes[1].plot(roc[:, 0], roc[:, 2], ls, label=f"{ver}: detection rate (unfolded, BFS>=3)")
axes[0].plot([0, 1], [0, 1], c="0.7", lw=0.8); axes[0].axvline(0.05, c="r", ls=":", lw=0.8)
axes[0].set_xlabel("false-alarm rate"); axes[0].set_ylabel("detection rate"); axes[0].set_title("M4 ROC, pairs with gap >= 60")
axes[1].set_xscale("log"); axes[1].set_xlabel("latent distance threshold"); axes[1].set_ylabel("rate"); axes[1].set_title("rates vs threshold")
for ax in axes:
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "dp_m4_roc.png"), dpi=110)
print("saved")
