# Spearman(Metric A F1, Metric B % improvement) under each label set of
# results_metric_a_coarse.json; Metric B values identical to compare_metric_a_b.py.
import json
import os

from scipy.stats import spearmanr

from compare_metric_a_b import HERE

NAMES = ["signal_6", "signal_7", "signal_8", "signal_10", "signal_11", "signal_12", "signal_13",
         "top_down", "bottom_up", "fixed", "random", "oracle"]


def main():
    ab = json.load(open(os.path.join(HERE, "results_metric_a_vs_b.json")))
    metric_b = {r["name"]: r["metric_b_pct_improve"] for r in ab["table"]}
    coarse = json.load(open(os.path.join(HERE, "results_metric_a_coarse.json")))["summary"]
    out = {}
    for ls, table in coarse["prf1"].items():
        a = [table[n]["overall"]["f1"] for n in NAMES]
        rho, p = spearmanr(a, [metric_b[n] for n in NAMES])
        auroc = coarse["auroc_ap"][ls]
        sig = [n for n in NAMES if n.startswith("signal_")]
        rho_au, p_au = spearmanr([auroc[n]["auroc_mean"] for n in sig], [metric_b[n] for n in sig])
        out[ls] = {"spearman_f1_vs_b": rho, "p": p, "spearman_auroc_vs_b_signals_only": rho_au, "p_auroc": p_au,
                   "f1": dict(zip(NAMES, a))}
        print(f"{ls:<18} rho(F1,B)={rho:+.3f} (p={p:.4f})  rho(AUROC,B; 7 signals)={rho_au:+.3f} (p={p_au:.3f})")
    out["metric_b_pct_improve"] = metric_b
    json.dump(out, open(os.path.join(HERE, "results_metric_a_coarse_vs_b.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
