# A.3/A.4: compare results computed with raw (buggy, *_v1.json) inputs vs the
# normalized inputs (preprocess.py). Writes results_bugfix_comparison.json.
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset", "main")


def j(name):
    return json.load(open(os.path.join(HERE, name)))


def gate(g, key):
    d = g[key]
    return {
        "copy_baseline_ratio": d["ratio_pred_over_copy"]["ratio_of_means"],
        "bias_norm_over_pred_error": d["bias_decomposition"]["global_bias_vector_norm"] / d["pred_error_l2"]["mean"],
        "residual_after_bias_pct": d["bias_decomposition"]["residual_after_bias_removal_pct_of_original"],
        "direction_cosine_mean": d["direction_cosine"]["mean"],
        "pred_error_l2_mean": d["pred_error_l2"]["mean"],
        "copy_l2_mean": d["copy_baseline_l2"]["mean"],
    }


def inv(i, key):
    d = i[key]
    return {
        "shuffle_ratio_1step": d["ratio_real_over_shuffled_1step"],
        "shuffle_ratio_5step": d["five_step_shuffle"]["ratio_real_over_shuffled"],
        "direction_cosine_bias_corrected": d["direction_cosine_bias_corrected_mean"],
        "r_squared": d["r_squared"],
    }


def metric_b(sp, am, s16):
    out = {}
    for name, key in (("signal_6", "signal_6_bocpd"), ("signal_7", "signal_7_speed"), ("signal_8", "signal_8_direction"),
                      ("signal_10", "signal_10_action_delta"), ("fixed", "fixed"), ("random", "random"), ("dp_segmentation", "oracle")):
        out[name] = sp["summary"][key]["improvement_over_fixed"]
    for name, key in (("signal_11", "signal_11_curvature"), ("signal_12", "signal_12_norm_curvature"),
                      ("signal_13", "signal_13_chord_dev"), ("bottom_up_minseg8", "bottom_up_minseg8"), ("top_down_minseg8", "top_down_minseg8")):
        if key in am:
            out[name] = am[key]["improvement_over_fixed"]
    for name in ("signal_16_v2_obs", "signal_16_v2_proprio"):
        if name in s16.get("summary", s16):
            out[name] = s16.get("summary", s16)[name]["improvement_over_fixed"]
    return out


def dp_agreement(new_sp):
    old = torch.load(os.path.join(DATA, "changepoints_minseg8.pt"), weights_only=False)  # computed from raw inputs
    within2, exact, all5 = [], [], []
    for ep, nb in new_sp["oracle_boundaries_by_episode"].items():
        ob = [int(x) for x in old[int(ep)]]
        hits = [min(abs(b - n) for n in nb) <= 2 for b in ob]
        within2.append(np.mean(hits))
        exact.append(np.mean([b in nb for b in ob]))
        all5.append(all(hits))
    return {"n_episodes": len(within2), "frac_boundaries_within2": float(np.mean(within2)),
            "frac_boundaries_exact": float(np.mean(exact)), "frac_episodes_all5_within2": float(np.mean(all5))}


def main():
    res = {}
    g1, g2 = j("results_gate_check_1b_v1.json"), j("results_gate_check_1b.json")
    for k, label in (("obs_only_matches_training_and_planner", "obs"), ("proprio_only_matches_training", "proprio"), ("fused_state_original_1b", "fused")):
        res[f"gate_{label}"] = {"v1_raw_inputs": gate(g1, k), "fixed": gate(g2, k)}
    i1, i2 = j("results_investigate_1b_bias_v1.json"), j("results_investigate_1b_bias.json")
    for k in ("obs", "proprio"):
        res[f"investigate_{k}"] = {"v1_raw_inputs": inv(i1, k), "fixed": inv(i2, k)}
    res["metric_b_improvement_over_fixed"] = {
        "v1_raw_inputs": metric_b(j("results_stage2_predictor_free_v1.json"), j("results_stage2_amendment3_v1.json"), j("results_signal16_v2_v1.json")),
        "fixed": metric_b(j("results_stage2_predictor_free.json"), j("results_stage2_amendment3.json"), j("results_signal16_v2.json")),
        "note": "v1 16v2 used its own 50-episode sample (seed 7); fixed run uses the shared 300-episode sample. v1 bottom_up_minseg8 predates the bottom_up_merge bug fix.",
    }
    res["dp_segmentation_boundary_agreement_old_cache_vs_fixed"] = dp_agreement(j("results_stage2_predictor_free.json"))
    json.dump(res, open(os.path.join(HERE, "results_bugfix_comparison.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
