# Boundary-signal study, Amendment-2 gate check: is 1b measuring what we
# think it is, before building Stage 2's predictor-based signals on it?
# See PREREGISTRATION.md Amendment 2 and PROGRESS.md for the user's exact
# request. Computes, over the full main+probe corpus:
#   (1) copy-baseline comparison: ||z_t - z_{t+1}|| vs 1b
#   (2) bias decomposition: global mean error vector, residual-after-bias
#   (3) direction check: cosine(predicted step, real step)
#   (5) consistency check against the REAL training objective classes
#
# KEY CODE FINDING (item 4, see PROGRESS.md for full citations): the actual
# training config (large_diverse_25maps_l2.yaml:113-115) uses
# `ObjectiveType.PredictionObs` + `ObjectiveType.PredictionProprio` -- NOT
# plain `ObjectiveType.Prediction` (pred_attr="state"). PredictionObjective
# with pred_attr="state" (what compute_error_series/signal1_common's err1/
# err1b have been computing all along, in the FUSED 18-channel
# obs+proprio-concatenated "encodings" space) is a real class in this
# codebase but is NOT what this checkpoint was trained with. The MPPI
# planner's cost function also only compares `cost_entity="obs_component"`
# (16 channels, proprio excluded) -- pldm/planning/planners/mppi_planner.py
# line 286. So this script computes 1b in THREE spaces for comparison:
# fused "state" (original), obs-only (matches training's PredictionObs AND
# the planner's cost), and proprio-only (matches training's
# PredictionProprio, reported for completeness).
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, iter_episodes, compute_episode_full, WINDOW  # noqa: E402
from pldm.objectives.prediction import PredictionObjective, PredictionObjectiveConfig  # noqa: E402

HERE = os.path.dirname(__file__)
N_OBS_CHANNELS = 16  # confirmed empirically: encodings = cat([obs_component(16ch), proprio_component(2ch)], dim=2)


def l2(vec_2d):
    """vec_2d: (N, D) -> (N,) L2 norms."""
    return np.linalg.norm(vec_2d, axis=1)


def run_space_from_stats(name, copy_l2, pred_l2, cos_sim, bias_norm, residual_l2, out):
    """Same report as before, but built from precomputed per-sample scalar
    arrays instead of raw (N, D) vectors -- see main()'s streaming rewrite
    (memory-safe: never holds full-D vectors for more than one episode at a
    time; see PROGRESS.md for why the original all-in-RAM version crashed
    the whole machine around episode 600/2250)."""
    ratio_of_means = float(pred_l2.mean() / copy_l2.mean())
    mean_ratio = float((pred_l2 / np.clip(copy_l2, 1e-8, None)).mean())
    pct_of_original = float(residual_l2.mean() / pred_l2.mean() * 100)

    result = {
        "n_samples": int(len(copy_l2)),
        "copy_baseline_l2": {"mean": float(copy_l2.mean()), "median": float(np.median(copy_l2))},
        "pred_error_l2": {"mean": float(pred_l2.mean()), "median": float(np.median(pred_l2))},
        "ratio_pred_over_copy": {"ratio_of_means": ratio_of_means, "mean_of_per_sample_ratios": mean_ratio},
        "bias_decomposition": {
            "global_bias_vector_norm": bias_norm,
            "residual_after_bias_removal_pct_of_original": pct_of_original,
        },
        "direction_cosine": {
            "mean": float(cos_sim.mean()),
            "median": float(np.median(cos_sim)),
            "std": float(cos_sim.std()),
            "frac_positive": float((cos_sim > 0).mean()),
        },
    }
    out[name] = result
    print(f"\n=== space: {name} (N={result['n_samples']}) ===")
    print(f"copy baseline L2: mean={result['copy_baseline_l2']['mean']:.4f}, median={result['copy_baseline_l2']['median']:.4f}")
    print(f"pred error L2:    mean={result['pred_error_l2']['mean']:.4f}, median={result['pred_error_l2']['median']:.4f}")
    print(f"ratio (pred/copy): of means={ratio_of_means:.3f}, mean of per-sample ratios={mean_ratio:.3f}")
    print(f"bias vector norm: {bias_norm:.4f}; residual-after-bias-removal is {pct_of_original:.1f}% of original error")
    print(f"direction cosine(pred_step, real_step): mean={cos_sim.mean():.4f}, median={np.median(cos_sim):.4f}, frac_positive={result['direction_cosine']['frac_positive']:.3f}")
    return result


def consistency_check(model):
    """(5): compare our manual err1 (fused, signal 1's own compounding
    rollout) against the REAL PredictionObjective(pred_attr='state') class,
    on the SAME ForwardResult -- a pure code-correctness check. Also
    computes what the REAL training loss actually was (PredictionObs +
    PredictionProprio), for comparison."""
    obj_state = PredictionObjective(PredictionObjectiveConfig(), pred_attr="state")
    obj_obs = PredictionObjective(PredictionObjectiveConfig(global_coeff=2.416154262252218), pred_attr="obs")
    obj_proprio = PredictionObjective(PredictionObjectiveConfig(global_coeff=2.416154262252218), pred_attr="proprio")

    print("\n=== (5) consistency check: manual computation vs official PredictionObjective class ===")
    for ep_idx, ep, images, ep_start in iter_episodes("main", limit=3):
        _, _, _, result = compute_episode_full(model, ep, images, ep_start)

        manual_state_err = (
            (result.backbone_output.encodings[1:] - result.pred_output.predictions[1:]).pow(2).mean()
        )
        official_state_loss = obj_state(None, [result]).pred_loss
        official_obs_loss = obj_obs(None, [result]).pred_loss
        official_proprio_loss = obj_proprio(None, [result]).pred_loss

        print(
            f"ep {ep_idx}: manual_state_mse={manual_state_err.item():.6f}, "
            f"official_state_mse={official_state_loss.item():.6f}, "
            f"match={torch.allclose(manual_state_err, official_state_loss)}"
        )
        print(
            f"        REAL training loss components: PredictionObs={official_obs_loss.item():.6f} "
            f"(x global_coeff=2.416 -> {official_obs_loss.item()*2.416154262252218:.4f}), "
            f"PredictionProprio={official_proprio_loss.item():.6f} "
            f"(x global_coeff -> {official_proprio_loss.item()*2.416154262252218:.4f}) "
            f"-- fused 'state' MSE (never actually trained on): {official_state_loss.item():.6f}"
        )


N_MAIN_EPISODES = 1250
N_PROBE_EPISODES = 1000
N_TOTAL_EPISODES = N_MAIN_EPISODES + N_PROBE_EPISODES  # 2250
N_TOTAL_SAMPLES = N_TOTAL_EPISODES * 60  # 135000
H, W = 43, 43
D_OBS = N_OBS_CHANNELS * H * W  # 29584
D_PROPRIO_START = D_OBS
D_FUSED = 18 * H * W  # 33282

MEMMAP_PATH = os.path.join(HERE, "gate_check_pred_err_fused.f32.memmap")
CKPT_PATH = os.path.join(HERE, "gate_check_checkpoint.npz")


def _new_stat_lists():
    return {sp: {"copy_l2": [], "pred_l2": [], "cos_sim": []} for sp in ("fused", "obs", "proprio")}


def main():
    model = get_model()

    print("Item (4)/(5) key code finding: large_diverse_25maps_l2.yaml:113-115 "
          "trains with ObjectiveType.PredictionObs + PredictionProprio (NOT "
          "plain Prediction/'state'). Confirmed encodings = cat(obs_component"
          "[16ch], proprio_component[2ch]) exactly (checked empirically).")

    consistency_check(model)

    # Memory-safe, resumable streaming pass. The original version held
    # (N=135000, D=33282) float32 arrays x3 in RAM (~54GB) -- this machine
    # has ~14GB free, which is almost certainly why the process (and
    # possibly the whole WSL VM under the resulting memory pressure) died
    # around episode 600 twice in a row. Fix: never hold more than one
    # episode's (60, D) arrays at a time; persist only the small per-sample
    # scalars in RAM, and the (N, D_fused) pred_err array on DISK via
    # memmap (needed for the bias-removal residual, which requires a global
    # mean computed from ALL samples first -- a genuine two-pass
    # requirement, not avoidable by more RAM alone at this N).
    stats = _new_stat_lists()
    bias_sum_fused = np.zeros(D_FUSED, dtype=np.float64)
    n_samples_seen = 0
    start_episode = 0

    if os.path.exists(CKPT_PATH):
        ckpt = np.load(CKPT_PATH, allow_pickle=True)
        start_episode = int(ckpt["next_episode"])
        n_samples_seen = int(ckpt["n_samples_seen"])
        bias_sum_fused = ckpt["bias_sum_fused"]
        stats = ckpt["stats"].item()
        print(f"resuming from checkpoint: episode {start_episode}/{N_TOTAL_EPISODES}", flush=True)

    pred_err_memmap = np.memmap(
        MEMMAP_PATH, dtype="float32", mode="r+" if os.path.exists(MEMMAP_PATH) else "w+",
        shape=(N_TOTAL_SAMPLES, D_FUSED),
    )

    def episode_iterator():
        g = 0
        for ep_idx, ep, images, ep_start in iter_episodes("main"):
            yield g, ep, images, ep_start
            g += 1
        for ep_idx, ep, images, ep_start in iter_episodes("probe"):
            yield g, ep, images, ep_start
            g += 1

    for g, ep, images, ep_start in episode_iterator():
        if g < start_episode:
            continue

        z_t, z_tp1, pred, _result = compute_episode_full(model, ep, images, ep_start)  # each (60, D_FUSED)
        pred_err = pred - z_tp1  # (60, D_FUSED)
        copy_err = z_tp1 - z_t
        pred_step = pred - z_t
        real_step = copy_err

        row0 = g * 60
        pred_err_memmap[row0 : row0 + 60] = pred_err
        bias_sum_fused += pred_err.sum(axis=0, dtype=np.float64)
        n_samples_seen += 60

        slices = {"fused": (0, D_FUSED), "obs": (0, D_OBS), "proprio": (D_PROPRIO_START, D_FUSED)}
        for sp, (a, b) in slices.items():
            c_l2 = l2(copy_err[:, a:b])
            p_l2 = l2(pred_err[:, a:b])
            denom = l2(pred_step[:, a:b]) * l2(real_step[:, a:b])
            cos = (pred_step[:, a:b] * real_step[:, a:b]).sum(axis=1) / np.clip(denom, 1e-8, None)
            stats[sp]["copy_l2"].append(c_l2)
            stats[sp]["pred_l2"].append(p_l2)
            stats[sp]["cos_sim"].append(cos)

        if (g + 1) % 100 == 0:
            pred_err_memmap.flush()
            np.savez(
                CKPT_PATH,
                next_episode=g + 1,
                n_samples_seen=n_samples_seen,
                bias_sum_fused=bias_sum_fused,
                stats=stats,
            )
            print(f"{g + 1}/{N_TOTAL_EPISODES} episodes done (checkpointed)", flush=True)

    pred_err_memmap.flush()
    print(f"pass 1 complete: {n_samples_seen} samples", flush=True)

    # Pass 2 (cheap, no model): residual-after-bias-removal, chunked read
    # from the on-disk memmap so we never load all 135000x33282 at once.
    bias_vec_fused = (bias_sum_fused / n_samples_seen).astype(np.float32)
    residual_l2 = {"fused": [], "obs": [], "proprio": []}
    chunk = 5000
    for i in range(0, N_TOTAL_SAMPLES, chunk):
        block = np.asarray(pred_err_memmap[i : i + chunk])  # (chunk, D_FUSED), materialized copy
        for sp, (a, b) in (("fused", (0, D_FUSED)), ("obs", (0, D_OBS)), ("proprio", (D_PROPRIO_START, D_FUSED))):
            res = block[:, a:b] - bias_vec_fused[a:b][None, :]
            residual_l2[sp].append(l2(res))
    for sp in residual_l2:
        residual_l2[sp] = np.concatenate(residual_l2[sp])
    print("pass 2 (residual-after-bias-removal) complete", flush=True)

    out = {}
    bias_norms = {
        "fused": float(np.linalg.norm(bias_vec_fused)),
        "obs": float(np.linalg.norm(bias_vec_fused[:D_OBS])),
        "proprio": float(np.linalg.norm(bias_vec_fused[D_PROPRIO_START:])),
    }
    name_map = {
        "fused": "fused_state_original_1b",
        "obs": "obs_only_matches_training_and_planner",
        "proprio": "proprio_only_matches_training",
    }
    for sp in ("fused", "obs", "proprio"):
        copy_l2 = np.concatenate(stats[sp]["copy_l2"])
        pred_l2 = np.concatenate(stats[sp]["pred_l2"])
        cos_sim = np.concatenate(stats[sp]["cos_sim"])
        run_space_from_stats(name_map[sp], copy_l2, pred_l2, cos_sim, bias_norms[sp], residual_l2[sp], out)

    verdict_ratio = out["fused_state_original_1b"]["ratio_pred_over_copy"]["ratio_of_means"]
    verdict = "TRIGGERED" if verdict_ratio >= 3.0 else "not triggered"
    print(f"\n=== VERDICT (fused/original 1b vs copy baseline ratio = {verdict_ratio:.3f}): {verdict} ===")

    out_path = os.path.join(HERE, "results_gate_check_1b.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")

    # cleanup: the memmap (~18GB) and checkpoint were only scratch space
    del pred_err_memmap
    if os.path.exists(MEMMAP_PATH):
        os.remove(MEMMAP_PATH)
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)


if __name__ == "__main__":
    main()
