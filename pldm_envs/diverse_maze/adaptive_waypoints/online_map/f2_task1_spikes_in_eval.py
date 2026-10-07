# Follow-up 2, task 1: do encoder "spike" frames also occur during evaluation?
# V0 stage-1 replan images (rendered with the evaluator's own renderer) encoded
# exactly as the evaluator does (per-frame Normalizer.normalize_state on the uint8
# CHW render, then stacked). Consecutive replans are 4 MPC steps apart, so r50 is
# compared at the same spacing (frames t, t+4, ...) as well as at spacing 1.
# spike = obs-latent MSE to BOTH temporal neighbours > 1.0.
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AW, "followup"))
sys.path.insert(0, os.path.join(AW, "inference_study"))
from common import DATA, D_OBS_CH, encode, episode_starts  # noqa: E402
from grid_utils import obs_to_ij  # noqa: E402
from signal1_common import get_model  # noqa: E402
from run_stage2_predictor_free import sample_episodes  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import eval_normalizer, normalize_images  # noqa: E402
import analyze_inference as ai  # noqa: E402

SPIKE = 1.0
RENDER = os.path.abspath(os.path.join(AW, "..", "..", "..", "..", "experiments", "kaggle_render_v0_states", "output", "v0_replan_images.npz"))


def eval_path_encode(model, imgs_uint8_chw, bs=256):
    norm = eval_normalizer()
    out = []
    for s in range(0, len(imgs_uint8_chw), bs):
        x = torch.stack([norm.normalize_state(torch.from_numpy(f)) for f in imgs_uint8_chw[s:s + bs]])
        with torch.no_grad():
            enc = model.level1.backbone(x, proprio=torch.zeros(len(x), 2)).encodings
        out.append(enc[:, :D_OBS_CH].flatten(1))
    return torch.cat(out)


def spikes(seq_d):
    """seq_d: distances between consecutive items (len n-1) -> bool spike flags for n items."""
    n = len(seq_d) + 1
    prev = np.r_[0.0, seq_d]
    nxt = np.r_[seq_d, 0.0]
    return (prev > SPIKE) & (nxt > SPIKE)


def stats(d):
    d = np.asarray(d)
    return {"n": int(len(d)), "median": float(np.median(d)), "q90": float(np.percentile(d, 90)), "q99": float(np.percentile(d, 99)),
            "share_gt_1": float((d > SPIKE).mean())}


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    model = get_model()
    r = np.load(RENDER)
    imgs, tids, ridx = r["images"], r["trial_id"], r["replan_index"]
    z = eval_path_encode(model, imgs)
    trials = ai.load_variant("v0")
    cons, spike_flags, backward_flags, fail_flags = [], [], [], []
    for tid in sorted(set(tids.tolist())):
        sel = np.where(tids == tid)[0]
        sel = sel[np.argsort(ridx[sel])]
        zz = z[sel]
        d = (zz[1:] - zz[:-1]).pow(2).mean(1).numpy()
        cons += d.tolist()
        sp = spikes(d)
        t = trials[tid]
        ts = ai.trial_series(t)
        f = ts["field"]
        for k, q in enumerate(sel):
            rep = ts["reps"][ridx[q]]
            if rep["wp1_xy"] is None:
                continue
            ai_, aj = obs_to_ij(np.asarray(rep["agent_xy"]))
            wi, wj = obs_to_ij(np.asarray(rep["wp1_xy"]))
            inside = 0 <= wi < f.shape[0] and 0 <= wj < f.shape[1] and f[wi, wj] >= 0
            backward_flags.append(bool(inside and f[ai_, aj] - f[wi, wj] < 0))
            spike_flags.append(bool(sp[k]))
            fail_flags.append(not t["success"])
    spike_flags, backward_flags, fail_flags = map(np.array, (spike_flags, backward_flags, fail_flags))
    res = {"v0_replans": {"consecutive_obs_mse": stats(cons), "spike_rate": float(spike_flags.mean()), "n_replans": int(len(spike_flags))}}
    res["backward_carrot_by_spike"] = {}
    for g, m in (("all", np.ones_like(fail_flags)), ("failed_trials", fail_flags), ("successful_trials", ~fail_flags)):
        m = m.astype(bool)
        res["backward_carrot_by_spike"][g] = {
            "backward_rate_at_spike": float(backward_flags[m & spike_flags].mean()) if (m & spike_flags).any() else None,
            "backward_rate_at_non_spike": float(backward_flags[m & ~spike_flags].mean()),
            "n_spike": int((m & spike_flags).sum()), "n_non_spike": int((m & ~spike_flags).sum()),
            "spike_rate_at_backward": float(spike_flags[m & backward_flags].mean()) if (m & backward_flags).any() else None,
            "spike_rate_at_other": float(spike_flags[m & ~backward_flags].mean()),
        }
    # r50 reference at spacing 1 and 4 (same 300-episode sample as the boundary study)
    splits, chosen = sample_episodes()
    images = np.load(os.path.join(DATA, "main", "images.npy"), mmap_mode="r")
    starts = episode_starts(splits)
    c1, c4, s1, s4 = [], [], [], []
    for e in chosen:
        T = len(splits[e]["observations"])
        x = normalize_images(torch.from_numpy(np.array(images[starts[e]:starts[e] + T])).float().permute(0, 3, 1, 2))
        with torch.no_grad():
            zz = model.level1.backbone(x, proprio=torch.zeros(T, 2)).encodings[:, :D_OBS_CH].flatten(1)
        d1 = (zz[1:] - zz[:-1]).pow(2).mean(1).numpy()
        z4 = zz[::4]
        d4 = (z4[1:] - z4[:-1]).pow(2).mean(1).numpy()
        c1 += d1.tolist(); c4 += d4.tolist()
        s1 += spikes(d1).tolist(); s4 += spikes(d4).tolist()
    res["r50_spacing1"] = {"consecutive_obs_mse": stats(c1), "spike_rate": float(np.mean(s1))}
    res["r50_spacing4"] = {"consecutive_obs_mse": stats(c4), "spike_rate": float(np.mean(s4))}
    json.dump(res, open(os.path.join(HERE, "results_f2_task1_spikes_in_eval.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
