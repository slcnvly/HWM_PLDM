# Follow-up 1 + 2 (2026-10-06), corrected (normalized) inputs, frozen L1.
# 1. Action sensitivity over k = 1, 5, 10 predictor steps: from the same z_t, roll
#    the L1 predictor with the real action sequence and with a shuffled one (the
#    real sequence starting at another time in the same episode). Report, for obs
#    and proprio separately: prediction error vs the real future (and the
#    real/shuffled ratio), and the direct action effect ||pred_real - pred_shuf||
#    relative to the size of the predicted change ||pred_real - z_t||.
# 2. Surprise signal: one-step prediction error (obs, proprio) per frame vs the
#    Stage-1 junction-arrival label (exact frame and +/-2 dilated), plus means by
#    the agent's cell type.
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import DATA, D_OBS_CH, WINDOW, cell_types_for_frames, encode, episode_inputs, episode_starts
from event_labels import label_episode
from metric_a import dilate_events
from run_stage2_predictor_free import sample_episodes
from signal1_common import get_model

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (1, 5, 10)


def rollout(model, z0, pc0, acts, k):
    """z0: (B,18,43,43), pc0: (B,2,43,43), acts: (k,B,2) -> prediction at step k (B,18,43,43)."""
    with torch.no_grad():
        out = model.level1.predictor.forward_multiple(
            state_encs=z0.unsqueeze(0), actions=acts, T=k, proprio=pc0.unsqueeze(0), compute_posterior=False)
    return out.predictions[k]


def norms(x):
    return x.flatten(1).norm(dim=1)


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    model = get_model()
    splits, chosen = sample_episodes()
    images = np.load(os.path.join(DATA, "main", "images.npy"), mmap_mode="r")
    maps = torch.load(os.path.join(DATA, "main", "train_maps.pt"), weights_only=False)
    starts = episode_starts(splits)
    rng = np.random.default_rng(0)

    acc = {k: {c: {"err_real": [], "err_shuf": [], "effect": [], "pred_change": [], "copy": []} for c in ("obs", "proprio")} for k in KS}
    surprise = {"obs": [], "proprio": []}
    labels_exact, labels_dil, ctypes = [], [], []
    per_ep_auroc = {"obs": [], "proprio": []}
    for n, e in enumerate(chosen):
        ep = splits[e]
        img, pv, acts = episode_inputs(ep, images, starts[e])
        enc, pc = encode(model, img, pv)
        layout = maps[int(ep["map_idx"])].split("\\")
        # ---- 1. action sensitivity (starts 0..50 so that t+10 <= 60)
        tt = np.arange(0, 51)
        perm = tt.copy()
        while np.any(perm == tt):
            perm = rng.permutation(tt)
        for k in KS:
            a_real = torch.stack([acts[t:t + k] for t in tt], dim=1)  # (k, B, 2)
            a_shuf = torch.stack([acts[s:s + k] for s in perm], dim=1)
            p_real = rollout(model, enc[tt], pc[tt], a_real, k)
            p_shuf = rollout(model, enc[tt], pc[tt], a_shuf, k)
            target = enc[tt + k]
            for c, sl in (("obs", slice(0, D_OBS_CH)), ("proprio", slice(D_OBS_CH, None))):
                acc[k][c]["err_real"].append(norms(p_real[:, sl] - target[:, sl]))
                acc[k][c]["err_shuf"].append(norms(p_shuf[:, sl] - target[:, sl]))
                acc[k][c]["effect"].append(norms(p_real[:, sl] - p_shuf[:, sl]))
                acc[k][c]["pred_change"].append(norms(p_real[:, sl] - enc[tt][:, sl]))
                acc[k][c]["copy"].append(norms(enc[tt][:, sl] - target[:, sl]))
        # ---- 2. one-step surprise for frames 1..60 (index i -> frame i+1, event_labels convention)
        t60 = np.arange(0, 60)
        p1 = rollout(model, enc[t60], pc[t60], acts[t60].unsqueeze(0), 1)
        ev = label_episode(ep["observations"], layout)
        jl = ev["junction_arrival"]
        jd = dilate_events(jl, 2)
        labels_exact.append(jl)
        labels_dil.append(jd)
        ctypes += cell_types_for_frames(ep["observations"][1:WINDOW, :2], layout)
        for c, sl in (("obs", slice(0, D_OBS_CH)), ("proprio", slice(D_OBS_CH, None))):
            s = norms(p1[:, sl] - enc[t60 + 1][:, sl]).numpy()
            surprise[c].append(s)
            if jl.any() and not jl.all():
                per_ep_auroc[c].append(roc_auc_score(jl, s))
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(chosen)}", flush=True)

    res = {"n_episodes": len(chosen), "action_sensitivity": {}, "surprise": {}}
    for k in KS:
        res["action_sensitivity"][k] = {}
        for c in ("obs", "proprio"):
            d = {m: torch.cat(v).numpy() for m, v in acc[k][c].items()}
            res["action_sensitivity"][k][c] = {
                "err_real_mean": float(d["err_real"].mean()), "err_shuffled_mean": float(d["err_shuf"].mean()),
                "ratio_real_over_shuffled": float(d["err_real"].mean() / d["err_shuf"].mean()),
                "copy_baseline_mean": float(d["copy"].mean()),
                "ratio_real_over_copy": float(d["err_real"].mean() / d["copy"].mean()),
                "action_effect_mean": float(d["effect"].mean()),
                "action_effect_over_predicted_change": float(d["effect"].mean() / d["pred_change"].mean()),
                "action_effect_over_err_real": float(d["effect"].mean() / d["err_real"].mean()),
            }
    yl = np.concatenate(labels_exact)
    yd = np.concatenate(labels_dil)
    ct = np.array(ctypes)
    for c in ("obs", "proprio"):
        s = np.concatenate(surprise[c])
        res["surprise"][c] = {
            "auroc_junction_arrival_exact_pooled": float(roc_auc_score(yl, s)),
            "auroc_junction_arrival_pm2_pooled": float(roc_auc_score(yd, s)),
            "auroc_junction_arrival_exact_per_episode_mean": float(np.mean(per_ep_auroc[c])),
            "n_episodes_with_arrival": len(per_ep_auroc[c]),
            "mean_by_cell_type": {t: float(s[ct == t].mean()) for t in ("straight", "corner", "dead_end", "junction")},
            "mean_at_arrival_vs_not": [float(s[yl].mean()), float(s[~yl].mean())],
        }
    res["label_rates"] = {"junction_arrival_exact": float(yl.mean()), "junction_arrival_pm2": float(yd.mean()),
                          "cell_type_share": {t: float((ct == t).mean()) for t in ("straight", "corner", "dead_end", "junction")}}
    json.dump(res, open(os.path.join(HERE, "results_action_sensitivity_surprise.json"), "w"), indent=2)
    np.savez(os.path.join(HERE, "surprise_per_frame.npz"), obs=np.concatenate(surprise["obs"]), proprio=np.concatenate(surprise["proprio"]),
             junction_exact=yl, junction_pm2=yd, cell_type=ct, episodes=np.array(chosen))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
