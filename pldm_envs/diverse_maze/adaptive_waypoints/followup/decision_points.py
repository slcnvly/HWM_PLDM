# Follow-up 3 (2026-10-06): decision-point detection by the spread of L2 futures.
# From a state z_t, sample K=64 L2 latent actions z ~ the planner's
# l2_use_latent_mean_std distribution (mean = data mean, per-dim std = data std x
# noise_sigma(10), clamped to [p2+0.1, p98-0.1]; mppi_planner.py:309-326,
# utils.py:59-72) and, as a secondary, the plain data distribution (std x 1).
# Roll the L2 predictor H=1,2,3 steps; spread = sum over dims of the variance
# across samples of (a) the predicted obs latent and (b) its prober-decoded xy.
#
# modes:  r50   -> 300-episode sample, frames 1..60 step 2, junction labels
#         v0    -> V0 stage-1 replan states rendered by kaggle_render_v0_states
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from common import (DATA, D_OBS_CH, WINDOW, cell_types_for_frames, decode_xy, encode, episode_inputs, episode_starts,
                    load_full_model, load_prober)
from event_labels import label_episode
from metric_a import dilate_events
from run_stage2_predictor_free import sample_episodes
from signals_predictor_free import signal_7_latent_speed
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_actions, normalize_images, normalize_proprio_vel

HERE = os.path.dirname(os.path.abspath(__file__))
K, HS, NOISE_SIGMA, PCT = 64, (1, 2, 3), 10.0, 2.0
FRAMES = list(range(1, 61, 2))


def l2_latent_stats(model, n_windows_per_ep=4, seed=0):
    """Posterior means over normalized 10-step action chunks of r50 main (fixed stride),
    as compute_l2_latent_bounds does (normalizer.py:448-560) but on random windows of every episode."""
    data = torch.load(os.path.join(DATA, "main", "data.p"), weights_only=False)
    rng = np.random.default_rng(seed)
    chunks = []
    for ep in data:
        a = normalize_actions(torch.from_numpy(ep["actions"]).float())
        for _ in range(n_windows_per_ep):
            s = int(rng.integers(0, len(a) - 60 + 1))
            chunks.append(a[s:s + 60].reshape(6, 20))
    with torch.no_grad():
        mu, _ = model.level2.predictor.posterior_model(torch.cat(chunks))
    mu = mu.numpy()
    return {"mean": mu.mean(0), "std": mu.std(0), "lo": np.percentile(mu, PCT, axis=0) + 0.1,
            "hi": np.percentile(mu, 100 - PCT, axis=0) - 0.1, "n_chunks": len(mu)}


def sample_latents(stats, n, std_mult, gen):
    z = torch.from_numpy(stats["mean"]).float() + torch.randn((n, 8), generator=gen) * torch.from_numpy(stats["std"]).float() * std_mult
    return torch.max(torch.min(z, torch.from_numpy(stats["hi"]).float()), torch.from_numpy(stats["lo"]).float())


def spreads(model, prober, enc, pc, stats, std_mult, seed):
    """enc: (N,18,43,43). Returns dict h -> (latent_spread (N,), xy_spread (N,))."""
    gen = torch.Generator().manual_seed(seed)
    out = {h: ([], []) for h in HS}
    for i in range(enc.shape[0]):
        lat = torch.stack([sample_latents(stats, K, std_mult, gen) for _ in HS])  # (H, K, 8)
        z0 = enc[i:i + 1].expand(K, -1, -1, -1)
        p0 = pc[i:i + 1].expand(K, -1, -1, -1)
        with torch.no_grad():
            r = model.level2.predictor.forward_multiple(z0.unsqueeze(0), actions=None, T=len(HS), proprio=p0.unsqueeze(0), latents=lat)
        for h in HS:
            obs = r.obs_component[h]  # (K,16,43,43)
            out[h][0].append(float(obs.flatten(1).var(dim=0, unbiased=False).sum()))
            out[h][1].append(float(decode_xy(prober, obs).var(dim=0, unbiased=False).sum()))
    return {h: (np.array(a), np.array(b)) for h, (a, b) in out.items()}


def auroc(y, s):
    y = np.asarray(y, bool)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None


def run_r50(model, prober, stats):
    # 100-episode stratified sample (4 per map; CPU budget, see PROGRESS.md), a subset of the
    # 300-episode sample whose per-frame surprise values were saved by follow-up 2
    splits, chosen = sample_episodes(n_target=100)
    images = np.load(os.path.join(DATA, "main", "images.npy"), mmap_mode="r")
    maps = torch.load(os.path.join(DATA, "main", "train_maps.pt"), weights_only=False)
    starts = episode_starts(splits)
    sur = np.load(os.path.join(HERE, "surprise_per_frame.npz"))
    sur_pos = {int(e): i for i, e in enumerate(sur["episodes"])}
    rows = {"jexact": [], "jpm2": [], "at_junction": [], "ctype": [], "sig7": [], "sur_obs": [], "sur_prop": []}
    sp = {m: {h: ([], []) for h in HS} for m in ("planner", "data")}
    for n, e in enumerate(chosen):
        ep = splits[e]
        img, pv, _ = episode_inputs(ep, images, starts[e])
        enc, pc = encode(model, img, pv)
        layout = maps[int(ep["map_idx"])].split("\\")
        ev = label_episode(ep["observations"], layout)
        jd = dilate_events(ev["junction_arrival"], 2)
        s7 = signal_7_latent_speed(enc.flatten(1).numpy())
        ct = cell_types_for_frames(ep["observations"][FRAMES, :2], layout)
        for k, f in enumerate(FRAMES):
            rows["jexact"].append(bool(ev["junction_arrival"][f - 1]))
            rows["jpm2"].append(bool(jd[f - 1]))
            rows["at_junction"].append(ct[k] == "junction")
            rows["ctype"].append(ct[k])
            rows["sig7"].append(float(s7[f - 1]))
            if e in sur_pos:
                rows["sur_obs"].append(float(sur["obs"][sur_pos[e] * 60 + f - 1]))
                rows["sur_prop"].append(float(sur["proprio"][sur_pos[e] * 60 + f - 1]))
            else:
                rows["sur_obs"].append(np.nan)
                rows["sur_prop"].append(np.nan)
        for m, mult in (("planner", NOISE_SIGMA), ("data", 1.0)):
            s = spreads(model, prober, enc[FRAMES], pc[FRAMES], stats, mult, seed=int(e))
            for h in HS:
                sp[m][h][0].append(s[h][0])
                sp[m][h][1].append(s[h][1])
        if (n + 1) % 25 == 0:
            print(f"r50 {n + 1}/{len(chosen)}", flush=True)
    ct = np.array(rows["ctype"])
    signals = {"signal_7": np.array(rows["sig7"]), "surprise_obs": np.array(rows["sur_obs"]), "surprise_proprio": np.array(rows["sur_prop"])}
    for m in ("planner", "data"):
        for h in HS:
            signals[f"spread_latent_{m}_H{h}"] = np.concatenate(sp[m][h][0])
            signals[f"spread_xy_{m}_H{h}"] = np.concatenate(sp[m][h][1])
    res = {"n_episodes": len(chosen), "n_points": len(ct), "frames": "1..59 step 2",
           "label_rates": {k: float(np.mean(rows[k])) for k in ("jexact", "jpm2", "at_junction")}, "signals": {}}
    for name, s in signals.items():
        ok = ~np.isnan(s)
        s, ctn = s[ok], ct[ok]
        lab = {k: np.array(rows[k])[ok] for k in ("jexact", "jpm2", "at_junction")}
        res["signals"][name] = {"n_points": int(ok.sum()),
            "auroc_junction_arrival_exact": auroc(lab["jexact"], s),
            "auroc_junction_arrival_pm2": auroc(lab["jpm2"], s),
            "auroc_at_junction_cell": auroc(lab["at_junction"], s),
            "mean_by_cell_type": {t: float(s[ctn == t].mean()) for t in ("straight", "corner", "dead_end", "junction")},
        }
    np.savez(os.path.join(HERE, "decision_points_r50_signals.npz"), ctype=ct, **{k: np.array(v) for k, v in rows.items() if k != "ctype"}, **signals)
    return res


def run_v0(model, prober, stats, npz_path):
    """V0 stage-1 replan states: image rendered at the agent position (exact renderer),
    velocity from the last two recorded positions scaled by the r50 displacement/velocity fit."""
    import analyze_inference as ai
    import carrot_direction  # noqa: F401  (same backward definition)
    from grid_utils import obs_to_ij
    r = np.load(npz_path)
    imgs, tids, ridx = r["images"], r["trial_id"], r["replan_index"]
    scale = velocity_scale()
    sp_rows = {"planner": {h: ([], []) for h in HS}}
    trials = ai.load_variant("v0")
    cand = []  # (image row, velocity, backward, trial, success, step)
    for tid in sorted(set(tids.tolist())):
        t = trials[tid]
        ts = ai.trial_series(t)
        steps = np.array([s_[:2] for s_ in t["diag"]["steps"]["s1"]])
        reps = ts["reps"]
        f = ts["field"]
        for q in np.where(tids == tid)[0]:
            rep = reps[ridx[q]]
            if rep["wp1_xy"] is None:
                continue
            i = rep["step"]
            if i >= 2:
                v = (steps[i - 1] - steps[i - 2]) * scale
            elif i == 1:
                v = (steps[0] - np.asarray(t["diag"]["start_xy"])) * scale
            else:
                v = np.zeros(2)
            ai_, aj = obs_to_ij(np.asarray(rep["agent_xy"]))
            wi, wj = obs_to_ij(np.asarray(rep["wp1_xy"]))
            inside = 0 <= wi < f.shape[0] and 0 <= wj < f.shape[1] and f[wi, wj] >= 0
            cand.append((int(q), v, bool(inside and f[ai_, aj] - f[wi, wj] < 0), tid, int(t["success"]), i))
    # all backward-carrot replans + an equal-size random sample of the other replans (CPU budget)
    rng = np.random.default_rng(0)
    back = [c for c in cand if c[2]]
    other = [c for c in cand if not c[2]]
    pick = back + [other[k] for k in rng.choice(len(other), size=min(len(back), len(other)), replace=False)]
    lab, meta = [], []
    for tid in sorted(set(c[3] for c in pick)):
        rows_t = [c for c in pick if c[3] == tid]
        img = normalize_images(torch.from_numpy(imgs[[c[0] for c in rows_t]]).float())
        pv = normalize_proprio_vel(torch.from_numpy(np.array([c[1] for c in rows_t])).float())
        enc, pc = encode(model, img, pv)
        s = spreads(model, prober, enc, pc, stats, NOISE_SIGMA, seed=int(tid))
        for h in HS:
            sp_rows["planner"][h][0].append(s[h][0])
            sp_rows["planner"][h][1].append(s[h][1])
        lab += [c[2] for c in rows_t]
        meta += [(c[3], c[4], c[5]) for c in rows_t]
    lab = np.array(lab)
    succ = np.array([m[1] for m in meta])
    res = {"n_replans_total": len(cand), "backward_rate_all_replans": float(np.mean([c[2] for c in cand])),
           "n_analysed": len(lab), "n_backward": int(lab.sum()), "velocity_scale": scale, "by_signal": {}}
    for h in HS:
        for kind, j in (("latent", 0), ("xy", 1)):
            s = np.concatenate(sp_rows["planner"][h][j])
            entry = {"mean_backward": float(s[lab].mean()), "mean_other": float(s[~lab].mean()),
                     "auroc_backward": auroc(lab, s),
                     "mannwhitney_p": float(mannwhitneyu(s[lab], s[~lab]).pvalue)}
            for g, gm in (("failed_trials", succ == 0), ("successful_trials", succ == 1)):
                entry[f"auroc_backward_{g}"] = auroc(lab[gm], s[gm])
            res["by_signal"][f"spread_{kind}_planner_H{h}"] = entry
    return res


def velocity_scale():
    """qvel -> per-step displacement factor from r50: dxy_t ~ c * v_{t+1} (least squares)."""
    data = torch.load(os.path.join(DATA, "main", "data.p"), weights_only=False)
    num = den = 0.0
    for ep in data[:300]:
        o = ep["observations"]
        d = o[1:, :2] - o[:-1, :2]
        v = o[1:, 2:4]
        num += float((d * v).sum())
        den += float((v * v).sum())
    return den / num  # v ~ dxy * scale


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    mode = sys.argv[1]
    model = load_full_model()
    prober = load_prober()
    stats = l2_latent_stats(model)
    stats_json = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()}
    if mode == "r50":
        res = run_r50(model, prober, stats)
    else:
        res = run_v0(model, prober, stats, sys.argv[2])
    res["l2_latent_stats"] = stats_json
    json.dump(res, open(os.path.join(HERE, f"results_decision_points_{mode}.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "l2_latent_stats"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
