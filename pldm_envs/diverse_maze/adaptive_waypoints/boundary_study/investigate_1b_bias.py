# Boundary-signal study: follow-up root-cause investigation into the
# Amendment-2 gate-check finding (1b ~= constant bias). Per the user's
# explicit scope: exactly 3 checks, no further digging. Recorded in
# PROGRESS.md as a diagnostic, not a new preregistration amendment (doesn't
# change the signal list).
#
# (1) Training window start position: file:line, see PROGRESS.md -- already
#     confirmed by reading pldm_envs/diverse_maze/d4rl.py:236-237
#     (D4RLDataset.__getitem__): idx indexes a flat cumulative range over
#     ALL (episode, frame) pairs in the dataset, start_idx = idx minus the
#     episode's cumulative offset -- NOT restricted to frame 0. Combined
#     with a shuffling DataLoader, training windows start at arbitrary
#     in-episode positions. This REFUTES the Amendment-2 "only ever
#     anchored at t=0" hypothesis (user's counter-example (a)).
#
# (2) Normalization layers: file:line, see PROGRESS.md -- confirmed by
#     reading pldm/models/utils.py:46-84 (build_conv, used by BOTH MeNet6's
#     encoder conv trunk AND ConvPredictor's conv layers): always
#     nn.GroupNorm (per-sample statistics, batch-independent), never
#     nn.BatchNorm2d, regardless of BackboneConfig.backbone_norm="batch_norm"
#     (encoders/enums.py:28)'s default value -- that config field is not
#     actually consumed by this architecture's conv-building path. No
#     Dropout either (RESULTS.md SS8 already confirmed dropout=0.0
#     everywhere; ConvPredictor itself has no dropout call at all).
#     GroupNorm/LayerNorm/no-dropout should mean model.eval() and
#     model.train() are numerically IDENTICAL for this model -- empirically
#     verified below on a real batch, using a fresh deep copy so no running
#     stats (there aren't any, but just in case) get mutated.
#
# (3) Action-shuffle baseline + bias-corrected cosine + R-squared.
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from signal1_common import get_model, iter_episodes, WINDOW  # noqa: E402

HERE = os.path.dirname(__file__)
N_OBS_CHANNELS = 16
H, W = 43, 43
D_OBS = N_OBS_CHANNELS * H * W
D_FUSED = 18 * H * W
N_MAIN_EPISODES = 1250
N_PROBE_EPISODES = 1000
N_TOTAL_EPISODES = N_MAIN_EPISODES + N_PROBE_EPISODES
N_TOTAL_SAMPLES = N_TOTAL_EPISODES * 60

PRED_ERR_MEMMAP_PATH = os.path.join(HERE, "investigate_pred_err.f32.memmap")
Z_TP1_MEMMAP_PATH = os.path.join(HERE, "investigate_z_tp1.f32.memmap")
REAL_STEP_MEMMAP_PATH = os.path.join(HERE, "investigate_real_step.f32.memmap")
CKPT_PATH = os.path.join(HERE, "investigate_checkpoint.npz")

N_5STEP_EPISODES = 300  # expensive-signal cap, matches PREREGISTRATION.md SS7


def l2(vec_2d):
    return np.linalg.norm(vec_2d, axis=1)


def one_step_batched(model, state_encs_b, actions_b, proprio_b):
    with torch.no_grad():
        out = model.level1.predictor.forward_multiple(
            state_encs=state_encs_b, actions=actions_b, T=1, proprio=proprio_b, compute_posterior=False
        )
    return out.predictions[1]  # (batch, C, H, W)


def check_2_eval_vs_train(model):
    """(2)'s empirical half: same batch through model.eval() and a deep-copied
    model.train() (no_grad throughout, so no gradients/weight updates -- only
    checking whether the forward computation itself differs)."""
    print("\n=== (2) empirical eval() vs train() comparison ===")
    model_train_copy = copy.deepcopy(model)
    model_train_copy.train()

    for ep_idx, ep, images, ep_start in iter_episodes("main", limit=1):
        obs = ep["observations"][:WINDOW]
        proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1)
        img_seq = torch.from_numpy(np.array(images[ep_start : ep_start + WINDOW])).float().permute(0, 3, 1, 2)
        states = img_seq.unsqueeze(1)
        actions = torch.from_numpy(ep["actions"][: WINDOW - 1]).float().unsqueeze(1)

        with torch.no_grad():
            result_eval = model.level1.forward_posterior(states, actions, proprio_vel=proprio_vel, encode_only=False)
            result_train = model_train_copy.level1.forward_posterior(
                states, actions, proprio_vel=proprio_vel, encode_only=False
            )

        enc_eval = result_eval.backbone_output.encodings
        enc_train = result_train.backbone_output.encodings
        pred_eval = result_eval.pred_output.predictions
        pred_train = result_train.pred_output.predictions

        # encodings should be identical (backbone is a pure function of
        # input if no batch-dependent layers exist)
        enc_diff = (enc_eval - enc_train).abs().max().item()
        pred_diff = (pred_eval - pred_train).abs().max().item()

        obs_bias_eval = (enc_eval[1:, :, :N_OBS_CHANNELS] - pred_eval[1:, :, :N_OBS_CHANNELS]).mean().item()
        obs_bias_train = (enc_train[1:, :, :N_OBS_CHANNELS] - pred_train[1:, :, :N_OBS_CHANNELS]).mean().item()

        print(f"ep {ep_idx}: max|encodings_eval - encodings_train| = {enc_diff:.8f}")
        print(f"        max|predictions_eval - predictions_train| = {pred_diff:.8f}")
        print(f"        obs-channel mean bias (eval): {obs_bias_eval:.6f}, (train): {obs_bias_train:.6f}")
        print(f"        CONCLUSION: {'IDENTICAL (no train/eval discrepancy)' if pred_diff < 1e-5 else 'DIFFERENT -- investigate further'}")


def check_5step_shuffle(model):
    """(3)'s 5-step half, capped at N_5STEP_EPISODES (expensive-signal
    precedent, PREREGISTRATION.md SS7). Starting points t=0,5,...,50 (11 per
    episode) to keep compute bounded; for each, real 5-action sequence vs a
    random permutation of those same 5 actions."""
    print(f"\n=== (3) 5-step rollout shuffle check ({N_5STEP_EPISODES} episodes) ===")
    rng = np.random.default_rng(1)
    starts = list(range(0, 55, 5))  # 11 starts: 0,5,...,50

    real_l2 = {"obs": [], "proprio": []}
    shuffled_l2 = {"obs": [], "proprio": []}

    n_done = 0
    for ep_idx, ep, images, ep_start in iter_episodes("main", limit=N_5STEP_EPISODES):
        obs = ep["observations"][:WINDOW]
        proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1)
        img_seq = torch.from_numpy(np.array(images[ep_start : ep_start + WINDOW])).float().permute(0, 3, 1, 2)
        states = img_seq.unsqueeze(1)
        actions = torch.from_numpy(ep["actions"][: WINDOW - 1]).float().unsqueeze(1)

        with torch.no_grad():
            result = model.level1.forward_posterior(states, actions, proprio_vel=proprio_vel, encode_only=False)
        full_encodings = result.backbone_output.encodings
        proprio_component = result.backbone_output.proprio_component

        for t in starts:
            state_encs_b = full_encodings[t : t + 1]  # (1, 1, C, H, W)
            real_actions_5 = actions[t : t + 5]  # (5, 1, A)
            perm5 = rng.permutation(5)
            shuffled_actions_5 = real_actions_5[perm5]
            proprio_in = proprio_component[t : t + 1] if proprio_component is not None else None

            with torch.no_grad():
                pred_real = model.level1.predictor.forward_multiple(
                    state_encs=state_encs_b, actions=real_actions_5, T=5, proprio=proprio_in, compute_posterior=False
                ).predictions[-1].flatten(1).cpu().numpy()  # (1, D)
                pred_shuffled = model.level1.predictor.forward_multiple(
                    state_encs=state_encs_b, actions=shuffled_actions_5, T=5, proprio=proprio_in, compute_posterior=False
                ).predictions[-1].flatten(1).cpu().numpy()

            z_target = full_encodings[t + 5].flatten(1).cpu().numpy()  # (1, D)
            err_real = pred_real - z_target
            err_shuffled = pred_shuffled - z_target

            for sp, (a, b) in (("obs", (0, D_OBS)), ("proprio", (D_OBS, D_FUSED))):
                real_l2[sp].append(l2(err_real[:, a:b])[0])
                shuffled_l2[sp].append(l2(err_shuffled[:, a:b])[0])

        n_done += 1
        if n_done % 100 == 0:
            print(f"(3) 5-step: {n_done}/{N_5STEP_EPISODES} episodes done", flush=True)

    result = {}
    for sp in ("obs", "proprio"):
        r = np.array(real_l2[sp])
        s = np.array(shuffled_l2[sp])
        ratio = float(r.mean() / s.mean())
        result[sp] = {"n_samples": len(r), "real_l2_mean": float(r.mean()), "shuffled_l2_mean": float(s.mean()), "ratio_real_over_shuffled": ratio}
        print(f"5-step {sp}: real={r.mean():.4f}, shuffled={s.mean():.4f}, ratio={ratio:.4f}")
    return result


def episode_iterator():
    g = 0
    for ep_idx, ep, images, ep_start in iter_episodes("main"):
        yield g, ep, images, ep_start
        g += 1
    for ep_idx, ep, images, ep_start in iter_episodes("probe"):
        yield g, ep, images, ep_start
        g += 1


def main():
    model = get_model()

    print("(1) Training window start position: pldm_envs/diverse_maze/d4rl.py:236-237 "
          "(D4RLDataset.__getitem__) -- idx indexes a FLAT cumulative range over ALL "
          "(episode, frame) pairs; start_idx = idx - episode's cumulative offset, NOT "
          "restricted to 0. Training windows start at arbitrary in-episode positions.")
    print("(2) Normalization: pldm/models/utils.py:46-84 (build_conv) always uses "
          "nn.GroupNorm (batch-independent), never nn.BatchNorm2d, for both the MeNet6 "
          "encoder and ConvPredictor. No Dropout in ConvPredictor either.")

    check_2_eval_vs_train(model)

    # (3) Action-shuffle baseline (1-step, full corpus) + bias-corrected
    # cosine + R^2. Same memory-safe streaming/checkpoint pattern as
    # gate_check_1b.py (see PROGRESS.md for why that's necessary on this
    # machine).
    stats = {
        "real": {"copy_l2": [], "pred_l2": [], "cos_sim": []},
        "shuffled": {"copy_l2": [], "pred_l2": []},
    }
    bias_sum_fused = np.zeros(D_FUSED, dtype=np.float64)
    z_tp1_sum_fused = np.zeros(D_FUSED, dtype=np.float64)
    real_step_sum_fused = np.zeros(D_FUSED, dtype=np.float64)
    n_samples_seen = 0
    start_episode = 0

    if os.path.exists(CKPT_PATH):
        ckpt = np.load(CKPT_PATH, allow_pickle=True)
        start_episode = int(ckpt["next_episode"])
        n_samples_seen = int(ckpt["n_samples_seen"])
        bias_sum_fused = ckpt["bias_sum_fused"]
        z_tp1_sum_fused = ckpt["z_tp1_sum_fused"]
        real_step_sum_fused = ckpt["real_step_sum_fused"]
        stats = ckpt["stats"].item()
        print(f"resuming (3) from checkpoint: episode {start_episode}/{N_TOTAL_EPISODES}", flush=True)

    pred_err_memmap = np.memmap(
        PRED_ERR_MEMMAP_PATH, dtype="float32",
        mode="r+" if os.path.exists(PRED_ERR_MEMMAP_PATH) else "w+", shape=(N_TOTAL_SAMPLES, D_FUSED),
    )
    z_tp1_memmap = np.memmap(
        Z_TP1_MEMMAP_PATH, dtype="float32",
        mode="r+" if os.path.exists(Z_TP1_MEMMAP_PATH) else "w+", shape=(N_TOTAL_SAMPLES, D_FUSED),
    )
    real_step_memmap = np.memmap(
        REAL_STEP_MEMMAP_PATH, dtype="float32",
        mode="r+" if os.path.exists(REAL_STEP_MEMMAP_PATH) else "w+", shape=(N_TOTAL_SAMPLES, D_FUSED),
    )

    rng = np.random.default_rng(0)

    for g, ep, images, ep_start in episode_iterator():
        if g < start_episode:
            continue

        obs = ep["observations"][:WINDOW]
        proprio_vel = torch.from_numpy(obs[:, 2:4]).float().unsqueeze(1)
        img_seq = torch.from_numpy(np.array(images[ep_start : ep_start + WINDOW])).float().permute(0, 3, 1, 2)
        states = img_seq.unsqueeze(1)
        actions = torch.from_numpy(ep["actions"][: WINDOW - 1]).float().unsqueeze(1)

        with torch.no_grad():
            result = model.level1.forward_posterior(states, actions, proprio_vel=proprio_vel, encode_only=False)
        full_encodings = result.backbone_output.encodings
        proprio_component = result.backbone_output.proprio_component

        state_encs_b = full_encodings[:60].squeeze(1).unsqueeze(0)
        actions_b = actions[:60].squeeze(1).unsqueeze(0)
        proprio_b = proprio_component[:60].squeeze(1).unsqueeze(0) if proprio_component is not None else None

        pred_real = one_step_batched(model, state_encs_b, actions_b, proprio_b).flatten(1).cpu().numpy()

        perm = rng.permutation(60)
        actions_shuffled_b = actions_b[:, perm]
        pred_shuffled = one_step_batched(model, state_encs_b, actions_shuffled_b, proprio_b).flatten(1).cpu().numpy()

        z_t = full_encodings[:60].squeeze(1).flatten(1).cpu().numpy()
        z_tp1 = full_encodings[1:61].squeeze(1).flatten(1).cpu().numpy()

        real_pred_err = pred_real - z_tp1
        shuffled_pred_err = pred_shuffled - z_tp1
        copy_err = z_tp1 - z_t
        pred_step = pred_real - z_t
        real_step = copy_err

        row0 = g * 60
        pred_err_memmap[row0 : row0 + 60] = real_pred_err
        z_tp1_memmap[row0 : row0 + 60] = z_tp1
        real_step_memmap[row0 : row0 + 60] = real_step
        bias_sum_fused += real_pred_err.sum(axis=0, dtype=np.float64)
        z_tp1_sum_fused += z_tp1.sum(axis=0, dtype=np.float64)
        real_step_sum_fused += real_step.sum(axis=0, dtype=np.float64)
        n_samples_seen += 60

        for sp, (a, b) in (("obs", (0, D_OBS)), ("proprio", (D_OBS, D_FUSED))):
            stats.setdefault(f"real_{sp}", {"copy_l2": [], "pred_l2": [], "cos_sim": []})
            stats.setdefault(f"shuffled_{sp}", {"pred_l2": []})
            c_l2 = l2(copy_err[:, a:b])
            p_l2 = l2(real_pred_err[:, a:b])
            ps_l2 = l2(shuffled_pred_err[:, a:b])
            denom = l2(pred_step[:, a:b]) * l2(real_step[:, a:b])
            cos = (pred_step[:, a:b] * real_step[:, a:b]).sum(axis=1) / np.clip(denom, 1e-8, None)
            stats[f"real_{sp}"]["copy_l2"].append(c_l2)
            stats[f"real_{sp}"]["pred_l2"].append(p_l2)
            stats[f"real_{sp}"]["cos_sim"].append(cos)
            stats[f"shuffled_{sp}"]["pred_l2"].append(ps_l2)

        if (g + 1) % 100 == 0:
            pred_err_memmap.flush()
            z_tp1_memmap.flush()
            real_step_memmap.flush()
            np.savez(
                CKPT_PATH, next_episode=g + 1, n_samples_seen=n_samples_seen,
                bias_sum_fused=bias_sum_fused, z_tp1_sum_fused=z_tp1_sum_fused,
                real_step_sum_fused=real_step_sum_fused, stats=stats,
            )
            print(f"(3) {g + 1}/{N_TOTAL_EPISODES} episodes done (checkpointed)", flush=True)

    pred_err_memmap.flush()
    z_tp1_memmap.flush()
    real_step_memmap.flush()
    print(f"(3) pass 1 complete: {n_samples_seen} samples", flush=True)

    bias_vec_fused = (bias_sum_fused / n_samples_seen).astype(np.float32)
    mean_z_tp1_fused = (z_tp1_sum_fused / n_samples_seen).astype(np.float32)
    mean_real_step_fused = (real_step_sum_fused / n_samples_seen).astype(np.float32)
    mean_pred_step_fused = bias_vec_fused + mean_real_step_fused  # pred_step = pred_err + real_step (linearity)

    residual_l2 = {"obs": [], "proprio": []}
    target_dev_l2 = {"obs": [], "proprio": []}
    debiased_cos = {"obs": [], "proprio": []}
    chunk = 5000
    for i in range(0, N_TOTAL_SAMPLES, chunk):
        pe_block = np.asarray(pred_err_memmap[i : i + chunk])
        zt_block = np.asarray(z_tp1_memmap[i : i + chunk])
        rs_block = np.asarray(real_step_memmap[i : i + chunk])
        pred_step_block = pe_block + rs_block  # reconstruct: pred_step = pred_err + real_step
        for sp, (a, b) in (("obs", (0, D_OBS)), ("proprio", (D_OBS, D_FUSED))):
            res = pe_block[:, a:b] - bias_vec_fused[a:b][None, :]
            residual_l2[sp].append(l2(res))
            dev = zt_block[:, a:b] - mean_z_tp1_fused[a:b][None, :]
            target_dev_l2[sp].append(l2(dev))
            # bias-corrected direction check: subtract the GLOBAL mean
            # predicted-step vector (the model's "average" predicted
            # direction regardless of input), then re-check cosine against
            # the real step -- does the RESIDUAL variation in the model's
            # prediction correlate with real movement, once its constant
            # average direction is removed?
            debiased_pred_step = pred_step_block[:, a:b] - mean_pred_step_fused[a:b][None, :]
            real_step_sp = rs_block[:, a:b]
            denom = l2(debiased_pred_step) * l2(real_step_sp)
            cos = (debiased_pred_step * real_step_sp).sum(axis=1) / np.clip(denom, 1e-8, None)
            debiased_cos[sp].append(cos)
    for sp in ("obs", "proprio"):
        residual_l2[sp] = np.concatenate(residual_l2[sp])
        target_dev_l2[sp] = np.concatenate(target_dev_l2[sp])
        debiased_cos[sp] = np.concatenate(debiased_cos[sp])
    print("(3) pass 2 (R^2 / bias-corrected residuals+cosine) complete", flush=True)

    out = {}
    for sp in ("obs", "proprio"):
        copy_l2 = np.concatenate(stats[f"real_{sp}"]["copy_l2"])
        pred_l2 = np.concatenate(stats[f"real_{sp}"]["pred_l2"])
        cos_sim = np.concatenate(stats[f"real_{sp}"]["cos_sim"])
        shuffled_l2 = np.concatenate(stats[f"shuffled_{sp}"]["pred_l2"])

        ss_res = float((residual_l2[sp] ** 2).sum())
        ss_tot = float((target_dev_l2[sp] ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot

        pred_l2_over_shuffled = float(pred_l2.mean() / shuffled_l2.mean())

        out[sp] = {
            "n_samples": int(len(copy_l2)),
            "real_action_pred_l2_mean": float(pred_l2.mean()),
            "shuffled_action_pred_l2_mean": float(shuffled_l2.mean()),
            "ratio_real_over_shuffled_1step": pred_l2_over_shuffled,
            "direction_cosine_mean": float(cos_sim.mean()),
            "direction_cosine_bias_corrected_mean": float(debiased_cos[sp].mean()),
            "r_squared": r_squared,
            "bias_vector_norm": float(np.linalg.norm(bias_vec_fused[0:D_OBS] if sp == "obs" else bias_vec_fused[D_OBS:D_FUSED])),
        }
        print(f"\n=== space: {sp} ===")
        print(f"real-action pred L2 mean: {pred_l2.mean():.4f}, shuffled-action pred L2 mean: {shuffled_l2.mean():.4f}")
        print(f"ratio (real/shuffled, 1-step): {pred_l2_over_shuffled:.4f} (near 1.0 = model ignores action identity)")
        print(f"direction cosine (real action): {cos_sim.mean():.4f}, bias-corrected: {debiased_cos[sp].mean():.4f}")
        print(f"R^2 (real action, debiased): {r_squared:.4f}")

    five_step_result = check_5step_shuffle(model)
    for sp in ("obs", "proprio"):
        out[sp]["five_step_shuffle"] = five_step_result[sp]

    with open(os.path.join(HERE, "results_investigate_1b_bias.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote results_investigate_1b_bias.json")

    for p in (PRED_ERR_MEMMAP_PATH, Z_TP1_MEMMAP_PATH, REAL_STEP_MEMMAP_PATH, CKPT_PATH):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
