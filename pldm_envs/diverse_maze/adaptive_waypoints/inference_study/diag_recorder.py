# Per-replan diagnostics for the inference-design study (INFERENCE_PREREG.md SS5).
# Called from pldm/planning/mpc.py only when config.diag_path is set; it reads
# planner state and never changes what the planner does.
import json
import os

import torch

from pldm.models.misc import Prober
from pldm.models.utils import flatten_conv_output

LOC_MEAN = torch.tensor([4.3646, 4.2948])
LOC_STD = torch.tensor([2.3662, 2.3378])


def _ess(ctrl):
    w = ctrl.omega
    return None if w is None else float(1.0 / (w.double() ** 2).sum())


def _mse(a, b):
    return (flatten_conv_output(a) - flatten_conv_output(b)).pow(2).mean(dim=-1)


class DiagRecorder:
    def __init__(self, config, model, normalizer):
        self.path = config.diag_path
        self.device = next(model.parameters()).device
        self.prober = None
        if getattr(config, "diag_prober_path", "") and os.path.exists(config.diag_prober_path):
            p = Prober((16, 43, 43), arch="conv", output_shape=2, input_dim=(16, 43, 43), arch_subclass="c")
            p.load_state_dict(torch.load(config.diag_prober_path, map_location="cpu", weights_only=False)["state_dict"])
            self.prober = p.to(self.device).eval()
        self.records = {}
        self.trial_ids = []
        self.prev_wp1 = None

    @torch.no_grad()
    def _decode(self, obs):
        """obs: (N,16,43,43) -> (N,2) raw xy, or None without a prober."""
        if self.prober is None:
            return None
        out = self.prober(obs.float().to(self.device)).cpu()
        return out * LOC_STD + LOC_MEAN

    def on_chunk_start(self, envs, trial_ids):
        self.trial_ids = list(trial_ids)
        self.prev_wp1 = [None] * len(envs)
        for j, env in enumerate(envs):
            u = env.unwrapped
            self.records[self.trial_ids[j]] = {
                "trial_id": self.trial_ids[j],
                "map_key": getattr(u, "map_key", None),
                "start_xy": [float(x) for x in getattr(env, "start_xy", env.get_pos())[:2]],
                "target_xy": [float(x) for x in env.get_target()[:2]],
                "block_dist": getattr(u, "block_dist", None),
                "turns": getattr(u, "turns", None),
                "replans": [],
                "steps": {"s1": [], "s2": []},
            }

    @torch.no_grad()
    def on_replan(self, stage, step, planner, planning_result, envs):
        n = len(envs)
        agent = [[float(x) for x in e.get_pos()[:2]] for e in envs]
        if stage == "s1":
            l2, l1 = planning_result.level2, planning_result.level1
            wps = l2.pred_obs  # (T+1, B, 16, 43, 43)
            T = wps.shape[0] - 1
            idx = min(planner.l1_waypoint_index, T)
            wp1 = wps[1]
            dec1 = self._decode(wp1)
            dec2 = self._decode(wps[2]) if T >= 2 else None
            target = wps[idx]
            l1_pred = l1.pred_obs  # (H+1, B, 16, 43, 43)
            final_to_target = _mse(l1_pred[-1], target)
            final_to_wp1 = _mse(l1_pred[-1], wp1)
            per_t = torch.stack([_mse(l1_pred[t], target) for t in range(1, l1_pred.shape[0])])
            for j in range(n):
                move = None if self.prev_wp1[j] is None else float(_mse(wp1[j : j + 1], self.prev_wp1[j])[0])
                self.prev_wp1[j] = wp1[j : j + 1].clone()
                c2, c1 = planner.l2_planner.ctrls[j], planner.l1_planner.ctrls[j]
                self.records[self.trial_ids[j]]["replans"].append({
                    "stage": stage, "step": step, "agent_xy": agent[j],
                    "l2_T": int(T), "l1_target_idx": int(idx),
                    "wp1_xy": None if dec1 is None else dec1[j].tolist(),
                    "wp2_xy": None if dec2 is None else dec2[j].tolist(),
                    "l1_final_cost": float(final_to_target[j]),
                    "l1_final_cost_wp1": float(final_to_wp1[j]),
                    "l1_min_cost": float(per_t[:, j].min()),
                    "l2_best_cost": float(c2.cost_total.min()),
                    "l1_best_cost": float(c1.cost_total.min()),
                    "wp1_move_latent": move,
                    "ess_l2": _ess(c2), "ess_l1": _ess(c1),
                    "l2_clamp_frac": c2.last_clamp_frac,
                })
        else:
            pred = planning_result.pred_obs
            tgt = planner.objective.target_enc
            final = (flatten_conv_output(pred[-1]) - flatten_conv_output(tgt)).pow(2).mean(dim=-1)
            for j in range(n):
                c = planner.ctrls[j]
                self.records[self.trial_ids[j]]["replans"].append({
                    "stage": stage, "step": step, "agent_xy": agent[j],
                    "flat_final_cost": float(final[j]), "flat_best_cost": float(c.cost_total.min()),
                    "ess_flat": _ess(c),
                })

    def on_step(self, stage, step, infos, rewards):
        for j, info in enumerate(infos):
            self.records[self.trial_ids[j]]["steps"][stage].append(
                [float(info["location"][0]), float(info["location"][1]), int(rewards[j] > 0)]
            )

    def on_chunk_end(self):
        existing = {}
        if os.path.exists(self.path):
            existing = json.load(open(self.path))
        for tid in self.trial_ids:
            existing[str(tid)] = self.records[tid]
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(existing, f)
