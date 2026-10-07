# Follow-up 2, task 2: is the encoder "spike" a stride-aliasing effect? The agent is
# rendered (evaluator renderer, zero velocity) at 0.01-cell steps along corridors;
# obs latents (eval path) are compared with the reference position and with the
# previous step. The agent's pixel position is measured from the renders to convert
# the latent-jump spacing into pixels and compare it with the encoder's cumulative
# stride (98 -> conv5 s1 -> 94 -> conv5 s2 -> 45 -> conv3 -> 43 -> conv3 p1 -> 1x1: stride 2).
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from f2_task1_spikes_in_eval import eval_path_encode, get_model  # noqa: E402
from f2_task3_latent_vs_maze_by_range import SWEEP  # noqa: E402

INPUTS = os.path.abspath(os.path.join(os.path.dirname(SWEEP), "..", "..", "hwm_sweep_render_inputs", "sweep_render_inputs.json"))


def agent_pixel_centroid(frames):
    """frames (N,3,H,W) uint8 of one sweep: the ball is what differs from the per-pixel median."""
    f = frames.astype(np.float64)
    bg = np.median(f, axis=0)
    out = []
    for x in f:
        w = np.abs(x - bg).sum(0)
        w = np.where(w > 0.5 * w.max(), w, 0)
        ys, xs = np.indices(w.shape)
        out.append([(w * xs).sum() / w.sum(), (w * ys).sum() / w.sum()])
    return np.array(out)  # (col, row)


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 6)))
    model = get_model()
    sw = np.load(SWEEP)
    meta = json.load(open(INPUTS))["sweeps"]
    fig, axes = plt.subplots(len(meta), 1, figsize=(11, 3 * len(meta)))
    res = {"encoder_cumulative_stride_px": 2, "sweeps": []}
    for k, s in enumerate(meta):
        frames = sw[f"sweep{k}"]
        z = eval_path_encode(model, frames)
        offs = np.array(s["offsets_cells"])
        ref = int(np.argmin(np.abs(offs)))
        d_ref = (z - z[ref]).pow(2).mean(1).numpy()
        d_step = np.r_[np.nan, (z[1:] - z[:-1]).pow(2).mean(1).numpy()]
        cen = agent_pixel_centroid(frames)
        coord = cen[:, 0] if np.ptp(cen[:, 0]) > np.ptp(cen[:, 1]) else cen[:, 1]
        px_per_cell = float(np.polyfit(offs, coord, 1)[0])
        jumps = np.where(d_step > 1.0)[0]
        jump_px = coord[jumps]
        # spacing between successive jump positions (in pixels), and their phase modulo 2 px
        spacing = np.diff(np.sort(jump_px)) if len(jump_px) > 1 else np.array([])
        # periodicity of d_ref via FFT over position (cycles per pixel)
        x_px = (offs - offs[0]) * abs(px_per_cell)
        sig = d_ref - d_ref.mean()
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=float(np.mean(np.diff(x_px))))
        top = freqs[1 + int(np.argmax(spec[1:]))]
        res["sweeps"].append({
            "map": s["map"], "axis": s["axis"], "cell": s["cell"], "px_per_cell": px_per_cell,
            "step_px": float(abs(px_per_cell) * 0.01),
            "n_steps_with_latent_jump_gt1": int(len(jumps)), "share_steps_jump": float(len(jumps) / (len(offs) - 1)),
            "jump_positions_px_mod2": np.round(np.mod(jump_px, 2.0), 2).tolist(),
            "jump_spacing_px": np.round(spacing, 2).tolist(),
            "median_jump_spacing_px": float(np.median(spacing)) if len(spacing) else None,
            "d_ref_fft_peak_period_px": float(1 / top) if top > 0 else None,
            "d_ref_at_offsets": {f"{o:+.2f}": float(v) for o, v in zip(offs[::10], d_ref[::10])},
            "d_ref_frac_gt1": float((d_ref > 1).mean()),
        })
        ax = axes[k]
        ax.plot(offs, d_ref, ".-", ms=3, lw=0.8, label="latent MSE to reference (offset 0)")
        ax.plot(offs, d_step, "r.", ms=4, label="latent MSE to previous step")
        ax2 = ax.secondary_xaxis("top", functions=(lambda c, p=px_per_cell: c * abs(p), lambda q, p=px_per_cell: q / abs(p)))
        ax2.set_xlabel("agent shift (pixels)", fontsize=8)
        ax.set_xlabel("agent shift (cells)"); ax.set_title(f"map {s['map']}, sweep along {s['axis']} through cell {tuple(s['cell'])}  ({px_per_cell:.2f} px/cell)", fontsize=9)
        ax.legend(fontsize=7)
        print(k, json.dumps({kk: v for kk, v in res["sweeps"][-1].items() if kk not in ("d_ref_at_offsets", "jump_positions_px_mod2")}), flush=True)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "f2_task2_position_sweep.png"), dpi=110)
    json.dump(res, open(os.path.join(HERE, "results_f2_task2_position_sweep.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
