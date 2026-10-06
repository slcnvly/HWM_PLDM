# A.2: the shared preprocessing (preprocess.py) must reproduce the evaluation
# pipeline's model inputs exactly. Same batch of real r50 frames, encoded through
#   (eval)  per-frame uint8 HWC -> CHW tensor -> Normalizer.normalize_state, built by
#           Normalizer.build_normalizer(normalizer_hardset=True) as in training
#           (maze_draw.py:86-105), qvel -> normalize_proprio_vel (wrappers.py:297-302)
#   (new)   preprocess.normalize_images / normalize_proprio_vel on the batch
#   (old)   the raw inputs the boundary-study scripts used before the fix
# then the same L1 encoder; torch.equal on inputs and encodings.
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "boundary_study"))
from pldm_envs.utils.normalizer import Normalizer  # noqa: E402
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel, normalize_actions  # noqa: E402
from signal1_common import get_model  # noqa: E402

DATA = os.path.join(ROOT, "pldm_envs/diverse_maze/datasets/r50_local/r50_dataset/main")


class _FakeDS:
    """Minimal dataset for build_normalizer: with normalizer_hardset=True the
    statistics come from STATS; the iteration only has to succeed."""

    def __init__(self):
        self.config = SimpleNamespace(env_name="maze2d_large_diverse")
        self.dataset = self

    def __iter__(self):
        while True:
            yield SimpleNamespace(states=torch.zeros(1, 2, 3, 4, 4), locations=torch.zeros(1, 2, 2),
                                  actions=torch.zeros(1, 2, 2), proprio_vel=torch.zeros(1, 2, 2), proprio_pos=None)


def main():
    torch.set_num_threads(8)
    norm = Normalizer.build_normalizer(_FakeDS(), n_samples=1, normalizer_hardset=True, image_based=True)
    data = torch.load(os.path.join(DATA, "data.p"), weights_only=False)
    images = np.load(os.path.join(DATA, "images.npy"), mmap_mode="r")
    ep = data[0]
    T = 61
    frames = np.array(images[0:T])  # uint8 (T, 98, 98, 3)
    qvel = ep["observations"][:T, 2:4]
    acts = ep["actions"][:T - 1]

    eval_states = torch.stack([norm.normalize_state(torch.from_numpy(np.array(f)).permute(2, 0, 1)) for f in frames])
    eval_pvel = torch.stack([norm.normalize_proprio_vel(torch.from_numpy(q).float()) for q in qvel])
    eval_acts = norm.normalize_action(torch.from_numpy(acts).float())

    new_states = normalize_images(torch.from_numpy(frames).float().permute(0, 3, 1, 2))
    new_pvel = normalize_proprio_vel(torch.from_numpy(qvel).float())
    new_acts = normalize_actions(torch.from_numpy(acts).float())

    old_states = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
    old_pvel = torch.from_numpy(qvel).float()

    model = get_model()
    with torch.no_grad():
        enc_eval = model.level1.backbone(eval_states, proprio=eval_pvel).encodings
        enc_new = model.level1.backbone(new_states, proprio=new_pvel).encodings
        enc_old = model.level1.backbone(old_states, proprio=old_pvel).encodings
        # full forward_posterior on the window (encoder + predictor) as the scripts call it
        fp_eval = model.level1.forward_posterior(eval_states.unsqueeze(1), eval_acts.unsqueeze(1), proprio_vel=eval_pvel.unsqueeze(1))
        fp_new = model.level1.forward_posterior(new_states.unsqueeze(1), new_acts.unsqueeze(1), proprio_vel=new_pvel.unsqueeze(1))

    res = {
        "states_equal": bool(torch.equal(eval_states, new_states)),
        "proprio_vel_equal": bool(torch.equal(eval_pvel, new_pvel)),
        "actions_equal": bool(torch.equal(eval_acts, new_acts)),
        "encodings_equal_eval_vs_new": bool(torch.equal(enc_eval, enc_new)),
        "forward_posterior_predictions_equal": bool(torch.equal(fp_eval.pred_output.predictions, fp_new.pred_output.predictions)),
        "old_vs_eval_encoding_rel_diff": float((enc_old - enc_eval).norm() / enc_eval.norm()),
        "old_input_range": [float(old_states.min()), float(old_states.max())],
        "eval_input_range": [float(eval_states.min()), float(eval_states.max())],
    }
    json.dump(res, open(os.path.join(HERE, "verify_preprocess_result.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
