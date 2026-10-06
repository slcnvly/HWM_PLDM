import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
from pldm_envs.diverse_maze.adaptive_waypoints.compute_changepoints import load_level1
from pldm_envs.diverse_maze.adaptive_waypoints.preprocess import normalize_images, normalize_proprio_vel, normalize_actions  # noqa: E402

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "..", "..", "datasets", "r50_local", "r50_dataset")

model = load_level1(
    os.path.join(HERE, "..", "..", "..", "..", "pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml"),
    os.path.join(DATA_ROOT, "3-9-1-seed248_epoch=3_sample_step=15465472.ckpt"),
    "cpu",
)

splits = torch.load(os.path.join(DATA_ROOT, "main", "data.p"), weights_only=False)
images = np.load(os.path.join(DATA_ROOT, "main", "images.npy"), mmap_mode="r")

window = 61
ep = splits[0]
obs = ep["observations"][:window]
proprio_vel = normalize_proprio_vel(torch.from_numpy(obs[:, 2:4]).float()).unsqueeze(1)
img_seq = normalize_images(torch.from_numpy(np.array(images[0:window])).float().permute(0, 3, 1, 2))
states = img_seq.unsqueeze(1)
actions = normalize_actions(torch.from_numpy(ep["actions"][: window - 1]).float()).unsqueeze(1)

with torch.no_grad():
    result = model.level1.forward_posterior(states, actions, proprio_vel=proprio_vel, encode_only=False)

print("model.level1.predictor.prior_model is None:", model.level1.predictor.prior_model is None)
print("model.level1.predictor.posterior_model is None:", model.level1.predictor.posterior_model is None)
print("model.level1.predictor.z_stochastic:", getattr(model.level1.predictor, "z_stochastic", "N/A"))
print("encodings shape:", result.backbone_output.encodings.shape)
print("proprio_component:", None if result.backbone_output.proprio_component is None else result.backbone_output.proprio_component.shape)
print("location_component:", result.backbone_output.location_component)
print("raw_locations:", result.backbone_output.raw_locations)
print("predictions shape:", result.pred_output.predictions.shape)

# Now try an open-loop 5-step rollout from t=10 using forward_multiple directly.
t = 10
state_encs = result.backbone_output.encodings[t : t + 1]  # (1, B, D)
proprio_in = None
if result.backbone_output.proprio_component is not None:
    proprio_in = result.backbone_output.proprio_component[t : t + 5]
roll_actions = actions[t : t + 5]  # (5, B, A)

with torch.no_grad():
    pred_out = model.level1.predictor.forward_multiple(
        state_encs=state_encs,
        actions=roll_actions,
        T=5,
        proprio=proprio_in,
        compute_posterior=False,
    )
print("rollout predictions shape:", pred_out.predictions.shape)
actual_z_t5 = result.backbone_output.encodings[t + 5]
predicted_z_t5 = pred_out.predictions[-1]
diff = (actual_z_t5 - predicted_z_t5).pow(2).flatten(1).mean(dim=-1)
print("5-step cumulative error at t=10:", diff)

# confirm the rollout actually evolves step to step (not a degenerate no-op
# that just repeats state_encs[0])
step_diffs = [
    (pred_out.predictions[k] - pred_out.predictions[k - 1]).pow(2).mean().item()
    for k in range(1, pred_out.predictions.shape[0])
]
print("rollout step-to-step mean squared deltas (should be nonzero, non-degenerate):", step_diffs)

# sanity: 1-step-only rollout from forward_multiple should roughly match
# compute_error_series's err[t] (teacher-forced 1-step, so should be IDENTICAL
# for the very first step since there's no compounding yet)
err_direct = (result.backbone_output.encodings[t + 1] - result.pred_output.predictions[t]).pow(2).flatten(1).mean(dim=-1)
pred_out_1 = model.level1.predictor.forward_multiple(
    state_encs=result.backbone_output.encodings[t : t + 1],
    actions=actions[t : t + 1],
    T=1,
    proprio=result.backbone_output.proprio_component[t : t + 1] if result.backbone_output.proprio_component is not None else None,
    compute_posterior=False,
)
err_via_rollout = (result.backbone_output.encodings[t + 1] - pred_out_1.predictions[-1]).pow(2).flatten(1).mean(dim=-1)
print("err_direct (from forward_posterior):", err_direct)
print("err_via_rollout (1-step forward_multiple):", err_via_rollout)
print("match:", torch.allclose(err_direct, err_via_rollout, atol=1e-4))
