"""PARTITION_ORACLE runner (see ../PARTITION_ORACLE_PREREG.md). No training.

Subcommands (all run inside the repo's conda env, from the repo root):

  prep    Build the model / normalizer exactly the way `python -m pldm.train` (eval_only)
          does -- same config file, same --values overrides, same L2 latent-bound
          computation under the same RNG seed -- and pickle them into one bundle so the
          light worker processes do not each need the 15 GB dataset in RAM.
            python run.py prep --bundle B.pt --configs CFG --values k=v ...

  gen     Sample start/goal instances with the original rule
          (Maze2DEnvsGenerator: start = uniform sample of the map, goal = random cell at
          BFS block distance 13..16), without building environments.
            python run.py gen --data_path PROBE_DIR --n N --seed S --out F.pt

  worker  Run trials from a task list. One trial = one instance, one partition, one
          planner seed, one environment. Appends one JSON line per finished trial to
          --out and skips every (set, instance, partition, seed) already present there.
          Several workers can share the same task list: a trial is claimed by creating
          <claims>/<key> exclusively.
            python run.py worker --bundle B.pt --tasks T.json --out R.jsonl --claims DIR
                                 --deadline UNIX_TS --wid K

Task list format (JSON): {"sets": {name: path_to_instances.pt}, "tasks": [
    {"set": name, "inst": i, "part": "P47", "l2_len": 47, "seed": 0, "budget": 485}, ...]}
The order of "tasks" is the execution order.
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(AW, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, AW)

LEVEL = "hard"


# ----------------------------------------------------------------------------- grid helpers
def layout_of(map_key):
    return map_key.split("\\")


def cell_of(xy):
    from boundary_study.grid_utils import obs_to_ij  # corrected grid (boundary_study/grid_utils.py:21)
    import numpy as np

    ij = obs_to_ij(np.asarray(xy, dtype=np.float64)[:2])
    return int(ij[0]), int(ij[1])


def bfs_from(layout, src):
    """BFS distances from one cell; same neighborhood rule as latent_law/ab_saturation.py:45 bfs_all."""
    d = {src: 0}
    q = deque([src])
    while q:
        i, j = q.popleft()
        for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + a, j + b)
            if n not in d and 0 <= n[0] < len(layout) and 0 <= n[1] < len(layout[0]) and layout[n[0]][n[1]] == "O":
                d[n] = d[(i, j)] + 1
                q.append(n)
    return d


def open_cell_of(map_key, xy):
    """Cell of a continuous position; if that cell is a wall under the corrected grid
    (the generator samples goals with the uncorrected one, so a goal can sit just across
    a cell border), the open cell whose center is nearest."""
    import numpy as np
    from boundary_study.grid_utils import GRID_SIZE, OBS_MIN_TOTAL

    layout = layout_of(map_key)
    c = cell_of(xy)
    if 0 <= c[0] < len(layout) and 0 <= c[1] < len(layout[0]) and layout[c[0]][c[1]] == "O":
        return c
    xy = np.asarray(xy, dtype=np.float64)[:2]
    best, best_d = None, 1e18
    for i in range(len(layout)):
        for j in range(len(layout[0])):
            if layout[i][j] == "O":
                d = float(((OBS_MIN_TOTAL + (np.array([i, j]) + 0.5) * GRID_SIZE - xy) ** 2).sum())
                if d < best_d:
                    best, best_d = (i, j), d
    return best


def bfs_cells(map_key, a, b):
    """BFS distance between two cells, or -1 if either is not an open cell / unreachable."""
    layout = layout_of(map_key)
    for c in (a, b):
        if not (0 <= c[0] < len(layout) and 0 <= c[1] < len(layout[0]) and layout[c[0]][c[1]] == "O"):
            return -1
    return bfs_from(layout, b).get(a, -1)


# ----------------------------------------------------------------------------- prep
def cmd_prep(args, rest):
    import torch
    from pldm.train import TrainConfig, Trainer, fixed_rng
    from pldm.evaluation.evaluator import Evaluator
    from pldm_envs.utils.normalizer import compute_l2_latent_bounds

    sys.argv = [sys.argv[0]] + rest
    config = TrainConfig.parse_from_command_line()
    trainer = Trainer(config)
    eval_seed = int(config.seed + trainer.epoch * 1009 + 17)  # as Trainer.validate (pldm/train.py:630)
    with fixed_rng(eval_seed):
        trainer.model.eval()
        assert config.eval_cfg.l2_latent_bounds_percentile > 0
        compute_l2_latent_bounds(
            model=trainer.model,
            dataset=trainer.ds,
            normalizer=trainer.ds.normalizer,
            config=config,
            percentile=config.eval_cfg.l2_latent_bounds_percentile,
            device="cuda",
            max_batches=100,
        )
        evaluator = Evaluator(
            config=config.eval_cfg,
            model=trainer.model,
            quick_debug=config.quick_debug,
            normalizer=trainer.ds.normalizer,
            epoch=trainer.epoch,
            probing_datasets=trainer.datasets.probing_datasets,
            l2_probing_datasets=trainer.datasets.l2_probing_datasets,
            load_checkpoint_path=config.load_checkpoint_path,
            output_path=config.output_path,
        )
        probers, probers_l2 = evaluator.evaluate_loc_probing()
    model = getattr(trainer.model, "_orig_mod", trainer.model)  # undo torch.compile's wrapper
    norm = trainer.ds.normalizer
    bundle = {
        "model": model,
        "normalizer": norm,
        "probers": probers,
        "probers_l2": probers_l2,
        "eval_cfg": config.eval_cfg,
        "quick_debug": config.quick_debug,
        "argv": rest,
    }
    torch.save(bundle, args.bundle)
    info = {
        "bundle_bytes": os.path.getsize(args.bundle),
        "l2_latent_min_bounds": [float(x) for x in torch.as_tensor(norm.l2_latent_min_bounds).flatten().tolist()],
        "l2_latent_max_bounds": [float(x) for x in torch.as_tensor(norm.l2_latent_max_bounds).flatten().tolist()],
        "step_skip": int(model.config.step_skip),
        "param_checksum": float(sum(p.double().sum() for p in model.parameters())),
    }
    json.dump(info, open(args.bundle + ".info.json", "w"), indent=1)
    print("PREP_OK", json.dumps(info)[:600], flush=True)


# ----------------------------------------------------------------------------- gen
def cmd_gen(args, rest):
    import random

    import numpy as np
    import torch
    from pldm_envs.diverse_maze.evaluation.envs_generator import sample_location
    from pldm_envs.diverse_maze.evaluation.maze2d_envs_generator import Maze2DEnvsGenerator
    from pldm_envs.diverse_maze.utils import load_uniform

    env_name = "maze2d_large_diverse"
    random.seed(args.seed)  # the goal cell is drawn with the global `random` (utils.py:293)
    np.random.seed(args.seed)
    gen = Maze2DEnvsGenerator(
        env_name=env_name, n_envs=args.n, min_block_radius=args.min_r, max_block_radius=args.max_r,
        action_repeat=4, action_repeat_mode="id", seed=args.seed, stack_states=1, image_obs=True,
        data_path=args.data_path, trials_path=None, unique_shortest_path=False, normalizer=None,
    )
    uniform_ds = load_uniform(env_name, args.data_path)
    map_layouts = torch.load(f"{args.data_path}/train_maps.pt")
    map_keys = list(map_layouts.keys())
    env_keys = map_keys[: args.n] if args.n <= len(map_keys) else np.resize(map_keys, args.n)
    trials = {"starts": [], "targets": [], "map_layouts": [], "block_dists": [], "turns": []}
    extra = []
    for map_idx in env_keys:  # same order of draws as EnvsGenerator.__call__ (envs_generator.py:107-131)
        start = sample_location(uniform_ds[map_idx], rng=gen.rng)[:2]
        target, block_dist, turns = gen._sample_nearby_location(
            anchor=start[:2], map_key=map_layouts[map_idx], min_block_radius=args.min_r, max_block_radius=args.max_r,
        )
        trials["starts"].append(start)
        trials["targets"].append(target)
        trials["map_layouts"].append(map_layouts[map_idx])
        trials["block_dists"].append(block_dist)
        trials["turns"].append(turns)
        sc, tc = open_cell_of(map_layouts[map_idx], start), open_cell_of(map_layouts[map_idx], target)
        extra.append({
            "map_idx": int(map_idx) if not isinstance(map_idx, str) else map_idx,
            "map_key": map_layouts[map_idx],
            "start_xy": [float(x) for x in start[:2]], "target_xy": [float(x) for x in target[:2]],
            "start_cell": list(sc), "target_cell": list(tc),
            "bfs": bfs_cells(map_layouts[map_idx], sc, tc),
            "generator_block_dist": int(block_dist), "turns": int(turns),
        })
    torch.save(trials, args.out)
    json.dump({"seed": args.seed, "n": args.n, "min_r": args.min_r, "max_r": args.max_r, "instances": extra},
              open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print("GEN_OK", args.out, "n=", len(extra), "bfs=", sorted(e["bfs"] for e in extra), flush=True)


# ----------------------------------------------------------------------------- worker
class ForwardCounter:
    """Counts world-model predictor evaluations made through the MPPI dynamics wrapper
    (pldm/planning/planners/mppi_planner.py LearnedDynamics.__call__). One call is one
    batched rollout segment of T predictor steps for K samples; rows = K * T."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.c = {"l1_calls": 0, "l1_rows": 0, "l2_calls": 0, "l2_rows": 0}

    def install(self):
        from pldm.planning.planners import mppi_planner

        orig = mppi_planner.LearnedDynamics.__call__
        counter = self

        def counted(dyn, state, proprio, location, raw_location, action, *a, **k):
            lvl = "l2" if dyn.model is counter.model.level2 else "l1"
            T = 1 if action.dim() < 3 else int(action.shape[0])
            counter.c[f"{lvl}_calls"] += 1
            counter.c[f"{lvl}_rows"] += int(state.shape[1]) * T
            return orig(dyn, state, proprio, location, raw_location, action, *a, **k)

        mppi_planner.LearnedDynamics.__call__ = counted


def task_key(t):
    return f"{t['set']}|{t['inst']}|{t['part']}|{t['seed']}"


def read_done(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                try:
                    done.add(task_key(json.loads(line)))
                except Exception:
                    pass  # a torn last line is simply redone
    return done


def cmd_worker(args, rest):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import types

    import torch

    torch.set_num_threads(1)
    from pldm.evaluation.evaluator import Evaluator
    from pldm.evaluation.evaluator import D4RLPixelMapper
    from pldm_envs.diverse_maze.evaluation import envs_generator

    # EnvsGenerator.__call__ loads the 1000-episode probe data.p on every call but only
    # uses it when it has to sample starts; with trials_path set it is never read.
    envs_generator.load_uniform = lambda *a, **k: {}
    import pldm.planning.mpc as mpc_module

    mpc_module.tqdm = lambda it, **k: it  # no per-step progress bars in the worker logs

    spec = json.load(open(args.tasks))
    bundle = torch.load(args.bundle, weights_only=False)
    model = bundle["model"].cuda().eval()
    eval_cfg = bundle["eval_cfg"]
    base_h = eval_cfg.h_d4rl_planning
    stub = types.SimpleNamespace(
        h_planning_config=base_h,
        quick_debug=bundle["quick_debug"],
        config=eval_cfg,
        normalizer=bundle["normalizer"],
        model=model,
        pixel_mapper=D4RLPixelMapper(env_name=eval_cfg.env_name),
        probers=bundle["probers"],
        probers_l2=bundle["probers_l2"],
        _merge_dataclass_overrides=Evaluator._merge_dataclass_overrides,
    )
    counter = ForwardCounter(model)
    counter.install()

    sets = {name: torch.load(path, weights_only=False) for name, path in spec["sets"].items()}
    tmp_dir = os.path.join(args.claims, f"inst_w{args.wid}")
    os.makedirs(tmp_dir, exist_ok=True)
    n_run = 0
    done = read_done(args.out)  # trials finished in an earlier session; claims cover this one
    for t in spec["tasks"]:
        if args.deadline and time.time() >= args.deadline:
            print(f"[w{args.wid}] deadline reached, stopping", flush=True)
            break
        key = task_key(t)
        if key in done:
            continue
        try:
            fd = os.open(os.path.join(args.claims, key.replace("|", "__") + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        trials = sets[t["set"]]
        i = int(t["inst"])
        inst_file = os.path.join(tmp_dir, "inst.pt")
        torch.save({k: [v[i]] for k, v in trials.items()}, inst_file)
        level_cfg = dataclasses.replace(getattr(base_h, LEVEL), n_envs=1, set_start_target_path=inst_file)
        stub.h_planning_config = dataclasses.replace(
            base_h,
            **{LEVEL: level_cfg},
            n_envs_batch_size=1,
            visualize_planning=False,
            crn=True,
            crn_seed=int(t["seed"]),
            crn_trial_id_offset=i,
            deterministic_algorithms=True,
            diag_path="",
            l2_plan_length_override=int(t["l2_len"]),
            stop_when_all_done=True,
            step_budget=int(t["budget"]),
        )
        counter.reset()
        t0 = time.time()
        ev = Evaluator._create_l2_planning_evaluator(stub, LEVEL, level_cfg)
        assert ev.crn and ev.config.step_budget == int(t["budget"]) and len(ev.envs) == 1
        # Same planner construction and CRN seeding as MPCEvaluator._perform_mpc_in_chunks
        # (pldm/planning/mpc.py:189-210) for a chunk holding this one trial.
        trial_ids = [ev.crn_trial_id_offset]
        planner = ev._construct_planner(n_envs=1)
        planner.set_crn(ev.crn_seed, trial_ids, "flat")
        h_planner = ev._construct_h_planner(n_envs=1)
        h_planner.l1_planner.set_crn(ev.crn_seed, trial_ids, "l1")
        h_planner.l2_planner.set_crn(ev.crn_seed, trial_ids, "l2")
        result = ev._perform_h_mpc(h_planner=h_planner, planner=planner, envs=ev.envs)
        rewards = [bool(r[0]) for r in result.reward_history]
        steps_used = len(rewards)
        term_index = rewards.index(True) if any(rewards) else -1  # as MPCReport.terminations
        success = int(term_index >= 0)
        env = ev.envs[0]
        final_xy = [float(x) for x in env.get_pos()[:2]]
        target_xy = [float(x) for x in env.get_target()[:2]]
        map_key = trials["map_layouts"][i]
        rec = {
            "set": t["set"], "inst": i, "part": t["part"], "l2_len": int(t["l2_len"]), "seed": int(t["seed"]),
            "budget": int(t["budget"]),
            "success": success,
            "steps": term_index + 1 if success else steps_used,  # env steps used (to success, or all executed)
            "term_index": term_index,
            "steps_executed": steps_used,
            "fwd_rows": counter.c["l1_rows"] + counter.c["l2_rows"],
            "fwd_calls": counter.c["l1_calls"] + counter.c["l2_calls"],
            **counter.c,
            "final_xy": final_xy,
            "final_bfs": bfs_cells(map_key, open_cell_of(map_key, final_xy), open_cell_of(map_key, target_xy)),
            "seconds": round(time.time() - t0, 2),
            "wid": args.wid,
            "t_end": round(time.time(), 1),
        }
        ev.close()
        del ev, result, planner, h_planner
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n_run += 1
        print(f"[w{args.wid}] {key} success={rec['success']} steps={rec['steps']} "
              f"final_bfs={rec['final_bfs']} {rec['seconds']}s", flush=True)
    print(f"[w{args.wid}] finished, ran {n_run} trials", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("--bundle", required=True)
    a = sub.add_parser("gen")
    a.add_argument("--data_path", required=True)
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--seed", type=int, required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--min_r", type=int, default=13)
    a.add_argument("--max_r", type=int, default=16)
    a = sub.add_parser("worker")
    a.add_argument("--bundle", required=True)
    a.add_argument("--tasks", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--claims", required=True)
    a.add_argument("--deadline", type=float, default=0)
    a.add_argument("--wid", type=int, default=0)
    args, rest = p.parse_known_args()
    {"prep": cmd_prep, "gen": cmd_gen, "worker": cmd_worker}[args.cmd](args, rest)


if __name__ == "__main__":
    main()
