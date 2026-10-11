"""Merge downloaded Kaggle outputs (kaggle_out/<session>/out/results/...) into results/ and
append the per-task sweep results to sweep_raw.jsonl (existing lines are kept; a (task, horizon,
seed) already present is skipped). Run after every session download.

  python analysis/collect.py
"""
import glob
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pre = json.load(open(f"{ROOT}/prereg.json"))
RAW = f"{ROOT}/sweep_raw.jsonl"

n_copied = 0
for src in sorted(glob.glob(f"{ROOT}/kaggle_out/*/out/results/**/*", recursive=True)):
    if not os.path.isfile(src):
        continue
    rel = src.split("/out/results/", 1)[1]
    dst = f"{ROOT}/results/{rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if rel.endswith("metrics.yaml") or rel.endswith("eval_config.yaml") or rel.endswith("planning_compute.json"):
        # for sweep runs these describe the last tiered invocation; keep the one from the latest session
        if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            shutil.copy2(src, dst)
    elif not os.path.exists(dst):
        shutil.copy2(src, dst)
        n_copied += 1

have = set()
if os.path.exists(RAW):
    for line in open(RAW):
        r = json.loads(line)
        have.add((r["task"], r["horizon"], r["seed"]))
added = 0
with open(RAW, "a") as out:
    for seed in pre["seeds_A"] + pre["seeds_B"]:  # seed outer, horizon inner, as run
        for h in pre["horizons_priority"]:
            d = f"{ROOT}/results/ant_hjepa_l3/seed42/sweep/h{h}_s{seed}/chunks"
            for f in sorted(glob.glob(f"{d}/tasks_*.json")):
                task = int(os.path.basename(f)[len("tasks_"):].split("-")[0])
                if (task, h, seed) in have:
                    continue
                c = json.load(open(f))
                assert len(c["episode_successes"]) == 1
                s = c["steps_to_success"][0]
                out.write(json.dumps({"task": task, "horizon": h, "seed": seed, "split": "A" if seed in pre["seeds_A"] else "B",
                                      "success": int(bool(c["episode_successes"][0])),
                                      "env_steps": int(s) if s is not None and s == s else None,
                                      "wall_time": round(c["evaluation_time"], 2)}) + "\n")
                added += 1
print(f"copied {n_copied} new result files, appended {added} trials, sweep_raw.jsonl now has {len(have) + added} trials")
cnt = {}
for line in open(RAW):
    r = json.loads(line)
    cnt[(r["horizon"], r["seed"])] = cnt.get((r["horizon"], r["seed"]), 0) + 1
print({f"h{h}_s{s}": n for (h, s), n in sorted(cnt.items(), key=lambda x: (x[0][1], pre["horizons_priority"].index(x[0][0])))})
