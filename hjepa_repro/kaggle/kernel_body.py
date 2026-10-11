"""H-JEPA Visual AntMaze: minimal reproduction + top-level horizon sweep, Kaggle runner.

build_kernels.py prepends ROLE / JOBS / limits to this file. The H-JEPA repository is used
unmodified at REPO_COMMIT; every eval is the README command (plus the horizon / task-file
overrides listed in PREREG.md for the sweep). No training.

Roles:
  gen    stage 4: generate 100 extra eval tasks with the OGBench expert (README 3.2 command)
  jobs   run the eval jobs in JOBS (order = priority) with N_WORKERS parallel processes,
         worker k on GPU k % n_gpus; resumable (eval.py skips finished per-task chunk files,
         and results found under /kaggle/input from an earlier session are restored first)
"""
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

REPO_URL = "https://github.com/kevinghst/H-JEPA"
REPO_COMMIT = "97054ecaff83e13839195ec4b92d74c08a0127c5"
TEMP = "/kaggle/temp"
REPO = f"{TEMP}/H-JEPA"
HJ = f"{REPO}/h_jepa"
VENV = f"{TEMP}/venv311"
HJEPA_HOME = f"{TEMP}/hjepa_data"
CK = f"{HJEPA_HOME}/ckpts"
OUT = "/kaggle/working/out"
RES = f"{OUT}/results"
LOGS = f"{OUT}/logs"
T0 = time.time()
for d in (TEMP, HJEPA_HOME, RES, LOGS):
    os.makedirs(d, exist_ok=True)
LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(*a):
    print(f"[{now()} +{time.time() - T0:.0f}s]", *a, flush=True)


def sh(name, cmd, env=None, cwd=None, check=True):
    say(f"$ ({name}) {cmd}")
    t = time.time()
    with open(f"{LOGS}/{name}.log", "a") as log:
        p = subprocess.Popen("set -o pipefail; " + cmd, shell=True, env=env, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, executable="/bin/bash")
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = p.wait()
    say(f"({name}) rc={rc} {time.time() - t:.1f}s")
    if check and rc != 0:
        raise RuntimeError(f"STEP FAILED: {name} rc={rc}")
    return rc


# ----------------------------------------------------------------------------- install (stage 1)
base = dict(os.environ)
base.update({"UV_CACHE_DIR": f"{TEMP}/uvcache", "UV_PYTHON_INSTALL_DIR": f"{TEMP}/uvpython", "PYTHONUNBUFFERED": "1"})
sh("nvidia_smi", "nvidia-smi; nproc; free -g | head -2; df -h /kaggle/working /kaggle/temp | cat", env=base, check=False)
if shutil.which("uv") is None:
    sh("uv_install", f"curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR={TEMP}/uvbin UV_NO_MODIFY_PATH=1 sh", env=base)
    base["PATH"] = f"{TEMP}/uvbin:{TEMP}/uvbin/bin:" + base["PATH"]
sh("clone", f"rm -rf {REPO} && git clone -q {REPO_URL} {REPO} && cd {REPO} && git checkout -q {REPO_COMMIT} && git log -1 --format='%H %ci'", env=base)
sh("venv", f"uv venv --python 3.11.13 {VENV}", env=base)
sh("install", f"cd {REPO} && uv pip install --python {VENV}/bin/python -e '.[train,env]' "
              "--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match 2>&1 | tail -n 30", env=base)
env = dict(base)
env["PATH"] = f"{VENV}/bin:" + env["PATH"]
env["VIRTUAL_ENV"] = VENV
env["MUJOCO_GL"] = "egl"
env["MPLBACKEND"] = "Agg"
env["HJEPA_HOME"] = HJEPA_HOME
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"
env.pop("PYTHONPATH", None)
sh("versions", "python -V && python -c \"import torch, mujoco; print('torch', torch.__version__, 'cuda', torch.version.cuda, "
               "'available', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count(), "
               "[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'mujoco', mujoco.__version__)\"", env=env, cwd=REPO)

# ----------------------------------------------------------------------------- downloads (stage 2)
hf = "hf" if os.path.exists(f"{VENV}/bin/hf") else "huggingface-cli"
sh("dl_tasks", f"{hf} download jepa-world-models/h-jepa --repo-type dataset --local-dir assets --include 'eval_trajs/*' 2>&1 | tail -n 3 "
               "&& (cd assets/eval_trajs && sha256sum -c SHA256SUMS)", env=env, cwd=HJ)
sh("dl_ckpt", f"{hf} download jepa-world-models/h-jepa --local-dir {CK} --include '{CKPT_INCLUDE}' 2>&1 | tail -n 3 "
              f"&& find {CK} -name '*.ckpt' -o -name 'normalizer.pt' | sort | xargs ls -la", env=env, cwd=HJ)
OFFICIAL = f"{HJ}/assets/eval_trajs/ant/expert_grid_d3_n50.pt"


def find_input(pattern):
    """Files under /kaggle/input matching a glob suffix; kernel-source _output_.zip archives are unpacked first."""
    for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
        dst = f"{TEMP}/unz_{abs(hash(z))}"
        if not os.path.exists(dst):
            os.makedirs(dst)
            subprocess.run(f"unzip -q -o {z} -d {dst}", shell=True)
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) + glob.glob(f"{TEMP}/unz_*/**/{pattern}", recursive=True)
    return sorted(hits)


# ----------------------------------------------------------------------------- stage 4: extra tasks
if ROLE == "gen":
    os.makedirs(f"{OUT}/tasks", exist_ok=True)
    gen_out = f"{OUT}/tasks/expert_grid_d3_n100_s7.pt"
    status = {"ok": False}
    try:
        sh("expert_dl", f"mkdir -p {HJEPA_HOME}/ogbench_experts && wget -qO- https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz "
                        f"| tar xz -C {HJEPA_HOME}/ogbench_experts --strip-components=1 experts/ant && ls -la {HJEPA_HOME}/ogbench_experts/ant | head", env=env)
        t = time.time()
        sh("generate", f"python scripts/generate_maze_expert_grid_eval_tasks.py --config-name ant_flat --output-path {gen_out} "
                       f"--d-low 3 --d-high 3 --num-episodes 100 --rollout-budget 125 --seed 7 > {LOGS}/generate_full.log 2>&1; rc=$?; "
                       f"grep -v 'EGL\\|OpenGL\\|eglDestroy\\|cArguments\\|baseOperation\\|result = 0\\|^)$\\|Exception ignored\\|__del__\\|self.free\\|self.close\\|_gl_context\\|raise self._errorClass' "
                       f"{LOGS}/generate_full.log | tail -n 80; exit $rc", env=env, cwd=HJ)
        status = {"ok": os.path.exists(gen_out), "seconds": round(time.time() - t, 1)}
    except Exception as e:
        status["error"] = repr(e)
        say("GENERATION FAILED", repr(e))
    summ = f"""
import json, torch
def tasks(p):
    d = torch.load(p, map_location='cpu', weights_only=False)
    info = d.get('task_info', {{}})
    ts = info.get('tasks', []) if isinstance(info, dict) else []
    out = [{{k: (v if isinstance(v, (int, float, str, list)) else str(v)) for k, v in t.items() if k in
            ('start_ij', 'goal_ij', 'grid_distance', 'segment_len', 'seed', 'start_xy', 'goal_xy')}} for t in ts]
    return {{'n': len(d['data']), 'keys': sorted(d.keys()), 'columns': list(d.get('columns', [])), 'tasks': out,
            'task_info_keys': sorted(info.keys()) if isinstance(info, dict) else None,
            'pixels_shape': list(d['data'][0]['pixels'].shape) if 'pixels' in d['data'][0] else None}}
res = {{'official': tasks('{OFFICIAL}')}}
import os
if os.path.exists('{gen_out}'):
    res['generated'] = tasks('{gen_out}')
    key = lambda t: (tuple(t.get('start_ij', [])), tuple(t.get('goal_ij', [])))
    off = {{key(t) for t in res['official']['tasks']}}
    res['generated_same_start_goal_cells_as_an_official_task'] = [i for i, t in enumerate(res['generated']['tasks']) if key(t) in off]
    res['generated_first50_same_as_official'] = [i for i in res['generated_same_start_goal_cells_as_an_official_task'] if i < 50]
json.dump(res, open('{OUT}/tasks/task_sets.json', 'w'), indent=1)
print('official n', res['official']['n'], 'generated n', res.get('generated', {{}}).get('n'),
      'overlap', len(res.get('generated_same_start_goal_cells_as_an_official_task', [])))
"""
    open(f"{TEMP}/summ.py", "w").write(summ)
    sh("task_summary", f"python {TEMP}/summ.py", env=env, cwd=HJ, check=False)
    json.dump(status, open(f"{OUT}/tasks/gen_status.json", "w"))
    say("GEN_STATUS", json.dumps(status))

# ----------------------------------------------------------------------------- eval jobs (stages 3, 5, 7)
if ROLE == "jobs":
    task_file = None
    if TASK_FILE_NAME:
        hits = find_input(TASK_FILE_NAME)
        assert hits, f"{TASK_FILE_NAME} not found under /kaggle/input"
        task_file = f"{TEMP}/{TASK_FILE_NAME}"
        shutil.copy(hits[0], task_file)
        sh("task_file", f"ls -la {task_file} && md5sum {task_file}", env=env)

    def run_dir(job):  # where eval.py writes: beside the checkpoint
        return f"{CK}/ant/ant_{job['model']}/seed{job['ckpt_seed']}/{job['out']}"

    def res_dir(job):  # what is kept as kernel output
        return f"{RES}/ant_{job['model']}/seed{job['ckpt_seed']}/{job['out']}"

    def n_chunks(d):
        return len(glob.glob(f"{d}/chunks/tasks_*.json"))

    # restore results of earlier sessions (kernel output mounted as input)
    for job in JOBS:
        rel = f"results/ant_{job['model']}/seed{job['ckpt_seed']}/{job['out']}"
        best = max(find_input(f"{rel}/chunks") or [None], key=lambda p: len(os.listdir(p)) if p else -1)
        if best:
            os.makedirs(f"{run_dir(job)}/chunks", exist_ok=True)
            for f in glob.glob(f"{best}/*.json"):
                shutil.copy(f, f"{run_dir(job)}/chunks/")
            say(f"[restore] {job['name']}: {n_chunks(run_dir(job))} chunks from {best}")
    prior_status = {}
    for p in find_input("status.json"):
        try:
            for k, v in json.load(open(p)).get("jobs", {}).items():
                if v.get("wall_seconds", 0) >= prior_status.get(k, {}).get("wall_seconds", -1):
                    prior_status[k] = v
        except Exception:
            pass

    state = {j["name"]: {"state": "pending", "wall_seconds": prior_status.get(j["name"], {}).get("wall_seconds", 0.0),
                         "attempts": 0, "fails": 0} for j in JOBS}
    procs = {}
    stop = threading.Event()
    deadline = T0 + WALL_LIMIT_S

    def sync():
        with LOCK:
            raw = []
            for job in JOBS:
                src, dst = run_dir(job), res_dir(job)
                if not os.path.isdir(src):
                    continue
                os.makedirs(f"{dst}/chunks", exist_ok=True)
                for f in glob.glob(f"{src}/chunks/tasks_*.json"):
                    if not os.path.exists(f"{dst}/chunks/{os.path.basename(f)}"):
                        shutil.copy(f, f"{dst}/chunks/")
                for name in ("metrics.yaml", "eval_config.yaml", "planning_compute.json"):
                    if os.path.exists(f"{src}/{name}"):
                        shutil.copy(f"{src}/{name}", dst)
                state[job["name"]]["chunks"] = n_chunks(dst)
                for f in sorted(glob.glob(f"{dst}/chunks/tasks_*.json")):
                    c = json.load(open(f))
                    start = int(os.path.basename(f)[len("tasks_"):].split("-")[0])
                    for k, ok in enumerate(c["episode_successes"]):
                        s = c["steps_to_success"][k]
                        raw.append({"job": job["name"], "model": job["model"], "ckpt_seed": job["ckpt_seed"], "task": start + k,
                                    "horizon": job.get("horizon"), "seed": job["seed"], "split": job.get("split"),
                                    "success": int(bool(ok)), "env_steps": int(s) if s == s and s is not None else None,
                                    "wall_time": c["evaluation_time"] / len(c["episode_successes"]),
                                    "wall_clock_to_success": c["wall_clock_to_success"][k]})
            with open(f"{OUT}/trials_raw.jsonl", "w") as f:
                for r in raw:
                    f.write(json.dumps(r) + "\n")
            json.dump({"time": now(), "elapsed_s": round(time.time() - T0), "jobs": state}, open(f"{OUT}/status.json", "w"), indent=1)

    def summarize(job):
        d = run_dir(job)
        cs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/chunks/tasks_*.json"))]
        if not cs:
            return "no chunks"
        succ = sum(int(bool(x)) for c in cs for x in c["episode_successes"])
        n = sum(len(c["episode_successes"]) for c in cs)
        tt = sum(c["evaluation_time"] for c in cs)
        return f"tasks={n} successes={succ} eval_time_sum={tt:.0f}s mean_per_task={tt / n:.1f}s"

    # Work items: (tier, job) in order -- every job is first brought to `tier` finished tasks before
    # any job goes on to the next tier, so that a time-limited run leaves all jobs with (nearly) the
    # same number of finished tasks. A tier run is the same eval.py command with eval.num_eval=tier:
    # eval.py evaluates the first num_eval tasks, one chunk file per task, each task seeded by
    # planner seed + task index, and skips chunk files that exist. TIER_STEP = 0: one tier (all tasks).
    def tiers_of(job):
        n = job["n_tasks"]
        if not TIER_STEP:
            return [n]
        if isinstance(TIER_STEP, (list, tuple)):  # explicit tier list
            return sorted({t for t in TIER_STEP if t < n} | {n})
        return sorted(set(list(range(TIER_STEP, n, TIER_STEP)) + [n]))

    all_tiers = sorted({t for j in JOBS for t in tiers_of(j)})
    ITEMS = [(t, j) for t in all_tiers for j in JOBS if t in tiers_of(j)]
    busy = set()

    def item_done(tier, job):
        if n_chunks(run_dir(job)) < tier:
            return False
        return tier < job["n_tasks"] or os.path.exists(f"{run_dir(job)}/metrics.yaml")

    def worker(k, n_gpus):
        wenv = dict(env)
        wenv["CUDA_VISIBLE_DEVICES"] = str(k % n_gpus)
        while not stop.is_set():
            with LOCK:
                pick, waiting = None, False
                for tier, job in ITEMS:
                    st = state[job["name"]]
                    if st["state"].startswith("skipped") or item_done(tier, job):
                        continue
                    if job["name"] in busy:
                        waiting = True
                        continue
                    pick = (tier, job)
                    break
                if pick:
                    tier, job = pick
                    st = state[job["name"]]
                    busy.add(job["name"])
                    st["state"] = "running"
                    st["worker"] = k
            if not pick:
                if not waiting:
                    return
                time.sleep(10)
                continue
            policy = f"{CK}/ant/ant_{job['model']}/seed{job['ckpt_seed']}/ant_{job['model']}_object.ckpt"
            ovr = [o for o in job.get("overrides", []) if not o.startswith("eval.num_eval=")]
            if tier != 50 or any(o.startswith("eval.num_eval=") for o in job.get("overrides", [])):
                ovr.append(f"eval.num_eval={tier}")
            cmd = ["python", "-u", "eval.py", "--config-name", job["config"], f"seed={job['seed']}"] + ovr + \
                  [f"policy={policy}", f"output.dir={run_dir(job)}"]
            st["cmd"] = " ".join(cmd)
            st["attempts"] += 1
            t = time.time()
            say(f"[w{k} gpu{k % n_gpus}] START {job['name']} tier={tier} (attempt {st['attempts']}, chunks so far {n_chunks(run_dir(job))}): {' '.join(cmd)}")
            with open(f"{LOGS}/job_{job['name']}.log", "a") as log:
                log.write(f"\n===== attempt {st['attempts']} tier {tier} {now()} =====\n")
                log.flush()
                p = subprocess.Popen(cmd, cwd=HJ, env=wenv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                procs[k] = p
                rc = p.wait()
            st["wall_seconds"] += time.time() - t
            if stop.is_set():
                st["state"] = "interrupted"
            elif rc == 0 and item_done(tier, job):
                st["fails"] = 0
                st["state"] = "done" if tier >= job["n_tasks"] else "pending"
                say(f"[w{k}] DONE {job['name']} tier={tier}: {summarize(job)} wall={st['wall_seconds']:.0f}s")
            else:
                st["fails"] += 1
                tail = subprocess.run(f"grep -v 'policy call number\\|GradientSolver\\|planning horizon' {LOGS}/job_{job['name']}.log | tail -n 15",
                                      shell=True, capture_output=True, text=True).stdout
                say(f"[w{k}] FAIL {job['name']} tier={tier} rc={rc} (consecutive fails {st['fails']})\n{tail}")
                st["last_error"] = tail[-1500:]
                st["state"] = "skipped_after_3_failures" if st["fails"] >= 3 else "pending"
            with LOCK:
                busy.discard(job["name"])
            sync()

    n_gpus = max(1, len(subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.strip().splitlines()))
    threads = [threading.Thread(target=worker, args=(k, n_gpus), daemon=True) for k in range(N_WORKERS)]
    for th in threads:
        th.start()
        time.sleep(5)
    last = time.time()
    while any(th.is_alive() for th in threads):
        time.sleep(20)
        if time.time() >= deadline and not stop.is_set():
            say("WALL LIMIT reached: stopping eval processes")
            stop.set()
            for p in list(procs.values()):
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        if time.time() - last > 900:
            last = time.time()
            sync()
            gpu = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True,
                                 capture_output=True, text=True).stdout.strip().replace("\n", " | ")
            prog = ", ".join(f"{j['name']}:{state[j['name']]['state'][:4]}:{n_chunks(run_dir(j))}" for j in JOBS
                             if state[j["name"]]["state"] != "pending")
            say(f"PROGRESS gpu[{gpu}] {prog}")
            for j in JOBS:
                if state[j["name"]]["state"] == "running":
                    say(f"  {j['name']}: {summarize(j)}")
    sync()
    say("FINAL", json.dumps({k: (v["state"], v.get("chunks")) for k, v in state.items()}))

shutil.rmtree(f"{TEMP}/uvcache", ignore_errors=True)
say("===== DONE =====")
