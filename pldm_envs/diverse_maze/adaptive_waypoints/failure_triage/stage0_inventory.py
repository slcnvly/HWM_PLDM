"""FAILURE_TRIAGE stage 0: inventory of what was saved for the two 90.8% conditions.

Reads only files already in the repo (no model, no GPU):
  A  causal student   -> failure_triage/causal_student_stage5_log_lines.txt
                         (the 12 "stage 5 chunkN" lines of the kernel log; nothing else survived)
  B  surprise minseg8 -> results_hard_surprise_progress.json
  reference pair with per-trial data (other conditions, not A/B):
     results_hard_dp_progress.json vs results_hard_fixed_reeval_progress.json

Per-episode outcomes do not exist for A or B, so the stage 1 set comparison cannot be
done. The only thing the saved per-chunk counts allow is a bound on the overlap, which
is what this script computes. No statistical test is run.

usage: python failure_triage/stage0_inventory.py   (from adaptive_waypoints/)
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
AW = os.path.dirname(HERE)
CHUNK = 20
N_CHUNKS = 6


def chunk_fail_counts_a():
    k = {}
    per_trial_errors = 0
    for line in open(os.path.join(HERE, "causal_student_stage5_log_lines.txt")):
        m = re.search(r"stage 5 chunk(\d): k=(\d+)/(\d+)", line)
        if m:
            assert int(m.group(3)) == CHUNK
            k[int(m.group(1))] = int(m.group(2))
        if "could not extract per-trial data" in line:
            per_trial_errors += 1
    assert sorted(k) == list(range(N_CHUNKS)), k
    return [CHUNK - k[i] for i in range(N_CHUNKS)], per_trial_errors


def chunk_fail_counts_b():
    d = json.load(open(os.path.join(AW, "results_hard_surprise_progress.json")))
    fails, has_per_trial = [], []
    for i in range(N_CHUNKS):
        c = d[f"chunk{i}"]
        assert c["n"] == CHUNK
        fails.append(c["n"] - c["k"])
        has_per_trial.append(c.get("per_trial_success") is not None)
    return fails, has_per_trial


def per_trial_failures(fname):
    d = json.load(open(os.path.join(AW, fname)))
    out = []
    for i in range(N_CHUNKS):
        succ = d[f"chunk{i}"]["per_trial"]["success"]
        assert len(succ) == CHUNK
        out += [CHUNK * i + j for j, s in enumerate(succ) if not s]
    return out


def main():
    fa, a_errors = chunk_fail_counts_a()
    fb, b_has = chunk_fail_counts_b()
    max_both = sum(min(x, y) for x, y in zip(fa, fb))
    min_a_only = sum(max(x - y, 0) for x, y in zip(fa, fb))
    min_b_only = sum(max(y - x, 0) for x, y in zip(fa, fb))
    na, nb = sum(fa), sum(fb)
    res = {
        "note": "Bounds from per-chunk failure counts only. Chunk i = episodes 20i..20i+19. No test.",
        "A_causal_student": {"fail_per_chunk": fa, "n_fail": na, "per_trial_available": False,
                             "per_trial_extraction_errors_in_log": a_errors},
        "B_surprise": {"fail_per_chunk": fb, "n_fail": nb, "per_trial_available": any(b_has)},
        "overlap_bounds": {
            "both_fail_min": 0, "both_fail_max": max_both,
            "A_only_min": min_a_only, "B_only_min": min_b_only,
            "jaccard_min": 0.0, "jaccard_max": max_both / (na + nb - max_both),
            "identical_sets_possible": min_a_only == 0 and min_b_only == 0,
            "chunks_forcing_A_only": [i for i in range(N_CHUNKS) if fa[i] > fb[i]],
            "chunks_forcing_B_only": [i for i in range(N_CHUNKS) if fb[i] > fa[i]],
        },
    }
    dp = set(per_trial_failures("results_hard_dp_progress.json"))
    fx = set(per_trial_failures("results_hard_fixed_reeval_progress.json"))
    res["reference_pair_dp_vs_fixed_reeval"] = {
        "note": "Different conditions (RESULTS.md SS9), same 120 instances, no CRN. Reference only.",
        "dp_only": sorted(dp - fx), "fixed_only": sorted(fx - dp), "both": sorted(dp & fx),
        "jaccard": len(dp & fx) / len(dp | fx),
    }
    with open(os.path.join(HERE, "results_ft_stage0.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
