from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, stdev

B_EXP = Path("experiments/methodB_rank_traincalib_seeds12")
RESULTS = B_EXP / "autograder_results"

rows = []

def parse_name(name: str):
    m = re.search(r"seed(\d+)_step(\d+)_eval(\d+)", name)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

for p in sorted(RESULTS.glob("*.json")):
    seed, step, eval_id = parse_name(p.name)

    data = json.loads(p.read_text(encoding="utf-8"))

    part2 = None
    for t in data.get("tests", []):
        if t.get("name") == "part2_best":
            part2 = t
            break

    if part2 is None:
        continue

    output = part2.get("output", "")

    m_online = re.search(r"online=([0-9.]+)", output)
    m_threshold = re.search(r"threshold\s+([0-9.]+)", output)
    m_usable = re.search(r"usable=([0-9]+)", output)

    rows.append({
        "seed": seed,
        "step": step,
        "eval": eval_id,
        "online": float(m_online.group(1)) if m_online else None,
        "threshold": float(m_threshold.group(1)) if m_threshold else None,
        "usable": int(m_usable.group(1)) if m_usable else None,
        "status": part2.get("status"),
        "file": p.name,
        "output": output,
    })

rows = sorted(rows, key=lambda r: (r["seed"], r["step"], r["eval"]))

print("Clean Method B seed1/seed2 results:")
for r in rows:
    print(
        f"seed={r['seed']} step={r['step']} eval={r['eval']} "
        f"online={r['online']} status={r['status']} usable={r['usable']}"
    )
    print("  " + r["output"])

groups = {}
for r in rows:
    if r["online"] is not None and r["seed"] is not None and r["step"] is not None:
        groups.setdefault((r["seed"], r["step"]), []).append(r["online"])

summary_rows = []
for (seed, step), vals in sorted(groups.items()):
    summary_rows.append({
        "seed": seed,
        "step": step,
        "n": len(vals),
        "mean_online": mean(vals),
        "min_online": min(vals),
        "max_online": max(vals),
        "std_online": stdev(vals) if len(vals) > 1 else 0.0,
        "values": vals,
        "num_passed": sum(v >= 0.75 for v in vals),
    })

print("\nAggregated:")
for s in summary_rows:
    print(
        f"seed={s['seed']} step={s['step']} n={s['n']} "
        f"mean={s['mean_online']:.4f} min={s['min_online']:.4f} "
        f"max={s['max_online']:.4f} std={s['std_online']:.4f} "
        f"passed={s['num_passed']}/{s['n']}"
    )

summary = {
    "method": "Method B: train-calibrated rank-advantage GRPO, seeds 1 and 2",
    "threshold": 0.75,
    "rows": rows,
    "summary_by_seed_step": summary_rows,
}

(B_EXP / "methodB_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

with (B_EXP / "methodB_results.tsv").open("w", encoding="utf-8") as f:
    f.write("seed\tstep\teval\tonline\tstatus\tusable\tfile\toutput\n")
    for r in rows:
        f.write(
            f"{r['seed']}\t{r['step']}\t{r['eval']}\t{r['online']}\t"
            f"{r['status']}\t{r['usable']}\t{r['file']}\t{r['output']}\n"
        )

with (B_EXP / "methodB_summary_table.md").open("w", encoding="utf-8") as f:
    f.write("| Seed | Step | N evals | Mean online | Min | Max | Std | Passes |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in summary_rows:
        f.write(
            f"| {s['seed']} | {s['step']} | {s['n']} | "
            f"{s['mean_online']:.4f} | {s['min_online']:.4f} | "
            f"{s['max_online']:.4f} | {s['std_online']:.4f} | "
            f"{s['num_passed']}/{s['n']} |\n"
        )

print("\nWrote:")
print(B_EXP / "methodB_summary.json")
print(B_EXP / "methodB_results.tsv")
print(B_EXP / "methodB_summary_table.md")
