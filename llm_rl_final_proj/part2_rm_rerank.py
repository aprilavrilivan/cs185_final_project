from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from llm_rl_final_proj.models.load import load_reward_model_and_tokenizer, resolve_adapter_path
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs


def split_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def split_float_csv(text: str) -> list[float]:
    return [float(x) for x in split_csv(text)]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_id(row: dict[str, Any]) -> str:
    if "row_id" not in row:
        raise KeyError("row_id missing")
    return str(row["row_id"])


def load_prompt_file(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        rid = row_id(row)
        if rid in by_id:
            raise ValueError(f"duplicate prompt row_id: {rid}")
        if "prompt_messages" not in row:
            raise ValueError(f"prompt row {rid} has no prompt_messages")
        if "prompt_text" not in row:
            raise ValueError(f"prompt row {rid} has no prompt_text")
        by_id[rid] = row

    return rows, by_id


def load_candidate_file(path: str | Path, name: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        rid = row_id(row)
        if rid in by_id:
            raise ValueError(f"duplicate row_id in candidate {name}: {rid}")
        if "response_text" not in row:
            raise ValueError(f"candidate {name}, row_id={rid}, has no response_text")
        by_id[rid] = row

    return by_id


def small_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline fixed-candidate calibrated RM reranking.")
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--candidate_jsonls", required=True)
    ap.add_argument("--candidate_names", required=True)

    ap.add_argument("--reward_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--reward_adapter_paths", required=True)
    ap.add_argument("--reward_calibration_chosen_means", required=True)
    ap.add_argument("--reward_calibration_rejected_means", required=True)

    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--max_prompt_tokens", type=int, default=700)
    ap.add_argument("--max_response_tokens", type=int, default=512)
    ap.add_argument("--per_device_batch_size", type=int, default=8)
    args = ap.parse_args()

    candidate_paths = split_csv(args.candidate_jsonls)
    candidate_names = split_csv(args.candidate_names)
    adapter_paths = split_csv(args.reward_adapter_paths)
    chosen_means = split_float_csv(args.reward_calibration_chosen_means)
    rejected_means = split_float_csv(args.reward_calibration_rejected_means)

    if len(candidate_paths) != len(candidate_names):
        raise ValueError("candidate_jsonls and candidate_names lengths differ")
    if len(adapter_paths) != len(chosen_means) or len(adapter_paths) != len(rejected_means):
        raise ValueError("reward adapters and calibration lists must have the same length")

    centers: list[float] = []
    scales: list[float] = []
    for i, (chosen, rejected) in enumerate(zip(chosen_means, rejected_means)):
        center = 0.5 * (chosen + rejected)
        scale = 0.5 * (chosen - rejected)
        if scale <= 0:
            raise ValueError(
                f"bad calibration for RM {i}: chosen={chosen}, rejected={rejected}, scale={scale}"
            )
        centers.append(center)
        scales.append(scale)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    prompt_rows, prompt_by_id = load_prompt_file(args.prompts_jsonl)
    candidate_maps = [
        load_candidate_file(path, name)
        for path, name in zip(candidate_paths, candidate_names)
    ]

    scoring_rows: list[dict[str, Any]] = []
    flat_items: list[dict[str, Any]] = []

    for prompt_row in prompt_rows:
        rid = row_id(prompt_row)

        for name, cand_map in zip(candidate_names, candidate_maps):
            if rid not in cand_map:
                raise ValueError(f"candidate {name} missing row_id={rid}")

            cand = cand_map[rid]
            response = str(cand.get("response_text") or "").strip()
            if not response:
                response = "[no response]"

            scoring_rows.append(
                {
                    "row_id": f"{rid}:{name}",
                    "prompt_messages": prompt_by_id[rid]["prompt_messages"],
                    "prompt_text": prompt_by_id[rid]["prompt_text"],
                    "response_text": response,
                }
            )
            flat_items.append(
                {
                    "row_id": rid,
                    "candidate_name": name,
                    "candidate_row": cand,
                    "response_text": response,
                }
            )

    total_scores = [0.0 for _ in scoring_rows]
    rm_summaries: list[dict[str, Any]] = []

    for rm_idx, adapter_path in enumerate(adapter_paths):
        adapter_path_resolved = resolve_adapter_path(adapter_path)

        loaded = load_reward_model_and_tokenizer(
            args.reward_model_name,
            device=device,
            dtype=dtype,
            adapter_path=adapter_path_resolved,
        )
        loaded.model.eval()

        raw_scores = score_prompt_response_pairs(
            loaded.model,
            loaded.tokenizer,
            scoring_rows,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            per_device_batch_size=args.per_device_batch_size,
            device=device,
        )

        calibrated_scores = [
            (float(s) - centers[rm_idx]) / scales[rm_idx]
            for s in raw_scores
        ]

        weight = 1.0 / float(len(adapter_paths))
        for j, s in enumerate(calibrated_scores):
            total_scores[j] += weight * s

        rm_summaries.append(
            {
                "adapter_path": adapter_path,
                "center": centers[rm_idx],
                "scale": scales[rm_idx],
                "raw_score": small_stats([float(x) for x in raw_scores]),
                "calibrated_score": small_stats(calibrated_scores),
            }
        )

        del loaded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    num_candidates = len(candidate_names)
    expected = len(prompt_rows) * num_candidates
    if len(total_scores) != expected:
        raise RuntimeError(f"score count mismatch: got {len(total_scores)}, expected {expected}")

    out_rows: list[dict[str, Any]] = []
    chosen_counts: Counter[str] = Counter()
    scores_by_candidate: defaultdict[str, list[float]] = defaultdict(list)
    per_prompt_selection: list[dict[str, Any]] = []

    for i, prompt_row in enumerate(prompt_rows):
        start = i * num_candidates
        stop = start + num_candidates
        local_scores = total_scores[start:stop]

        best_k = max(range(num_candidates), key=lambda k: local_scores[k])
        best_item = flat_items[start + best_k]
        best_name = str(best_item["candidate_name"])
        best_row = dict(best_item["candidate_row"])
        best_response = str(best_item["response_text"])

        chosen_counts[best_name] += 1
        for k, name in enumerate(candidate_names):
            scores_by_candidate[name].append(float(local_scores[k]))

        submit_row: dict[str, Any] = {
            "row_id": prompt_row["row_id"],
            "prompt_text": prompt_row["prompt_text"],
            "response_text": best_response,
        }

        if best_row.get("generated_num_tokens") is not None:
            submit_row["generated_num_tokens"] = best_row["generated_num_tokens"]

        out_rows.append(submit_row)

        per_prompt_selection.append(
            {
                "row_id": prompt_row["row_id"],
                "selected_candidate": best_name,
                "scores": {
                    candidate_names[k]: float(local_scores[k])
                    for k in range(num_candidates)
                },
            }
        )

    write_jsonl(args.output_jsonl, out_rows)

    meta_path = Path(args.output_jsonl).with_suffix(".meta.json")
    meta = {
        "method": "fixed_candidate_calibrated_reward_model_reranking",
        "num_prompts": len(prompt_rows),
        "candidate_names": candidate_names,
        "candidate_jsonls": candidate_paths,
        "reward_model_name": args.reward_model_name,
        "reward_adapter_paths": adapter_paths,
        "chosen_means": chosen_means,
        "rejected_means": rejected_means,
        "centers": centers,
        "scales": scales,
        "selection_counts": dict(chosen_counts),
        "score_by_candidate": {
            name: small_stats(values)
            for name, values in scores_by_candidate.items()
        },
        "reward_model_summaries": rm_summaries,
        "per_prompt_selection": per_prompt_selection,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_jsonl": args.output_jsonl,
                "meta_json": str(meta_path),
                "num_prompts": len(out_rows),
                "selection_counts": dict(chosen_counts),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
