from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import load_dataset

from llm_rl_final_proj.models.load import load_reward_model_and_tokenizer
from llm_rl_final_proj.reward_model.evaluation import score_prompt_response_pairs
from llm_rl_final_proj.utils.hardware import (
    require_cuda_if_requested,
    resolve_device_and_dtype,
)
from llm_rl_final_proj.utils.seed import set_seed


RewardModelEntry = tuple[torch.nn.Module, Any, str]


@dataclass
class Config:
    reward_model_name: str
    reward_adapter_paths: str
    dataset_name: str
    split: str = "test_prefs"

    reward_calibration_chosen_means: str = ""
    reward_calibration_rejected_means: str = ""
    reward_calibration_eps: float = 1e-6
    reward_ensemble_lambda: float = 0.0

    max_prompt_tokens: int = 700
    max_response_tokens: int = 256
    reward_batch_size: int = 16
    limit: int = 0
    seed: int = 0
    output_json: str = ""


def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Evaluate calibrated reward-model ensemble pairwise accuracy on preference split."
    )

    ap.add_argument("--reward_model_name", type=str, required=True)
    ap.add_argument("--reward_adapter_paths", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, required=True)
    ap.add_argument("--split", type=str, default="test_prefs")

    ap.add_argument("--reward_calibration_chosen_means", type=str, required=True)
    ap.add_argument("--reward_calibration_rejected_means", type=str, required=True)
    ap.add_argument("--reward_calibration_eps", type=float, default=1e-6)
    ap.add_argument("--reward_ensemble_lambda", type=float, default=0.0)

    ap.add_argument("--max_prompt_tokens", type=int, default=700)
    ap.add_argument("--max_response_tokens", type=int, default=256)
    ap.add_argument("--reward_batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_json", type=str, default="")

    args = ap.parse_args()
    return Config(**vars(args))


def _parse_csv_paths(raw: str) -> List[str]:
    paths = [x.strip() for x in raw.split(",") if x.strip()]
    if not paths:
        raise ValueError("--reward_adapter_paths must contain at least one path.")
    return paths


def _parse_csv_floats(raw: str, name: str) -> List[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} is required.")
    try:
        return [float(x) for x in vals]
    except ValueError as exc:
        raise ValueError(f"Failed to parse {name}: {raw}") from exc


def _compute_calibration_params(
    chosen_means: Sequence[float],
    rejected_means: Sequence[float],
    *,
    eps: float,
) -> tuple[List[float], List[float]]:
    if len(chosen_means) != len(rejected_means):
        raise ValueError(
            f"chosen_means length {len(chosen_means)} != rejected_means length {len(rejected_means)}"
        )
    if eps <= 0:
        raise ValueError("--reward_calibration_eps must be positive.")

    centers: List[float] = []
    scales: List[float] = []

    for i, (mu_pos, mu_neg) in enumerate(zip(chosen_means, rejected_means)):
        center = 0.5 * (mu_pos + mu_neg)
        scale = 0.5 * (mu_pos - mu_neg)

        if scale <= 0:
            raise ValueError(
                f"Invalid calibration scale for RM {i}: "
                f"chosen_mean={mu_pos}, rejected_mean={mu_neg}, scale={scale}. "
                "Expected chosen_mean > rejected_mean."
            )

        centers.append(float(center))
        scales.append(float(max(scale, eps)))

    return centers, scales


def _load_preference_split(dataset_name: str, split: str):
    """
    Load a preference split.

    Important:
    If dataset_name is a local directory, do NOT call load_dataset(dataset_name, split=split)
    directly, because the directory may contain both *_gen.jsonl and *_prefs.jsonl files
    with different schemas. Instead, explicitly load only {split}.jsonl.
    """
    dataset_path = Path(dataset_name)

    if dataset_path.is_dir():
        split_file = dataset_path / f"{split}.jsonl"
        if not split_file.is_file():
            available = sorted(p.name for p in dataset_path.glob("*.jsonl"))
            raise FileNotFoundError(
                f"Could not find split file: {split_file}\n"
                f"Available jsonl files under {dataset_path}: {available}"
            )

        return load_dataset(
            "json",
            data_files={split: str(split_file)},
            split=split,
        )

    return load_dataset(dataset_name, split=split)


def _load_reward_model_entries(
    *,
    reward_model_name: str,
    adapter_paths: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
) -> List[RewardModelEntry]:
    entries: List[RewardModelEntry] = []

    for adapter_path in adapter_paths:
        loaded = load_reward_model_and_tokenizer(
            reward_model_name,
            device=device,
            dtype=dtype,
            adapter_path=adapter_path,
        )
        loaded.model.eval()
        for p in loaded.model.parameters():
            p.requires_grad_(False)

        entries.append((loaded.model, loaded.tokenizer, adapter_path))

    return entries


def _get_field(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"Could not find any of fields {candidates} in row keys: {list(row.keys())}")


def _normalize_response_text(value: Any) -> str:
    """
    Preference datasets sometimes store chosen/rejected as plain strings,
    and sometimes as message lists. This keeps the scorer input stable.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        # If it is a chat message list, use the last assistant-like content if possible.
        for item in reversed(value):
            if isinstance(item, dict) and "content" in item:
                return str(item["content"])
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        if "text" in value:
            return str(value["text"])
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def _build_pref_rows(
    dataset_name: str,
    split: str,
    limit: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ds = _load_preference_split(dataset_name, split)

    if limit and limit > 0:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))

    chosen_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(ds):
        row = dict(row)

        prompt_messages = _get_field(row, ["prompt_messages", "prompt"])
        chosen_text = _get_field(row, ["chosen_text", "chosen", "chosen_response", "response_chosen"])
        rejected_text = _get_field(row, ["rejected_text", "rejected", "rejected_response", "response_rejected"])

        prompt_text = row.get("prompt_text", "")

        chosen_rows.append(
            {
                "row_id": str(row.get("row_id", idx)),
                "prompt_messages": prompt_messages,
                "prompt_text": prompt_text,
                "response_text": _normalize_response_text(chosen_text),
            }
        )
        rejected_rows.append(
            {
                "row_id": str(row.get("row_id", idx)),
                "prompt_messages": prompt_messages,
                "prompt_text": prompt_text,
                "response_text": _normalize_response_text(rejected_text),
            }
        )

    return chosen_rows, rejected_rows


@torch.no_grad()
def score_with_calibrated_ensemble(
    reward_models: Sequence[RewardModelEntry],
    scoring_rows: Sequence[Dict[str, Any]],
    *,
    calibration_centers: Sequence[float],
    calibration_scales: Sequence[float],
    reward_ensemble_lambda: float,
    max_prompt_tokens: int,
    max_response_tokens: int,
    per_device_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, Dict[str, float]]:
    if len(reward_models) != len(calibration_centers):
        raise ValueError("Number of reward models and calibration centers does not match.")
    if len(reward_models) != len(calibration_scales):
        raise ValueError("Number of reward models and calibration scales does not match.")
    if reward_ensemble_lambda < 0:
        raise ValueError("--reward_ensemble_lambda must be non-negative.")

    raw_tensors: List[torch.Tensor] = []
    calibrated_tensors: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}

    for idx, ((model, tokenizer, adapter_path), center, scale) in enumerate(
        zip(reward_models, calibration_centers, calibration_scales)
    ):
        scores = score_prompt_response_pairs(
            model,
            tokenizer,
            scoring_rows,
            max_prompt_tokens=max_prompt_tokens,
            max_response_tokens=max_response_tokens,
            per_device_batch_size=per_device_batch_size,
            device=device,
        )

        raw = torch.tensor(scores, device=device, dtype=torch.float32)
        calibrated = (raw - float(center)) / float(scale)

        raw_tensors.append(raw)
        calibrated_tensors.append(calibrated)

        metrics[f"member_{idx}_raw_mean"] = float(raw.mean().item())
        metrics[f"member_{idx}_raw_std"] = float(raw.std(unbiased=False).item())
        metrics[f"member_{idx}_calibrated_mean"] = float(calibrated.mean().item())
        metrics[f"member_{idx}_calibrated_std"] = float(calibrated.std(unbiased=False).item())
        metrics[f"member_{idx}_calibration_center"] = float(center)
        metrics[f"member_{idx}_calibration_scale"] = float(scale)
        metrics[f"member_{idx}_adapter_path"] = adapter_path

    raw_matrix = torch.stack(raw_tensors, dim=0)
    calibrated_matrix = torch.stack(calibrated_tensors, dim=0)

    raw_mean = raw_matrix.mean(dim=0)
    raw_disagreement = raw_matrix.std(dim=0, unbiased=False)

    calibrated_mean = calibrated_matrix.mean(dim=0)
    calibrated_disagreement = calibrated_matrix.std(dim=0, unbiased=False)

    combined = calibrated_mean - float(reward_ensemble_lambda) * calibrated_disagreement

    metrics["raw_mean_score_mean"] = float(raw_mean.mean().item())
    metrics["raw_mean_score_std"] = float(raw_mean.std(unbiased=False).item())
    metrics["raw_disagreement_mean"] = float(raw_disagreement.mean().item())
    metrics["raw_disagreement_max"] = float(raw_disagreement.max().item())

    metrics["calibrated_mean_score_mean"] = float(calibrated_mean.mean().item())
    metrics["calibrated_mean_score_std"] = float(calibrated_mean.std(unbiased=False).item())
    metrics["calibrated_disagreement_mean"] = float(calibrated_disagreement.mean().item())
    metrics["calibrated_disagreement_max"] = float(calibrated_disagreement.max().item())

    metrics["combined_score_mean"] = float(combined.mean().item())
    metrics["combined_score_std"] = float(combined.std(unbiased=False).item())
    metrics["reward_ensemble_lambda"] = float(reward_ensemble_lambda)

    return combined, metrics


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    require_cuda_if_requested()

    adapter_paths = _parse_csv_paths(cfg.reward_adapter_paths)
    chosen_means = _parse_csv_floats(
        cfg.reward_calibration_chosen_means,
        "--reward_calibration_chosen_means",
    )
    rejected_means = _parse_csv_floats(
        cfg.reward_calibration_rejected_means,
        "--reward_calibration_rejected_means",
    )

    if len(adapter_paths) != len(chosen_means) or len(adapter_paths) != len(rejected_means):
        raise ValueError(
            f"Length mismatch: adapter_paths={len(adapter_paths)}, "
            f"chosen_means={len(chosen_means)}, rejected_means={len(rejected_means)}"
        )

    centers, scales = _compute_calibration_params(
        chosen_means,
        rejected_means,
        eps=cfg.reward_calibration_eps,
    )

    device, dtype = resolve_device_and_dtype()

    print(f"[setup] device={device} dtype={dtype}")
    print(f"[setup] reward_model_name={cfg.reward_model_name}")
    print(f"[setup] dataset_name={cfg.dataset_name}")
    print(f"[setup] split={cfg.split}")
    print(f"[setup] num_reward_models={len(adapter_paths)}")
    print(f"[setup] reward_ensemble_lambda={cfg.reward_ensemble_lambda}")
    print("[setup] calibration:")
    for i, path in enumerate(adapter_paths):
        print(
            f"  member_{i}: center={centers[i]:.6f}, scale={scales[i]:.6f}, "
            f"chosen_mean={chosen_means[i]:.6f}, rejected_mean={rejected_means[i]:.6f}, "
            f"path={path}"
        )

    print("[data] loading preference rows...")
    chosen_rows, rejected_rows = _build_pref_rows(
        cfg.dataset_name,
        cfg.split,
        cfg.limit,
        cfg.seed,
    )

    if not chosen_rows:
        raise RuntimeError("No preference examples found.")

    print(f"[data] loaded {len(chosen_rows)} preference pairs")

    reward_models = _load_reward_model_entries(
        reward_model_name=cfg.reward_model_name,
        adapter_paths=adapter_paths,
        device=device,
        dtype=dtype,
    )

    print("[eval] scoring chosen responses...")
    chosen_scores, chosen_metrics = score_with_calibrated_ensemble(
        reward_models,
        chosen_rows,
        calibration_centers=centers,
        calibration_scales=scales,
        reward_ensemble_lambda=cfg.reward_ensemble_lambda,
        max_prompt_tokens=cfg.max_prompt_tokens,
        max_response_tokens=cfg.max_response_tokens,
        per_device_batch_size=cfg.reward_batch_size,
        device=device,
    )

    print("[eval] scoring rejected responses...")
    rejected_scores, rejected_metrics = score_with_calibrated_ensemble(
        reward_models,
        rejected_rows,
        calibration_centers=centers,
        calibration_scales=scales,
        reward_ensemble_lambda=cfg.reward_ensemble_lambda,
        max_prompt_tokens=cfg.max_prompt_tokens,
        max_response_tokens=cfg.max_response_tokens,
        per_device_batch_size=cfg.reward_batch_size,
        device=device,
    )

    margins = chosen_scores - rejected_scores
    correct = margins > 0
    ties = margins == 0

    accuracy = correct.float().mean().item()
    tie_rate = ties.float().mean().item()

    result = {
        "dataset_name": cfg.dataset_name,
        "split": cfg.split,
        "num_pairs": len(chosen_rows),
        "reward_model_name": cfg.reward_model_name,
        "reward_adapter_paths": adapter_paths,
        "reward_ensemble_lambda": cfg.reward_ensemble_lambda,
        "calibration_chosen_means": chosen_means,
        "calibration_rejected_means": rejected_means,
        "calibration_centers": centers,
        "calibration_scales": scales,
        "pair_accuracy": float(accuracy),
        "tie_rate": float(tie_rate),
        "chosen_score_mean": float(chosen_scores.mean().item()),
        "chosen_score_std": float(chosen_scores.std(unbiased=False).item()),
        "rejected_score_mean": float(rejected_scores.mean().item()),
        "rejected_score_std": float(rejected_scores.std(unbiased=False).item()),
        "margin_mean": float(margins.mean().item()),
        "margin_std": float(margins.std(unbiased=False).item()),
        "margin_min": float(margins.min().item()),
        "margin_max": float(margins.max().item()),
        "chosen_metrics": chosen_metrics,
        "rejected_metrics": rejected_metrics,
    }

    print("\n========== Calibrated Ensemble Preference Evaluation ==========")
    print(f"dataset_name:        {cfg.dataset_name}")
    print(f"split:               {cfg.split}")
    print(f"num_pairs:           {result['num_pairs']}")
    print(f"pair_accuracy:       {result['pair_accuracy']:.6f}")
    print(f"tie_rate:            {result['tie_rate']:.6f}")
    print(f"chosen_score_mean:   {result['chosen_score_mean']:.6f}")
    print(f"chosen_score_std:    {result['chosen_score_std']:.6f}")
    print(f"rejected_score_mean: {result['rejected_score_mean']:.6f}")
    print(f"rejected_score_std:  {result['rejected_score_std']:.6f}")
    print(f"margin_mean:         {result['margin_mean']:.6f}")
    print(f"margin_std:          {result['margin_std']:.6f}")
    print(f"margin_min:          {result['margin_min']:.6f}")
    print(f"margin_max:          {result['margin_max']:.6f}")
    print("==============================================================\n")

    if cfg.output_json.strip():
        out_path = Path(cfg.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[save] wrote metrics to {out_path}")


if __name__ == "__main__":
    main()