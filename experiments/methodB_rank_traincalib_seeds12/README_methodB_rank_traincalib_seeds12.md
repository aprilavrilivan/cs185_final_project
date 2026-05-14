# Method B: Train-calibrated rank-advantage GRPO, seeds 1 and 2

## Goal

Method B tests whether GRPO can use within-prompt reward ordering instead of calibrated reward magnitude.

Standard GRPO computes group-relative advantages from reward values. Method B replaces this with rank-only advantages. With group size 4, responses receive:

`[-1, -1/3, 1/3, 1]`

according to their calibrated ensemble reward rank.

## Reward ensemble

We reuse the reward ensemble from Method A:

1. Three reward models trained on `train_prefs`.
2. Reward-model checkpoints selected using held-out `test_prefs` diagnostics.
3. Calibration centers/scales computed from `train_prefs` chosen/rejected score statistics.

The calibrated score for reward model i is:

`z_i(x, y) = (r_i(x, y) - c_i) / s_i`

The ensemble reward is the mean of calibrated scores.

## Online training setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Algorithm: GRPO
- Advantage mode: rank
- Reward ensemble lambda: 0.0
- Train split: `train_gen`
- Eval split: `test_gen`
- Policy seeds included here: 1 and 2
- Evaluated checkpoints: step 75 and step 100
- Submission prompt file: `public_test_gen_prompts_128.jsonl`
- Submission decoding: deterministic, `temperature=0.0`, `top_p=1.0`
- `max_prompt_tokens=700`
- `max_new_tokens=256`

## Cleaned experiment scope

This cleaned directory intentionally excludes the earlier/default B run. It only contains the newly labeled seed1 and seed2 experiments.

## Files

- `candidates/`: deterministic public-eval JSONLs.
- `autograder_results/`: clean local autograder results.
- `candidate_manifest.json`: row counts and hashes for candidates.
- `reward_config/methodB_reward_config.json`: reward model and calibration configuration.
- `methodB_results.tsv`: flat result table.
- `methodB_summary.json`: structured result summary.
- `methodB_summary_table.md`: report-ready table.
- `code_snapshot/`: snapshot/diff of the rank-advantage implementation.

## Interpretation

If the seed1/seed2 rank-GRPO runs pass the online threshold but remain weaker than the original calibrated-ensemble GRPO online best, we treat Method B as a meaningful online ablation rather than the final online-best method.

The main interpretation is that rank-only advantages reduce sensitivity to reward scale but may discard useful calibrated reward-magnitude information.
