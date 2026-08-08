# Preserved Results

> **Scope.** These are preserved experiment results, not a transcript of a course
> grade. The GPT-5.4 values below are local head-to-head evaluation win rates recorded
> by the project artifacts.

## Reward Model

The preserved reward-model pair accuracy is **0.84375**.

## Method A: Calibrated Reward-Ensemble Reranking

Method A scored a fixed candidate family with a calibrated three-model reward
ensemble and selected the highest-scoring response for each prompt.

| Repeated evaluation | Local win rate |
|---:|---:|
| 1 | 0.8636 |
| 2 | 0.8621 |
| 3 | 0.8571 |
| **Mean** | **0.8609** |

The detailed prompt/response judgments are intentionally not included in the portfolio
snapshot. The aggregate source remains in
[`methodA_repeated_gpt_summary.json`](experiments/methodA_traincalib/methodA_repeated_gpt_summary.json).

## Method B: Rank-Advantage GRPO

The best preserved Method B configuration is policy seed 2 at step 100. Its three
repeated evaluations were:

| Repeated evaluation | Local win rate | Above 0.75 |
|---:|---:|:---:|
| 1 | 0.7258 | No |
| 2 | 0.7636 | Yes |
| 3 | 0.7705 | Yes |
| **Mean** | **0.7533** | **2 of 3** |

The phrase "2 of 3" refers to repeated evaluations of this configuration, not to
training seeds. The aggregate source remains in
[`methodB_summary.json`](experiments/methodB_rank_traincalib_seeds12/methodB_summary.json).

## Interpretation Boundary

The preserved results support two narrow observations:

1. Method A was consistent across its three recorded local evaluations, whose range
   was `0.0065`.
2. Method B's best recorded configuration had a mean above `0.75`, while one of its
   three repeated evaluations was below `0.75`.

No claim about statistical significance, official grading outcome, or performance on
unrecorded evaluations is made here.
