# Calibrated Reward Ensembles for LLM RLHF

**CS 185/285 final project - portfolio edition**

> **Portfolio provenance.** This repository is a cleaned presentation snapshot prepared
> after course completion from Yifan Xu's original code, experiments, and submitted
> materials. It is not the report submitted for grading and is not an official course
> specification.

This project studies offline preference optimization, reward modeling, and online
reinforcement learning for open-ended instruction following. Experiments use
`Qwen/Qwen2.5-1.5B-Instruct` and a benchmark of 5,000 preference pairs.

The portfolio snapshot keeps the implementation, experiment configurations,
reproduction notes, manifests, and aggregate results. Raw prompts, responses,
per-example generations, model checkpoints, and machine-generated caches are omitted
to reduce privacy and repository-hygiene risk.

## Results at a Glance

| Component | Preserved result |
|---|---:|
| Reward model | 0.84375 pair accuracy |
| Method A: calibrated reward-ensemble reranking | 0.8609 mean local GPT-5.4 win rate |
| Method B: rank-advantage GRPO | 0.7533 best three-evaluation mean |

Method A's three local evaluations were `0.8636`, `0.8621`, and `0.8571`.
For Method B's best preserved configuration, two of three repeated evaluations
exceeded `0.75`; this statement refers to repeated evaluations, not to the number of
training seeds.

See [RESULTS.md](RESULTS.md) for the complete preserved summary and
[the portfolio technical report](report/llm-rl-final-project-portfolio-report.pdf)
for a compact project narrative.

## Project Structure

```text
llm_rl_final_proj/       Training, sampling, reward-model, and RL implementations
scripts/                 Modal training and evaluation entry points
experiments/             Method A/B configurations and aggregate summaries
dataset/                 Dataset metadata only; raw examples are excluded
public_eval/             Evaluation manifest only; prompts and responses are excluded
student_autograder/      Local evaluation harness (requires authorized evaluation assets)
report/                  Post-course portfolio technical report and build script
```

## Implemented Methods

The codebase contains offline preference objectives (DPO, IPO, and AOT), a
Bradley-Terry-style reward-model pipeline, and online policy optimization code for
GRPO, DrGRPO, and GSPO.

The two project investigations preserved here are:

- **Method A - calibrated reward-ensemble reranking.** Three reward models are
  normalized with training-set score statistics, averaged, and used to select among a
  fixed family of policy candidates for each prompt.
- **Method B - rank-advantage GRPO.** Within each four-response group, calibrated
  ensemble scores are converted to rank-only advantages
  `[-1, -1/3, 1/3, 1]` before the GRPO update.

The method-specific records are in
[`experiments/methodA_traincalib`](experiments/methodA_traincalib) and
[`experiments/methodB_rank_traincalib_seeds12`](experiments/methodB_rank_traincalib_seeds12).

## Reproduction Scope

[`README_for_experiments.md`](README_for_experiments.md) preserves the commands used
for the original experiment workflow. The public portfolio snapshot is intentionally
not self-contained: reproduction requires separately authorized dataset and evaluation
assets as well as the relevant model adapters or checkpoints. Those artifacts are not
distributed here.

Environment metadata remains in `pyproject.toml` and `uv.lock`. No API credentials are
required merely to inspect the repository. Training and LLM-based evaluation require
users to configure their own service credentials outside the repository.

## Data and Privacy

Raw dataset and evaluation JSONL files are deliberately excluded. The original files
contained open-domain third-party text, including strings matching common contact-data
patterns. Only aggregate statistics and non-content manifests remain. See
[`dataset/README.md`](dataset/README.md) and
[`public_eval/README.md`](public_eval/README.md).

This cleanup applies to the current repository tree only. Git history has not been
rewritten, so earlier commits may still contain removed artifacts. A fully history-clean
public distribution should be created from this allowlisted tree in a new repository,
unless a separately reviewed history-rewrite process is explicitly approved.

## Attribution and Disclosure

This work was completed in the context of UC Berkeley CS 185/285. Portions of the
repository derive from course starter code; attribution and the applicable MIT terms
are retained in [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

The PDF in `report/` is a post-course portfolio artifact, not the original graded
submission. Editorial organization and typesetting for that report were assisted by
OpenAI Codex; all experimental claims were limited to values preserved in Yifan Xu's
original experiment records.
