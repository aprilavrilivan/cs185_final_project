# README for Experiment Reproduction

This file records the commands used to reproduce our main Part 1 and Part 2 experiments. Before running these commands, upload the dataset to the Modal volume:

```bash
uv run modal volume put llm-rl-final-project-volume dataset/wildchat_min4_judged_5k_v1 /synthetic_datasets/
```

The dataset will be available inside Modal at:

```text
/vol/synthetic_datasets/wildchat_min4_judged_5k_v1
```

All generated submission files are written to:

```text
/vol/submissions/
```

---

# Part 1 DPO

## Training

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo dpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --generation_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_dpo_beta005_v1 \
  --beta 0.005 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --generation_eval_limit 32 \
  --generation_eval_max_new_tokens 256 \
  --generation_eval_every 100 \
  --eval_interval 100 \
  --save_interval 100 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_dpo_beta005_v1
```

## Using the best ckpt to generate response

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_dpo_beta005_v1/checkpoints/step_000500/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/dpo.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Download the generated response

```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations

uv run modal volume get llm-rl-final-project-volume \
  /submissions/dpo.jsonl \
  llm_rl_final_proj_public_submission/policy_generations/
```

---

# Part 1 IPO

## Training

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo ipo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --generation_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_ipo_beta005_v1 \
  --beta 0.005 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --generation_eval_limit 32 \
  --generation_eval_max_new_tokens 256 \
  --generation_eval_every 100 \
  --eval_interval 100 \
  --save_interval 100 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_ipo_beta005_v1
```

## Using the best ckpt to generate response

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_ipo_beta005_v1/checkpoints/step_000300/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/ipo.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Download the generated response

```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations

uv run modal volume get llm-rl-final-project-volume \
  /submissions/ipo.jsonl \
  llm_rl_final_proj_public_submission/policy_generations/
```

---

# Part 1 AOT

## Training

```bash
uv run modal run scripts/modal_train.py::train_remote -- \
  --algo aot \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --generation_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_aot_beta02_v1 \
  --beta 0.2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --grad_accum_steps 4 \
  --lr 5e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --generation_eval_limit 32 \
  --generation_eval_max_new_tokens 256 \
  --generation_eval_every 50 \
  --eval_interval 50 \
  --save_interval 50 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_aot_beta02_v1
```

## Using the best ckpt to generate response

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_aot_beta02_v1/checkpoints/step_000750/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/aot.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Download the generated response

```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations

uv run modal volume get llm-rl-final-project-volume \
  /submissions/aot.jsonl \
  llm_rl_final_proj_public_submission/policy_generations/
```

---

# Part 1 Reward Model Training

## Training

```bash
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --output_dir /vol/runs/wildchat_min4_judged_5k_reward_model_v1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --grad_accum_steps 4 \
  --lr 3e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_reward_model_v1
```

## Using the best ckpt to generate judge

```bash
uv run modal run scripts/modal_train.py::build_reward_model_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000400/adapter \
  --prefs_jsonl /root/project/public_eval/public_test_prefs_256.jsonl \
  --output_jsonl /vol/submissions/public_test_pref_scores.jsonl \
  --max_prompt_tokens 700 \
  --max_response_tokens 512
```

## Download the generated judge

```bash
mkdir -p llm_rl_final_proj_public_submission/reward_model

uv run modal volume get llm-rl-final-project-volume \
  /submissions/public_test_pref_scores.jsonl \
  llm_rl_final_proj_public_submission/reward_model/
```

---

# Part 1 GRPO

## Training

We used the stronger reward-model checkpoint at step 400 and extended online training to 150 steps. Evaluation and checkpointing were performed every 10 steps.

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000400/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_grpo_rm400_steps150_v1 \
  --steps 150 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 256 \
  --eval_limit 32 \
  --eval_interval 10 \
  --save_interval 10 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_grpo_rm400_steps150_v1
```

## Using the best ckpt to generate response

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_grpo_rm400_steps150_v1/checkpoints/step_000150/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/grpo.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Download the generated response

```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations

uv run modal volume get llm-rl-final-project-volume \
  /submissions/grpo.jsonl \
  llm_rl_final_proj_public_submission/policy_generations/
```

---

# Part 1 DrGRPO

## Training

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo dr_grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000400/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_drgrpo_rm400_steps150_v1 \
  --steps 150 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 256 \
  --eval_limit 32 \
  --eval_interval 10 \
  --save_interval 10 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_drgrpo_rm400_steps150_v1
```

## Using the best ckpt to generate response

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_drgrpo_rm400_steps150_v1/checkpoints/step_000120/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/drgrpo.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Download the generated response

```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations

uv run modal volume get llm-rl-final-project-volume \
  /submissions/drgrpo.jsonl \
  llm_rl_final_proj_public_submission/policy_generations/
```

---

# Part 1 GSPO

## Training

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo gspo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_path /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000400/adapter \
  --output_dir /vol/runs/wildchat_min4_judged_5k_gspo_rm400_steps150_v1 \
  --steps 150 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 256 \
  --eval_limit 32 \
  --eval_interval 10 \
  --save_interval 10 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_gspo_rm400_steps150_v1
```

## Using the best ckpt to generate response

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_gspo_rm400_steps150_v1/checkpoints/step_000150/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/gspo.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Download the generated response

```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations

uv run modal volume get llm-rl-final-project-volume \
  /submissions/gspo.jsonl \
  llm_rl_final_proj_public_submission/policy_generations/
```

---

# Part 2 Additional Reward Model Training

We trained two additional reward models using different random seeds. Other hyperparameters were kept the same as the Part 1 reward-model training setup.

```bash
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --output_dir /vol/runs/wildchat_min4_judged_5k_reward_model_seed1_v1 \
  --seed 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --grad_accum_steps 4 \
  --lr 3e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_reward_model_seed1_v1
```

```bash
uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_prefs \
  --eval_split test_prefs \
  --output_dir /vol/runs/wildchat_min4_judged_5k_reward_model_seed2_v1 \
  --seed 2 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --grad_accum_steps 4 \
  --lr 3e-5 \
  --num_train_epochs 3 \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --eval_interval 25 \
  --save_interval 25 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name wildchat_min4_judged_5k_reward_model_seed2_v1
```

---

# Part 2 Reward Ensemble GRPO Training

The calibrated reward ensemble uses three independently trained reward models:

- seed 0: step 400
- seed 2: step 325
- seed 1: step 350

The calibration statistics used in the run were:

```text
chosen means:   3.09375, 5.62500, 2.73438
rejected means: -2.21875, 0.55078, -1.80469
```

## Training

```bash
uv run modal run scripts/modal_train.py::rm_grpo_train_remote -- \
  --algo grpo \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_paths /vol/runs/wildchat_min4_judged_5k_reward_model_v1/checkpoints/step_000400/adapter,/vol/runs/wildchat_min4_judged_5k_reward_model_seed2_v1/checkpoints/step_000325/adapter,/vol/runs/wildchat_min4_judged_5k_reward_model_seed1_v1/checkpoints/step_000350/adapter \
  --reward_calibration_chosen_means=3.09375,5.62500,2.73438 \
  --reward_calibration_rejected_means=-2.21875,0.55078,-1.80469 \
  --reward_calibration_eps 1e-6 \
  --reward_ensemble_lambda 0.0 \
  --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
  --train_split train_gen \
  --eval_split test_gen \
  --output_dir /vol/runs/wildchat_min4_judged_5k_grpo_calibrated_ensemble_v1 \
  --steps 100 \
  --batch_size 16 \
  --group_size 4 \
  --min_new_tokens 32 \
  --max_new_tokens 256 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 1e-5 \
  --grad_accum_steps 2 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --clip_eps 0.2 \
  --kl_coef 0.01 \
  --max_prompt_tokens 700 \
  --max_response_tokens 256 \
  --eval_limit 32 \
  --eval_interval 10 \
  --save_interval 10 \
  --wandb_enabled \
  --wandb_project llm-rl-final-project \
  --wandb_name grpo_calibrated_ensemble_v1
```

## Using the best ckpt to generate Part 2 online response

```bash
uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter_path /vol/runs/wildchat_min4_judged_5k_grpo_calibrated_ensemble_v1/checkpoints/step_000100/adapter \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --output_jsonl /vol/submissions/online_best.jsonl \
  --max_prompt_tokens 700 \
  --max_new_tokens 256 \
  --temperature 0.0 \
  --top_p 1.0
```

## Download the Part 2 online response

```bash
mkdir -p llm_rl_final_proj_public_submission/part2

uv run modal volume get llm-rl-final-project-volume \
  /submissions/online_best.jsonl \
  llm_rl_final_proj_public_submission/part2/
```

---

# Part 2 Fixed-Candidate Reward-Model Reranking

This Part 2 offline experiment uses a train-calibrated reward-model ensemble to select among a fixed candidate family. It does **not** train a new policy adapter.

The candidate family is:

```text
DPO deterministic public generation
IPO deterministic public generation
AOT deterministic public generation
```

For each public prompt, the reranker scores the three candidates with the calibrated reward ensemble and selects the highest-scoring response. We do not sample extra candidates for the public prompts, and we do not use judge feedback for per-prompt selection.

For this experiment, reward-model checkpoints are selected using held-out `test_prefs` diagnostics. Calibration centers and scales are then computed from `train_prefs` score statistics.

```text
test_prefs  -> reward-model checkpoint selection
train_prefs -> reward-scale calibration
public_eval -> final artifact evaluation
```

Selected reward-model checkpoints:

```text
seed0 step 425: /vol/runs/rm__seed0/checkpoints/step_000425/adapter
seed1 step 350: /vol/runs/rm__seed1/checkpoints/step_000350/adapter
seed2 step 325: /vol/runs/rm__seed2/checkpoints/step_000325/adapter
```

Train-prefs calibration values:

```bash
export RM_PATHS="/vol/runs/rm__seed0/checkpoints/step_000425/adapter,/vol/runs/rm__seed1/checkpoints/step_000350/adapter,/vol/runs/rm__seed2/checkpoints/step_000325/adapter"
export CHOSEN_MEANS="-2.187500,2.375000,4.281250"
export REJECTED_MEANS="-9.062500,-3.281250,-2.046875"
```

Because some calibration values are negative, use the equals-sign form when passing them to command-line flags:

```bash
--reward_calibration_chosen_means="$CHOSEN_MEANS"
--reward_calibration_rejected_means="$REJECTED_MEANS"
```

## Train reward models for reranking and rank-advantage GRPO

These reward models are separate from the Part 1 reward-model submission artifact. They are used for the fixed-candidate reranking and rank-advantage GRPO experiments.

```bash
for SEED in 0 1 2; do
  uv run modal run scripts/modal_train.py::reward_model_train_remote -- \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
    --train_split train_prefs \
    --eval_split test_prefs \
    --output_dir /vol/runs/rm__seed${SEED} \
    --seed ${SEED} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --grad_accum_steps 4 \
    --lr 3e-5 \
    --num_train_epochs 3 \
    --max_steps 450 \
    --max_prompt_tokens 700 \
    --max_response_tokens 512 \
    --eval_interval 25 \
    --save_interval 25 \
    --wandb_enabled \
    --wandb_project llm-rl-final-project \
    --wandb_name rm__seed${SEED}
done
```

If W&B credentials are not available, replace `--wandb_enabled` with `--no-wandb_enabled`.

## Evaluate reward-model checkpoints on test_prefs

We evaluate checkpoints on `test_prefs` and choose one checkpoint per seed using pair accuracy, with margin mean as a tie-breaker.

```bash
for SEED in 0 1 2; do
  for STEP in 300 325 350 375 400 425 450; do
    PAD=$(printf "%06d" ${STEP})

    uv run modal run scripts/modal_train.py::reward_model_eval_remote -- \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
      --eval_split test_prefs \
      --adapter_path /vol/runs/rm__seed${SEED}/checkpoints/step_${PAD}/adapter \
      --max_prompt_tokens 700 \
      --max_response_tokens 512 \
      --per_device_eval_batch_size 8 \
      --save_json /vol/rm_eval/no_tag/seed${SEED}_step${STEP}.json
  done
done
```

Download the evaluation JSON files:

```bash
mkdir -p rm_eval_no_tag

for SEED in 0 1 2; do
  for STEP in 300 325 350 375 400 425 450; do
    uv run modal volume get llm-rl-final-project-volume \
      /rm_eval/no_tag/seed${SEED}_step${STEP}.json \
      rm_eval_no_tag/
  done
done
```

The selected checkpoints from our run are recorded in:

```text
experiments/methodA_traincalib/rm_eval_test_prefs/
experiments/methodA_traincalib/selected_rms_no_tag.json
```

## Compute train_prefs calibration statistics

After fixing the selected checkpoints, compute calibration statistics on the full `train_prefs` split. The `--eval_limit 0` flag evaluates the full split.

```bash
export RM_LABELS="seed0_step425,seed1_step350,seed2_step325"
export RM_PATHS="/vol/runs/rm__seed0/checkpoints/step_000425/adapter,/vol/runs/rm__seed1/checkpoints/step_000350/adapter,/vol/runs/rm__seed2/checkpoints/step_000325/adapter"

IFS=',' read -r -a ADAPTERS <<< "$RM_PATHS"
IFS=',' read -r -a LABELS <<< "$RM_LABELS"

for IDX in "${!ADAPTERS[@]}"; do
  ADAPTER="${ADAPTERS[$IDX]}"
  LABEL="${LABELS[$IDX]}"

  uv run modal run scripts/modal_train.py::reward_model_eval_remote -- \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
    --eval_split train_prefs \
    --eval_limit 0 \
    --adapter_path "$ADAPTER" \
    --max_prompt_tokens 700 \
    --max_response_tokens 512 \
    --per_device_eval_batch_size 8 \
    --save_json /vol/rm_calib_train/no_tag/${LABEL}.json
done
```

Download the calibration JSON files:

```bash
mkdir -p rm_calib_train_no_tag

IFS=',' read -r -a LABELS <<< "$RM_LABELS"

for LABEL in "${LABELS[@]}"; do
  uv run modal volume get llm-rl-final-project-volume \
    /rm_calib_train/no_tag/${LABEL}.json \
    rm_calib_train_no_tag/
done
```

The resulting train-prefs calibration files are stored in:

```text
experiments/methodA_traincalib/rm_calib_train_prefs/
experiments/methodA_traincalib/methodA_reward_models_no_tag.json
```

## Prepare fixed DPO / IPO / AOT candidates

We used the deterministic Part 1 offline generations from `llm_zbest_ckpts/`.

```bash
uv run python - <<'PY'
import json
from pathlib import Path

names = ["dpo", "ipo", "aot"]
row_sets = {}

for name in names:
    p = Path(f"llm_zbest_ckpts/{name}.jsonl")
    assert p.exists(), f"missing {p}"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(name, "rows =", len(rows), "keys =", sorted(rows[0].keys()))
    assert len(rows) == 128
    assert all("row_id" in r for r in rows)
    assert all("response_text" in r for r in rows)
    row_sets[name] = {str(r["row_id"]) for r in rows}

base = row_sets["dpo"]
for name in names[1:]:
    assert row_sets[name] == base, f"row_id mismatch: dpo vs {name}"

print("DPO / IPO / AOT fixed candidates look OK.")
PY
```

Upload the candidates to Modal:

```bash
export CAND_REMOTE=/part2_candidates_methodA_no_tag_20260512_1319
export CAND_DIR=/vol${CAND_REMOTE}

uv run modal volume put llm-rl-final-project-volume \
  llm_zbest_ckpts/dpo.jsonl \
  ${CAND_REMOTE}/dpo.jsonl

uv run modal volume put llm-rl-final-project-volume \
  llm_zbest_ckpts/ipo.jsonl \
  ${CAND_REMOTE}/ipo.jsonl

uv run modal volume put llm-rl-final-project-volume \
  llm_zbest_ckpts/aot.jsonl \
  ${CAND_REMOTE}/aot.jsonl
```

If this path already exists, either reuse it or choose a new Modal path and update `CAND_REMOTE` / `CAND_DIR`.

## Run fixed-candidate reward-model reranking

```bash
export METHOD_A_TAG=methodA_no_tag_20260512_1319

uv run modal run scripts/modal_train.py::part2_rm_rerank_remote -- \
  --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
  --candidate_jsonls ${CAND_DIR}/dpo.jsonl,${CAND_DIR}/ipo.jsonl,${CAND_DIR}/aot.jsonl \
  --candidate_names dpo,ipo,aot \
  --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
  --reward_adapter_paths "$RM_PATHS" \
  --reward_calibration_chosen_means="$CHOSEN_MEANS" \
  --reward_calibration_rejected_means="$REJECTED_MEANS" \
  --output_jsonl /vol/submissions/${METHOD_A_TAG}/offline_best.jsonl \
  --max_prompt_tokens 700 \
  --max_response_tokens 512 \
  --per_device_batch_size 8
```

If this runs out of memory, use `--per_device_batch_size 4`.

Download the reranking output and metadata:

```bash
mkdir -p llm_rl_final_proj_public_submission/part2
mkdir -p experiments/methodA_traincalib/artifacts
mkdir -p experiments/methodA_traincalib/debug/methodA_debug_${METHOD_A_TAG}

uv run modal volume get llm-rl-final-project-volume \
  /submissions/${METHOD_A_TAG}/offline_best.jsonl \
  llm_rl_final_proj_public_submission/part2/offline_best.jsonl

uv run modal volume get llm-rl-final-project-volume \
  /submissions/${METHOD_A_TAG}/offline_best.meta.json \
  experiments/methodA_traincalib/debug/methodA_debug_${METHOD_A_TAG}/offline_best.meta.json

cp llm_rl_final_proj_public_submission/part2/offline_best.jsonl \
   experiments/methodA_traincalib/artifacts/offline_best_traincalib.jsonl
```

Expected reranker selection counts:

```text
DPO: 45
IPO: 34
AOT: 49
```

## Evaluate fixed-candidate reranking

We evaluated the same `offline_best.jsonl` three times to estimate judge variance.

```bash
export A_EXP_DIR=experiments/methodA_traincalib
export A_OFFLINE_JSONL=${A_EXP_DIR}/artifacts/offline_best_traincalib.jsonl
mkdir -p ${A_EXP_DIR}/autograder_results

for EVAL_ID in 1 2 3; do
  TMP_DIR=tmp_eval_methodA_offline_eval${EVAL_ID}
  OUT_JSON=${A_EXP_DIR}/autograder_results/methodA_offline_eval${EVAL_ID}.json

  rm -rf ${TMP_DIR}
  mkdir -p ${TMP_DIR}/part2
  mkdir -p ${TMP_DIR}/reward_model

  cp llm_rl_final_proj_public_submission/reward_model/public_test_pref_scores.jsonl \
     ${TMP_DIR}/reward_model/public_test_pref_scores.jsonl

  cp ${A_OFFLINE_JSONL} \
     ${TMP_DIR}/part2/offline_best.jsonl

  uv run python student_autograder/run_local_autograder.py \
    --submission_dir ${TMP_DIR} \
    --output_json ${OUT_JSON}
done
```

Inspect the stored repeated-evaluation summary:

```bash
cat experiments/methodA_traincalib/methodA_repeated_gpt_summary_table.md
```

Expected result:

```text
Eval 1: offline=0.8636, PASS
Eval 2: offline=0.8621, PASS
Eval 3: offline=0.8571, PASS
Mean: 0.8609
```

---

# Part 2 Online Ablation: Rank-Advantage GRPO

Rank-Advantage GRPO reuses the same train-calibrated reward ensemble as fixed-candidate reranking. It changes only the advantage construction: instead of using calibrated reward magnitudes, it assigns advantages by within-group reward rank.

With group size 4, the possible rank advantages are:

```text
[-1, -1/3, 1/3, 1]
```

This is an online ablation. It is not used as the final `online_best.jsonl`; the final online artifact remains the stronger calibrated reward-ensemble GRPO output from the previous section.

## Train rank-advantage GRPO

We trained this variant with policy seeds 1 and 2.

```bash
for POLICY_SEED in 1 2; do
  RUN_TAG=methodB_rank_grpo_traincalib_seed${POLICY_SEED}_v1

  uv run modal run scripts/modal_train.py::rm_grpo_train_remote_h200 -- \
    --algo grpo \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name /vol/synthetic_datasets/wildchat_min4_judged_5k_v1 \
    --train_split train_gen \
    --eval_split test_gen \
    --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
    --reward_adapter_paths "$RM_PATHS" \
    --reward_calibration_chosen_means="$CHOSEN_MEANS" \
    --reward_calibration_rejected_means="$REJECTED_MEANS" \
    --reward_ensemble_lambda 0.0 \
    --output_dir /vol/runs/${RUN_TAG} \
    --seed ${POLICY_SEED} \
    --steps 100 \
    --batch_size 16 \
    --group_size 4 \
    --min_new_tokens 32 \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_p 0.95 \
    --lr 1e-5 \
    --grad_accum_steps 2 \
    --ppo_epochs 2 \
    --minibatch_size 8 \
    --clip_eps 0.2 \
    --kl_coef 0.01 \
    --advantage_mode rank \
    --max_prompt_tokens 700 \
    --max_response_tokens 256 \
    --reward_batch_size 8 \
    --eval_limit 32 \
    --eval_interval 25 \
    --save_interval 25 \
    --wandb_enabled \
    --wandb_project llm-rl-final-project \
    --wandb_name ${RUN_TAG}
done
```

If H200 memory is insufficient, reduce only the batch-related settings:

```bash
--batch_size 8 \
--grad_accum_steps 4 \
--reward_batch_size 4
```

## Build deterministic rank-advantage GRPO submissions

We evaluated steps 75 and 100 for each seed.

```bash
for POLICY_SEED in 1 2; do
  RUN_TAG=methodB_rank_grpo_traincalib_seed${POLICY_SEED}_v1

  for STEP in 75 100; do
    PAD=$(printf "%06d" ${STEP})

    uv run modal run scripts/modal_train.py::build_policy_submission_remote -- \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --adapter_path /vol/runs/${RUN_TAG}/checkpoints/step_${PAD}/adapter \
      --prompts_jsonl /root/project/public_eval/public_test_gen_prompts_128.jsonl \
      --output_jsonl /vol/submissions/${RUN_TAG}/rank_grpo_step_${STEP}.jsonl \
      --max_prompt_tokens 700 \
      --max_new_tokens 256 \
      --temperature 0.0 \
      --top_p 1.0
  done
done
```

Download the candidate files:

```bash
export B_EXP_DIR=experiments/methodB_rank_traincalib_seeds12
mkdir -p ${B_EXP_DIR}/candidates

for POLICY_SEED in 1 2; do
  RUN_TAG=methodB_rank_grpo_traincalib_seed${POLICY_SEED}_v1

  for STEP in 75 100; do
    uv run modal volume get llm-rl-final-project-volume \
      /submissions/${RUN_TAG}/rank_grpo_step_${STEP}.jsonl \
      ${B_EXP_DIR}/candidates/rank_grpo_seed${POLICY_SEED}_step${STEP}.jsonl
  done
done
```

## Evaluate rank-advantage GRPO

Each candidate is evaluated in a temporary submission directory containing only `online_best.jsonl`. This is useful for debugging one method at a time; missing files are simply marked as failures.

```bash
export B_EXP_DIR=experiments/methodB_rank_traincalib_seeds12
mkdir -p ${B_EXP_DIR}/autograder_results

for SEED in 1 2; do
  for STEP in 75 100; do
    TMP_DIR=tmp_clean_eval_methodB_seed${SEED}_step${STEP}
    OUT_JSON=${B_EXP_DIR}/autograder_results/seed${SEED}_step${STEP}_eval1.json

    rm -rf ${TMP_DIR}
    mkdir -p ${TMP_DIR}/part2
    mkdir -p ${TMP_DIR}/reward_model

    cp llm_rl_final_proj_public_submission/reward_model/public_test_pref_scores.jsonl \
       ${TMP_DIR}/reward_model/public_test_pref_scores.jsonl

    cp ${B_EXP_DIR}/candidates/rank_grpo_seed${SEED}_step${STEP}.jsonl \
       ${TMP_DIR}/part2/online_best.jsonl

    uv run python student_autograder/run_local_autograder.py \
      --submission_dir ${TMP_DIR} \
      --output_json ${OUT_JSON}
  done
done
```

We repeated step 100 evaluations to estimate judge variance:

```bash
for SEED in 1 2; do
  for EVAL_ID in 2 3; do
    TMP_DIR=tmp_clean_eval_methodB_seed${SEED}_step100_eval${EVAL_ID}
    OUT_JSON=${B_EXP_DIR}/autograder_results/seed${SEED}_step100_eval${EVAL_ID}.json

    rm -rf ${TMP_DIR}
    mkdir -p ${TMP_DIR}/part2
    mkdir -p ${TMP_DIR}/reward_model

    cp llm_rl_final_proj_public_submission/reward_model/public_test_pref_scores.jsonl \
       ${TMP_DIR}/reward_model/public_test_pref_scores.jsonl

    cp ${B_EXP_DIR}/candidates/rank_grpo_seed${SEED}_step100.jsonl \
       ${TMP_DIR}/part2/online_best.jsonl

    uv run python student_autograder/run_local_autograder.py \
      --submission_dir ${TMP_DIR} \
      --output_json ${OUT_JSON}
  done
done
```

Summarize the results:

```bash
uv run python experiments/methodB_rank_traincalib_seeds12/summarize_methodB.py
cat experiments/methodB_rank_traincalib_seeds12/methodB_summary_table.md
```

Expected summary:

```text
seed1 step75:  mean online = 0.5833, 0/1 pass
seed1 step100: mean online = 0.6360, 0/3 pass
seed2 step75:  mean online = 0.7021, 0/1 pass
seed2 step100: mean online = 0.7533, 2/3 pass
```

---

# Final Part 2 Artifact Choices

Use fixed-candidate reward-model reranking as the final offline best:

```bash
mkdir -p llm_rl_final_proj_public_submission/part2

cp experiments/methodA_traincalib/artifacts/offline_best_traincalib.jsonl \
   llm_rl_final_proj_public_submission/part2/offline_best.jsonl
```

Use the stronger calibrated reward-ensemble GRPO output as the final online best, not Rank-Advantage GRPO:

```bash
cp llm_zbest_ckpts/online_best.jsonl \
   llm_rl_final_proj_public_submission/part2/online_best.jsonl
```

Verify final Part 2 artifacts:

```bash
cmp -s llm_rl_final_proj_public_submission/part2/offline_best.jsonl \
       experiments/methodA_traincalib/artifacts/offline_best_traincalib.jsonl \
  && echo "OK: offline_best is fixed-candidate reranking"

cmp -s llm_rl_final_proj_public_submission/part2/online_best.jsonl \
       llm_zbest_ckpts/online_best.jsonl \
  && echo "OK: online_best is calibrated reward-ensemble GRPO"
```

---

# Download All Part 1 Submission Files

```bash
mkdir -p llm_rl_final_proj_public_submission/policy_generations
mkdir -p llm_rl_final_proj_public_submission/reward_model
mkdir -p llm_rl_final_proj_public_submission/part2

uv run modal volume get llm-rl-final-project-volume /submissions/dpo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/ipo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/aot.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/grpo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/drgrpo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/gspo.jsonl llm_rl_final_proj_public_submission/policy_generations/
uv run modal volume get llm-rl-final-project-volume /submissions/public_test_pref_scores.jsonl llm_rl_final_proj_public_submission/reward_model/
```

---

# Zip Final Submission

```bash
zip -r llm_rl_final_proj_public_submission.zip llm_rl_final_proj_public_submission -x "*.DS_Store"
```

---

