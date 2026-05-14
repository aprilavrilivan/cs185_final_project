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

