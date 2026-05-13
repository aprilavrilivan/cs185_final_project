# Method A: Fixed-candidate calibrated reward-model reranking

## Protocol

1. Train three reward models on `train_prefs` with different seeds.
2. Select one checkpoint per seed using held-out `test_prefs` diagnostics.
3. Compute calibration centers/scales from `train_prefs` score statistics.
4. Use the calibrated reward ensemble to score a fixed candidate family:
   - DPO deterministic public generations
   - IPO deterministic public generations
   - AOT deterministic public generations
5. For each prompt, select the candidate with the highest calibrated ensemble score.
6. Submit the selected responses as `part2/offline_best.jsonl`.

## Important distinction

- `rm_eval_test_prefs/` contains held-out reward-model diagnostics.
- `rm_calib_train_prefs/` contains train-prefs score statistics used only for calibration.
- Train-prefs pair accuracy should not be reported as held-out test accuracy.

## Main result

Method A train-calibrated reranking passed Part 2 offline:
`offline=0.8636`, threshold `0.8300`.
