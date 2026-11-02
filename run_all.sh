#!/bin/bash
# Reproduces every number in the manuscript. Safe to re-run: finished (run, subject, method) results and the
# fitness caches of interrupted searches are reused.
set -e
export HAR_DATA=${HAR_DATA:-data}

# SVM, RF, untuned SAE and the five search strategies (beta = 0)
python cv_protocol.py --dataset wisdm  --out results/wisdm  --runs 3
python cv_protocol.py --dataset ucihar --out results/ucihar --runs 1
python cv_protocol.py --dataset pamap2 --out results/pamap2 --runs 1

# 1D-CNN and LSTM baselines on the same folds and validation subjects
python cv_dl_baselines.py --dataset wisdm  --out results/wisdm  --runs 3
python cv_dl_baselines.py --dataset ucihar --out results/ucihar --runs 1
python cv_dl_baselines.py --dataset pamap2 --out results/pamap2 --runs 1

# size-penalty ablation (beta = 0.15), WISDM, run 1
python cv_protocol.py --dataset wisdm --out results/wisdm_beta015 --runs 1 --beta 0.15 --methods SO,PSO,GWO,GA,Random

# tables, significance tests and figures
python analyze_results.py --out results/wisdm  --labels Walking Jogging Upstairs Downstairs Sitting Standing
python analyze_results.py --out results/ucihar --labels Walking Upstairs Downstairs Sitting Standing Lying
python analyze_results.py --out results/pamap2 --labels Lying Sitting Standing Walking Running Cycling Nordic_walking \
    Ascending_stairs Descending_stairs Vacuum_cleaning Ironing Rope_jumping
python make_figs.py
