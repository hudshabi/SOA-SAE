# SOA-SAE: Snake-Optimizer-Tuned Stacked Autoencoders for Indoor Activity Monitoring

Code, per-subject predictions, logs, and figures for the manuscript

> **Intelligent Indoor Activity Monitoring Using Deep Learning and Snake Optimizer: A Potential Framework for Supporting Visually Impaired Persons**

The study tunes the hyperparameters of a Stacked Autoencoder (SAE) with the Snake Optimizer (SO; Hashim & Hussien, 2022). It compares SO with PSO, GA, GWO, and random search under an **equal search budget**, and benchmarks all tuned SAEs against an untuned SAE, SVM, Random Forest, 1D-CNN, and LSTM. The datasets are WISDM, UCI-HAR, and PAMAP2, and evaluation uses **subject-wise cross-validation with nested validation subjects**.

Everything is implemented from scratch in NumPy and runs on a CPU. PyTorch is needed only for the CNN/LSTM baselines.

## Main results

These are pooled macro-F1 scores (%) over all test subjects. WISDM figures are the mean ± std over 3 runs. UCI-HAR and PAMAP2 figures come from a single run.

| Method | WISDM | UCI-HAR | PAMAP2 |
|---|---|---|---|
| SVM (RBF) | 77.72 ± 1.40 | 90.26 | 72.37 |
| Random Forest | 67.24 ± 0.90 | 88.23 | 66.92 |
| 1D-CNN | **80.87 ± 0.64** | **90.89** | **75.48** |
| LSTM | 79.31 ± 3.13 | 89.81 | 67.35 |
| SAE (untuned) | 74.83 ± 0.69 | 89.15 | 68.69 |
| Random-SAE | 76.98 ± 1.63 | 89.73 | 69.58 |
| PSO-SAE | 75.86 ± 0.92 | 88.86 | 65.52 |
| GA-SAE | 75.46 ± 1.97 | 88.71 | 65.88 |
| GWO-SAE | 75.57 ± 1.13 | 89.17 | 68.00 |
| **SOA-SAE** | 75.36 ± 0.80 | 89.61 | 62.59 |

The paired per-subject t-tests and Wilcoxon tests were Holm-adjusted. Against that standard:

- SOA-SAE is **not** significantly different from the untuned SAE, PSO-SAE, GA-SAE, or GWO-SAE on any dataset.
- SOA-SAE is significantly **worse than random search on WISDM**, by −1.99 points per subject (p = 0.013 for the t-test, 0.005 for the Wilcoxon test).
- Under this protocol, SO tuning is competitive with simpler alternatives but not superior to them.

We report this negative result deliberately. Our recommendation is that any metaheuristic tuner for HAR be benchmarked against random search at a matched budget.

Full tables are in `results/<dataset>/tables.json` and `analysis.txt`. The figures are in `figures/`.

**Results archive.** The `results/` folder holds 1,664 small files: per-subject predictions, logs, and tables. It is distributed as a single archive, `results.zip`. Unzip it in the repository root before running `analyze_results.py` or `make_figs.py`:

```bash
unzip results.zip
```

## Protocol

| Item | Setting |
|---|---|
| Outer evaluation | Subject-wise 5-fold CV. Subjects are shuffled with the run seed, and every subject is tested exactly once per run. |
| Inner validation | 20% of the outer-training subjects (at least 2). These subjects score the fitness and drive early stopping. Z-score statistics come from the inner-training subjects only. |
| Search space | 8 variables in [0, 1]: depth {2, 3}, first width, two width ratios, learning rate, weight decay, dropout, batch size (Table 2 of the paper). |
| Fitness | F = (1 − β)(1 − macro-F1_val) + β · P/P_max. Main experiments use β = 0; the ablation uses β = 0.15. |
| Budget | 10 agents × 21 generations = 210 evaluations per optimizer and outer fold, identical for SO, PSO, GA, GWO, and random search. |
| Low-fidelity search | At most 2,000 training and 1,500 validation windows (stratified); 3 pre-training epochs per layer; at most 15 fine-tuning epochs (patience 4). |
| Final model | All inner-training windows; 10 pre-training epochs per layer; at most 60 fine-tuning epochs (patience 8). |
| Runs | WISDM: seeds 42, 123, 456. UCI-HAR and PAMAP2: seed 42. |
| Statistics | Paired t-test and Wilcoxon signed-rank on per-subject macro-F1, with Holm–Bonferroni correction and Shapiro–Wilk checks. |

Windowing (2.56 s windows, 50% overlap) and the sensor channels used per dataset:

| Dataset | Sampling | Channels | Input size |
|---|---|---|---|
| UCI-HAR | 50 Hz | total acceleration + gyroscope | 128 × 6 |
| WISDM | 20 Hz | accelerometer | 51 × 3 |
| PAMAP2 | 100 Hz | chest IMU: acc ±16 g, gyro, mag; 12 protocol activities | 256 × 9 |

## Repository layout

```
prep_data.py           build windowed datasets -> data/cache/*.npz
sae_so.py              NumPy SAE + optimizers (SO, PSO, GA, GWO, random search) and search-space decoding
cv_protocol.py         nested subject-wise protocol: SVM, RF, untuned SAE, five tuned SAEs (resumable)
cv_dl_baselines.py     1D-CNN and LSTM on the same folds / validation subjects / normalization
analyze_results.py     Tables 3–5, Holm-adjusted tests, convergence curves, timings -> tables.json
make_figs.py           Figures 2–4 of the paper
run_all.sh             reproduces every result in the paper
extended_protocol/     PyTorch models used by cv_dl_baselines.py; run_full_protocol.py also offers a
                       larger LOSO protocol (GPU recommended; not used for the reported results)
results/               per-subject predictions, logs, tables.json and analysis.txt for each experiment
figures/               figures used in the manuscript
data/                  place the raw datasets here (see data/README.md)
```

Each prediction file `results/<exp>/preds/run{r}_subj{s}_{method}.npz` contains:

- `y_true` and `y_pred` for one test subject.
- An `info` JSON record holding the fold, test and validation subjects, the selected configuration, the parameter count, the search curve, and the timings.

## Reproducing the results

```bash
pip install -r requirements.txt
# download the datasets into data/ (see data/README.md), then
python prep_data.py --root data
bash run_all.sh
```

To check the setup in a few minutes, run a smoke test. It trains only the untuned SAE on WISDM, which is deterministic and should reproduce the per-fold values in `results/wisdm/progress.log`:

```bash
python cv_protocol.py --dataset wisdm --out smoke --runs 1 --methods SAE
```

These are the options of `cv_protocol.py`:

| Option | Meaning |
|---|---|
| `--runs` | Number of runs |
| `--N`, `--T` | Population size and iterations |
| `--beta` | Size-penalty weight |
| `--methods` | Comma-separated subset, e.g. `SO,Random` |

All scripts resume automatically. Finished (run, subject, method) results are skipped, and the fitness values of an interrupted search are cached in `<out>/cache/`. Set `HAR_DATA` if your data folder is not `./data`.

**Compute.** The reported results were produced on a 2-core CPU virtual machine with 7 GB RAM and no GPU. Typical search time per optimizer and fold:

| Dataset | Search time per optimizer and fold |
|---|---|
| WISDM | about 3 min |
| UCI-HAR | 11–21 min |
| PAMAP2 | 13–57 min |

The complete pipeline takes roughly 2–3 days on such a machine. PAMAP2 accounts for most of that time.

## Notes and limitations

- UCI-HAR and PAMAP2 were run once, so Table 3 has no run-to-run spread for them.
- PAMAP2 has only 9 subjects, so its inner validation uses 2 subjects and its significance tests have low power.
- One PAMAP2 fold contains only subject 109: 48 windows, all of a single activity. Per-fold scores for that fold are not meaningful, and all metrics pool predictions over subjects.
- Timings are indicative only. The machine was shared between jobs, and some searches were resumed from cached fitness values.

## Citation

If you use this code, please cite the paper (citation details will be added upon publication).

## License

Released under the MIT License (see `LICENSE`).
