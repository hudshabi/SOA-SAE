"""Compute-feasible protocol for SOA-SAE (NumPy, CPU) — the protocol reported in the manuscript.

  * outer: subject-wise 5-fold cross-validation (subjects shuffled with the run seed, every subject tested once)
  * inner validation: 20% of the outer-training subjects (at least two); z-score from inner-training subjects only
  * search space = Table 2; fitness = 1 - macroF1 on the inner validation subjects (beta = 0; --beta for the ablation)
  * equal budget: N = 10, T = 20 -> 210 evaluations per optimizer and fold
  * low-fidelity search: <= 2000 inner-training / <= 1500 validation windows (stratified), 3 pre-training epochs per
    layer, <= 15 fine-tuning epochs (patience 4)
  * final model: all inner-training windows, 10 pre-training epochs per layer, <= 60 fine-tuning epochs (patience 8)
  * 3 runs (seeds 42, 123, 456)
Per-subject prediction files are written to <out>/preds/run{r}_subj{s}_{method}.npz (format of analyze_results.py).
Fitness values are cached per (run, fold, method), so an interrupted search resumes without repeating work.
"""
import argparse, json, os, time
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sae_so import (SAE, decode, snake_optimizer, pso, gwo, ga, random_search, strat_cap, SMAX, UNTUNED)

SEEDS = [42, 123, 456]
OPT = dict(SO=snake_optimizer, PSO=pso, GWO=gwo, GA=ga, Random=random_search)


def folds_for(subjects, seed, k=5):
    rng = np.random.default_rng(seed)
    s = rng.permutation(subjects)
    return [np.sort(f) for f in np.array_split(s, k)]


def save_preds(out, run, method, yt, p, st, info):
    for u in np.unique(st):
        m = st == u
        np.savez(os.path.join(out, 'preds', f'run{run}_subj{int(u)}_{method}.npz'),
                 y_true=yt[m], y_pred=p[m], info=json.dumps(info))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(SMAX))
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--N', type=int, default=10)
    ap.add_argument('--T', type=int, default=20)
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--methods', default='SVM,RF,SAE,SO,PSO,GWO,GA,Random')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'preds'), exist_ok=True); os.makedirs(os.path.join(a.out, 'cache'), exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
    d = np.load(os.path.join(os.environ.get('HAR_DATA', 'data'), 'cache', f'{a.dataset}.npz'))
    X, y, s = d['X'], d['y'], d['s']; k = len(d['labels']); dim = X.shape[1]
    smax = SMAX[a.dataset]
    pmax = SAE(decode([1, 1, 1, 1, 0, 0, 0, 0], smax), k, 0).n_params(dim)
    log = open(os.path.join(a.out, 'progress.log'), 'a')

    def say(msg):
        print(msg, flush=True); log.write(msg + '\n'); log.flush()

    for run, seed in enumerate(SEEDS[:a.runs]):
        for fi, test_s in enumerate(folds_for(np.unique(s), seed)):
            done = lambda m: all(os.path.exists(os.path.join(a.out, 'preds', f'run{run}_subj{int(u)}_{m}.npz')) for u in test_s)
            todo = [m for m in a.methods.split(',') if not done(m)]
            if not todo:
                continue
            rng = np.random.default_rng(seed * 100 + fi)
            train_s = np.setdiff1d(np.unique(s), test_s)
            val_s = rng.choice(train_s, max(2, int(round(0.2 * len(train_s)))), replace=False)
            inner_s = np.setdiff1d(train_s, val_s)
            mi, mv, mt = np.isin(s, inner_s), np.isin(s, val_s), np.isin(s, test_s)
            mu, sd = X[mi].mean(0), X[mi].std(0) + 1e-6
            Xi, Xv, Xt = ((X[m] - mu) / sd for m in (mi, mv, mt))
            yi, yv, yt, st = y[mi], y[mv], y[mt], s[mt]
            si, sv = strat_cap(yi, 2000, rng), strat_cap(yv, 1500, rng)
            base = dict(run=run, seed=seed, fold=fi, test_subjects=test_s.tolist(), val_subjects=val_s.tolist())
            for meth in todo:
                t0 = time.time(); info = dict(base, method=meth)
                if meth in ('SVM', 'RF'):
                    clf = SVC(C=10, gamma='scale', random_state=seed) if meth == 'SVM' else \
                        RandomForestClassifier(300, n_jobs=-1, random_state=seed)
                    clf.fit(np.vstack([Xi, Xv]), np.r_[yi, yv]); info['train_s'] = time.time() - t0
                    t1 = time.time(); p = clf.predict(Xt)
                else:
                    if meth == 'SAE':
                        cfg = dict(h=UNTUNED[a.dataset], lr=1e-3, wd=1e-4, dropout=0.2, batch=64)
                    else:
                        cfile = os.path.join(a.out, 'cache', f'run{run}_fold{fi}_{meth}.json')
                        cache = json.load(open(cfile)) if os.path.exists(cfile) else {}
                        n_new = [0]

                        def fobj(x):
                            key = ','.join(f'{v:.6f}' for v in np.clip(x, 0, 1))
                            if key in cache:
                                return cache[key]
                            c = decode(x, smax)
                            m = SAE(c, k, seed).fit(Xi[si], yi[si], Xv[sv], yv[sv], pre_epochs=3, ft_epochs=15, patience=4)
                            f = (1 - a.beta) * (1 - m.best_val_f1) + a.beta * m.n_params(dim) / pmax
                            cache[key] = float(f); n_new[0] += 1
                            if n_new[0] % 5 == 0:
                                json.dump(cache, open(cfile, 'w'))
                            return f
                        xb, fb, curve = OPT[meth](fobj, a.N, a.T, np.random.default_rng(seed + 7))
                        json.dump(cache, open(cfile, 'w'))
                        cfg = decode(xb, smax)
                        info.update(search_s=time.time() - t0, best_fitness=float(fb), curve=[float(c) for c in curve])
                    t2 = time.time()
                    model = SAE(cfg, k, seed).fit(Xi, yi, Xv, yv, pre_epochs=10, ft_epochs=60, patience=8)
                    info.update(train_s=time.time() - t2, cfg=cfg, params=int(model.n_params(dim)),
                                val_f1=float(model.best_val_f1), epochs=int(model.epochs_run))
                    t1 = time.time(); p = model.predict(Xt)
                info['infer_ms_per_window'] = (time.time() - t1) / len(yt) * 1000
                save_preds(a.out, run, meth, yt, p, st, info)
                f1 = __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(yt, p, average='macro')
                say(f'{a.dataset} run {run} fold {fi} {meth}: macroF1={f1:.4f} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
