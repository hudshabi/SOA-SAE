"""1D-CNN and LSTM baselines under exactly the folds, validation subjects and normalization of cv_protocol.py.
Adam (lr 1e-3), batch 64, <= 60 epochs with early stopping on inner-validation macro-F1 (patience 8) — the same rule
as the final SAE models."""
import argparse, json, os, sys, time
import numpy as np
import torch
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extended_protocol'))
from run_full_protocol import CNN1D, LSTMNet, train_classifier, predict, CHANNELS
from cv_protocol import SEEDS, folds_for, save_preds

torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--methods', default='CNN,LSTM')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'preds'), exist_ok=True)
    d = np.load(os.path.join(os.environ.get('HAR_DATA', 'data'), 'cache', f'{a.dataset}.npz'))
    X, y, s = d['X'], d['y'], d['s']; k = len(d['labels'])
    C = CHANNELS[a.dataset]; Tw = X.shape[1] // C
    log = open(os.path.join(a.out, 'progress_dl.log'), 'a')
    for run, seed in enumerate(SEEDS[:a.runs]):
        for fi, test_s in enumerate(folds_for(np.unique(s), seed)):
            rng = np.random.default_rng(seed * 100 + fi)        # identical draw order to cv_protocol.py
            train_s = np.setdiff1d(np.unique(s), test_s)
            val_s = rng.choice(train_s, max(2, int(round(0.2 * len(train_s)))), replace=False)
            inner_s = np.setdiff1d(train_s, val_s)
            mi, mv, mt = np.isin(s, inner_s), np.isin(s, val_s), np.isin(s, test_s)
            mu, sd = X[mi].mean(0), X[mi].std(0) + 1e-6
            Xi, Xv, Xt = (torch.tensor((X[m] - mu) / sd, dtype=torch.float32) for m in (mi, mv, mt))
            yi, yv = torch.tensor(y[mi]), torch.tensor(y[mv])
            yt, st = y[mt], s[mt]
            for meth in a.methods.split(','):
                if all(os.path.exists(os.path.join(a.out, 'preds', f'run{run}_subj{int(u)}_{meth}.npz')) for u in test_s):
                    continue
                t0 = time.time()
                torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
                model = (CNN1D if meth == 'CNN' else LSTMNet)(Tw, C, k)
                vf1, eps = train_classifier(model, Xi, yi, Xv, yv, 1e-3, 0.0, 64, 60, 8, gen)
                ttrain = time.time() - t0
                t1 = time.time(); p = predict(model, Xt); tinf = (time.time() - t1) / len(yt) * 1000
                info = dict(run=run, seed=seed, fold=fi, method=meth, test_subjects=test_s.tolist(),
                            val_subjects=val_s.tolist(), val_f1=float(vf1), epochs=int(eps), train_s=ttrain,
                            infer_ms_per_window=tinf, params=int(sum(q.numel() for q in model.parameters())))
                save_preds(a.out, run, meth, yt, p, st, info)
                msg = f'{a.dataset} run {run} fold {fi} {meth}: macroF1={f1_score(yt, p, average="macro"):.4f} ({ttrain:.0f}s, {eps} ep)'
                print(msg, flush=True); log.write(msg + '\n'); log.flush()


if __name__ == '__main__':
    main()
