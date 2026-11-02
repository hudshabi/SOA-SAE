"""Builds the manuscript tables from the per-fold prediction files written by run_full_protocol.py.

  Table 3 : accuracy / macro-F1 per method, mean ± std across runs (each run pools predictions over all LOSO folds)
  Table 4 : SOA-SAE vs each baseline, per-subject macro-F1 averaged over runs, paired t-test and Wilcoxon,
            Holm-adjusted; Shapiro-Wilk p-value of the paired differences
  Table 5 : row-normalized confusion matrix of SOA-SAE pooled over folds, averaged over runs
  Also    : mean convergence curve per optimizer, search / training / inference times, selected configurations

Usage: python analyze_results.py --out results_ucihar --labels Walking Upstairs Downstairs Sitting Standing Lying
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASELINES = ['SAE', 'Random', 'PSO', 'GA', 'GWO']
NAMES = dict(SVM='SVM', RF='RF', CNN='CNN', LSTM='LSTM', SAE='SAE (untuned)', Random='Random-SAE',
             PSO='PSO-SAE', GA='GA-SAE', GWO='GWO-SAE', SO='SOA-SAE')


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--labels', nargs='+', required=True)
    a = ap.parse_args()
    k = len(a.labels)
    data = defaultdict(lambda: defaultdict(dict))         # method -> run -> subject -> (y, p, info)
    for fn in glob.glob(os.path.join(a.out, 'preds', '*.npz')):
        m = re.match(r'run(\d+)_subj(\d+)_(\w+)\.npz', os.path.basename(fn))
        z = np.load(fn, allow_pickle=False)
        data[m.group(3)][int(m.group(1))][int(m.group(2))] = (z['y_true'], z['y_pred'], json.loads(str(z['info'])))
    rep = {}

    # ---- Table 3
    t3 = {}
    for meth, runs in data.items():
        accs, f1s = [], []
        for r, subj in runs.items():
            y = np.concatenate([v[0] for v in subj.values()]); p = np.concatenate([v[1] for v in subj.values()])
            accs.append(accuracy_score(y, p)); f1s.append(f1_score(y, p, average='macro'))
        t3[meth] = dict(acc_mean=100 * np.mean(accs), acc_std=100 * np.std(accs, ddof=1) if len(accs) > 1 else 0.0,
                        f1_mean=100 * np.mean(f1s), f1_std=100 * np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0,
                        runs=len(accs), folds=len(next(iter(runs.values()))))
    rep['table3'] = t3

    # ---- Table 4 (subject is the unit)
    def subj_f1(meth):
        out = defaultdict(list)
        for subj in data[meth].values():
            for s, (y, p, _) in subj.items():
                out[s].append(f1_score(y, p, average='macro', labels=np.unique(y)))
        return {s: np.mean(v) for s, v in out.items()}
    t4 = {}
    if 'SO' in data:
        so = subj_f1('SO'); rows, pt, pw = [], [], []
        for b in BASELINES:
            if b not in data:
                continue
            bb = subj_f1(b); common = sorted(set(so) & set(bb))
            diff = np.array([so[s] - bb[s] for s in common])
            pt.append(stats.ttest_rel([so[s] for s in common], [bb[s] for s in common]).pvalue)
            pw.append(stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0)
            rows.append(dict(baseline=b, n_subjects=len(common), mean_diff=float(diff.mean()),
                             shapiro_p=float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else None))
        for r, a1, a2, r1, r2 in zip(rows, holm(pt), holm(pw), pt, pw):
            r.update(t_p=float(r1), t_p_holm=float(a1), wilcoxon_p=float(r2), wilcoxon_p_holm=float(a2))
            t4[r['baseline']] = r
    rep['table4'] = t4

    # ---- Table 5, curves, timings, configs
    if 'SO' in data:
        cms = []
        for subj in data['SO'].values():
            y = np.concatenate([v[0] for v in subj.values()]); p = np.concatenate([v[1] for v in subj.values()])
            cm = confusion_matrix(y, p, labels=range(k)).astype(float); cms.append(100 * cm / cm.sum(1, keepdims=True))
        rep['table5_confusion_pct'] = np.mean(cms, 0).round(2).tolist()
    rep['convergence'] = {m: np.mean([v[2]['curve'] for subj in data[m].values() for v in subj.values()], 0).tolist()
                          for m in ['SO', 'PSO', 'GWO', 'GA', 'Random'] if m in data}
    rep['timing'] = {m: {key: float(np.mean([v[2][key] for subj in data[m].values() for v in subj.values() if key in v[2]]))
                         for key in ['search_s', 'train_s', 'infer_ms_per_window']
                         if any(key in v[2] for subj in data[m].values() for v in subj.values())}
                     for m in data}
    rep['configs'] = {m: [v[2]['cfg'] for subj in data[m].values() for v in subj.values()]
                      for m in data if any('cfg' in v[2] for subj in data[m].values() for v in subj.values())}
    json.dump(rep, open(os.path.join(a.out, 'tables.json'), 'w'), indent=1)

    print('\nTable 3 (mean ± std across runs, %)')
    for m in ['SVM', 'RF', 'CNN', 'LSTM', 'SAE', 'Random', 'PSO', 'GA', 'GWO', 'SO']:
        if m in t3:
            v = t3[m]
            print(f"  {NAMES[m]:14s} {v['acc_mean']:.2f} ± {v['acc_std']:.2f} / {v['f1_mean']:.2f} ± {v['f1_std']:.2f}")
    print('\nTable 4 (Holm-adjusted p, per-subject macro-F1)')
    for b, r in t4.items():
        print(f"  SOA-SAE vs {NAMES[b]:14s} n={r['n_subjects']}  t: {r['t_p_holm']:.4g}  Wilcoxon: {r['wilcoxon_p_holm']:.4g}"
              f"  mean diff {100 * r['mean_diff']:+.2f} pts")


if __name__ == '__main__':
    main()
