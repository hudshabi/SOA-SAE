"""Full experimental protocol of the manuscript (PyTorch, GPU recommended).

Implements Sections 3-4 exactly:
  * nested Leave-One-Subject-Out (outer fold = one test subject; 20% of the remaining subjects, at least two,
    = inner validation)
  * z-score statistics from inner-training subjects only
  * search space = Table 2 (8 hyperparameters, weights NOT searched); fitness F = (1-beta)(1 - macroF1_val) + beta P/P_max,
    beta = 0 in the main experiments (--beta 0.15 for the size-penalty ablation)
  * equal budget for SO, PSO, GWO, GA, Random: N = 20, T = 30 -> 620 evaluations per outer fold
  * search schedule: 5 pre-training epochs per layer, <= 20 fine-tuning epochs (patience 5)
    final model: 50 pre-training epochs per layer, <= 200 fine-tuning epochs (patience 15)
  * baselines: untuned SAE, 1D-CNN, LSTM, SVM (RBF, C=10), RF (300 trees)
  * 5 runs (seeds 42, 123, 456, 789, 1024)
Every (run, fold, method) result is written to <out>/preds/ immediately, so the job can be stopped and resumed.
Analysis (Tables 3-5, convergence, timings): analyze_results.py

Usage:
  python prep_data.py                                   # builds data/cache/*.npz from the raw downloads
  python run_full_protocol.py --dataset ucihar --out results_ucihar
  python run_full_protocol.py --dataset ucihar --out results_ucihar --runs 1 --folds 2 --N 4 --T 2   # quick smoke test
Options --runs / --folds / --N / --T allow a reduced budget; any reduction must be reported in the paper.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
SMAX = dict(wisdm=256, ucihar=1024, pamap2=1536)
UNTUNED = dict(wisdm=[128, 64, 32], ucihar=[512, 256, 128], pamap2=[1536, 768, 384])
CHANNELS = dict(wisdm=3, ucihar=6, pamap2=9)          # windows are flattened time-major: (T, C) -> T*C
D = 8                                                  # number of decision variables


# ============================================================== search space (Table 2)
def decode(x, smax):
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    L = 2 + int(round(x[0]))
    s1 = int(round(64 + x[1] * (smax - 64)))
    s2 = max(8, int(round((0.25 + 0.5 * x[2]) * s1)))
    s3 = max(8, int(round((0.25 + 0.5 * x[3]) * s2)))
    return dict(h=[s1, s2, s3][:L], lr=float(10 ** (-4 + 2 * x[4])), wd=float(10 ** (-6 + 3 * x[5])),
                dropout=float(0.5 * x[6]), batch=[32, 64, 128][min(2, int(x[7] * 3))])


def n_params(d, h, k):
    dims = [d] + h
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:])) + dims[-1] * k + k


# ============================================================== models
def batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen, device='cpu').to(DEV)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model, X, bs=8192):
    model.eval()
    return torch.cat([model(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)]).cpu().numpy()


def train_classifier(model, X, y, Xv, yv, lr, wd, bs, epochs, patience, gen):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    best, best_state, wait, ep = -1.0, None, 0, 0
    yv_np = yv.cpu().numpy()
    for ep in range(epochs):
        model.train()
        for bi in batches(len(X), bs, gen):
            opt.zero_grad(); lossf(model(X[bi]), y[bi]).backward(); opt.step()
        f1 = f1_score(yv_np, predict(model, Xv), average='macro')
        if f1 > best + 1e-4:
            best, wait = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return best, ep + 1


class SAEClassifier(nn.Module):
    def __init__(self, d, h, k, dropout):
        super().__init__()
        self.enc = nn.ModuleList([nn.Linear(a, b) for a, b in zip([d] + h[:-1], h)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(h[-1], k)

    def forward(self, x):
        for lin in self.enc:
            x = self.drop(torch.relu(lin(x)))
        return self.out(x)


def fit_sae(cfg, X, y, Xv, yv, k, pre_epochs, ft_epochs, patience, seed):
    """Greedy layer-wise pre-training (ReLU encoder, linear decoder, MSE + L2) then supervised fine-tuning."""
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = SAEClassifier(X.shape[1], cfg['h'], k, cfg['dropout']).to(DEV)
    H = X
    for lin in model.enc:
        dec = nn.Linear(lin.out_features, lin.in_features).to(DEV)
        opt = torch.optim.Adam(list(lin.parameters()) + list(dec.parameters()), lr=cfg['lr'], weight_decay=cfg['wd'])
        for _ in range(pre_epochs):
            for bi in batches(len(H), cfg['batch'], gen):
                opt.zero_grad(); nn.functional.mse_loss(dec(torch.relu(lin(H[bi]))), H[bi]).backward(); opt.step()
        with torch.no_grad():
            H = torch.relu(lin(H))
    val_f1, eps = train_classifier(model, X, y, Xv, yv, cfg['lr'], cfg['wd'], cfg['batch'], ft_epochs, patience, gen)
    return model, val_f1, eps


class CNN1D(nn.Module):
    def __init__(self, T, C, k):
        super().__init__()
        self.T, self.C = T, C
        self.f = nn.Sequential(nn.Conv1d(C, 64, 5, padding=2), nn.ReLU(), nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(),
                               nn.MaxPool1d(2), nn.Flatten(), nn.Linear(64 * (T // 2), 128), nn.ReLU(), nn.Dropout(0.5),
                               nn.Linear(128, k))

    def forward(self, x):
        return self.f(x.view(-1, self.T, self.C).transpose(1, 2))


class LSTMNet(nn.Module):
    def __init__(self, T, C, k):
        super().__init__()
        self.T, self.C = T, C
        self.lstm = nn.LSTM(C, 64, num_layers=2, batch_first=True)
        self.out = nn.Linear(64, k)

    def forward(self, x):
        o, _ = self.lstm(x.view(-1, self.T, self.C))
        return self.out(o[:, -1])


# ============================================================== optimizers (budget N*(T+1))
def snake_optimizer(f, N, T, rng, c1=0.5, c2=0.05, c3=2.0):
    Nm = N // 2; Nf = N - Nm
    Xm, Xf = rng.random((Nm, D)), rng.random((Nf, D))
    fm, ff = np.array([f(x) for x in Xm]), np.array([f(x) for x in Xf])
    bm, bf = fm.argmin(), ff.argmin()
    Xbm, fbm, Xbf, fbf = Xm[bm].copy(), fm[bm], Xf[bf].copy(), ff[bf]
    Xfood, ffood = (Xbm.copy(), fbm) if fbm < fbf else (Xbf.copy(), fbf)
    curve = [ffood]; sgn = lambda: rng.choice([-1.0, 1.0])
    for t in range(1, T + 1):
        Temp = np.exp(-t / T); Q = min(1.0, c1 * np.exp((t - T) / T))
        Ym, Yf = Xm.copy(), Xf.copy()
        if Q < 0.25:                                        # exploration
            for i in range(Nm):
                r = rng.integers(Nm); Ym[i] = Xm[r] + sgn() * c2 * np.exp(-fm[r] / (fm[i] + 1e-12)) * rng.random(D)
            for i in range(Nf):
                r = rng.integers(Nf); Yf[i] = Xf[r] + sgn() * c2 * np.exp(-ff[r] / (ff[i] + 1e-12)) * rng.random(D)
        elif Temp > 0.6:                                    # hot: move to food
            for i in range(Nm):
                Ym[i] = Xfood + sgn() * c3 * Temp * rng.random(D) * (Xfood - Xm[i])
            for i in range(Nf):
                Yf[i] = Xfood + sgn() * c3 * Temp * rng.random(D) * (Xfood - Xf[i])
        elif rng.random() > 0.6:                            # cold: fight
            for i in range(Nm):
                Ym[i] = Xm[i] + c3 * np.exp(-fbf / (fm[i] + 1e-12)) * rng.random(D) * (Q * Xbf - Xm[i])
            for i in range(Nf):
                Yf[i] = Xf[i] + c3 * np.exp(-fbm / (ff[i] + 1e-12)) * rng.random(D) * (Q * Xbm - Xf[i])
        else:                                               # cold: mating + egg hatch
            for i in range(min(Nm, Nf)):
                Ym[i] = Xm[i] + c3 * np.exp(-ff[i] / (fm[i] + 1e-12)) * rng.random(D) * (Q * Xf[i] - Xm[i])
                Yf[i] = Xf[i] + c3 * np.exp(-fm[i] / (ff[i] + 1e-12)) * rng.random(D) * (Q * Xm[i] - Xf[i])
            if rng.random() > 0.5:
                Ym[fm.argmax()] = rng.random(D); Yf[ff.argmax()] = rng.random(D)
        Ym, Yf = np.clip(Ym, 0, 1), np.clip(Yf, 0, 1)
        for i in range(Nm):
            v = f(Ym[i])
            if v < fm[i]: Xm[i], fm[i] = Ym[i], v
        for i in range(Nf):
            v = f(Yf[i])
            if v < ff[i]: Xf[i], ff[i] = Yf[i], v
        bm, bf = fm.argmin(), ff.argmin()
        if fm[bm] < fbm: Xbm, fbm = Xm[bm].copy(), fm[bm]
        if ff[bf] < fbf: Xbf, fbf = Xf[bf].copy(), ff[bf]
        if min(fbm, fbf) < ffood:
            Xfood, ffood = (Xbm.copy(), fbm) if fbm < fbf else (Xbf.copy(), fbf)
        curve.append(ffood)
    return Xfood, ffood, curve


def pso(f, N, T, rng, w=0.7, c=1.5):
    X = rng.random((N, D)); V = np.zeros_like(X)
    F = np.array([f(x) for x in X]); P, PF = X.copy(), F.copy()
    g = PF.argmin(); G, GF = P[g].copy(), PF[g]; curve = [GF]
    for _ in range(T):
        V = w * V + c * rng.random((N, D)) * (P - X) + c * rng.random((N, D)) * (G - X)
        X = np.clip(X + V, 0, 1); F = np.array([f(x) for x in X])
        imp = F < PF; P[imp], PF[imp] = X[imp], F[imp]
        g = PF.argmin()
        if PF[g] < GF: G, GF = P[g].copy(), PF[g]
        curve.append(GF)
    return G, GF, curve


def gwo(f, N, T, rng):
    X = rng.random((N, D)); F = np.array([f(x) for x in X])
    o = F.argsort(); A_, B_, C_ = (X[o[j]].copy() for j in range(3))
    best_x, best_f = A_.copy(), F[o[0]]; curve = [best_f]
    for t in range(T):
        a = 2 - 2 * t / T
        for i in range(N):
            xs = []
            for Ld in (A_, B_, C_):
                A = 2 * a * rng.random(D) - a; Cc = 2 * rng.random(D)
                xs.append(Ld - A * np.abs(Cc * Ld - X[i]))
            X[i] = np.clip(np.mean(xs, 0), 0, 1)
        F = np.array([f(x) for x in X]); o = F.argsort()
        A_, B_, C_ = (X[o[j]].copy() for j in range(3))
        if F[o[0]] < best_f: best_x, best_f = X[o[0]].copy(), F[o[0]]
        curve.append(best_f)
    return best_x, best_f, curve


def ga(f, N, T, rng, pc=0.8, pm=0.1, sigma=0.1):
    X = rng.random((N, D)); F = np.array([f(x) for x in X]); curve = [F.min()]
    for _ in range(T):
        def tour():
            c = rng.choice(N, min(3, N), replace=False); return X[c[F[c].argmin()]]
        new = []
        while len(new) < N:
            a, b = tour().copy(), tour().copy()
            if rng.random() < pc:
                m = rng.random(D) < 0.5; a[m], b[m] = b[m].copy(), a[m].copy()
            for ch in (a, b):
                mm = rng.random(D) < pm; ch[mm] += rng.normal(0, sigma, mm.sum()); new.append(np.clip(ch, 0, 1))
        newX = np.array(new[:N]); newF = np.array([f(x) for x in newX])
        e = F.argmin()
        if F[e] < newF.min():
            w = newF.argmax(); newX[w], newF[w] = X[e], F[e]
        X, F = newX, newF; curve.append(F.min())
    b = F.argmin()
    return X[b], F[b], curve


def random_search(f, N, T, rng):
    best_x, best_f, curve = None, np.inf, []
    for j in range(N * (T + 1)):
        x = rng.random(D); v = f(x)
        if v < best_f: best_x, best_f = x, v
        if (j + 1) % N == 0: curve.append(best_f)
    return best_x, best_f, curve


OPTIMIZERS = dict(SO=snake_optimizer, PSO=pso, GWO=gwo, GA=ga, Random=random_search)


# ============================================================== protocol
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(SMAX))
    ap.add_argument('--cache', default='data/cache')
    ap.add_argument('--out', required=True)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--folds', type=int, default=0, help='0 = all subjects (full LOSO)')
    ap.add_argument('--N', type=int, default=20)
    ap.add_argument('--T', type=int, default=30)
    ap.add_argument('--methods', default='SVM,RF,CNN,LSTM,SAE,SO,PSO,GWO,GA,Random')
    ap.add_argument('--beta', type=float, default=0.0,
                    help='weight of the size penalty in the fitness; 0 in the main experiments, 0.15 for the ablation')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'preds'), exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, 'config.json'), 'w'), indent=1)

    d = np.load(os.path.join(a.cache, f'{a.dataset}.npz'))
    X_all, y_all, s_all = d['X'].astype(np.float32), d['y'].astype(np.int64), d['s']
    k = len(d['labels']); dim = X_all.shape[1]; C = CHANNELS[a.dataset]; Tw = dim // C
    subjects = np.unique(s_all)
    if a.folds:
        subjects = subjects[:a.folds]
    pmax = n_params(dim, decode([1, 1, 1, 1, 0, 0, 0, 0], SMAX[a.dataset])['h'], k)
    print(f'{a.dataset}: {X_all.shape}, {len(np.unique(s_all))} subjects, device {DEV}', flush=True)

    for run, seed in enumerate(SEEDS[:a.runs]):
        for test_subj in subjects:
            rng = np.random.default_rng(seed * 1000 + int(test_subj))
            train_s = np.setdiff1d(np.unique(s_all), [test_subj])
            val_s = rng.choice(train_s, max(2, int(round(0.2 * len(train_s)))), replace=False)   # >= 2 subjects
            inner_s = np.setdiff1d(train_s, val_s)
            mi, mv, mt = np.isin(s_all, inner_s), np.isin(s_all, val_s), s_all == test_subj
            mu, sd = X_all[mi].mean(0), X_all[mi].std(0) + 1e-6
            tX = lambda m: torch.tensor((X_all[m] - mu) / sd, device=DEV)
            Xi, Xv, Xt = tX(mi), tX(mv), tX(mt)
            yi, yv = torch.tensor(y_all[mi], device=DEV), torch.tensor(y_all[mv], device=DEV)
            yt = y_all[mt]
            for meth in a.methods.split(','):
                fn = os.path.join(a.out, 'preds', f'run{run}_subj{int(test_subj)}_{meth}.npz')
                if os.path.exists(fn):
                    continue
                info = dict(run=run, seed=seed, subject=int(test_subj), method=meth, val_subjects=val_s.tolist())
                t0 = time.time()
                if meth in ('SVM', 'RF'):
                    clf = SVC(C=10, gamma='scale', random_state=seed) if meth == 'SVM' else \
                        RandomForestClassifier(300, n_jobs=-1, random_state=seed)
                    Xtr = np.vstack([Xi.cpu().numpy(), Xv.cpu().numpy()]); ytr = np.r_[y_all[mi], y_all[mv]]
                    clf.fit(Xtr, ytr); info['train_s'] = time.time() - t0
                    t1 = time.time(); p = clf.predict(Xt.cpu().numpy())
                elif meth in ('CNN', 'LSTM'):
                    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
                    model = (CNN1D if meth == 'CNN' else LSTMNet)(Tw, C, k).to(DEV)
                    info['val_f1'], info['epochs'] = train_classifier(model, Xi, yi, Xv, yv, 1e-3, 0.0, 64, 200, 15, gen)
                    info['train_s'] = time.time() - t0
                    t1 = time.time(); p = predict(model, Xt)
                else:
                    if meth == 'SAE':
                        cfg = dict(h=UNTUNED[a.dataset], lr=1e-3, wd=1e-4, dropout=0.2, batch=64)
                    else:
                        n_ev = [0]
                        def fobj(x):
                            c = decode(x, SMAX[a.dataset])
                            _, f1v, _ = fit_sae(c, Xi, yi, Xv, yv, k, 5, 20, 5, seed)
                            n_ev[0] += 1
                            return (1 - a.beta) * (1 - f1v) + a.beta * n_params(dim, c['h'], k) / pmax
                        xb, fb, curve = OPTIMIZERS[meth](fobj, a.N, a.T, np.random.default_rng(seed + 7))
                        cfg = decode(xb, SMAX[a.dataset])
                        info.update(search_s=time.time() - t0, n_evals=n_ev[0], best_fitness=float(fb),
                                    curve=[float(c) for c in curve])
                    t2 = time.time()
                    model, info['val_f1'], info['epochs'] = fit_sae(cfg, Xi, yi, Xv, yv, k, 50, 200, 15, seed)
                    info.update(train_s=time.time() - t2, cfg=cfg, params=n_params(dim, cfg['h'], k))
                    t1 = time.time(); p = predict(model, Xt)
                info['infer_ms_per_window'] = (time.time() - t1) / max(1, len(yt)) * 1000
                np.savez(fn, y_true=yt, y_pred=p, info=json.dumps(info))
                print(f"run {run} subj {int(test_subj)} {meth}: macroF1={f1_score(yt, p, average='macro'):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
