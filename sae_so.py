"""Pilot experiment for SOA-SAE (NumPy, CPU).

Protocol (pilot, one run per dataset):
  * fixed subject-independent split: UCI-HAR official (21 train / 9 test subjects); WISDM 11 of 36 test
    subjects; PAMAP2 3 of subjects 101-108 test (subject 109 has 48 windows only and stays in training)
  * inner validation: 20% of training subjects (seeded); z-score statistics from inner-training subjects only
  * search space = manuscript Table 2; fitness F = 0.85 (1 - macroF1_val) + 0.15 P(x)/P_max
  * equal budget for every tuner: N = 10, T = 10 -> 110 fitness evaluations; low-fidelity search
    (<= 2000 training / 1500 validation windows, 3 pre-training epochs per layer, <= 15 fine-tuning epochs)
  * final model: best configuration retrained on all inner-training windows (10 pre-training epochs per
    layer, <= 60 fine-tuning epochs, patience 8, early stopping on inner-validation subjects), tested once
Results are appended to results/<dataset>.json after every method so the job can resume.
"""
import argparse, json, os, time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

F32 = np.float32


# ------------------------------------------------------------------ SAE (ReLU, linear decoder, MSE + L2)
class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(w) for w in params]; self.v = [np.zeros_like(w) for w in params]

    def step(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for i, (w, g) in enumerate(zip(self.p, grads)):
            self.m[i] *= b1; self.m[i] += (1 - b1) * g
            self.v[i] *= b2; self.v[i] += (1 - b2) * g * g
            w -= (self.lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + eps)).astype(F32)


def he(rng, a, b):
    return (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(F32)


class SAE:
    def __init__(self, cfg, k, seed):
        self.c, self.k, self.rng = cfg, k, np.random.default_rng(seed)

    def n_params(self, d):
        dims = [d] + self.c['h']
        return sum(a * b + b for a, b in zip(dims[:-1], dims[1:])) + dims[-1] * self.k + self.k

    def _pretrain(self, H, nh, epochs):
        c, rng = self.c, self.rng
        d = H.shape[1]
        W, b, W2, b2 = he(rng, d, nh), np.zeros(nh, F32), he(rng, nh, d), np.zeros(d, F32)
        opt = Adam([W, b, W2, b2], c['lr'])
        for _ in range(epochs):
            idx = rng.permutation(len(H))
            for s in range(0, len(H), c['batch']):
                xb = H[idx[s:s + c['batch']]]; m = len(xb)
                z = xb @ W + b; a = np.maximum(z, 0)
                r = a @ W2 + b2
                dr = (r - xb) * (2.0 / (m * d))
                da = (dr @ W2.T) * (z > 0)
                opt.step([xb.T @ da + c['wd'] * W, da.sum(0), a.T @ dr + c['wd'] * W2, dr.sum(0)])
        return W, b

    def forward(self, X):
        H = X
        for W, b in zip(self.Ws, self.bs):
            H = np.maximum(H @ W + b, 0)
        return H @ self.Wo + self.bo

    def predict(self, X, bs=4096):
        return np.concatenate([self.forward(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])

    def fit(self, X, y, Xv, yv, pre_epochs, ft_epochs, patience):
        c = self.c
        self.Ws, self.bs = [], []
        H = X
        for nh in c['h']:
            W, b = self._pretrain(H, nh, pre_epochs)
            self.Ws.append(W); self.bs.append(b); H = np.maximum(H @ W + b, 0)
        self.Wo = he(self.rng, H.shape[1], self.k); self.bo = np.zeros(self.k, F32)
        params = self.Ws + self.bs + [self.Wo, self.bo]
        opt = Adam(params, c['lr'])
        Y = np.eye(self.k, dtype=F32)[y]
        L = len(self.Ws); best, best_state, wait = -1.0, None, 0
        keep = 1.0 - c['dropout']
        for ep in range(ft_epochs):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), c['batch']):
                bi = idx[s:s + c['batch']]
                acts, zs, masks = [X[bi]], [], []
                for W, b in zip(self.Ws, self.bs):
                    z = acts[-1] @ W + b; zs.append(z)
                    mk = (self.rng.random(z.shape) < keep).astype(F32) / keep
                    masks.append(mk); acts.append(np.maximum(z, 0) * mk)
                o = acts[-1] @ self.Wo + self.bo; o -= o.max(1, keepdims=True)
                p = np.exp(o); p /= p.sum(1, keepdims=True)
                do = (p - Y[bi]) / len(bi)
                gWo, gbo = acts[-1].T @ do + c['wd'] * self.Wo, do.sum(0)
                dl = do @ self.Wo.T
                gW, gb = [None] * L, [None] * L
                for l in range(L - 1, -1, -1):
                    dl = dl * masks[l] * (zs[l] > 0)
                    gW[l] = acts[l].T @ dl + c['wd'] * self.Ws[l]; gb[l] = dl.sum(0)
                    dl = dl @ self.Ws[l].T
                opt.step(gW + gb + [gWo, gbo])
            f1 = f1_score(yv, self.predict(Xv), average='macro')
            if f1 > best + 1e-4:
                best, wait, best_state = f1, 0, [w.copy() for w in params]
            else:
                wait += 1
                if wait >= patience:
                    break
        for w, bw in zip(params, best_state):
            w[...] = bw
        self.best_val_f1, self.epochs_run = best, ep + 1
        return self


# ------------------------------------------------------------------ search space (manuscript Table 2)
def decode(x, smax):
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    L = 2 + int(round(x[0]))
    s1 = int(round(64 + x[1] * (smax - 64)))
    s2 = max(8, int(round((0.25 + 0.5 * x[2]) * s1)))
    s3 = max(8, int(round((0.25 + 0.5 * x[3]) * s2)))
    h = [s1, s2, s3][:L]
    return dict(h=h, lr=float(10 ** (-4 + 2 * x[4])), wd=float(10 ** (-6 + 3 * x[5])),
                dropout=float(0.5 * x[6]), batch=[32, 64, 128][min(2, int(x[7] * 3))])


D = 8


# ------------------------------------------------------------------ optimizers (equal budget N*(T+1))
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
        if Q < 0.25:
            for i in range(Nm):
                r = rng.integers(Nm); A = np.exp(-fm[r] / (fm[i] + 1e-12))
                Ym[i] = Xm[r] + sgn() * c2 * A * rng.random(D)
            for i in range(Nf):
                r = rng.integers(Nf); A = np.exp(-ff[r] / (ff[i] + 1e-12))
                Yf[i] = Xf[r] + sgn() * c2 * A * rng.random(D)
        elif Temp > 0.6:
            for i in range(Nm):
                Ym[i] = Xfood + sgn() * c3 * Temp * rng.random(D) * (Xfood - Xm[i])
            for i in range(Nf):
                Yf[i] = Xfood + sgn() * c3 * Temp * rng.random(D) * (Xfood - Xf[i])
        elif rng.random() > 0.6:
            for i in range(Nm):
                Ym[i] = Xm[i] + c3 * np.exp(-fbf / (fm[i] + 1e-12)) * rng.random(D) * (Q * Xbf - Xm[i])
            for i in range(Nf):
                Yf[i] = Xf[i] + c3 * np.exp(-fbm / (ff[i] + 1e-12)) * rng.random(D) * (Q * Xbm - Xf[i])
        else:
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
    o = F.argsort(); A_, B_, C_ = (X[o[k]].copy() for k in range(3)); fa = F[o[0]]
    best_x, best_f = A_.copy(), fa; curve = [best_f]
    for t in range(T):
        a = 2 - 2 * t / T
        for i in range(N):
            xs = []
            for Ld in (A_, B_, C_):
                A = 2 * a * rng.random(D) - a; Cc = 2 * rng.random(D)
                xs.append(Ld - A * np.abs(Cc * Ld - X[i]))
            X[i] = np.clip(np.mean(xs, 0), 0, 1)
        F = np.array([f(x) for x in X]); o = F.argsort()
        A_, B_, C_ = (X[o[k]].copy() for k in range(3))
        if F[o[0]] < best_f: best_x, best_f = X[o[0]].copy(), F[o[0]]
        curve.append(best_f)
    return best_x, best_f, curve


def ga(f, N, T, rng, pc=0.8, pm=0.1, sigma=0.1):
    X = rng.random((N, D)); F = np.array([f(x) for x in X])
    curve = [F.min()]
    for _ in range(T):
        def tour():
            c = rng.choice(N, min(3, N), replace=False); return X[c[F[c].argmin()]]
        new = []
        while len(new) < N:
            a, b = tour().copy(), tour().copy()
            if rng.random() < pc:
                m = rng.random(D) < 0.5; a[m], b[m] = b[m].copy(), a[m].copy()
            for ch in (a, b):
                mm = rng.random(D) < pm; ch[mm] += rng.normal(0, sigma, mm.sum())
                new.append(np.clip(ch, 0, 1))
        newX = np.array(new[:N]); newF = np.array([f(x) for x in newX])
        e = F.argmin()
        if F[e] < newF.min():            # elitism: keep the best parent
            w = newF.argmax(); newX[w], newF[w] = X[e], F[e]
        X, F = newX, newF
        curve.append(F.min())
    b = F.argmin()
    return X[b], F[b], curve


def random_search(f, N, T, rng):
    best_x, best_f, curve = None, np.inf, []
    for k in range(N * (T + 1)):
        x = rng.random(D); v = f(x)
        if v < best_f: best_x, best_f = x, v
        if (k + 1) % N == 0: curve.append(best_f)
    return best_x, best_f, curve


# ------------------------------------------------------------------ helpers
def metrics(y, p):
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p, average='macro', zero_division=0),
                recall=recall_score(y, p, average='macro', zero_division=0), f1=f1_score(y, p, average='macro'),
                kappa=cohen_kappa_score(y, p))


def per_subject(y, p, s):
    return {int(u): dict(f1=f1_score(y[s == u], p[s == u], average='macro'),
                         acc=accuracy_score(y[s == u], p[s == u]), n=int((s == u).sum())) for u in np.unique(s)}


def strat_cap(y, cap, rng):
    if len(y) <= cap:
        return np.arange(len(y))
    idx = []
    for c in np.unique(y):
        ci = np.flatnonzero(y == c); k = max(1, int(round(cap * len(ci) / len(y))))
        idx.append(rng.choice(ci, min(k, len(ci)), replace=False))
    return np.sort(np.concatenate(idx))


SMAX = dict(wisdm=256, ucihar=1024, pamap2=1536)
UNTUNED = dict(wisdm=[128, 64, 32], ucihar=[512, 256, 128], pamap2=[1536, 768, 384])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(SMAX))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--N', type=int, default=10)
    ap.add_argument('--T', type=int, default=10)
    ap.add_argument('--methods', default='baselines,SAE,SO,PSO,GWO,GA,Random')
    ap.add_argument('--beta', type=float, default=0.15, help='complexity weight in the fitness (alpha = 1 - beta)')
    a = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    out = f'results/{a.dataset}_s{a.seed}' + ('' if a.beta == 0.15 else f'_beta{a.beta:g}') + '.json'
    R = json.load(open(out)) if os.path.exists(out) else {}

    d = np.load(os.path.join(os.environ.get('HAR_DATA', 'data'), 'cache', f'{a.dataset}.npz'))
    X, y, s, labels = d['X'], d['y'], d['s'], [str(l) for l in d['labels']]
    k = len(labels); rng = np.random.default_rng(a.seed)
    subs = np.unique(s)
    if a.dataset == 'ucihar':
        test_s = np.loadtxt(os.path.join(os.environ.get('HAR_DATA', 'data'), 'ucihar', 'UCI HAR Dataset', 'test', 'subject_test.txt')).astype(int)
        test_s = np.unique(test_s)
    elif a.dataset == 'wisdm':
        test_s = rng.choice(subs, 11, replace=False)
    else:
        test_s = rng.choice(subs[subs != 109], 3, replace=False)
    train_s = np.setdiff1d(subs, test_s)
    val_s = rng.choice(train_s, max(1, int(round(0.2 * len(train_s)))), replace=False)
    inner_s = np.setdiff1d(train_s, val_s)
    mi, mv, mt = np.isin(s, inner_s), np.isin(s, val_s), np.isin(s, test_s)
    mu, sd = X[mi].mean(0), X[mi].std(0) + 1e-6
    Xi, Xv, Xt = ((X[m] - mu) / sd for m in (mi, mv, mt))
    yi, yv, yt, st = y[mi], y[mv], y[mt], s[mt]
    R['protocol'] = dict(test_subjects=test_s.tolist(), val_subjects=val_s.tolist(), inner_train_subjects=inner_s.tolist(),
                         n_inner=int(mi.sum()), n_val=int(mv.sum()), n_test=int(mt.sum()), dim=int(X.shape[1]),
                         labels=labels, N=a.N, T=a.T, evals=a.N * (a.T + 1), seed=a.seed)
    print(a.dataset, R['protocol'], flush=True)

    si, sv = strat_cap(yi, 2000, rng), strat_cap(yv, 1500, rng)
    Xs_, ys_, Xsv, ysv = Xi[si], yi[si], Xv[sv], yv[sv]
    pmax = SAE(decode(np.array([1, 1, 1, 1, 0, 0, 0, 0]), SMAX[a.dataset]), k, 0).n_params(X.shape[1])

    def save():
        json.dump(R, open(out, 'w'), indent=1, default=float)

    def final_eval(name, cfg, extra):
        t0 = time.time()
        m = SAE(cfg, k, a.seed).fit(Xi, yi, Xv, yv, pre_epochs=10, ft_epochs=60, patience=8)
        ttrain = time.time() - t0
        t0 = time.time(); p = m.predict(Xt); tinf = (time.time() - t0) / len(Xt) * 1000
        R[name] = dict(metrics=metrics(yt, p), per_subject=per_subject(yt, p, st), cm=confusion_matrix(yt, p, labels=range(k)).tolist(),
                       cfg=cfg, params=m.n_params(X.shape[1]), final_train_s=ttrain, infer_ms_per_window=tinf,
                       val_f1=m.best_val_f1, ft_epochs=m.epochs_run, **extra)
        save(); print(name, R[name]['metrics'], f'train {ttrain:.0f}s', flush=True)

    for meth in a.methods.split(','):
        if meth in R:
            print('skip', meth); continue
        if meth == 'baselines':
            for nm, clf in [('SVM', SVC(C=10, gamma='scale', random_state=a.seed)),
                            ('RF', RandomForestClassifier(300, n_jobs=-1, random_state=a.seed))]:
                t0 = time.time(); clf.fit(np.vstack([Xi, Xv]), np.r_[yi, yv]); p = clf.predict(Xt)
                R[nm] = dict(metrics=metrics(yt, p), per_subject=per_subject(yt, p, st), train_s=time.time() - t0)
                save(); print(nm, R[nm]['metrics'], flush=True)
            R['baselines'] = True; save(); continue
        if meth == 'SAE':
            cfg = dict(h=UNTUNED[a.dataset], lr=1e-3, wd=1e-4, dropout=0.2, batch=64)
            final_eval('SAE', cfg, {}); continue
        evals = []
        def fobj(x):
            cfg = decode(x, SMAX[a.dataset])
            m = SAE(cfg, k, a.seed).fit(Xs_, ys_, Xsv, ysv, pre_epochs=3, ft_epochs=15, patience=4)
            f = (1 - a.beta) * (1 - m.best_val_f1) + a.beta * m.n_params(X.shape[1]) / pmax
            evals.append(f); return f
        opt = dict(SO=snake_optimizer, PSO=pso, GWO=gwo, GA=ga, Random=random_search)[meth]
        t0 = time.time(); xb, fb, curve = opt(fobj, a.N, a.T, np.random.default_rng(a.seed + 7))
        tsearch = time.time() - t0
        print(f'{meth}: search {tsearch:.0f}s, {len(evals)} evals, best fitness {fb:.4f}', flush=True)
        final_eval(f'{meth}-SAE', decode(xb, SMAX[a.dataset]),
                   dict(search_s=tsearch, n_evals=len(evals), best_fitness=fb, curve=[float(c) for c in curve]))


if __name__ == '__main__':
    main()
