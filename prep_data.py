"""Build windowed, flattened datasets exactly as described in the manuscript (Section 3.1).

UCI-HAR : 2.56 s windows at 50 Hz (128 samples), official segmentation; channels = total_acc xyz + body_gyro xyz -> 768
WISDM   : 20 Hz, 51-sample windows (2.56 s), 50% overlap within contiguous user-activity segments; acc xyz -> 153
PAMAP2  : 100 Hz, 256-sample windows (2.56 s), 50% overlap within contiguous activity segments; chest IMU
          acc(+-16g) xyz + gyro xyz + mag xyz -> 2304; 12 protocol activities (activity 0 = transient, removed)
Outputs float32 arrays X (n, d), y (n,), subject (n,) to data/cache/<name>.npz
"""
import os, re, glob
import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--root', default='data', help='folder holding ucihar/, wisdm/, pamap2/ (extracted downloads)')
ROOT = _ap.parse_args().root
OUT = os.path.join(ROOT, 'cache'); os.makedirs(OUT, exist_ok=True)


def ucihar():
    base = os.path.join(ROOT, 'ucihar', 'UCI HAR Dataset')
    ch = ['total_acc_x', 'total_acc_y', 'total_acc_z', 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    Xs, ys, ss = [], [], []
    for split in ('train', 'test'):
        sig = [np.loadtxt(os.path.join(base, split, 'Inertial Signals', f'{c}_{split}.txt'), dtype=np.float32) for c in ch]
        Xs.append(np.stack(sig, axis=2).reshape(len(sig[0]), -1))       # (n, 128*6), time-major
        ys.append(np.loadtxt(os.path.join(base, split, f'y_{split}.txt')).astype(int) - 1)
        ss.append(np.loadtxt(os.path.join(base, split, f'subject_{split}.txt')).astype(int))
    labels = ['Walking', 'Upstairs', 'Downstairs', 'Sitting', 'Standing', 'Lying']
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss), labels


def wisdm(win=51, step=25):
    path = os.path.join(ROOT, 'wisdm', 'WISDM_ar_v1.1', 'WISDM_ar_v1.1_raw.txt')
    labels = ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
    lab = {a: i for i, a in enumerate(labels)}
    users, acts, sig = [], [], []
    with open(path) as f:
        for rec in f.read().replace('\n', '').split(';'):
            p = rec.strip().strip(',').split(',')
            if len(p) < 6:
                continue
            try:
                u, a = int(p[0]), p[1]
                xyz = [float(p[3]), float(p[4]), float(re.sub(r'[^0-9eE.+-]', '', p[5]))]
            except ValueError:
                continue
            if a not in lab:
                continue
            users.append(u); acts.append(lab[a]); sig.append(xyz)
    users, acts, sig = np.array(users), np.array(acts), np.array(sig, dtype=np.float32)
    X, y, s = [], [], []
    b = np.r_[0, np.flatnonzero((np.diff(users) != 0) | (np.diff(acts) != 0)) + 1, len(users)]
    for i0, i1 in zip(b[:-1], b[1:]):
        for k in range(i0, i1 - win + 1, step):
            X.append(sig[k:k + win].reshape(-1)); y.append(acts[i0]); s.append(users[i0])
    return np.array(X, dtype=np.float32), np.array(y), np.array(s), labels


def pamap2(win=256, step=128):
    files = sorted(glob.glob(os.path.join(ROOT, 'pamap2', 'PAMAP2_Dataset', 'Protocol', 'subject1*.dat')))
    act_ids = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    names = ['Lying', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic walking',
             'Ascending stairs', 'Descending stairs', 'Vacuum cleaning', 'Ironing', 'Rope jumping']
    amap = {a: i for i, a in enumerate(act_ids)}
    cols = [21, 22, 23, 27, 28, 29, 30, 31, 32]   # chest acc16, gyro, mag
    X, y, s = [], [], []
    for fp in files:
        subj = int(re.search(r'subject(\d+)', fp).group(1))
        d = np.loadtxt(fp, dtype=np.float64)
        act = d[:, 1].astype(int)
        sig = d[:, cols]
        b = np.r_[0, np.flatnonzero(np.diff(act) != 0) + 1, len(act)]
        for i0, i1 in zip(b[:-1], b[1:]):
            a = act[i0]
            if a not in amap:
                continue
            seg = sig[i0:i1].copy()
            for c in range(seg.shape[1]):                     # linear interpolation of dropped samples
                v = seg[:, c]; m = np.isnan(v)
                if m.all():
                    seg = None; break
                if m.any():
                    v[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), v[~m])
            if seg is None:
                continue
            for k in range(0, len(seg) - win + 1, step):
                X.append(seg[k:k + win].astype(np.float32).reshape(-1)); y.append(amap[a]); s.append(subj)
        print(f'  {os.path.basename(fp)}: total windows so far {len(X)}', flush=True)
    return np.array(X, dtype=np.float32), np.array(y), np.array(s), names


if __name__ == '__main__':
    for name, fn in [('ucihar', ucihar), ('wisdm', wisdm), ('pamap2', pamap2)]:
        X, y, s, labels = fn()
        np.savez(os.path.join(OUT, f'{name}.npz'), X=X, y=y, s=s, labels=np.array(labels))
        print(name, X.shape, 'subjects', len(np.unique(s)), 'class counts', np.bincount(y).tolist(), flush=True)
