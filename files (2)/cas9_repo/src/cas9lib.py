"""Core library for the Cas9 (Spencer 2017, positive selection) experiment.

Everything here is sequence-derived. NO pretrained PLM, structure model or diffusion
model is used (weights/structures were not reachable in the sandbox) -> conclusions
about those components are NOT drawn from this code; see REPORT.pdf.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from Bio.Align import substitution_matrices

AA = "ACDEFGHIKLMNPQRSTVWY"
L = 1368
BUFFER = 15  # residues around held-out positions removed from training

# ---------------------------------------------------------------- amino-acid tables
_KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8,
           K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)          # Kyte-Doolittle
_VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7,
            L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
_CHG = {a: 0.0 for a in AA}; _CHG.update(K=1, R=1, D=-1, E=-1, H=0.1)
_POL = dict(A=8.1, R=10.5, N=11.6, D=13.0, C=5.5, Q=10.5, E=12.3, G=9.0, H=10.4, I=5.2, L=4.9,
            K=11.3, M=5.7, F=5.2, P=8.0, S=9.2, T=8.6, W=5.4, Y=6.2, V=5.9)              # Grantham polarity
_HEL = dict(A=1.42, R=.98, N=.67, D=1.01, C=.70, Q=1.11, E=1.51, G=.57, H=1.00, I=1.08, L=1.21,
            K=1.16, M=1.45, F=1.13, P=.57, S=.77, T=.83, W=1.08, Y=.69, V=1.06)          # Chou-Fasman P(alpha)
_BET = dict(A=.83, R=.93, N=.89, D=.54, C=1.19, Q=1.10, E=.37, G=.75, H=.87, I=1.60, L=1.30,
            K=.74, M=1.05, F=1.38, P=.55, S=.75, T=1.19, W=1.37, Y=1.47, V=1.70)         # Chou-Fasman P(beta)
PROPS = dict(kd=_KD, vol=_VOL, chg=_CHG, pol=_POL, hel=_HEL, bet=_BET)
_B62 = substitution_matrices.load("BLOSUM62")

# ---------------------------------------------------------------- genetic code / library artefact
_b = "TCAG"
_aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODE = {a + b_ + c: _aas[i] for i, (a, b_, c) in enumerate(itertools.product(_b, _b, _b))}
CODONS = {}
for k, v in CODE.items():
    CODONS.setdefault(v, []).append(k)
_TS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}


def reach(w, m):
    """(#single-nt paths wt->mut, #transition paths) summed over synonymous wt codons."""
    n = ts = 0
    for c1 in CODONS[w]:
        for c2 in CODONS[m]:
            diff = [(x, y) for x, y in zip(c1, c2) if x != y]
            if len(diff) == 1:
                n += 1
                ts += diff[0] in _TS
    return n, ts


# ---------------------------------------------------------------- domains (approximate, Nishimasu 2014 topology)
def domain_of(pos):
    if pos <= 59 or 718 <= pos <= 774 or 909 <= pos <= 1098:
        return "RuvC"
    if 60 <= pos <= 717:
        return "REC"
    if 775 <= pos <= 908:
        return "HNH"
    return "PI"


# ---------------------------------------------------------------- data
def load(data_dir):
    d = pd.read_csv(f"{data_dir}/CAS9_STRP1_Spencer_2017_positive.csv")
    seq = "".join(open(f"{data_dir}/CAS9_STRP1_WT.fasta").read().split("\n")[1:])
    d["wt"] = d.mutant.str[0]; d["mut"] = d.mutant.str[-1]; d["pos"] = d.mutant.str[1:-1].astype(int)
    assert len(seq) == L and all(seq[p - 1] == w for p, w in zip(d.pos, d.wt))
    d["y"] = d.DMS_score.values
    d["domain"] = d.pos.map(domain_of)
    return d.reset_index(drop=True), seq


# ---------------------------------------------------------------- features
def sub_features(d):
    cols = {}
    for a in AA:
        cols[f"wt_{a}"] = (d.wt == a).astype(float)
        cols[f"mut_{a}"] = (d.mut == a).astype(float)
    onehot = pd.DataFrame(cols)
    bl = pd.DataFrame({"blosum": [_B62[w][m] for w, m in zip(d.wt, d.mut)]})
    pr = {}
    for k, t in PROPS.items():
        pr[f"d_{k}"] = [t[m] - t[w] for w, m in zip(d.wt, d.mut)]
        pr[f"mut_{k}"] = [t[m] for m in d.mut]
        pr[f"wt_{k}"] = [t[w] for w in d.wt]
    pr["to_P"] = (d.mut == "P").astype(float); pr["from_G"] = (d.wt == "G").astype(float)
    pr["from_P"] = (d.wt == "P").astype(float); pr["to_G"] = (d.mut == "G").astype(float)
    return onehot, bl, pd.DataFrame(pr)


def reach_features(d):
    r = [reach(w, m) for w, m in zip(d.wt, d.mut)]
    return pd.DataFrame({"reach_n": [x[0] for x in r], "reach_ts": [x[1] for x in r],
                         "reach_tv": [x[0] - x[1] for x in r]})


def ctx_table(seq):
    """Per-position context features from the WT sequence only (index 1..L)."""
    rows = []
    P = {k: np.array([t[a] for a in seq]) for k, t in PROPS.items()}
    isGP = np.array([a in "GP" for a in seq], float); isAr = np.array([a in "FWY" for a in seq], float)
    isCh = np.array([a in "KRDE" for a in seq], float)
    feats = {}
    for w in (2, 5, 10, 20):
        k = np.ones(2 * w + 1) / (2 * w + 1)
        pad = lambda v: np.pad(v, w, mode="edge")
        conv = lambda v: np.convolve(pad(v), k, mode="valid")
        for name, v in [("kd", P["kd"]), ("vol", P["vol"]), ("chg", P["chg"]), ("hel", P["hel"]),
                        ("bet", P["bet"]), ("gp", isGP), ("aro", isAr), ("chd", isCh)]:
            feats[f"w{w}_{name}"] = conv(v)
    for off in (-4, -3, -2, -1, 1, 2, 3, 4):
        for name in ("kd", "chg"):
            v = P[name]; idx = np.clip(np.arange(L) + off, 0, L - 1)
            feats[f"o{off}_{name}"] = v[idx]
    df = pd.DataFrame(feats); df.index = np.arange(1, L + 1)
    return df


# ---------------------------------------------------------------- splits
def make_splits(d, seed=0):
    rng = np.random.RandomState(seed)
    s = pd.DataFrame({"mutant": d.mutant, "pos": d.pos, "domain": d.domain})
    s["fold_random"] = rng.permutation(np.arange(len(d)) % 10)
    pf = rng.permutation(np.arange(L) % 10)                       # random position -> fold
    s["fold_posgroup"] = pf[d.pos.values - 1]
    s["fold_block"] = np.minimum((d.pos.values - 1) // 137, 9)     # 10 contiguous blocks (137 aa)
    s["fold_domain"] = d.domain.map({"RuvC": 0, "REC": 1, "HNH": 2, "PI": 3})
    return s


DIST = np.abs(np.arange(1, L + 1)[:, None] - np.arange(1, L + 1)[None, :])


def train_mask(d, fold_col, k, buffer):
    """Training rows: fold != k and (if buffer>0) position farther than `buffer` from any held-out position."""
    te = fold_col == k
    tr = ~te
    if buffer > 0:
        held = np.zeros(L, bool); held[d.pos.values[te] - 1] = True
        near = (DIST[:, held] <= buffer).any(axis=1)              # positions near a held-out one
        tr &= ~near[d.pos.values - 1]
    return tr, te


# ---------------------------------------------------------------- metrics
def sp(a, b):
    if np.ptp(b) < 1e-9 or np.ptp(a) < 1e-9:   # constant input -> no ranking information
        return 0.0
    return spearmanr(a, b)[0]


def within_pos_spearman(d, pred, min_n=5):
    out = []
    for _, g in d.assign(p=pred).groupby("pos"):
        if len(g) >= min_n and g.p.nunique() > 1:
            out.append(spearmanr(g.p, g.y)[0])
        elif len(g) >= min_n:
            out.append(0.0)
    return float(np.nanmean(out))


def topdecile_enrichment(y, p):
    k = int(0.1 * len(y)); top_true = set(np.argsort(-y)[:k]); top_pred = np.argsort(-p)[:k]
    return np.mean([i in top_true for i in top_pred]) / 0.1


def metrics(d, pred, folds):
    ok = ~np.isnan(pred)
    dd, p, f = d[ok], pred[ok], folds[ok]
    per = [sp(p[f == k], dd.y.values[f == k]) for k in np.unique(f)]
    return dict(n=int(ok.sum()), spearman=sp(p, dd.y.values), fold_mean=float(np.mean(per)), fold_sd=float(np.std(per)),
                auroc=roc_auc_score(dd.DMS_score_bin.values, p), within_pos=within_pos_spearman(dd, p),
                top10_enrich=topdecile_enrichment(dd.y.values, p))


def cluster_boot(d, preds, B=300, seed=0, ref=None):
    """Bootstrap over positions. Returns CI of pooled Spearman per model and paired delta vs `ref` model."""
    rng = np.random.RandomState(seed)
    groups = [np.where(d.pos.values == p)[0] for p in range(1, L + 1)]
    y = d.y.values; res = {m: [] for m in preds}; dres = {m: [] for m in preds}
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.randint(0, L, L)])
        r = {m: sp(preds[m][idx], y[idx]) for m in preds}
        for m in preds:
            res[m].append(r[m])
            if ref:
                dres[m].append(r[m] - r[ref])
    out = {}
    for m in preds:
        lo, hi = np.percentile(res[m], [2.5, 97.5])
        row = dict(ci_lo=lo, ci_hi=hi)
        if ref and m != ref:
            dl, dh = np.percentile(dres[m], [2.5, 97.5]); row.update(delta_lo=dl, delta_hi=dh, delta_mean=np.mean(dres[m]))
        out[m] = row
    return out


# ---------------------------------------------------------------- models
def hgb(**kw):
    p = dict(learning_rate=0.05, max_iter=150, max_depth=3, min_samples_leaf=40, l2_regularization=5.0, random_state=0)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def winsor(y):
    lo, hi = np.percentile(y, [1, 99]); return np.clip(y, lo, hi)


def local_mean_baseline(d, tr, te, K=150):
    """Mean y of the K training variants nearest in sequence position (own position included if in train)."""
    trp = d.pos.values[tr]; try_ = d.y.values[tr]; order = np.argsort(trp); trp, try_ = trp[order], try_[order]
    cache = {}
    out = np.zeros(te.sum())
    for j, p in enumerate(d.pos.values[te]):
        if p not in cache:
            dist = np.abs(trp - p); idx = np.argsort(dist, kind="stable")[:K]; cache[p] = try_[idx].mean()
        out[j] = cache[p]
    return out


def site_model(d, ctxtab, tr, te, seed=0, use_window=True, perm=None, kind='hgb'):
    """Stage 1: predict position-level tolerance (mean winsorised score) from WT-sequence context.
    Returns s_hat for every position: out-of-fold (inner block CV, buffered) for training positions,
    fully-trained prediction for test positions."""
    ct = ctxtab if perm is None else pd.DataFrame(ctxtab.values[perm], index=ctxtab.index, columns=ctxtab.columns)
    wtprops = pd.DataFrame({f"wt_{k}": [t[a] for a in SEQ] for k, t in PROPS.items()}, index=ctxtab.index)
    X = (pd.concat([ct, wtprops], axis=1) if use_window else wtprops).values
    yw = winsor(d.y.values[tr]); dtr = d[tr].assign(yw=yw)
    pm = dtr.groupby("pos").yw.agg(["mean", "count"])
    trpos = pm.index.values; ymean = pm["mean"].values; w = pm["count"].values
    s_hat = np.full(L, np.nan)
    if kind == 'ridge':
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        class _R:
            def __init__(self): self.m = make_pipeline(StandardScaler(), Ridge(alpha=300.0))
            def fit(self, X, y, sample_weight=None): self.m.fit(X, y, ridge__sample_weight=sample_weight); return self
            def predict(self, X): return self.m.predict(X)
        mk = _R
    else:
        mk = lambda: hgb(max_iter=100, min_samples_leaf=20)
    # inner OOF for training positions
    chunk = (trpos - 1) // 60; inner = chunk % 5
    for f in range(5):
        te_i = trpos[inner == f]
        far = (DIST[np.ix_(trpos - 1, te_i - 1)] > BUFFER).all(axis=1) & (inner != f)
        m = mk().fit(X[trpos[far] - 1], ymean[far], sample_weight=w[far])
        s_hat[te_i - 1] = m.predict(X[te_i - 1])
    tepos = np.unique(d.pos.values[te])
    m = mk().fit(X[trpos - 1], ymean, sample_weight=w)
    s_hat[tepos - 1] = m.predict(X[tepos - 1])
    return s_hat


SEQ = None  # set by init()


def init(seq):
    global SEQ
    SEQ = seq
