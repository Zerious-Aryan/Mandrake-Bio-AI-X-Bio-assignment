"""Run all experiments. Usage: python run_experiments.py --data /path/to/data_pack --out results
Deterministic (seeds fixed). ~10-20 min on 1 CPU."""
import argparse, json, os, time
import numpy as np, pandas as pd
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import cas9lib as C

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--boot", type=int, default=300); args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
d, seq = C.load(args.data); C.init(seq)
splits = C.make_splits(d); splits.to_csv("split_assignments.csv", index=False)
ctx = C.ctx_table(seq)
oh, bl, pr = C.sub_features(d); rc = C.reach_features(d)
X_ctx = ctx.loc[d.pos.values].values
rng = np.random.RandomState(123); PERM = rng.permutation(C.L)          # position->context shuffle (control)
X_ctx_shuf = ctx.values[PERM][d.pos.values - 1]
B = {"oh": oh.values, "bl": bl.values, "pr": pr.values, "rc": rc.values}
SUB = np.hstack([B["oh"], B["bl"], B["pr"]])
yv = d.y.values

def fitpred(Xtr, ytr, Xte, seed=0):
    return C.hgb(random_state=seed).fit(Xtr, C.winsor(ytr)).predict(Xte)

def run_fold(name, tr, te, cache, d_shuf):
    """Return predictions for `te` rows for model `name`."""
    def shat(key, use_window=True, perm=None, dd=d, kind='hgb'):
        if key not in cache:
            cache[key] = C.site_model(dd, ctx, tr, te, use_window=use_window, perm=perm, kind=kind)
        return cache[key]
    def sh_cols(s, rows):
        v = s[d.pos.values - 1][rows]; return v
    def E_X(s):
        v = s[d.pos.values - 1][:, None]
        return np.hstack([SUB, v, v * B["bl"], v * np.abs(pr[["d_kd"]].values)])
    if name == "blosum": return B["bl"][te, 0].astype(float)
    if name == "local_mean": return C.local_mean_baseline(d, tr, te)
    if name == "mutaa_mean":
        m = d[tr].groupby("mut").y.mean(); return d.mut[te].map(m).values
    if name == "posmean":
        m = d[tr].groupby("pos").y.mean(); lm = C.local_mean_baseline(d, tr, te)
        r = d.pos[te].map(m).values; return np.where(np.isnan(r), lm, r)
    if name == "A2_reach": return fitpred(B["rc"][tr], yv[tr], B["rc"][te])
    if name == "A1_artifact":
        X = np.hstack([B["oh"], B["rc"]]); return fitpred(X[tr], yv[tr], X[te])
    if name == "M1_props":
        return fitpred(B["pr"][tr], yv[tr], B["pr"][te])
    if name == "M1_sub": return fitpred(SUB[tr], yv[tr], SUB[te])
    if name == "M2_sub_ctx":
        X = np.hstack([SUB, X_ctx]); return fitpred(X[tr], yv[tr], X[te])
    if name == "M2_shufctx":
        X = np.hstack([SUB, X_ctx_shuf]); return fitpred(X[tr], yv[tr], X[te])
    if name == "site_only": return shat("s")[d.pos.values[te] - 1]
    if name == "site_ridge": return shat("sr", kind="ridge")[d.pos.values[te] - 1]
    if name == "E_twostage":
        X = E_X(shat("s")); return fitpred(X[tr], yv[tr], X[te])
    if name == "E_ridge":  # POST-HOC variant (added after seeing GBM site model fail); disclosed in report
        X = E_X(shat("sr", kind="ridge")); return fitpred(X[tr], yv[tr], X[te])
    if name == "E_nowindow":
        X = E_X(shat("nw", use_window=False)); return fitpred(X[tr], yv[tr], X[te])
    if name == "E_shufctx":
        X = E_X(shat("sp", perm=PERM)); return fitpred(X[tr], yv[tr], X[te])
    if name == "M2_shuffleY":
        X = np.hstack([SUB, X_ctx]); return fitpred(X[tr], d_shuf[tr], X[te])
    if name == "E_shuffleY":
        dd = d.assign(y=d_shuf); s = shat("sy", dd=dd); X = E_X(s); return fitpred(X[tr], d_shuf[tr], X[te])
    raise KeyError(name)

PLAN = {
 "random":   dict(col="fold_random",   buf=0,        models=["blosum", "local_mean", "posmean", "M1_sub", "M2_sub_ctx", "E_twostage"]),
 "posgroup": dict(col="fold_posgroup", buf=0,        models=["blosum", "local_mean", "M1_sub", "M2_sub_ctx", "E_twostage"]),
 "block":    dict(col="fold_block",    buf=C.BUFFER, models=["blosum", "local_mean", "mutaa_mean", "A2_reach", "A1_artifact", "M1_props", "M1_sub",
                                                               "M2_sub_ctx", "M2_shufctx", "site_only", "site_ridge", "E_twostage", "E_ridge", "E_nowindow", "E_shufctx",
                                                               "M2_shuffleY", "E_shuffleY"]),
 "domain":   dict(col="fold_domain",   buf=C.BUFFER, models=["blosum", "local_mean", "A1_artifact", "M1_sub", "M2_sub_ctx", "site_only", "site_ridge", "E_twostage", "E_ridge", "E_shufctx"]),
}
only = os.environ.get("ONLY_SPLITS"); PLAN = {k: v for k, v in PLAN.items() if not only or k in only.split(",")}
rows, boots, allpred = [], [], pd.DataFrame({"mutant": d.mutant, "pos": d.pos, "domain": d.domain, "y": d.y})
stage1 = []
for sname, P in PLAN.items():
    folds = splits[P["col"]].values; preds = {m: np.full(len(d), np.nan) for m in P["models"]}
    for k in np.unique(folds):
        tr, te = C.train_mask(d, folds, k, P["buf"]); cache = {}
        ysh = yv.copy(); idx = np.where(tr)[0]; ysh[idx] = np.random.RandomState(k).permutation(yv[idx])
        for m in P["models"]:
            preds[m][te] = run_fold(m, tr, te, cache, ysh)
        if "s" in cache:
            tp = np.unique(d.pos.values[te]); pm = d[te].groupby("pos").y.mean().loc[tp].values
            stage1.append(dict(split=sname, fold=int(k), n_pos=len(tp), site_spearman=C.sp(cache["s"][tp - 1], pm)))
        print(f"[{time.time()-t0:6.0f}s] {sname} fold {k} done (train n={tr.sum()}, test n={te.sum()})", flush=True)
    ref = "M1_sub" if "M1_sub" in preds else "local_mean"
    bt = C.cluster_boot(d, preds, B=args.boot, ref=ref)
    for m, p in preds.items():
        r = dict(split=sname, model=m, **C.metrics(d, p, folds), **bt[m], ref=ref); rows.append(r)
        allpred[f"{sname}__{m}"] = p
    pd.DataFrame([r for r in rows if r["split"] == sname]).to_csv(f"{args.out}/metrics_{sname}.csv", index=False)
    allpred[[c for c in allpred.columns if c.startswith(sname + "__")]].to_csv(f"{args.out}/pred_{sname}.csv", index=False)
    pd.DataFrame([r for r in stage1 if r["split"] == sname]).to_csv(f"{args.out}/site_level_spearman_{sname}.csv", index=False)
res = pd.DataFrame(rows)
print(res[["split", "model", "spearman", "ci_lo", "ci_hi", "fold_mean", "auroc", "within_pos", "top10_enrich"]].round(3).to_string())
print("total seconds", round(time.time() - t0))
