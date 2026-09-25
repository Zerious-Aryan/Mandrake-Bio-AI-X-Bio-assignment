"""Merge per-split outputs -> results/metrics_all.csv, predictions.csv; make results/fig1.png"""
import glob, os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src"); import cas9lib as C
d, seq = C.load("data"); s = pd.read_csv("split_assignments.csv")
M = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/metrics_[a-z]*.csv")) if "all" not in f], ignore_index=True)
M.to_csv("results/metrics_all.csv", index=False)
P = s.merge(d[["mutant", "y", "DMS_score_bin"]], on="mutant")
for f in sorted(glob.glob("results/pred_*.csv")): P = pd.concat([P, pd.read_csv(f)], axis=1)
P.to_csv("predictions.csv", index=False)
pd.concat([pd.read_csv(f) for f in glob.glob("results/site_level_spearman_*.csv") if os.path.getsize(f) > 5]).to_csv("results/site_level_spearman.csv", index=False)
# ---- figure
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.05, 1.25]})
g = lambda sp, m: M[(M.split == sp) & (M.model == m)].iloc[0]
splits = ["random", "posgroup", "block", "domain"]; lab = ["random\nvariant", "unseen\nposition", "unseen\nblock", "unseen\ndomain"]
models = [("blosum", "BLOSUM62", "#999999"), ("local_mean", "nearest-position mean", "#c9a227"), ("M1_sub", "M1 substitution GBM", "#2a6f97"),
          ("M2_sub_ctx", "M2 +window context", "#c1440e"), ("E_twostage", "E two-stage (site x subst.)", "#6a4c93")]
w = 0.16
for i, (m, name, col) in enumerate(models):
    x, y, lo, hi = [], [], [], []
    for j, sp in enumerate(splits):
        M_ = M[(M.split == sp) & (M.model == m)]
        if len(M_): r = M_.iloc[0]; x.append(j + (i - 2) * w); y.append(r.spearman); lo.append(r.spearman - r.ci_lo); hi.append(r.ci_hi - r.spearman)
    ax[0].bar(x, y, w, yerr=[lo, hi], color=col, label=name, error_kw=dict(lw=.6))
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab); ax[0].set_ylabel("pooled Spearman (95% position-cluster CI)")
ax[0].axhline(0.162, ls="--", c="k", lw=.7); ax[0].text(3.45, 0.166, "LOO position-mean\noracle 0.16", ha="right", fontsize=6.5)
ax[0].legend(fontsize=6, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.28), ncol=2); ax[0].set_title("(a) Protocol changes the ranking of models", fontsize=8, pad=26)
abl = [("blosum", "BLOSUM62 only"), ("A2_reach", "A2 codon-reachability only"), ("A1_artifact", "A1 wt/mut identity + reach."), ("M1_props", "M1 phys-chem only"),
       ("M1_sub", "M1 full substitution"), ("M2_sub_ctx", "M2 + window context"), ("M2_shufctx", "M2, context shuffled"), ("site_only", "site model (GBM)"),
       ("site_ridge", "site model (ridge)*"), ("E_twostage", "E two-stage"), ("E_ridge", "E two-stage (ridge site)*"), ("E_shufctx", "E, context shuffled"),
       ("M2_shuffleY", "M2, labels shuffled"), ("E_shuffleY", "E, labels shuffled")]
for k, (m, name) in enumerate(abl[::-1]):
    r = g("block", m); ax[1].barh(k, r.spearman, xerr=[[r.spearman - r.ci_lo], [r.ci_hi - r.spearman]], color="#2a6f97" if "shuf" not in m else "#bbbbbb", error_kw=dict(lw=.6), height=.7)
ax[1].set_yticks(range(len(abl))); ax[1].set_yticklabels([a[1] for a in abl][::-1]); ax[1].axvline(0, c="k", lw=.5)
ax[1].set_xlabel("pooled Spearman, unseen-block split"); ax[1].set_title("(b) Ablations & controls (* post hoc)", fontsize=8, pad=26)
plt.tight_layout(); plt.savefig("results/fig1.png", dpi=220)
cols = ["split", "model", "spearman", "ci_lo", "ci_hi", "delta_mean", "delta_lo", "delta_hi", "ref", "fold_mean", "auroc", "within_pos", "top10_enrich"]
print(M[M.split.isin(["block", "domain"])][cols].round(3).to_string())
print(pd.read_csv("results/site_level_spearman.csv").groupby("split").site_spearman.agg(["mean", "std"]).round(3))
