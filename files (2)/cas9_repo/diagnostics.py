"""Dataset diagnostics that back the 'limitations' section. Usage: python diagnostics.py --data data --out results"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import cas9lib as C

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results"); a = ap.parse_args()
d, seq = C.load(a.data); C.init(seq); out = {}
y = d.y.values
out["n_variants"] = len(d); out["n_positions"] = int(d.pos.nunique())
out["coverage_of_19xL"] = len(d) / (19 * C.L)
out["variants_per_position_min_max"] = [int(d.groupby("pos").size().min()), int(d.groupby("pos").size().max())]
out["score_quantiles_1_25_50_75_99"] = [float(np.percentile(y, q)) for q in (1, 25, 50, 75, 99)]
out["score_min_max"] = [float(y.min()), float(y.max())]
out["kurtosis_excess"] = float(pd.Series(y).kurt())
out["binarization_cutoff_is_median"] = float(d.DMS_score[d.DMS_score_bin == 1].min()), float(d.DMS_score[d.DMS_score_bin == 0].max())
# library artefact: single-nucleotide reachability
R = np.array([C.reach(w, m)[0] for w, m in zip(d.wt, d.mut)])
allp = [(w, m) for w in C.CODONS if w != "*" for m in C.CODONS if m != "*" and m != w]
out["frac_measured_single_nt_reachable"] = float((R > 0).mean())
out["frac_all_aa_pairs_single_nt_reachable"] = float(np.mean([C.reach(w, m)[0] > 0 for w, m in allp]))
# how much of the 19xL space is even single-nt reachable given (unknown-codon) wt aa
out["possible_single_nt_subs_upper_bound"] = int(sum(sum(C.reach(w, m)[0] > 0 for m in C.AA if m != w) for w in seq))
# informative missingness
n = d.groupby("pos").size(); pm = d.groupby("pos").y.mean()
out["spearman_nvariants_vs_position_mean"] = float(spearmanr(n, pm.loc[n.index])[0])
# variance decomposition and position-level ceilings
g = d.groupby("pos").y; s, c = g.transform("sum"), g.transform("count")
loo = (s - d.y) / (c - 1)
out["R2_position_mean_in_sample"] = float(1 - ((y - g.transform("mean")) ** 2).sum() / ((y - y.mean()) ** 2).sum())
out["spearman_LOO_position_mean"] = float(spearmanr(loo, y)[0])
pmean = pm.reindex(range(1, C.L + 1))
for w in (3, 15, 30):
    nb = [np.nanmean([pmean.get(p + k, np.nan) for k in range(-w, w + 1) if k != 0]) for p in d.pos]
    out[f"spearman_neighbour_window_pm_{w}"] = float(spearmanr(nb, y)[0])
# domain means
out["domain_mean_score"] = d.groupby("domain").y.mean().round(3).to_dict()
out["domain_n"] = d.groupby("domain").size().to_dict()
out["mut_aa_mean_score_extremes"] = d.groupby("mut").y.mean().round(3).sort_values().iloc[[0, 1, 2, -3, -2, -1]].to_dict()
os.makedirs(a.out, exist_ok=True); json.dump(out, open(f"{a.out}/diagnostics.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
