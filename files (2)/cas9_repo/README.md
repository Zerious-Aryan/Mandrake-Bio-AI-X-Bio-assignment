# Predicting Cas9 gene-editing activity — evaluation of a PLM+diffusion proposal

Evaluates the proposal (pretrained protein-sequence model + diffusion structure trunk,
trunk replaced by an activity head) against `CAS9_STRP1_Spencer_2017_positive`
(ProteinGym), and implements a runnable predictor + ablation + stress test as a
substitute for the untestable parts of the proposal.

**Sandbox constraint (drives several design choices below):** this container has no
GPU, 1 CPU, 3 GB RAM, and network egress only to package indices (PyPI/npm/GitHub) —
`huggingface.co`, `dl.fbaipublicfiles.com` and `files.rcsb.org` all return
`403 host_not_allowed`. No pretrained PLM checkpoint, no MSA, and no PDB structure
(5F9R) could be downloaded. **All models here are hand-crafted sequence/physicochemical
features + gradient-boosted trees / ridge — there is no PLM and no structure model in
this code.** See `REPORT.md`/`REPORT.pdf` §4 for what a real implementation would use
and why that would need to run outside this sandbox.

## Contents
```
data/                       CAS9_STRP1_Spencer_2017_positive.csv, CAS9_STRP1_WT.fasta (copied from data pack)
src/cas9lib.py              feature engineering, splits, models, metrics, bootstrap
diagnostics.py              dataset limitation diagnostics -> results/diagnostics.json
run_experiments.py          trains/evaluates all models over all CV protocols
make_assets.py              merges per-split CSVs, builds results/fig1.png
split_assignments.csv       fold id per variant, all four protocols (written by run_experiments.py)
predictions.csv             held-out prediction for every (variant, model, protocol) - the deliverable
results/metrics_all.csv     Spearman/AUROC/within-position-Spearman/top-decile-enrichment + 95% CI per model/protocol
results/site_level_spearman_*.csv   stage-1 (position-tolerance) model accuracy per fold
results/diagnostics.json    numbers backing the limitations section
results/fig1.png            main figure
REPORT.md / REPORT.pdf      3-page report
TIME_LOG.md                 time/compute/AI-tool record
```

## Reproduce
```bash
pip install pandas numpy scipy scikit-learn biopython matplotlib --break-system-packages
python run_experiments.py --data data --out results --boot 300   # ~2-3 min total (or ONLY_SPLITS=block,domain,random,posgroup to run one at a time)
python diagnostics.py --data data --out results
python make_assets.py
```
Fully deterministic (all seeds fixed); reruns reproduce `results/metrics_all.csv` exactly.

## What the four CV protocols are (see `cas9lib.make_splits`)
- **random** — 10-fold over variants (rows). Positive control: gives every model access
  to same-position labels at train time.
- **unseen position** — 10-fold over positions, no buffer. Adjacent positions can still
  land in different folds.
- **unseen block (+15 buffer)** — sequence split into 10 contiguous ~137-aa blocks;
  training variants within 15 residues of a held-out block are also dropped. This is
  the protocol used for the headline numbers.
- **unseen domain** — leave-one-domain-out (RuvC/REC/HNH/PI, approximate boundaries
  from Nishimasu 2014 topology), same 15-residue buffer.

## Models (see `run_experiments.run_fold`)
`blosum` (BLOSUM62 score, no fitting) · `local_mean`/`posmean`/`mutaa_mean` (non-learned
lookup baselines) · `A1_artifact`/`A2_reach` (controls that use only wt/mut identity and
single-nucleotide codon-reachability — isolate the library-construction artefact) ·
`M1_sub`/`M1_props` (substitution-level GBM: one-hot wt/mut + BLOSUM62 +
physicochemical deltas) · `M2_sub_ctx` (M1 + a ±2/5/10/20-residue windowed
physicochemical-context featurisation of the WT sequence) · `site_only`/`site_ridge`
(stage-1 position-tolerance regressor alone) · `E_twostage`/`E_ridge` (**preferred
extension**: two-stage model — position-tolerance prediction from local WT context,
interacted with the substitution-level model) · `*_shufctx` (context columns
permuted across positions — tests whether context features carry positional
information or just party tricks) · `*_shuffleY` (training labels permuted —
sanity floor).

`site_ridge`/`E_ridge` were added **after** seeing that the GBM stage-1 model gave a
negative held-out Spearman under the buffered block split (see report §3, disclosed as
post hoc).
