# FlexNet: nucleotide-level structural flexibility in human precursor microRNAs

Code and data accompanying the manuscript. **FlexNet** is a dilated residual
encoder-decoder (1-D convolution + bidirectional GRU, multi-task heads) that
predicts per-nucleotide structural flexibility of human pri-miRNA hairpins from
sequence, predicted-structure, thermodynamic and ensemble features. It is trained
on DMS-MaPseq and SHAPE-MaP reactivity and evaluated cross-chemistry (train on
one chemistry, predict the other).

## Repository layout

```
project/
├── scripts/                  all analysis code (Python)
│   ├── model.py              FlexNet architecture, dataset loader, losses
│   ├── evalkit.py            metrics, flexible-region calling, resampling
│   ├── s01_*.py … s13_*.py   numbered pipeline stages (run in this order)
│   ├── data/                 raw sources, references, intermediate files
│   │   ├── raw/              GEO supplementary files (GSE254361, GSE262014)
│   │   ├── mirbase/          miRBase hairpin.fa, mature.fa, hsa.gff3
│   │   ├── processed/        mapped/annotated tables (stage s01–s02)
│   │   └── features/         model-ready .npz arrays (stage s03–s04)
│   ├── dataset/              frozen, published processed dataset (see below)
│   └── results/              trained checkpoints + all numerical outputs
├── manuscript/               manuscript text and final figure files
├── requirements.txt
└── README.md                 (this file)
```

> **Note on paths.** The scripts resolve all paths relative to the *parent*
> directory of `scripts/` — i.e. they expect `data/`, `dataset/` and `results/`
> to sit **next to** `scripts/`. In this snapshot those folders are nested
> *inside* `scripts/`; move them one level up (or adjust the `ROOT` lines at the
> top of each script) before re-running the pipeline.

## Requirements

* Python ≥ 3.10 (developed on 3.13)
* `pip install -r requirements.txt` — numpy, scipy, pandas, scikit-learn,
  matplotlib, torch, biopython, viennarna, statsmodels
* A CUDA GPU is optional: training runs on CPU but is slower.

## Data sources

| Source | GEO accession | Used for |
|---|---|---|
| SHAPE-MaP pri-miRNA atlas | GSE254361 | SHAPE reactivity + probe structures |
| DMS-MaPseq pri-miRNA atlas | GSE262014 | DMS reactivity |
| miRBase | — | hairpin/mature sequences, annotations |

Raw downloads live in `scripts/data/raw/` and `scripts/data/mirbase/`; stages
s01–s04 turn them into the model-ready arrays.

## Pipeline

Run the numbered scripts in order from the repository root
(`python scripts/s01_map.py`, …). Each stage reads its inputs from and writes
its outputs to the fixed folders above.

| Stage | Script | Purpose | Key outputs |
|---|---|---|---|
| 1 | `s01_map.py` | map miRBase hairpins + raw reactivity to human hairpins | `data/processed/*` |
| 1b | `s01b_shape_align.py` | SHAPE alignment audit / QC | `shape_alignment_qc.tsv` |
| 2 | `s02_annotate.py` | conservation, mature-arm and loop annotations | `conservation.tsv`, `arm_annotations.tsv` |
| 2b | `s02b_shape_constructs.py` | SHAPE construct sequences (cached genome fetches) | `shape_constructs.tsv` |
| 3 | `s03_features.py` | compute the 66 per-nucleotide feature channels | `data/features/*_features.npz` |
| 4 | `s04_dataset.py` | attach labels, family-aware train/val/test splits | `*_modelling.npz`, `family_splits.json` |
| 5 | `s05_train.py` | train FlexNet, protocols A/B × 3 seeds | `model_{A,B}_s{0-2}.pt`, `metrics_main.json`, `predictions.npz`, `call_calibration.json`, `training_history.json` |
| 5b–e | `s05b_diagnose.py`, `s05c_tune.py`, `s05d_aux.py`, `s05e_flexnet_ablate.py` | optional diagnostics, tuning, auxiliary-target and architecture ablations | `tuning.json`, `auxiliary_comparisons.json`, `flexnet_ablations.json` |
| 6 | `s06_baselines.py` | baseline models (GBM, logistic, etc.) | `baselines.json` |
| 7 | `s07_regions.py` | call flexible regions, enrichment vs experiment | `region_enrichment.json` |
| 8 | `s08_robustness.py` | label-permutation and robustness checks | `robustness.json` |
| 9 | `s09_figures.py`, `s09b_architecture.py` | main figures + architecture schematic | `results/figures/fig2-6*`, `figures/fig_architecture*` |
| 10 | `s10_learning_curves.py` | learning curves / data-efficiency | `learning_curves.json` |
| 11 | `s11_model_scores.py` | aggregate scores, model-comparison tables | `model_scores.npz`, `model_comparison.json`, `tables/` |
| 12 | `s12_figures_ext.py` | extended figures (ROC/PR, calibration, curves) | `results/figures/fig7-11*` |
| 13 | `s13_submission.py` | build the submission package | `submission/data/`, `submission/supplementary/` |

### Minimal reproduction (processed data already included)

The model-ready arrays and trained checkpoints are shipped, so analyses can be
re-run without retraining:

```bash
python scripts/s06_baselines.py     # baselines
python scripts/s07_regions.py       # region calls + enrichment
python scripts/s08_robustness.py    # robustness (uses the shipped .pt checkpoints)
python scripts/s09_figures.py       # main figures
python scripts/s11_model_scores.py  # comparison tables
```

To retrain FlexNet from the processed arrays, run `python scripts/s05_train.py`
(needs the modelling `.npz` files and, ideally, a GPU).

## Processed dataset (`scripts/dataset/`)

`dataset/` is the **frozen, published training dataset** — the final processed
data every model in the manuscript was trained and evaluated on (features,
labels, family-aware splits, hairpin sequences/structures, scaler and
metadata). See [`scripts/dataset/README.md`](scripts/dataset/README.md) for the
full file-by-file description, split statistics and label definitions.

It is produced by `s13_submission.py` from `data/features/` + `results/`; the
four core `.npz` files are therefore byte-identical copies of the working files
in `data/features/` — `dataset/` is kept as the stable download for readers,
`data/features/` is what the scripts read and write.

## Results (`scripts/results/`)

| File(s) | Contents |
|---|---|
| `model_{A,B}_s{0,1,2}.pt` | FlexNet weights, protocols A/B, three seeds |
| `metrics_main.json`, `training_history.json` | main metrics and per-epoch training traces |
| `predictions.npz`, `call_calibration.json` | per-nucleotide scores and the frozen region-caller thresholds |
| `baselines.json`, `model_comparison.json`, `model_scores.npz` | baseline and comparative results |
| `region_enrichment.json`, `robustness.json`, `learning_curves.json` | region, robustness and learning-curve analyses |
| `scaler_{A,B}.npy.npz` | feature-channel standardisation parameters |

## Citation

Please cite the accompanying manuscript when using this code or dataset.
