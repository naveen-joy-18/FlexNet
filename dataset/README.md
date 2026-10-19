# Processed training data

This folder contains the **final processed datasets used to train and evaluate every
model in the manuscript** - the reviewed, reproducible dataset built by the
scripts pipeline (stages s01-s04) from the raw GEO and miRBase sources.

## Files

* `dms_modelling.npz`, `shape_modelling.npz` - flat per-nucleotide arrays used by
  `model.RNADataset` (scripts/s04_dataset.py, scripts/s05_train.py):
  * `X` (n_positions x 66) - the 66 per-nucleotide features (Table S4), one row per
    position of every hairpin, concatenated in file order;
  * `y` - raw probe reactivity (DMS normalised 0-1; SHAPE normalised); NaN where
    unmeasured;
  * `y_bin` - flexible/rigid label: reactivity >= 0.5 (DMS) or
    >= 1.0 (SHAPE);
  * `y_rank` - within-dataset rank-normalised reactivity in [0,1] (auxiliary target);
  * `valid` - 1 where a reactivity measurement exists;
  * `ptr` - RNA boundaries into the flat arrays (ptr[i], ptr[i+1]);
  * `lengths` - per-RNA lengths;
  * `mirbase_id`, `exp_id`, `family`, `split` - per-RNA identifiers, miRNA family
    stem and family-aware train/validation/test assignment;
  * `feature_names` - the 66 feature labels;
  * `struct` - per-RNA JSON: MFE/centroid dot-bracket and per-position loop codes.
* `dms_features.npz`, `shape_features.npz` - the same features *before* labels are
  attached (scripts/s03_features.py); identical `X` layout to the modelling file.
* `scaler.npy.npz` - channel mean (`mu`) and standard deviation (`sd`) fitted on the
  **DMS training split only**; `X` is Z-standardised as `(X - mu) / sd` before
  modelling. `feature_scaler.tsv` lists them per feature.
* `feature_names.tsv` - index, name and family (sequence/structure/thermo/ensemble)
  of the 66 features.
* `rna_table.tsv` - one row per hairpin: dataset, index, identifiers, family, length,
  split and the number of measured / flexible nucleotides.
* `hairpin_sequences.fa` - the exact hairpin sequence of each modelled RNA
  (FASTA header: `dataset|exp_id|mirbase_id|family|split|length`).
* `structures.tsv` - the MFE and centroid dot-bracket string of each hairpin.
* `dataset_parameters.json` - label thresholds, the flexible-region caller
  (T, L_min) chosen on validation, the split summary and the family-overlap counts.

## Splits

Whole miRNA families are assigned atomically, so no family straddles a boundary
(family splits: `rna_table.tsv` / `dataset_parameters.json`).

| dataset | RNAs | positions | flexible rate | train | val | test |
|---|---|---|---|---|---|---|
| DMS-MaPseq | 1061 | 84530 | 0.2121 | 743 | 156 | 162 |
| SHAPE-MaP | 462 | 39955 | 0.0968 | 324 | 70 | 68 |

Protocol A trains on DMS and evaluates on the DMS test set (own) and the SHAPE test
set (other); Protocol B is the mirror image. The flexible-region caller is:

| protocol | threshold T | min region length | selected on |
|---|---|---|---|
| A | 0.3436 | 2 | DMS-MaPseq validation |
| B | 0.2751 | 2 | SHAPE-MaP validation |

Source experiments: published MA-Pseq atlases (see manuscript Methods 2.1); scripts
s01-s04 turn the raw GEO/MiRBase sources into these arrays. The per-nucleotide
labels used experimentally were DMS reactivity >= 0.5 and SHAPE
reactivity >= 1.0, matching the measured flexible populations.
