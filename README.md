# dice_variants

nnU-Net at batch size 1, with a full validation pass after every optimizer step, to
measure how much each training sample improves each validation patient, and how that
relates to structure size, per loss (Dice, CE, Dice+CE).

## Setup (once)

1. Put `COMET_KEY=<key>` in `.env` at the repo root (git-ignored).
2. Paths, venv and modules are in `env.sh`. Raw data defaults to
   `/nobackup/proj/disk/naiss2025-5-504/personal/attilas/{ACDC-2D-CL,wmh}`.
3. From the repo: `sbatch prepare.sh [acdc_dir] [wmh_dir]`. This writes
   `Dataset027_ACDC` and `Dataset028_WMH` (see `convert.py`), then plans and
   preprocesses `2d` and `3d_fullres`.
   - ACDC-2D-CL (PNG slices of the 100 ACDC training patients, ED + ES) is stacked
     back into one volume per frame with the spacing from `spacing_3d.pkl`. The
     provided split is kept: train (70 patients) / val (10) is fold 0, the only fold;
     test (20) goes to `imagesTs` / `labelsTs`.
   - wmh (MICCAI 2017): the 60 training subjects get 5 folds balanced over sites
     (48 / 12), the 110 test subjects go to `imagesTs` / `labelsTs`. Label 2 (other
     pathology) becomes nnU-Net's ignore label. T1 and label take the FLAIR's header,
     which they match up to float noise.

## Training

```bash
sbatch run.sh <ACDC|WMH> <dice|ce|dice_ce> [seed] [train.py options]

sbatch run.sh WMH dice
sbatch run.sh ACDC dice_ce 1 --momentum 0 --lr 1 --config 2d
```

`python train.py --help` lists the options (`--steps` 10000, `--lr` 1e-2, `--momentum` 0.99,
`--fold` 0, `--config` 3d_fullres, `--plans` nnUNetPlans). ACDC has only fold 0;
for a higher fold nnU-Net would silently draw a random case-level split, putting
frames of one patient in both train and val.

Everything else is nnU-Net's: plans, architecture, augmentation, foreground
oversampling (33%), deep supervision, SGD with Nesterov momentum, poly lr from 1e-2
(decayed per step), gradient clipping at 12. Differences: batch size 1; each case once
per pass over the training set (nnU-Net draws with replacement); fp32 without
autocast/GradScaler (which silently skips overflowing steps; convolutions still use
TF32 on Ampere/Hopper, PyTorch's default); a deterministic
cross-entropy (nnU-Net's has no deterministic CUDA kernel); no torch.compile.

## What is logged

Every step, to `<run>/steps.csv` (the complete record; `analyze.py` reads it):

- the training case, whether its patch was forced to contain foreground, and the
  voxels of every label in the augmented patch the loss saw (`vox_background`,
  `vox_WMH`, `vox_ignore`, ...)
- loss, lr, gradient norm before clipping, and the Dice gradient terms (below)
- hard and soft Dice of every validation case and label
- `importance`: mean change of validation Dice caused by the step (`importance_soft`:
  same on soft Dice, which also moves when no voxel flips)

Comet (project `dice-variants`) plots at most ~1000 points per metric, so it gets
1000 points per run: means over each window of `steps / 1000` steps (loss, lr,
gradient norm, importance, `coin/*` gradient terms, numbers of validation cases that
improved / worsened / did not change), the Dice of every validation case and label
at that step (`dice/<case>/<label>`) and its change over the window
(`delta/<case>/<label>`). In addition, every **jump** is logged under `jump/*` with
the full detail of that step (`case_index` into `cases.csv`, patch voxels, gradient
terms, and each validation case's change): a step whose soft importance is more
than 4 robust standard deviations (MAD) from the median of the previous pass. That
is typically a few percent of the steps.

Validation is nnU-Net's sliding window (step 0.5, Gaussian weighting, no mirroring)
on the preprocessed validation cases, which stay on the GPU; Dice is computed in the
preprocessed space, ignoring WMH label 2. Step 0 is the untrained network.

At the end: `case_importance.csv` (mean importance over the steps that drew each
training case), `cases.csv` (voxels per label of every whole case),
`checkpoint_final.pth`, and the hashes `steps_sha256` / `weights_sha256`.

Runs live in
`$nnUNet_results/<dataset>/StepTrainer__<plans>__<config>/fold_<fold>/<run>_<jobid>/`.

### Dice gradient terms

For each foreground label, every step also logs the two values of the soft-Dice
gradient on the training patch (arXiv:2304.04319): with s the softmax output,
I = Σ s·y and U = Σ y + Σ s, `alpha = 2/(U+ε)` and `beta = (2I+ε)/(U+ε)²`. The loss
gradient w.r.t. s is `grad_fg = w·(beta − alpha)` on every foreground voxel and
`grad_bg = w·beta` on every background voxel (w = deep-supervision weight × Dice
weight / number of labels; ignored voxels get 0). Also logged: `soft_intersection`
(I) and `soft_sum_pred` (Σ s). A test checks these against autograd through
nnU-Net's Dice loss.

## Analysis

```bash
python analyze.py <run dir> [<run dir> ...] --out figs
```

Writes five figures and `case_scores.csv` per run, plus `figs/summary.csv`.
`python analyze.py --help` explains the method; in short:

1. `1_trajectory`: validation Dice, and the raw per-step change, which shrinks by
   orders of magnitude during training. That's why importance is normalized per pass.
2. `2_drivers`: do a few training cases drive the validation Dice? Ranked case
   importance against a null where the drawn case is shuffled within each pass;
   split-half reliability (even vs odd passes; if ρ is not significant the per-case
   ranking is noise); concentration of the positive importance.
3. `3_influence`: training case × validation case matrix, both sorted by foreground
   size, with cells that differ from the null at |z| > 3.
4. `4_coin_terms`: alpha, beta and the summed foreground / background gradient
   against structure size, with the most and least important cases highlighted.
5. `5_size`: importance vs structure size in the patch, vs alpha, and the parameter
   gradient norm vs size.

Per-sample attribution is only clean with `--momentum 0`: with momentum 0.99 each
update carries about the last 100 samples, and the split-half test shows this.
Runs from before the gradient-term logging get all figures except 4; rerun them
(same command) to add it, and the rerun reproduces the old trajectory bitwise.

## Reproducibility

The seed alone determines the network init, the case order, the oversampling
decisions, the patch locations and all augmentation (each step's patch is a function
of (seed, step), independent of the data-loader workers), and all GPU kernels run in
deterministic mode. The same command twice gives identical `steps_sha256` and
`weights_sha256`, on the same GPU model and the same software versions (logged to
Comet). The same seed with a different `--loss` starts from the same weights and sees
the same augmented patches, so losses can be compared pairwise.

## Tests

```bash
ruff check . && ruff format --check . && pytest tests
```
