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
sbatch run.sh ACDC dice_ce 1 --steps 5000 --momentum 0 --config 2d
```

`python train.py --help` lists the options (`--steps` 2000, `--momentum` 0.99,
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

Per step, to Comet (project `dice-variants`) and `<run>/steps.csv`:

- the training case (`case_index` into `cases.csv`), whether its patch was forced to
  contain foreground, and the voxels of every label in the augmented patch the loss
  saw (`vox_background`, `vox_WMH`, `vox_ignore`, ...)
- loss, lr, gradient norm before clipping
- hard Dice of every validation case and label (`dice/<case>/<label>`), its change
  from the previous step (`delta/<case>/<label>`), the number of validation cases
  that improved / worsened / did not change; `steps.csv` also has soft Dice
- `importance`: mean change of validation Dice caused by the step (`importance_soft`:
  same on soft Dice, which also moves when no voxel flips)

Validation is nnU-Net's sliding window (step 0.5, Gaussian weighting, no mirroring)
on the preprocessed validation cases, which stay on the GPU; Dice is computed in the
preprocessed space, ignoring WMH label 2. Step 0 is the untrained network.

At the end: `case_importance.csv` (mean importance over the steps that drew each
training case), `cases.csv` (voxels per label of every whole case),
`checkpoint_final.pth`, and the hashes `steps_sha256` / `weights_sha256`.

Runs live in
`$nnUNet_results/<dataset>/StepTrainer__<plans>__<config>/fold_<fold>/<run>_<jobid>/`.

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
