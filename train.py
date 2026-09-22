"""nnU-Net training at batch size 1 with a full validation pass after every step.

Per optimizer step, logged to Comet and <run>/steps.csv: the training case drawn,
the voxel count of every label in its augmented patch, loss, lr, gradient norm, and
the hard and soft Dice of every validation case and label. A step's importance is
the mean change in validation Dice it caused; a training case's importance is the
mean over the steps that drew it (<run>/case_importance.csv).

Everything random (network init, case order, foreground oversampling, patch
location, augmentation) derives from --seed, and every GPU kernel is deterministic:
a rerun is bitwise identical on the same GPU model and software stack (compare
steps_sha256 / weights_sha256 in Comet). Runs with the same seed and different
--loss start from the same weights and see the same sequence of augmented patches.
"""

import argparse
import csv
import hashlib
import os
import random
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from importlib.metadata import version
from itertools import product
from pathlib import Path

import comet_ml  # must be imported before torch
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from nnunetv2.inference.sliding_window_prediction import (
    compute_gaussian,
    compute_steps_for_sliding_window,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import nn

from convert import DATASETS

# (weight_ce, weight_dice) of nnU-Net's DC_and_CE_loss
LOSSES = {"dice": (0, 1), "ce": (1, 0), "dice_ce": (1, 1)}
TILE_STEP = 0.5  # nnU-Net's default sliding-window overlap
# validation tiles per forward pass: as many as fit in this many voxels (at least 1),
# a function of the plans only, so the numerics do not depend on the GPU
VAL_BATCH_VOXELS = 2**21
N_WORKERS = 4
# Comet plots at most ~1000 points per metric: regular metrics are window means over
# steps // COMET_POINTS steps; steps.csv has every step
COMET_POINTS = 1000
# a step is a jump (logged to Comet under jump/) when its soft importance is this many
# robust standard deviations from the median of the previous pass
JUMP_Z = 4.0
COIN_TERMS = (
    "alpha",
    "beta",
    "grad_fg",
    "grad_bg",
    "soft_intersection",
    "soft_sum_pred",
)


def load_env(path: Path) -> None:
    """KEY=VALUE lines of a .env file into os.environ (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def enable_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class DeterministicCE(nn.Module):
    """nn.CrossEntropyLoss(ignore_index=...) without nll_loss, which has no
    deterministic CUDA kernel."""

    def __init__(self, ignore_index: int | None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        if self.ignore_index is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            valid = target != self.ignore_index
        index = torch.where(valid, target, 0)[:, None]
        nll = -torch.log_softmax(logits, 1).gather(1, index)[:, 0]
        return (nll * valid).sum() / valid.sum()


class StepTrainer(nnUNetTrainer):
    """nnU-Net's network, loss, augmentation and poly-lr SGD. The training loop is
    train() below; loss_weights and momentum are set before initialize()."""

    loss_weights = LOSSES["dice_ce"]
    momentum = 0.99

    def _do_i_compile(self) -> bool:
        return False

    def _build_loss(self) -> nn.Module:
        loss = super()._build_loss()  # DeepSupervisionWrapper(DC_and_CE_loss)
        loss.loss.weight_ce, loss.loss.weight_dice = self.loss_weights
        loss.loss.ce = DeterministicCE(self.label_manager.ignore_label)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.momentum > 0,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)


class StepLoader(nnUNetDataLoader):
    """nnU-Net's patch sampling and augmentation, for the case and oversampling
    decision set by the caller instead of drawn at random."""

    key: str = ""
    force_fg: bool = False

    def get_indices(self) -> list[str]:
        return [self.key]

    def forced_fg(self, sample_idx: int) -> bool:
        return self.force_fg


class Steps(torch.utils.data.Dataset):
    """Item i is the augmented patch of step i + 1: a function of (seed, i) only, so
    it does not depend on the number of workers or on which worker computes it."""

    def __init__(self, loader: StepLoader, schedule: list[tuple[str, bool]], seed: int):
        self.loader = loader
        self.schedule = schedule
        self.seed = seed

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, i: int) -> dict:
        self.loader.key, self.loader.force_fg = self.schedule[i]
        seed_everything(
            int(np.random.SeedSequence([self.seed, i]).generate_state(1)[0])
        )
        return self.loader.generate_train_batch()


def make_loader(trainer: StepTrainer, keys: list[str]) -> StepLoader:
    """The training half of nnUNetTrainer.get_dataloaders, at batch size 1."""
    cm, lm = trainer.configuration_manager, trainer.label_manager
    rotation, dummy_2d, initial_patch_size, mirror_axes = (
        trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    )
    transforms = trainer.get_training_transforms(
        cm.patch_size,
        rotation,
        trainer._get_deep_supervision_scales(),
        mirror_axes,
        dummy_2d,
        use_mask_for_norm=cm.use_mask_for_norm,
        foreground_labels=lm.foreground_labels,
        ignore_label=lm.ignore_label,
    )
    # nnU-Net's blur picks FFT or direct convolution by timing both; they differ
    # in the last bits, so the pick would make augmentation run-dependent
    for t in transforms.transforms:
        if isinstance(getattr(t, "transform", None), GaussianBlurTransform):
            t.transform.benchmark = False
    loader = StepLoader(
        trainer.dataset_class(trainer.preprocessed_dataset_folder, list(keys)),
        1,
        initial_patch_size,
        cm.patch_size,
        lm,
        trainer.oversample_foreground_percent,
        transforms=transforms,
    )
    loader.get_do_oversample = loader.forced_fg  # nnU-Net sets it per instance
    return loader


def make_schedule(
    keys: list[str], steps: int, oversample: float, seed: int
) -> list[tuple[str, bool]]:
    """(case, force foreground) per step. Every case once per pass over the training
    set, in a new order each pass; nnU-Net's foreground oversampling rate."""
    rng = np.random.default_rng(seed)
    passes = -(-steps // len(keys))
    order = np.concatenate([rng.permutation(len(keys)) for _ in range(passes)])[:steps]
    force_fg = rng.random(steps) < oversample
    return [(keys[i], bool(f)) for i, f in zip(order, force_fg)]


def val_tile_batch(patch_size: list[int]) -> int:
    return max(1, VAL_BATCH_VOXELS // int(np.prod(patch_size)))


def tile_slicers(shape: tuple[int, ...], patch_size: list[int]) -> list[tuple]:
    """nnU-Net's sliding-window tiles (nnUNetPredictor._internal_get_sliding_window_slicers).
    A 2D patch size tiles every slice of the volume."""
    if len(patch_size) == len(shape):
        steps = compute_steps_for_sliding_window(shape, patch_size, TILE_STEP)
        return [
            (slice(None), *(slice(s, s + p) for s, p in zip(start, patch_size)))
            for start in product(*steps)
        ]
    steps = compute_steps_for_sliding_window(shape[1:], patch_size, TILE_STEP)
    return [
        (slice(None), z, *(slice(s, s + p) for s, p in zip(start, patch_size)))
        for z in range(shape[0])
        for start in product(*steps)
    ]


@dataclass
class ValCase:
    key: str
    image: torch.Tensor  # (C, *padded shape)
    gt: torch.Tensor  # (n foreground labels, *shape) bool, False where ignored
    valid: torch.Tensor  # (*shape) bool, False on the ignore label
    tiles: list[tuple]
    weight: torch.Tensor  # (*padded shape), summed gaussian of the covering tiles
    unpad: tuple  # padded -> original spatial shape


def load_val_cases(
    trainer: StepTrainer, keys: list[str], gaussian: torch.Tensor
) -> list[ValCase]:
    """Preprocessed validation cases, kept on the GPU for the whole run."""
    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, list(keys))
    patch_size = trainer.configuration_manager.patch_size
    lm = trainer.label_manager
    device = gaussian.device
    cases = []
    for key in keys:
        data, seg, _, _ = dataset.load_case(key)
        image, unpad = pad_nd_image(
            torch.from_numpy(data[:]).float(),
            patch_size,
            "constant",
            {"value": 0},
            True,
        )
        tiles = tile_slicers(tuple(image.shape[1:]), patch_size)
        weight = torch.zeros(image.shape[1:], device=device)
        for tile in tiles:
            weight[tile[1:]] += gaussian
        # -1 marks voxels outside nnU-Net's nonzero crop: background
        seg = torch.from_numpy(seg[0].astype(np.int16)).to(device).clamp_min(0)
        if lm.has_ignore_label:
            valid = seg != lm.ignore_label
        else:
            valid = torch.ones_like(seg, dtype=torch.bool)
        gt = torch.stack([seg == c for c in lm.foreground_labels]) & valid
        cases.append(
            ValCase(key, image.to(device), gt, valid, tiles, weight, unpad[1:])
        )
    return cases


@torch.no_grad()
def validate(
    trainer: StepTrainer, cases: list[ValCase], gaussian: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Hard and soft Dice, (n cases, n foreground labels), of fp32 sliding-window
    predictions without mirroring, in the preprocessed (resampled) space."""
    labels = trainer.label_manager.foreground_labels
    n_heads = trainer.label_manager.num_segmentation_heads
    batch = val_tile_batch(trainer.configuration_manager.patch_size)
    network = trainer.network
    network.eval()
    trainer.set_deep_supervision_enabled(False)
    hard = np.zeros((len(cases), len(labels)))
    soft = np.zeros_like(hard)
    for i, case in enumerate(cases):
        logits = torch.zeros((n_heads, *case.image.shape[1:]), device=gaussian.device)
        for j in range(0, len(case.tiles), batch):
            tiles = case.tiles[j : j + batch]
            out = network(torch.stack([case.image[t] for t in tiles]))
            for t, o in zip(tiles, out):
                logits[t] += o * gaussian
        probs = torch.softmax((logits / case.weight)[(slice(None), *case.unpad)], 0)
        argmax = probs.argmax(0)
        pred = torch.stack([argmax == c for c in labels]) & case.valid
        probs = probs[labels] * case.valid
        dims = tuple(range(1, pred.ndim))
        n_gt = case.gt.sum(dims, dtype=torch.float64)
        intersection = (pred & case.gt).sum(dims, dtype=torch.float64)
        hard[i] = (2 * intersection / (n_gt + pred.sum(dims))).cpu().numpy()
        soft_intersection = (probs * case.gt).sum(dims, dtype=torch.float64)
        soft_total = n_gt + probs.sum(dims, dtype=torch.float64)
        soft[i] = (2 * soft_intersection / soft_total).cpu().numpy()
    network.train()
    trainer.set_deep_supervision_enabled(True)
    return hard, soft


def train_step(
    trainer: StepTrainer, batch: dict
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """nnUNetTrainer.train_step in fp32: without autocast's GradScaler, which silently
    skips steps whose gradients overflow. Returns loss, pre-clipping grad norm, and
    the full-resolution logits (before the update) and target."""
    data = batch["data"].to(trainer.device, non_blocking=True)
    target = [t.to(trainer.device, non_blocking=True) for t in batch["target"]]
    trainer.optimizer.zero_grad(set_to_none=True)
    output = trainer.network(data)
    loss = trainer.loss(output, target)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(trainer.network.parameters(), 12)
    trainer.optimizer.step()
    return loss.item(), grad_norm.item(), output[0].detach(), target[0]


def coin_terms(
    trainer: StepTrainer, logits: torch.Tensor, target: torch.Tensor
) -> dict[str, np.ndarray]:
    """The two values of the soft Dice gradient (arXiv:2304.04319) on one patch, per
    foreground label, for nnU-Net's Dice (smooth eps, ignore label masked out).

    With s the softmax output, y the one-hot target, I = sum(s * y) and
    U = sum(y) + sum(s), d(Dice)/ds = alpha * y - beta with alpha = 2 / (U + eps)
    and beta = (2 I + eps) / (U + eps)^2. The training loss (mean of -Dice over the
    labels, times the Dice and deep supervision weights) therefore has gradient
    grad_fg = scale * (beta - alpha) on every foreground voxel and
    grad_bg = scale * beta on every background voxel of the label.
    """
    eps = trainer.loss.loss.dc.smooth
    labels = trainer.label_manager.foreground_labels
    seg = target[0, 0]
    valid = torch.ones_like(seg, dtype=torch.bool)
    if trainer.label_manager.has_ignore_label:
        valid = seg != trainer.label_manager.ignore_label
    y = torch.stack([seg == c for c in labels]) & valid
    s = torch.softmax(logits[0], 0)[labels] * valid
    dims = tuple(range(1, y.ndim))
    intersection = (s * y).sum(dims, dtype=torch.float64)
    sum_pred = s.sum(dims, dtype=torch.float64)
    union = y.sum(dims) + sum_pred
    alpha = 2 / (union + eps)
    beta = (2 * intersection + eps) / (union + eps) ** 2
    scale = trainer.loss.weight_factors[0] * trainer.loss.loss.weight_dice / len(labels)
    terms = {
        "alpha": alpha,
        "beta": beta,
        "grad_fg": scale * (beta - alpha),
        "grad_bg": scale * beta,
        "soft_intersection": intersection,
        "soft_sum_pred": sum_pred,
    }
    return {k: v.cpu().numpy() for k, v in terms.items()}


def label_counts(seg: np.ndarray | torch.Tensor, n: int) -> list[int]:
    """Voxels per label value 0..n-1; -1 (outside the nonzero crop) counts as background."""
    seg = torch.as_tensor(np.asarray(seg)).flatten().long().clamp_min(0)
    return torch.bincount(seg, minlength=n).tolist()


def robust_z(value: float, recent: deque) -> float:
    """value against the median and MAD of recent values; NaN until there are 20."""
    if len(recent) < 20:
        return np.nan
    median = np.median(recent)
    mad = 1.4826 * np.median(np.abs(np.asarray(recent) - median))
    return (value - median) / mad if mad > 0 else np.nan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights_sha256(network: nn.Module) -> str:
    h = hashlib.sha256()
    for tensor in network.state_dict().values():
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()


def write_cases(
    path: Path, trainer: StepTrainer, tr_keys: list[str], val_keys: list[str]
) -> None:
    """Voxels per label of every whole (preprocessed) case; index is case_index."""
    names = {v: k for k, v in trainer.dataset_json["labels"].items()}
    n_values = max(names) + 1
    folder = trainer.preprocessed_dataset_folder
    dataset = trainer.dataset_class(folder, tr_keys + val_keys)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        vox = [f"vox_{names[v]}" for v in range(n_values)]
        writer.writerow(["case", "split", "index", *vox])
        for split, keys in (("train", tr_keys), ("val", val_keys)):
            for i, key in enumerate(keys):
                counts = label_counts(dataset.load_case(key)[1][:], n_values)
                writer.writerow([key, split, i, *counts])


def train(
    trainer: StepTrainer,
    args: argparse.Namespace,
    experiment: comet_ml.CometExperiment,
    run_dir: Path,
) -> None:
    lm = trainer.label_manager
    names = {v: k for k, v in trainer.dataset_json["labels"].items()}
    n_values = max(names) + 1
    fg_names = [names[c] for c in lm.foreground_labels]
    tr_keys, val_keys = trainer.do_split()
    tr_keys, val_keys = sorted(tr_keys), sorted(val_keys)
    write_cases(run_dir / "cases.csv", trainer, tr_keys, val_keys)
    experiment.log_asset(str(run_dir / "cases.csv"))

    loader = make_loader(trainer, tr_keys)

    schedule = make_schedule(
        tr_keys, args.steps, trainer.oversample_foreground_percent, args.seed
    )
    batches = torch.utils.data.DataLoader(
        Steps(loader, schedule, args.seed),
        batch_size=None,
        num_workers=N_WORKERS,
        pin_memory=True,
        prefetch_factor=4,
    )
    gaussian = compute_gaussian(
        tuple(trainer.configuration_manager.patch_size),
        sigma_scale=1.0 / 8,
        value_scaling_factor=10,
        dtype=torch.float32,
        device=trainer.device,
    )
    val_cases = load_val_cases(trainer, val_keys, gaussian)
    experiment.log_parameters(
        {
            "n_train": len(tr_keys),
            "n_val": len(val_keys),
            "patch_size": trainer.configuration_manager.patch_size,
            "val_tiles": sum(len(c.tiles) for c in val_cases),
        }
    )

    pairs = [(key, name) for key in val_keys for name in fg_names]
    header = [
        *("step", "case", "force_fg", "loss", "lr", "grad_norm"),
        *(f"vox_{names[v]}" for v in range(n_values)),
        *(f"{t}_{n}" for t in COIN_TERMS for n in fg_names),
        *("importance", "importance_soft"),
        *(f"dice_{k}_{n}" for k, n in pairs),
        *(f"soft_{k}_{n}" for k, n in pairs),
    ]

    def val_metrics(hard: np.ndarray) -> dict:
        metrics = {"val_dice": np.nanmean(hard)}
        for j, name in enumerate(fg_names):
            metrics[f"val_dice_{name}"] = np.nanmean(hard[:, j])
        for (key, name), value in zip(pairs, hard.ravel()):
            metrics[f"dice/{key}/{name}"] = value
        return metrics

    with open(run_dir / "steps.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        hard, soft = validate(trainer, val_cases, gaussian)
        n_blank = len(header) - 1 - 2 * len(pairs)  # nothing trained at step 0
        writer.writerow([0, *[""] * n_blank, *hard.ravel(), *soft.ravel()])
        experiment.log_metrics(val_metrics(hard), step=0)
        trainer.print_to_log_file(f"step 0: val dice {np.nanmean(hard):.4f}")

        importance = defaultdict(list)
        every = max(1, args.steps // COMET_POINTS)
        window = defaultdict(list)  # scalars since the last Comet point
        window_delta = np.zeros_like(hard)
        # importance_soft over the last pass (at least the 20 robust_z needs)
        recent = deque(maxlen=max(len(tr_keys), 20))
        for step, batch in enumerate(batches, start=1):
            key = batch["keys"][0]
            force_fg = schedule[step - 1][1]
            vox = label_counts(batch["target"][0], n_values)
            trainer.lr_scheduler.step(step - 1)
            lr = trainer.optimizer.param_groups[0]["lr"]
            start = time.perf_counter()
            loss, grad_norm, logits, target = train_step(trainer, batch)
            coin = coin_terms(trainer, logits, target)
            del logits, target
            trained = time.perf_counter()
            new_hard, new_soft = validate(trainer, val_cases, gaussian)
            validated = time.perf_counter()

            delta = new_hard - hard
            case_delta = np.nanmean(delta, 1)
            step_importance = np.nanmean(delta)
            step_importance_soft = np.nanmean(new_soft - soft)
            importance[key].append((step_importance, step_importance_soft))
            hard, soft = new_hard, new_soft

            writer.writerow(
                [step, key, int(force_fg), loss, lr, grad_norm, *vox]
                + [coin[t][j] for t in COIN_TERMS for j in range(len(fg_names))]
                + [step_importance, step_importance_soft, *hard.ravel(), *soft.ravel()]
            )
            f.flush()

            coin_metrics = {
                f"coin/{t}_{n}": coin[t][j]
                for t in COIN_TERMS
                for j, n in enumerate(fg_names)
            }
            scalars = {
                "loss": loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "force_fg": int(force_fg),
                **coin_metrics,
                "importance": step_importance,
                "importance_soft": step_importance_soft,
                "val_improved": int((case_delta > 0).sum()),
                "val_worsened": int((case_delta < 0).sum()),
                "val_unchanged": int((case_delta == 0).sum()),
                "sec_train": trained - start,
                "sec_val": validated - trained,
            }
            for name, value in scalars.items():
                window[name].append(value)
            window_delta += np.nan_to_num(delta)
            if step % every == 0 or step == args.steps:
                experiment.log_metrics(
                    {name: np.mean(values) for name, values in window.items()}
                    | val_metrics(hard)
                    | {
                        f"delta/{k}/{n}": d
                        for (k, n), d in zip(pairs, window_delta.ravel())
                    },
                    step=step,
                )
                window.clear()
                window_delta[:] = 0

            z = robust_z(step_importance_soft, recent)
            recent.append(step_importance_soft)
            if abs(z) > JUMP_Z:
                experiment.log_metrics(
                    {
                        "jump/z": z,
                        "jump/importance": step_importance,
                        "jump/importance_soft": step_importance_soft,
                        "jump/case_index": tr_keys.index(key),
                        "jump/force_fg": int(force_fg),
                        **{f"jump/vox_{names[v]}": vox[v] for v in range(n_values)},
                        **{f"jump/{k}": v for k, v in coin_metrics.items()},
                        **{
                            f"jump/delta/{k}/{n}": d
                            for (k, n), d in zip(pairs, delta.ravel())
                        },
                    },
                    step=step,
                )
            trainer.print_to_log_file(
                f"step {step}/{args.steps} {key} loss {loss:.4f} val dice "
                f"{np.nanmean(hard):.4f} importance {step_importance:+.2e} "
                f"({validated - start:.1f} s)"
                + (f" jump z={z:+.1f}" if abs(z) > JUMP_Z else "")
            )

    with open(run_dir / "case_importance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "n_draws", "importance", "importance_soft"])
        for key in tr_keys:
            draws = np.array(importance[key]).reshape(-1, 2)
            writer.writerow([key, len(draws), *draws.mean(0)])
    experiment.log_asset(str(run_dir / "steps.csv"))
    experiment.log_table(str(run_dir / "case_importance.csv"))

    trainer.save_checkpoint(str(run_dir / "checkpoint_final.pth"))
    hashes = {
        "steps_sha256": sha256(run_dir / "steps.csv"),
        "weights_sha256": weights_sha256(trainer.network),
    }
    experiment.log_others(hashes)
    trainer.print_to_log_file(hashes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--loss", choices=LOSSES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000, help="optimizer steps")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.99,
        help="SGD (Nesterov) momentum, nnU-Net: 0.99. 0 makes each update depend "
        "on the current sample only.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="initial (poly-decayed) lr, nnU-Net: 1e-2. Momentum m scales the "
        "effective step by 1/(1-m), so momentum 0 needs a larger lr.",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--plans", default="nnUNetPlans")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env(Path(__file__).with_name(".env"))
    comet_key = os.environ["COMET_KEY"]
    enable_determinism()

    preprocessed = Path(os.environ["nnUNet_preprocessed"]) / DATASETS[args.dataset]
    plans = load_json(str(preprocessed / f"{args.plans}.json"))
    plans["continue_training"] = False  # nnunetv2 >= 2.8 pops it in __init__
    trainer = StepTrainer(
        plans, args.config, args.fold, load_json(str(preprocessed / "dataset.json"))
    )
    run_name = (
        f"{args.dataset}_{args.loss}_lr{args.lr:g}_mom{args.momentum:g}_seed{args.seed}"
    )
    job = os.environ.get("SLURM_JOB_ID") or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(trainer.output_folder) / f"{run_name}_{job}"
    run_dir.mkdir()
    shutil.move(trainer.log_file, run_dir / "log.txt")
    trainer.output_folder = str(run_dir)
    trainer.log_file = str(run_dir / "log.txt")

    trainer.num_epochs = args.steps  # the poly lr decays per step
    trainer.loss_weights = LOSSES[args.loss]
    trainer.momentum = args.momentum
    trainer.initial_lr = args.lr
    seed_everything(args.seed)  # the network init is the first draw
    trainer.initialize()
    save_json(trainer.plans_manager.plans, str(run_dir / "plans.json"), sort_keys=False)
    save_json(trainer.dataset_json, str(run_dir / "dataset.json"), sort_keys=False)

    experiment = comet_ml.start(
        api_key=comet_key,
        project="dice-variants",
        experiment_config=comet_ml.ExperimentConfig(
            name=run_name,
            tags=[args.dataset, args.loss],
            auto_metric_logging=False,
            auto_param_logging=False,
        ),
    )
    experiment.log_parameters(
        vars(args)
        | {
            "run_dir": str(run_dir),
            "batch_size": 1,
            "initial_lr": trainer.initial_lr,
            "weight_decay": trainer.weight_decay,
            "oversample_foreground": trainer.oversample_foreground_percent,
            "val_tile_step": TILE_STEP,
            "val_tile_batch": val_tile_batch(trainer.configuration_manager.patch_size),
            "nnunetv2": version("nnunetv2"),
            "torch": torch.__version__,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(),
        }
    )
    train(trainer, args, experiment, run_dir)
    experiment.end()


if __name__ == "__main__":
    main()
