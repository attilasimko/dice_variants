"""Figures for training runs written by train.py.

    python analyze.py <run dir> [<run dir> ...] [--out figs] [--metric soft|hard]

Attribution: the change of every validation case's Dice at step t is credited to the
training case drawn at step t. With --momentum 0 that is the exact effect of one
update on that sample; with momentum m the update also carries the previous
~1/(1-m) samples, which blurs the attribution (split-half reliability shows how much).

Normalized importance: early steps change the Dice far more than late ones, so each
(validation case, label) series of changes is detrended by its moving median and
scaled by its moving std, both over one pass through the training set. Every case is
drawn once per pass; its importance is the mean over passes. The null model shuffles
which case was drawn among the steps of each pass (same training phase, same draws).

Per run, into <out>/<run>/:
  1_trajectory.png  validation Dice and raw step importance over training
  2_drivers.png     do a few training cases drive the validation Dice?
  3_influence.png   training case x validation case influence
  4_coin_terms.png  alpha, beta and the Dice gradient vs structure size (new runs)
  5_size.png        importance vs structure size and alpha
  case_scores.csv   per training case: importance (normalized), contribution (raw sum
                    of its changes, which over all steps add up to the Dice gain), size
and <out>/summary.csv with one row per run.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

INK, INK_2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb",
)  # fmt: skip
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # one per foreground label, fixed order
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("div", [NEGATIVE, "#f0efec", POSITIVE])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])
N_TOP = 5  # highlighted cases, at most a quarter of the training set


@dataclass
class Run:
    name: str
    steps: pd.DataFrame  # one row per optimizer step
    dice: np.ndarray  # (steps + 1, val cases, labels), row 0 before training
    delta: np.ndarray  # (steps, val cases, labels)
    cases: pd.DataFrame  # cases.csv, indexed by case
    train_keys: list[str]
    val_keys: list[str]
    labels: list[str]  # foreground label names
    case_index: np.ndarray  # (steps,) index into train_keys
    passes: np.ndarray  # (steps,) pass through the training set


def load_run(path: Path, metric: str) -> Run:
    table = pd.read_csv(path / "steps.csv")
    cases = pd.read_csv(path / "cases.csv", index_col="case")
    train_keys = cases[cases.split == "train"].sort_values("index").index.tolist()
    val_keys = cases[cases.split == "val"].sort_values("index").index.tolist()
    labels = [
        c.removeprefix("vox_")
        for c in table.columns
        if c.startswith("vox_") and c not in ("vox_background", "vox_ignore")
    ]
    prefix = {"soft": "soft", "hard": "dice"}[metric]
    dice = np.stack(
        [table[[f"{prefix}_{k}_{n}" for n in labels]].to_numpy() for k in val_keys], 1
    )
    steps = table.iloc[1:].reset_index(drop=True)
    index = {k: i for i, k in enumerate(train_keys)}
    return Run(
        name=path.name,
        steps=steps,
        dice=dice,
        delta=np.diff(dice, axis=0),
        cases=cases,
        train_keys=train_keys,
        val_keys=val_keys,
        labels=labels,
        case_index=steps.case.map(index).to_numpy(),
        passes=(steps.step.to_numpy() - 1) // len(train_keys),
    )


def normalize(delta: np.ndarray, window: int) -> np.ndarray:
    """(delta - moving median) / moving std along the steps, per column."""
    flat = pd.DataFrame(delta.reshape(len(delta), -1))
    kwargs = {"window": window, "center": True, "min_periods": window // 2}
    median = flat.rolling(**kwargs).median()
    std = flat.rolling(**kwargs).std()
    z = (flat - median) / std.where(std > 0)
    return z.to_numpy().reshape(delta.shape)


def group_mean(values: np.ndarray, groups: np.ndarray, n: int) -> np.ndarray:
    """Mean of values per group index 0..n-1, ignoring NaN; values may be 2D."""
    values = values.reshape(len(values), -1)
    out = np.full((n, values.shape[1]), np.nan)
    for j in range(values.shape[1]):
        ok = ~np.isnan(values[:, j])
        count = np.bincount(groups[ok], minlength=n)
        total = np.bincount(groups[ok], values[ok, j], minlength=n)
        out[count > 0, j] = total[count > 0] / count[count > 0]
    return out.squeeze(1) if out.shape[1] == 1 else out


def shuffle_within_passes(
    case_index: np.ndarray, passes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    out = case_index.copy()
    for p in np.unique(passes):
        idx = np.flatnonzero(passes == p)
        out[idx] = rng.permutation(out[idx])
    return out


def fg_voxels(run: Run, split: str) -> pd.Series:
    rows = run.cases[run.cases.split == split]
    return rows[[f"vox_{n}" for n in run.labels]].sum(1)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "font.family": "sans-serif",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(run: Run, out: Path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    steps = np.arange(len(run.dice))
    for j, name in enumerate(run.labels):
        top.plot(steps, np.nanmean(run.dice[:, :, j], 1), color=SERIES[j], label=name)
    top.set_ylabel("validation Dice (mean over cases)")
    top.set_title("Validation Dice over training")
    if len(run.labels) > 1:
        top.legend(loc="lower right")
    raw = np.nanmean(run.delta, (1, 2))
    window = len(run.train_keys)
    bottom.scatter(run.steps.step, raw, s=2, color=AXIS, label="per step")
    smooth = pd.Series(raw).rolling(window, center=True, min_periods=1).mean()
    bottom.plot(run.steps.step, smooth, color=INK_2, label="mean over one pass")
    bottom.set_yscale("symlog", linthresh=np.nanpercentile(np.abs(raw), 50) or 1e-6)
    bottom.axhline(0, color=AXIS, linewidth=1)
    bottom.set_xlabel("optimizer step")
    bottom.set_ylabel("change of validation Dice")
    bottom.set_title("Raw step importance (why it is normalized per pass)")
    bottom.legend(loc="upper right")
    save(fig, out / "1_trajectory.png")


def case_scores(run: Run, z_step: np.ndarray, keep: np.ndarray) -> pd.DataFrame:
    n = len(run.train_keys)
    cases, passes = run.case_index[keep], run.passes[keep]
    score = z_step[keep]
    mean = group_mean(score, cases, n)
    sq = group_mean(score**2, cases, n)
    draws = np.bincount(cases[~np.isnan(score)], minlength=n)
    raw = np.nanmean(run.delta, (1, 2))[keep]
    frame = pd.DataFrame(
        {
            "case": run.train_keys,
            "n_passes": draws,
            "importance": mean,
            "importance_se": np.sqrt((sq - mean**2) / np.maximum(draws - 1, 1)),
            "importance_even_passes": group_mean(
                np.where(passes % 2 == 0, score, np.nan), cases, n
            ),
            "importance_odd_passes": group_mean(
                np.where(passes % 2 == 1, score, np.nan), cases, n
            ),
            "contribution": np.bincount(cases, raw, minlength=n),
            "fg_voxels": fg_voxels(run, "train").loc[run.train_keys].to_numpy(),
        }
    )
    return frame.set_index("case")


def positive_share(importance: np.ndarray) -> np.ndarray:
    """Share of all positive importance held by the k most important cases (Lorenz)."""
    positive = np.sort(np.clip(np.nan_to_num(importance), 0, None))[::-1]
    return np.cumsum(positive) / positive.sum()


def plot_drivers(
    run: Run,
    scores: pd.DataFrame,
    z_step: np.ndarray,
    keep: np.ndarray,
    n_perm: int,
    out: Path,
) -> dict:
    rng = np.random.default_rng(0)
    n = len(run.train_keys)
    null_sorted, null_rho, null_share = [], [], []
    for _ in range(n_perm):
        perm = shuffle_within_passes(run.case_index, run.passes, rng)[keep]
        mean = group_mean(z_step[keep], perm, n)
        null_sorted.append(np.sort(mean)[::-1])
        even = group_mean(
            np.where(run.passes[keep] % 2 == 0, z_step[keep], np.nan), perm, n
        )
        odd = group_mean(
            np.where(run.passes[keep] % 2 == 1, z_step[keep], np.nan), perm, n
        )
        null_rho.append(spearmanr(even, odd, nan_policy="omit").statistic)
        null_share.append(positive_share(mean))
    null_sorted, null_share = np.array(null_sorted), np.array(null_share)

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.8))
    ranked = scores.sort_values("importance", ascending=False)
    rank = np.arange(1, n + 1)
    lo, hi = np.nanpercentile(null_sorted, [2.5, 97.5], axis=0)
    a.fill_between(rank, lo, hi, color=GRID, label="null (cases shuffled), 95%")
    a.errorbar(
        rank, ranked.importance, 1.96 * ranked.importance_se, fmt="o", ms=3,
        color=POSITIVE, ecolor=AXIS, elinewidth=1, label="case mean ± 95% CI",
    )  # fmt: skip
    a.axhline(0, color=AXIS, linewidth=1)
    for i, (case, row) in enumerate(ranked.head(3).iterrows()):
        a.annotate(case, (i + 1, row.importance), xytext=(6, 0),
                   textcoords="offset points", fontsize=7, color=INK_2)  # fmt: skip
    a.set_xlabel("training cases, ranked")
    a.set_ylabel("normalized importance (mean over passes)")
    a.set_title("Ranked case importance vs chance")
    a.legend(loc="upper right")

    rho = spearmanr(
        scores.importance_even_passes, scores.importance_odd_passes, nan_policy="omit"
    ).statistic
    p_rho = (np.sum(np.array(null_rho) >= rho) + 1) / (n_perm + 1)
    b.scatter(scores.importance_even_passes, scores.importance_odd_passes, s=14,
              color=POSITIVE, edgecolors=SURFACE, linewidths=0.5)  # fmt: skip
    b.axhline(0, color=AXIS, linewidth=1)
    b.axvline(0, color=AXIS, linewidth=1)
    b.set_xlabel("importance, even passes")
    b.set_ylabel("importance, odd passes")
    b.set_title(f"Is it reproducible? Spearman ρ = {rho:.2f} (p = {p_rho:.3f})")

    share = positive_share(scores.importance.to_numpy())
    lo, hi = np.nanpercentile(null_share, [2.5, 97.5], axis=0)
    c.fill_between(rank, lo, hi, color=GRID, label="null, 95%")
    c.plot(rank, share, color=POSITIVE, label="observed")
    c.axhline(1, color=AXIS, linewidth=1)
    k50, k80 = (int(np.argmax(share >= q)) + 1 for q in (0.5, 0.8))
    for q, k in ((0.5, k50), (0.8, k80)):
        c.annotate(f"{q:.0%}: {k} of {n} cases", (k, q), xytext=(8, -12),
                   textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip
        c.plot([k], [q], "o", ms=5, color=INK_2)
    c.set_xlabel("training cases, most important first")
    c.set_ylabel("share of all positive importance")
    c.set_title("How concentrated is the positive importance?")
    c.legend(loc="lower right")
    fig.suptitle(f"{run.name}: do a few training cases drive the validation Dice?",
                 color=INK, fontsize=12)  # fmt: skip
    save(fig, out / "2_drivers.png")
    return {
        "split_half_rho": rho,
        "split_half_p": p_rho,
        "cases_for_50pct_positive_importance": k50,
        "cases_for_80pct_positive_importance": k80,
        "frac_cases_negative_importance": float(np.mean(scores.importance < 0)),
    }


def plot_influence(
    run: Run, z: np.ndarray, keep: np.ndarray, n_perm: int, out: Path
) -> dict:
    rng = np.random.default_rng(1)
    n = len(run.train_keys)
    per_val = np.nanmean(z, 2)[keep]  # (steps, val cases)
    matrix = group_mean(per_val, run.case_index[keep], n).reshape(n, -1)
    null = np.array(
        [
            group_mean(
                per_val, shuffle_within_passes(run.case_index, run.passes, rng)[keep], n
            ).reshape(n, -1)
            for _ in range(n_perm)
        ]
    )
    zscore = (matrix - null.mean(0)) / null.std(0)
    rows = np.argsort(fg_voxels(run, "train").loc[run.train_keys].to_numpy())
    cols = np.argsort(fg_voxels(run, "val").loc[run.val_keys].to_numpy())
    matrix, zscore = matrix[rows][:, cols], zscore[rows][:, cols]

    fig, ax = plt.subplots(figsize=(3 + 0.3 * len(cols), 3 + 0.05 * n))
    vmax = np.nanpercentile(np.abs(matrix), 99)
    image = ax.imshow(matrix, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto",
                      interpolation="nearest")  # fmt: skip
    ys, xs = np.nonzero(np.abs(zscore) > 3)
    ax.scatter(xs, ys, s=6, color=INK)
    ax.set_xticks(range(len(cols)), [run.val_keys[i] for i in cols], rotation=90,
                  fontsize=6)  # fmt: skip
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("validation cases, small → large foreground")
    ax.set_ylabel("training cases, small → large foreground")
    expected = 0.0027 * matrix.size
    ax.set_title(
        f"{run.name}\nmean normalized change of each validation case when a training "
        f"case is drawn\ndots: {len(ys)} cells with |z| > 3 vs the null "
        f"({expected:.1f} expected by chance)",
        fontsize=9,
    )
    fig.colorbar(image, ax=ax, shrink=0.5, label="normalized importance")
    save(fig, out / "3_influence.png")
    return {
        "influence_cells_z_gt_3": len(ys),
        "influence_cells_expected": expected,
    }


def plot_coin_terms(run: Run, scores: pd.DataFrame, out: Path) -> None:
    steps = run.steps
    ranked = scores.sort_values("importance", ascending=False).index
    n_top = max(1, min(N_TOP, len(ranked) // 4))
    top = steps.case.isin(ranked[:n_top])
    bottom = steps.case.isin(ranked[-n_top:])
    patch = steps[[c for c in steps.columns if c.startswith("vox_")]].sum(1)
    valid = patch - steps.get("vox_ignore", 0)
    fig, axes = plt.subplots(len(run.labels), 4, figsize=(17, 3.8 * len(run.labels)),
                             squeeze=False)  # fmt: skip
    for j, name in enumerate(run.labels):
        size = steps[f"vox_{name}"]
        x = size + 1
        panels = {
            "α = 2 / (|y| + Σs)": steps[f"alpha_{name}"],
            "β = 2 I / (|y| + Σs)²": steps[f"beta_{name}"],
            "Σ |∂L/∂s| over foreground voxels": size * steps[f"grad_fg_{name}"].abs(),
            "Σ ∂L/∂s over background voxels": (valid - size) * steps[f"grad_bg_{name}"],
        }
        for ax, (title, y) in zip(axes[j], panels.items()):
            ax.scatter(x, y, s=3, color=AXIS, label="all steps", rasterized=True)
            ax.scatter(x[top], y[top], s=10, color=POSITIVE,
                       label=f"{n_top} most important cases")  # fmt: skip
            ax.scatter(x[bottom], y[bottom], s=10, color=NEGATIVE,
                       label=f"{n_top} least important cases")  # fmt: skip
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.set_xlabel(f"{name} voxels in the patch + 1")
        guide = np.geomspace(max(x.min(), 2), x.max(), 50)
        axes[j, 0].plot(guide, 2 / guide, color=INK_2, linewidth=1)
        axes[j, 0].plot(guide, 1 / guide, color=MUTED, linewidth=1)
        axes[j, 0].annotate("2/|y| (empty prediction)", (guide[5], 2 / guide[5]),
                            fontsize=7, color=INK_2)  # fmt: skip
        axes[j, 0].annotate("1/|y| (Σs = |y|)", (guide[-15], 1 / guide[-15]),
                            fontsize=7, color=MUTED)  # fmt: skip
    axes[0, 0].legend(loc="lower left")
    fig.suptitle(
        f"{run.name}: the two values of the Dice gradient per patch. Foreground voxels "
        "get β − α, background voxels β (both × deep-supervision and label weights)",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, out / "4_coin_terms.png")


def binned(x: np.ndarray, y: np.ndarray, n_bins: int = 8):
    """Means ± 95% CI of y in quantile bins of positive x, plus the x == 0 bin."""
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    groups = [x == 0]
    positive = x[x > 0]
    if len(positive):
        edges = np.unique(np.quantile(positive, np.linspace(0, 1, n_bins + 1)))
        which = np.digitize(x, edges[1:-1])
        groups += [(x > 0) & (which == i) for i in range(len(edges) - 1)]
    rows = []
    for g in groups:
        if g.sum() > 1:
            rows.append(
                (np.median(x[g]), y[g].mean(), 1.96 * y[g].std() / np.sqrt(g.sum()))
            )
    return np.array(rows).T


def plot_size(run: Run, z: np.ndarray, keep: np.ndarray, out: Path) -> None:
    steps = run.steps[keep]
    has_coin = f"alpha_{run.labels[0]}" in steps
    fig, axes = plt.subplots(1, 3 if has_coin else 2, figsize=(15, 4.5))
    for j, name in enumerate(run.labels):
        label_z = np.nanmean(z[keep][:, :, j], 1)
        size = steps[f"vox_{name}"].to_numpy()
        xs, ys, ci = binned(size, label_z)
        empty = xs == 0
        xs = np.where(empty, max(size[size > 0].min() / 3, 0.5), xs)
        axes[0].errorbar(xs, ys, ci, fmt="o-", ms=5, color=SERIES[j], label=name,
                         elinewidth=1, linewidth=1.5)  # fmt: skip
        if empty.any():
            axes[0].annotate("empty", (xs[0], ys[0]), xytext=(4, 6),
                             textcoords="offset points", fontsize=7, color=INK_2)  # fmt: skip
        if has_coin:
            xs, ys, ci = binned(steps[f"alpha_{name}"].to_numpy(), label_z)
            axes[1].errorbar(xs, ys, ci, fmt="o-", ms=5, color=SERIES[j], label=name,
                             elinewidth=1, linewidth=1.5)  # fmt: skip
    fg = steps[[f"vox_{n}" for n in run.labels]].sum(1) + 1
    image = axes[-1].scatter(fg, steps.grad_norm, s=3, c=steps.step, cmap=SEQUENTIAL,
                             rasterized=True)  # fmt: skip
    fig.colorbar(image, ax=axes[-1], label="optimizer step")
    axes[0].set_title("Importance vs structure size in the patch", fontsize=10)
    axes[0].set_xlabel("label voxels in the patch (quantile bins)")
    if has_coin:
        axes[1].set_title("Importance vs α", fontsize=10)
        axes[1].set_xlabel("α of the label (quantile bins)")
    for ax in axes[:-1]:
        ax.set_xscale("log")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_ylabel("normalized importance on that label")
        if len(run.labels) > 1:
            ax.legend()
    axes[-1].set_xscale("log")
    axes[-1].set_yscale("log")
    axes[-1].set_title("Parameter gradient norm (before clipping at 12)", fontsize=10)
    axes[-1].set_xlabel("foreground voxels in the patch + 1")
    axes[-1].set_ylabel("gradient norm")
    fig.suptitle(f"{run.name}: does structure size decide how useful a sample is?",
                 color=INK, fontsize=12)  # fmt: skip
    fig.tight_layout()
    save(fig, out / "5_size.png")


def analyze(path: Path, args: argparse.Namespace) -> dict:
    run = load_run(path, args.metric)
    n_passes = run.passes.max() + 1
    if n_passes < args.skip_passes + 2:
        raise ValueError(
            f"{run.name}: {n_passes} passes over the training set; the analysis needs "
            f"at least {args.skip_passes + 2} (--skip-passes + 2)"
        )
    out = args.out / run.name
    out.mkdir(parents=True, exist_ok=True)
    z = normalize(run.delta, len(run.train_keys))
    z_step = np.nanmean(z, (1, 2))
    keep = run.passes >= args.skip_passes
    scores = case_scores(run, z_step, keep)
    scores.to_csv(out / "case_scores.csv")

    plot_trajectory(run, out)
    summary = {
        "run": run.name,
        "steps": len(run.steps),
        "passes": int(run.passes.max()) + 1,
        "val_dice_start": np.nanmean(run.dice[0]),
        "val_dice_end": np.nanmean(run.dice[-1]),
    }
    summary |= plot_drivers(run, scores, z_step, keep, args.permutations, out)
    summary |= plot_influence(run, z, keep, args.permutations, out)
    if f"alpha_{run.labels[0]}" in run.steps:
        plot_coin_terms(run, scores, out)
    plot_size(run, z, keep, out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("figs"))
    parser.add_argument(
        "--metric", choices=("soft", "hard"), default="soft",
        help="soft Dice (default) also moves when no voxel flips",
    )  # fmt: skip
    parser.add_argument(
        "--skip-passes", type=int, default=1,
        help="leave out the first passes, where the network is still near its init",
    )  # fmt: skip
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    style()
    summary = pd.DataFrame([analyze(path, args) for path in args.runs])
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
