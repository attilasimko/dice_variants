import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import train


@pytest.mark.parametrize("ignore_index", [None, 2])
def test_deterministic_ce_matches_torch(ignore_index):
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 4, 5, 6, dtype=torch.float64)
    target = torch.randint(0, 3, (2, 4, 5, 6))

    ours = train.DeterministicCE(ignore_index)(logits, target)

    reference = torch.nn.functional.cross_entropy(
        logits, target, ignore_index=-100 if ignore_index is None else ignore_index
    )
    assert torch.allclose(ours, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deterministic_ce_runs_in_deterministic_mode():
    logits = torch.randn(1, 3, 4, 8, 8, device="cuda", requires_grad=True)
    target = torch.randint(0, 3, (1, 4, 8, 8), device="cuda")
    torch.use_deterministic_algorithms(True)
    try:
        grads = []
        for _ in range(2):
            logits.grad = None
            train.DeterministicCE(2)(logits, target).backward()
            grads.append(logits.grad.clone())
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.equal(grads[0], grads[1])


@pytest.mark.parametrize(
    "shape, patch_size",
    [
        ((20, 50, 37), [16, 32, 32]),
        ((16, 32, 32), [16, 32, 32]),
        ((5, 70, 40), [64, 32]),
    ],
)
def test_tile_slicers_match_nnunet_predictor(shape, patch_size):
    predictor = SimpleNamespace(
        configuration_manager=SimpleNamespace(patch_size=patch_size),
        tile_step_size=train.TILE_STEP,
        verbose=False,
    )
    expected = nnUNetPredictor._internal_get_sliding_window_slicers(predictor, shape)
    assert train.tile_slicers(shape, patch_size) == expected


def test_make_schedule_visits_every_case_once_per_pass():
    keys = [f"case{i}" for i in range(7)]

    schedule = train.make_schedule(keys, 30, 0.33, seed=3)

    assert schedule == train.make_schedule(keys, 30, 0.33, seed=3)
    assert schedule != train.make_schedule(keys, 30, 0.33, seed=4)
    assert len(schedule) == 30
    for start in range(0, 28, 7):
        assert sorted(k for k, _ in schedule[start : start + 7]) == keys
    force_fg = train.make_schedule(keys, 10_000, 0.33, seed=3)
    assert np.mean([f for _, f in force_fg]) == pytest.approx(0.33, abs=0.02)


def test_label_counts_counts_outside_crop_as_background():
    seg = np.array([[-1, 0, 1], [2, 2, 1]])
    assert train.label_counts(seg, 4) == [2, 2, 2, 0]


def test_load_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEP", "old")
    monkeypatch.delenv("COMET_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# comment\nCOMET_KEY = 'abc123'\nKEEP=new\n\n")

    train.load_env(env)

    assert os.environ["COMET_KEY"] == "abc123"
    assert os.environ["KEEP"] == "old"


class LabelMapNetwork(torch.nn.Module):
    """Predicts the label map stored in input channel 0, with logit 5 for the label."""

    def forward(self, x):
        labels = x[:, 0].long()
        return 5.0 * torch.nn.functional.one_hot(labels, 3).movedim(-1, 1).float()


class FakeDataset:
    def __init__(self, cases):
        self.cases = cases

    def load_case(self, key):
        return *self.cases[key], None, {}


def test_validate_matches_numpy_dice():
    rng = np.random.default_rng(0)
    shape, ignore = (6, 20, 17), 3
    gt = rng.integers(-1, 4, shape)  # -1: outside the nonzero crop
    pred = rng.integers(0, 3, shape)
    cases = {"c": (pred[None].astype(np.float32), gt[None].astype(np.int8))}
    trainer = SimpleNamespace(
        dataset_class=lambda folder, keys: FakeDataset(cases),
        preprocessed_dataset_folder="",
        configuration_manager=SimpleNamespace(patch_size=[4, 8, 8]),
        label_manager=SimpleNamespace(
            foreground_labels=[1, 2],
            num_segmentation_heads=3,
            has_ignore_label=True,
            ignore_label=ignore,
        ),
        network=LabelMapNetwork(),
        set_deep_supervision_enabled=lambda enabled: None,
    )
    gaussian = train.compute_gaussian((4, 8, 8), 1.0 / 8, 10, torch.float32, "cpu")

    val_cases = train.load_val_cases(trainer, ["c"], gaussian)
    hard, soft = train.validate(trainer, val_cases, gaussian)

    valid = gt != ignore
    p_label = np.exp(5) / (np.exp(5) + 2)
    for k, c in enumerate([1, 2]):
        g, p = (gt == c) & valid, (pred == c) & valid
        assert hard[0, k] == pytest.approx(2 * (g & p).sum() / (g.sum() + p.sum()))
        prob = np.where(pred == c, p_label, 1 / (np.exp(5) + 2)) * valid
        assert soft[0, k] == pytest.approx(
            2 * (prob * g).sum() / (prob.sum() + g.sum())
        )


def test_coin_terms_are_the_two_values_of_the_dice_gradient():
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

    torch.manual_seed(0)
    ignore, weight = 3, 0.6
    logits = torch.randn(1, 3, 4, 5, 6, dtype=torch.float64)
    target = torch.randint(0, 4, (1, 1, 4, 5, 6))
    dice = MemoryEfficientSoftDiceLoss(None, batch_dice=False, do_bg=False, smooth=1e-5)
    trainer = SimpleNamespace(
        loss=SimpleNamespace(
            loss=SimpleNamespace(dc=dice, weight_dice=1), weight_factors=[weight]
        ),
        label_manager=SimpleNamespace(
            foreground_labels=[1, 2], has_ignore_label=True, ignore_label=ignore
        ),
    )
    # as in DC_and_CE_loss: ignored voxels masked out, their target set to 0
    probs = torch.softmax(logits, 1).requires_grad_()
    mask = target != ignore
    (weight * dice(probs, torch.where(mask, target, 0), loss_mask=mask)).backward()

    terms = train.coin_terms(trainer, logits, target)

    for j, c in enumerate([1, 2]):
        grad = probs.grad[0, c]
        fg, bg = (target[0, 0] == c), mask[0, 0] & (target[0, 0] != c)
        assert torch.allclose(grad[fg], torch.tensor(terms["grad_fg"][j]))
        assert torch.allclose(grad[bg], torch.tensor(terms["grad_bg"][j]))
        assert torch.all(grad[~mask[0, 0]] == 0)


def test_robust_z():
    from collections import deque

    recent = deque(np.linspace(-1, 1, 21))
    assert np.isnan(train.robust_z(5.0, deque([0.0] * 5)))
    assert train.robust_z(0.0, recent) == 0
    assert train.robust_z(10.0, recent) > train.JUMP_Z
    assert np.isnan(train.robust_z(1.0, deque([0.0] * 30)))
