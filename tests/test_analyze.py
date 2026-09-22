import numpy as np

import analyze


def test_group_mean_ignores_nan():
    values = np.array([1.0, 3.0, np.nan, 5.0])
    groups = np.array([0, 0, 1, 2])
    means = analyze.group_mean(values, groups, 4)
    assert np.allclose(means[[0, 2]], [2.0, 5.0])
    assert np.isnan(means[[1, 3]]).all()


def test_shuffle_within_passes_keeps_each_pass_a_permutation():
    rng = np.random.default_rng(0)
    passes = np.repeat(np.arange(5), 7)
    cases = np.concatenate([rng.permutation(7) for _ in range(5)])
    shuffled = analyze.shuffle_within_passes(cases, passes, rng)
    assert not np.array_equal(shuffled, cases)
    for p in range(5):
        assert sorted(shuffled[passes == p]) == list(range(7))


def test_normalize_removes_the_training_phase_trend():
    rng = np.random.default_rng(0)
    steps = np.arange(2000)
    delta = (np.exp(-steps / 300) * (1 + 0.3 * rng.standard_normal(2000)))[
        :, None, None
    ]
    z = analyze.normalize(delta, 100)
    early, late = z[100:500, 0, 0], z[1500:1900, 0, 0]
    assert abs(np.nanmedian(early)) < 0.2 and abs(np.nanmedian(late)) < 0.2
    assert 0.7 < np.nanstd(early) / np.nanstd(late) < 1.4


def test_positive_share():
    share = analyze.positive_share(np.array([-1.0, 3.0, 1.0, np.nan]))
    assert np.allclose(share, [0.75, 1.0, 1.0, 1.0])
