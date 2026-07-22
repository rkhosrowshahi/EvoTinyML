"""Unit tests for task+L1 hypervolume with fixed ref [1, 1]."""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV

from evotinyml.fitness import L1_HV_REF, l1_val_front, mean_abs_weights
from evotinyml.moo.display import _hv_unit_front


def test_l1_hv_ref_is_unit():
    assert L1_HV_REF.shape == (2,)
    assert np.allclose(L1_HV_REF, [1.0, 1.0])


def test_l1_val_front_f1():
    X = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float64)
    F = l1_val_front("f1_l1", X, macro_f1=np.array([0.8, 0.4]))
    assert F.shape == (2, 2)
    assert np.isclose(F[0, 0], 0.2)
    assert np.isclose(F[1, 0], 0.6)
    assert np.isclose(F[0, 1], mean_abs_weights(X[0]))
    assert np.isclose(F[1, 1], mean_abs_weights(X[1]))


def test_l1_val_front_ce():
    X = np.zeros((2, 4), dtype=np.float64)
    F = l1_val_front("cross_entropy_l1", X, mean_ce=np.array([0.3, 0.9]))
    assert np.allclose(F[:, 0], [0.3, 0.9])
    assert np.allclose(F[:, 1], 0.0)


def test_hv_unit_front_matches_pymoo_ref_ones():
    # Single point (0.2, 0.3) vs ref (1, 1) → HV = 0.8 * 0.7 = 0.56
    F = np.array([[0.2, 0.3]], dtype=np.float64)
    hv = _hv_unit_front(
        F,
        n_obj=2,
        hv_exact=HV(ref_point=L1_HV_REF, norm_ref_point=False),
        hv_samples=1000,
        hv_seed=0,
    )
    assert np.isclose(hv, 0.56)


def test_hv_excludes_points_outside_ref():
    # Point with CE >= 1 contributes nothing alone.
    F = np.array([[1.2, 0.1], [0.2, 0.3]], dtype=np.float64)
    hv = _hv_unit_front(
        F,
        n_obj=2,
        hv_exact=HV(ref_point=L1_HV_REF, norm_ref_point=False),
        hv_samples=1000,
        hv_seed=0,
    )
    assert np.isclose(hv, 0.56)


def test_soft_f1_l1_uses_hard_f1_on_val_front():
    X = np.array([[0.1, -0.1]], dtype=np.float64)
    F = l1_val_front("soft_f1_l1", X, macro_f1=np.array([0.5]))
    assert np.isclose(F[0, 0], 0.5)
