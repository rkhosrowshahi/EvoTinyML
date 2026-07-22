"""Unit tests for UPGrad dual-cone aggregation."""

from __future__ import annotations

import numpy as np

from evotinyml.moo.mo_es import (
    _upgrad_aggregate,
    aggregate_mo_gradients,
    mgda_weights,
    upgrad_aggregate,
)


def test_upgrad_no_conflict_equals_mean():
    # Acute angle: both grads in each other's dual half-space.
    G = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    assert float(np.dot(G[0], G[1])) > 0
    d = upgrad_aggregate(G)
    mean = G.mean(axis=0)
    assert np.allclose(d, mean, atol=1e-5)


def test_upgrad_direction_in_dual_cone_under_conflict():
    # Obtuse angle: classic conflict.
    G = np.array([[1.0, 0.0], [-0.5, 1.0]], dtype=np.float64)
    assert float(np.dot(G[0], G[1])) < 0
    d, w = _upgrad_aggregate(G)
    # Non-conflicting: g_c · d >= 0 for all c.
    aligns = G @ d
    assert np.all(aligns >= -1e-6), aligns
    assert np.linalg.norm(d) > 1e-8
    assert w.shape == (2,)


def test_upgrad_linear_under_scaling():
    G = np.array([[2.0, 0.0], [-1.0, 2.0]], dtype=np.float64)
    d1 = upgrad_aggregate(G)
    d2 = upgrad_aggregate(3.0 * G)
    assert np.allclose(d2, 3.0 * d1, atol=1e-4)


def test_aggregate_mo_gradients_mgda_vs_upgrad():
    G = np.array([[1.0, 0.2, -0.1], [-0.8, 1.0, 0.3]], dtype=np.float64)
    d_m, w_m, _ = aggregate_mo_gradients(G, aggregator="mgda", normalize="none")
    d_u, w_u, _ = aggregate_mo_gradients(G, aggregator="upgrad", normalize="none")
    d_m = np.asarray(d_m)
    d_u = np.asarray(d_u)
    # Both should be non-conflicting common-descent directions.
    assert np.all(G @ d_m >= -1e-5)
    assert np.all(G @ d_u >= -1e-5)
    # MGDA weights on simplex; UPGrad dual coeffs need not sum to 1.
    assert np.isclose(float(np.sum(w_m)), 1.0, atol=1e-5)
    assert w_u.shape == (2,)


def test_mgda_still_works():
    G = np.eye(2)
    w = mgda_weights(G)
    assert np.allclose(w, [0.5, 0.5], atol=1e-5)
