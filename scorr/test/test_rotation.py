#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import numpy as np
import pytest

from scorr.extensions import rotation_distance


@pytest.mark.parametrize("x, y, z, output", [
    # table with sin and cos values
    # http://www2.hs-esslingen.de/~kamelzer/2011WS/Werte_sin_cos.pdf
    (1, 0, 0, [0, 0, 1]),
    (0, 1, 0, [0, 90, 1]),
    (0, 0, 1, [90, 0, 1]),
    (np.sqrt(1 / 2), np.sqrt(1 / 2), 0, [0, 45, 1]),
    (np.sqrt(1 / 2), 0, np.sqrt(1 / 2), [45, 0, 1]),
    ((np.sqrt(6) + np.sqrt(2)) / 4, 0, (np.sqrt(6) - np.sqrt(2)) / 4, [15, 0, 1]),
    (np.sqrt(3) / 2, 0, 1 / 2, [30, 0, 1]),
    (1 / 2, 0, np.sqrt(3) / 2, [60, 0, 1]),
    ((np.sqrt(6) - np.sqrt(2)) / 4, 0, (np.sqrt(6) + np.sqrt(2)) / 4, [75, 0, 1]),
    ((-np.sqrt(6) - np.sqrt(2)) / 4, 0, (np.sqrt(6) - np.sqrt(2)) / 4, [15, 180, 1]),
    (-1 / 2, 0, -np.sqrt(3) / 2, [-60, 180, 1]),
    (1 / 2, 0, -np.sqrt(3) / 2, [-60, 0, 1]),
    (0, np.sqrt(1 / 2), np.sqrt(1 / 2), [45, 90, 1]),
    (np.sqrt(1 / 4), np.sqrt(1 / 4), np.sqrt(1 / 2), [45, 45, 1]),
])
def test_from_cartesian_to_latlon_r(x, y, z, output):
    np.testing.assert_allclose(rotation_distance.from_cartesian_to_latlon_r(x, y, z), output, atol=1e-8)


@pytest.mark.parametrize("x, y, z, output", [
    (1, 0, 0, [0, 0]),
    (0, 1, 0, [0, 90]),
    (0, 0, 1, [90, 0]),
    (np.sqrt(1 / 2), np.sqrt(1 / 2), 0, [0, 45]),
    (np.sqrt(1 / 2), 0, np.sqrt(1 / 2), [45, 0]),
    (0, np.sqrt(1 / 2), np.sqrt(1 / 2), [45, 90]),
    (np.sqrt(1 / 4), np.sqrt(1 / 4), np.sqrt(1 / 2), [45, 45]),
])
def test_from_cartesian_to_latlon(x, y, z, output):
    np.testing.assert_allclose(rotation_distance.from_cartesian_to_latlon(x, y, z), output, atol=1e-8)


@pytest.mark.parametrize("lat, lon, input, output", [
    (0, 0, [1, 0, 0], [1, 0, 0]),
    (0, 0, [0, 1, 0], [0, 0, 1]),
    (0, 0, [0, 0, 1], [0, 1, 0]),
    (90, 0, [1, 0, 0], [0, -1, 0]),
    (90, 0, [0, 1, 0], [0, 0, 1]),
    (90, 0, [0, 0, 1], [1, 0, 0]),
    (0, 90, [1, 0, 0], [0, 0, -1]),
    (0, 90, [0, 1, 0], [1, 0, 0]),
    (0, 90, [0, 0, 1], [0, 1, 0])
])
def test_rotation_matrix_3d(lat, lon, input, output):
    """
    code from salvus_seismo
    https://gitlab.com/Salvus/salvus_seismo
    """
    # The matrix should go from x, y, z in ZNE.
    M = rotation_distance.get_transformation_matrix_3d(latitude=lat, longitude=lon)
    np.testing.assert_allclose(np.dot(M, input), output, atol=1e-8)

    # Also test the inverse.
    M_inv = rotation_distance.get_transformation_matrix_3d(latitude=lat, longitude=lon, inverse=True)
    np.testing.assert_allclose(np.dot(M_inv, np.dot(M, input)), input, atol=1e-8)
