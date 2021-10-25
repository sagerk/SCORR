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
from scorr.test.helpers import comm, size, MPI
from scorr.test.helpers import wavefield_file_exists
from scorr.wavefield.adjoint_wavefield import AdjointWavefield
from scorr.wavefield.green_function import GreenFunction
from scorr.wavefield.wavefield import Wavefield

try:
    green_function_1 = GreenFunction(
        wavefield_file_exists, starttime=0.0, endtime=1.0)
except:
    print("problem reading Green function")
    raise

file_not_exists = "does_not_exist.h5"


def test_wavefields():
    # starttime and endtime
    with pytest.raises(TypeError):
        Wavefield(file_not_exists)

    # missing endtime
    with pytest.raises(TypeError):
        Wavefield(file_not_exists, starttime=0.0)

    # missing starttime
    with pytest.raises(TypeError):
        Wavefield(file_not_exists, endtime=0.0)

    # file does not exist
    with pytest.raises(OSError):
        Wavefield(file_not_exists, starttime=0.0, endtime=1.0)


def test_green_function():
    # starttime positive
    with pytest.raises(AssertionError):
        GreenFunction(wavefield_file_exists, starttime=1.0, endtime=1.0)

    # starttime and endtime
    with pytest.raises(TypeError):
        GreenFunction(file_not_exists)

    # missing endtime
    with pytest.raises(TypeError):
        GreenFunction(file_not_exists, starttime=0.0)

    # missing starttime
    with pytest.raises(TypeError):
        GreenFunction(file_not_exists, endtime=0.0)

    # file does not exist
    with pytest.raises(OSError):
        GreenFunction(file_not_exists, starttime=0.0, endtime=1.0)


def test_adjoint_wavefield():
    # starttime positive
    with pytest.raises(AssertionError):
        AdjointWavefield(wavefield_file_exists, starttime=1.0, endtime=1.0)

    # starttime and endtime
    with pytest.raises(TypeError):
        AdjointWavefield(file_not_exists)

    # missing endtime
    with pytest.raises(TypeError):
        AdjointWavefield(file_not_exists, starttime=0.0)

    # missing starttime
    with pytest.raises(TypeError):
        AdjointWavefield(file_not_exists, endtime=0.0)

    # file does not exist
    with pytest.raises(OSError):
        AdjointWavefield(file_not_exists, starttime=-1.0, endtime=1.0)


def test_green_function_init_with_dict():
    pass


def test_adjoint_wavefield_init_with_dict():
    pass


def test_green_function_information():
    assert green_function_1.n_elements_global == 4
    assert green_function_1.n_gll == 25
    assert green_function_1.n_components == 3
    assert green_function_1.dt == pytest.approx(0.01, rel=1e-6)
    assert green_function_1.nt_stored == 101
    assert green_function_1.nt_longest_branch == 101

    sum_rank = np.sum(green_function_1.wavefield)
    sum_total = comm.allreduce(sum_rank, op=MPI.SUM)
    assert sum_total == pytest.approx(9.198063e-05, rel=1e-6)


def test_wavefield_rotation():
    # directly rotate wavefield by respective call of GreenFunction
    # internally done with np.einsum
    green_rotated = GreenFunction(
        wavefield_file_exists, starttime=0.0, endtime=1.0, rotate=True)

    # first create GreenFunction, then rotate in the most comprehensible way
    green_not_rotated = GreenFunction(
        wavefield_file_exists, starttime=0.0, endtime=1.0, rotate=False)

    for id_element_local in range(green_not_rotated.coordinates.shape[0]):
        for id_gll in range(green_not_rotated.n_gll):
            x, y, z = green_not_rotated.coordinates[id_element_local, id_gll, :]
            lat, lon = rotation_distance.from_cartesian_to_latlon(x, y, z)
            R = rotation_distance.get_transformation_matrix_3d(
                latitude=lat, longitude=lon)

            # rotate each timestep
            for id_time in range(green_not_rotated.nt_stored):
                green_not_rotated.wavefield[id_time, id_element_local, :, id_gll] = np.dot(
                    R, green_not_rotated.wavefield[id_time, id_element_local, :, id_gll])

    assert np.allclose(green_rotated.wavefield,
                       green_not_rotated.wavefield, rtol=1e-15)


def test_wavefield_get_coordinates_of_gll():
    coordinates = green_function_1.get_coordinates_of_gll(0, 0)
    assert coordinates.shape == (3,)

    if size == 1:
        assert green_function_1.coordinates.shape == (4, 25, 3)
        assert np.sum(green_function_1.coordinates).round() == 10000000.0
    if size == 2:
        assert green_function_1.coordinates.shape == (2, 25, 3)

        sum_rank = np.sum(green_function_1.coordinates)
        sum_total = np.zeros(1)
        comm.Allreduce([sum_rank, MPI.DOUBLE], [sum_total, MPI.DOUBLE])
        assert sum_total == 10000000.0
    if size == 4:
        assert green_function_1.coordinates.shape == (1, 25, 3)

        sum_rank = np.sum(green_function_1.coordinates)
        sum_total = np.zeros(1)
        comm.Allreduce([sum_rank, MPI.DOUBLE], [sum_total, MPI.DOUBLE])
        assert sum_total == 10000000.0


def test_wavefield_get_wavefield_comp_at_gll():
    wavefield_1, _ = green_function_1.get_wavefield_comp_at_gll(0, 0, 0)
    assert wavefield_1.shape == (101,)

    wavefield_2, _ = green_function_1.get_pad_wavefield_comp_at_gll(0, 0, 0)
    assert wavefield_2.shape == (201,)

    wavefield_3, _ = green_function_1.get_pad_rev_wavefield_comp_at_gll(
        0, 0, 0)
    assert wavefield_3.shape == (201,)

    assert all(wavefield_2[100:] == wavefield_1)
    assert all(np.flipud(wavefield_3) == wavefield_2)
