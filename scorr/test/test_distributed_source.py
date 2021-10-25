#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import os

import h5py
import numpy as np
import pytest

from scorr.distributed_source.correlation_source import CorrelationSource
from scorr.noise_source.distribution import Distribution
from scorr.noise_source.noise_source import NoiseSource
from scorr.test.helpers import comm, rank, MPI, green_function, spectrum, DIR_TEST_TMP, DIR_TEST_DATA


def test_correlation_source():
    distribution = Distribution(distribution=1.0)
    noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution)

    corr_source = CorrelationSource(noise_source, green_function)
    assert corr_source.nt_full == 201


def test_convolve_green_function_homogeneous():
    distribution = Distribution.init_distribution_with_file(filename=DIR_TEST_DATA / "noise_source_homogeneous.h5")
    noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution)

    corr_source = CorrelationSource(noise_source, green_function)
    corr_source.convolve_wavefield(id_component_wavefield=0, id_component_dist_source=0, verbose=False)

    sum_rank = np.sum(corr_source.distributed_source)
    sum_total = np.zeros(1)
    comm.Allreduce([sum_rank, MPI.DOUBLE], [sum_total, MPI.DOUBLE])
    assert sum_total == pytest.approx(3.44259e-3, rel=1e-5)


def test_convolve_green_function_gaussian():
    distribution = Distribution.init_distribution_with_file(filename=DIR_TEST_DATA / "noise_source_gaussian.h5")
    noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution)

    corr_source = CorrelationSource(noise_source, green_function)
    corr_source.convolve_wavefield(id_component_wavefield=0, id_component_dist_source=0, verbose=False)

    sum_rank = np.sum(corr_source.distributed_source)
    sum_total = np.zeros(1)
    comm.Allreduce([sum_rank, MPI.DOUBLE], [sum_total, MPI.DOUBLE])
    assert sum_total == pytest.approx(3.44918e-3, rel=1e-5)


def test_linearity():
    # set up correlation source for 2 models
    for i in range(1, 3):
        distribution = Distribution(distribution=i * 1.0e5)
        noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution)
        corr_source = CorrelationSource(noise_source, green_function)
        corr_source.convolve_wavefield(id_component_wavefield=0, id_component_dist_source=0)
        corr_source.write_distributed_source_to_file(os.path.join(DIR_TEST_TMP, "linearity_" + str(i) + ".h5"))

        with h5py.File(os.path.join(DIR_TEST_TMP, "linearity_" + str(i) + ".h5"), 'r') as fh:
            if i == 1: source = 2 * fh["/ELASTIC_BND/data"][:]
            if i == 2: source -= fh["/ELASTIC_BND/data"][:]

    assert np.sum(source) == 0.0


def test_write_correlation_source_file_homogeneous():
    distribution = Distribution(distribution=1.0)
    noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution)

    corr_source = CorrelationSource(noise_source, green_function)
    corr_source.convolve_wavefield(id_component_wavefield=0, id_component_dist_source=0, verbose=False)

    # parallel writing
    corr_source.write_distributed_source_to_file(os.path.join(DIR_TEST_TMP, "test_homogeneous.h5"))

    with h5py.File(os.path.join(DIR_TEST_TMP, "test_homogeneous.h5"), 'r') as hdf5:
        sum = np.sum(hdf5["/ELASTIC_BND/data"])
        assert sum == pytest.approx(3.44259e-3, rel=1e-5,
                                    abs=1e-6), "check if parallel support for hdf5 and h5py is installed"


def test_write_correlation_source_file_gaussian():
    distribution = Distribution.init_distribution_with_file(filename=DIR_TEST_DATA / "noise_source_gaussian.h5")
    noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution)

    corr_source = CorrelationSource(noise_source, green_function)
    corr_source.convolve_wavefield(id_component_wavefield=0, id_component_dist_source=0, verbose=False)

    corr_source.write_distributed_source_to_file(os.path.join(DIR_TEST_TMP, "test_gaussian.h5"))

    if rank == 0:
        with h5py.File(os.path.join(DIR_TEST_TMP, "test_gaussian.h5"), 'r') as hdf5:
            sum = np.sum(hdf5["/ELASTIC_BND/data"][:])
            assert sum == pytest.approx(3.44918e-3, rel=1e-5), "check if parallel support for hdf5 and h5py is installed"
