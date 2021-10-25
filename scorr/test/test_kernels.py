#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
from scorr.kernel.source_kernel import SourceKernel
from scorr.test.helpers import DIR_TEST_DATA, wavefield_file_exists
from scorr.wavefield.wavefield import Wavefield


def test_source_kernel():
    test1 = SourceKernel()
    assert test1.coordinates is None
    assert test1.connectivity is None
    assert test1.globalElementIds is None
    assert test1.n_elements_global is None
    assert test1.kernel == 0.0

    test2 = SourceKernel.init_with_kernel_file(DIR_TEST_DATA / "kernel_source.h5")
    wavefield = Wavefield(filename=wavefield_file_exists, starttime=0.0, endtime=1.0)

    print(test2.coordinates - wavefield.coordinates)

    assert (test2.coordinates == wavefield.coordinates).all()
    assert (test2.connectivity == wavefield.connectivity).all()
    assert (test2.globalElementIds == wavefield.globalElementIds).all()
    assert test2.n_elements_global == wavefield.n_elements_global


def test_add():
    test_empty = SourceKernel()
    test_full = SourceKernel.init_with_kernel_file(DIR_TEST_DATA / "kernel_source.h5")
    wavefield = Wavefield(filename=wavefield_file_exists, starttime=0.0, endtime=1.0)

    # add two empty source kernels
    test_sum = test_empty + test_empty
    assert test_sum.coordinates is None
    assert test_sum.connectivity is None
    assert test_sum.globalElementIds is None
    assert test_sum.n_elements_global is None
    assert test_sum.kernel == 0.0

    # empty + full
    test_sum = test_empty + test_full
    assert (test_sum.coordinates == test_full.coordinates).all()
    assert (test_sum.connectivity == test_full.connectivity).all()
    assert (test_sum.globalElementIds == test_full.globalElementIds).all()
    assert test_sum.n_elements_global == test_full.n_elements_global
    assert (test_sum.kernel == test_full.kernel).all()
    assert (test_sum.coordinates == wavefield.coordinates).all()
    assert (test_sum.connectivity == wavefield.connectivity).all()
    assert (test_sum.globalElementIds == wavefield.globalElementIds).all()
    assert test_sum.n_elements_global == wavefield.n_elements_global

    # full + empty
    test_sum = test_full + test_empty
    assert (test_sum.coordinates == test_full.coordinates).all()
    assert (test_sum.connectivity == test_full.connectivity).all()
    assert (test_sum.globalElementIds == test_full.globalElementIds).all()
    assert test_sum.n_elements_global == test_full.n_elements_global
    assert (test_sum.kernel == test_full.kernel).all()
    assert (test_sum.coordinates == wavefield.coordinates).all()
    assert (test_sum.connectivity == wavefield.connectivity).all()
    assert (test_sum.globalElementIds == wavefield.globalElementIds).all()
    assert test_sum.n_elements_global == wavefield.n_elements_global

    # full + full
    test_sum = test_full + test_full
    assert (test_sum.coordinates == test_full.coordinates).all()
    assert (test_sum.connectivity == test_full.connectivity).all()
    assert (test_sum.globalElementIds == test_full.globalElementIds).all()
    assert test_sum.n_elements_global == test_full.n_elements_global
    assert (test_sum.kernel == 2 * test_full.kernel).all()
    assert (test_sum.coordinates == wavefield.coordinates).all()
    assert (test_sum.connectivity == wavefield.connectivity).all()
    assert (test_sum.globalElementIds == wavefield.globalElementIds).all()
    assert test_sum.n_elements_global == wavefield.n_elements_global


def test_iadd():
    test_full_check = SourceKernel.init_with_kernel_file(DIR_TEST_DATA / "kernel_source.h5")
    wavefield = Wavefield(filename=wavefield_file_exists, starttime=0.0, endtime=1.0)

    # add two empty source kernels
    test_empty = SourceKernel()

    test_empty += test_empty
    assert test_empty.coordinates is None
    assert test_empty.connectivity is None
    assert test_empty.globalElementIds is None
    assert test_empty.n_elements_global is None
    assert test_empty.kernel == 0.0

    # empty + full
    test_empty = SourceKernel()
    test_full = SourceKernel.init_with_kernel_file(DIR_TEST_DATA / "kernel_source.h5")

    test_empty += test_full
    assert (test_empty.coordinates == test_full.coordinates).all()
    assert (test_empty.connectivity == test_full.connectivity).all()
    assert (test_empty.globalElementIds == test_full.globalElementIds).all()
    assert test_empty.n_elements_global == test_full.n_elements_global
    assert (test_empty.kernel == test_full.kernel).all()
    assert (test_empty.coordinates == wavefield.coordinates).all()
    assert (test_empty.connectivity == wavefield.connectivity).all()
    assert (test_empty.globalElementIds == wavefield.globalElementIds).all()
    assert test_empty.n_elements_global == wavefield.n_elements_global

    # full + empty
    test_empty = SourceKernel()
    test_full = SourceKernel.init_with_kernel_file(DIR_TEST_DATA / "kernel_source.h5")

    test_full += test_empty
    assert (test_full.kernel == test_full_check.kernel).all()
    assert (test_full.coordinates == wavefield.coordinates).all()
    assert (test_full.connectivity == wavefield.connectivity).all()
    assert (test_full.globalElementIds == wavefield.globalElementIds).all()
    assert test_full.n_elements_global == wavefield.n_elements_global

    # full + full
    test_full += test_full
    assert (test_full.kernel == 2 * test_full_check.kernel).all()
    assert (test_full.coordinates == wavefield.coordinates).all()
    assert (test_full.connectivity == wavefield.connectivity).all()
    assert (test_full.globalElementIds == wavefield.globalElementIds).all()
    assert test_full.n_elements_global == wavefield.n_elements_global
