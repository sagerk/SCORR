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
import scorr.addons
import scorr.api as api
import scorr.main as main
from click.testing import CliRunner
from shutil import rmtree

from scorr.extensions import scorr_extensions
from scorr.test import helpers
from scorr.test.helpers import comm, rank, DIR_TEST_DATA, DIR_TEST_TMP


@pytest.mark.parametrize("input, expected", [
    ([], 2),
    (["init_test"], 0),
])
def test_click_options_init_project(input, expected):
    if input:
        input = [os.path.join(DIR_TEST_TMP, input[0])]
        print(input)

    runner = CliRunner()
    result = runner.invoke(main.init_project, input)

    assert result.exit_code == expected
    if expected == 0:
        assert len(os.listdir(input[0])) == 3
        assert len(os.listdir(os.path.join(input[0], "job_tracker"))) == 1
        assert len(os.listdir(os.path.join(input[0], "config"))) == 4
        assert 'scorr.json' in os.listdir(os.path.join(input[0], "config"))
        assert 'noise_source.json' in os.listdir(os.path.join(input[0], "config"))
        assert 'site.json' in os.listdir(os.path.join(input[0], "config"))
        assert 'measurement.json' in os.listdir(os.path.join(input[0], "config"))

        # cleanup directory
        if rank == 0:
            rmtree(input[0])
        comm.Barrier()


@pytest.mark.parametrize("input, expected", [
    (["--json_file", "json"], 2),
    (["--wavefield_file", "green"], 2),
    (["--output_file", "output"], 2),
    (["--json_file", "json", "--wavefield_file", "green"], 2),
    (["--json_file", "json", "--output_file", "output"], 2),
    (["--json_file", "json", "--noise_source_file", "noise_source"], 2),
    (["--wavefield_file", "green", "--output_file", "output"], 2),
    (["--wavefield_file", "green", "--noise_source_file", "noise_source"], 2),
    (["--output_file", "green", "--noise_source_file", "noise_source"], 2),
    (["--json_file", "json", "--wavefield_file", "green", "--output_file", "output"], 2),
    (["--json_file", "json", "--noise_source_file", "noise_source", "--output_file", "output"], 2),
    (["--json_file", "json", "--noise_source_file", "noise_source", "--wavefield_file", "green", ], 2),
    (["--noise_source_file", "noise_source", "--wavefield_file", "green", "--output_file", "output"], 2),
    (["--json_file", "json", "--noise_source_file", "noise_source", "--wavefield_file", "green", "--output_file", "output"], -1),
])
def test_click_options_correlation_source(input, expected):
    runner = CliRunner()
    result = runner.invoke(main.make_correlation_source, input)

    assert result.exit_code == expected
    if expected == -1:
        assert result.exc_info[1].__class__ is FileNotFoundError


@pytest.mark.parametrize("input, expected", [
    (["--json_file", "json"], 2),
    (["--spectrum", "1.0,2.0,3.0"], 2),
    (["--green_file", "green"], 2),
    (["--output_file", "output"], 2),
    (["--adjoint_file", "adjoint"], 2),
    (["--json_file", "json", "--green_file", "green"], 2),
    (["--json_file", "json", "--output_file", "output"], 2),
    (["--json_file", "json", "--adjoint_file", "adjoint"], 2),
    (["--json_file", "json", "--spectrum", "1.0,2.0,3.0"], 2),
    (["--green_file", "green", "--output_file", "output"], 2),
    (["--green_file", "green", "--adjoint_file", "adjoint"], 2),
    (["--green_file", "green", "--spectrum", "1.0,2.0,3.0"], 2),
    (["--adjoint_file", "adjoint", "--output_file", "output"], 2),
    (["--adjoint_file", "adjoint", "--spectrum", "1.0,2.0,3.0"], 2),
    (["--output_file", "output", "--spectrum", "1.0,2.0,3.0"], 2),
    (["--json_file", "json", "--green_file", "green", "--adjoint_file", "adjoint",
      "--spectrum", "1.0,2.0,3.0", "--output_file", "output"], -1),
])
def test_click_options_build_source_kernel(input, expected):
    runner = CliRunner()
    result = runner.invoke(main.build_source_kernel, input)

    assert result.exit_code == expected
    if expected == -1:
        assert result.exc_info[1].__class__ is AssertionError or result.exc_info[1].__class__ is FileNotFoundError


def test_make_correlation_source_homogeneous():
    config = scorr_extensions.load_configuration(helpers.TEST_config, type="scorr")
    config["noise_source"]["filename"] = DIR_TEST_DATA / "noise_source_homogeneous.h5"
    scorr.addons.write_json_file(DIR_TEST_TMP / "test_config_homogeneous.json", configuration=config)

    api.make_correlation_source(json_file=DIR_TEST_TMP / "test_config_homogeneous.json",
                                noise_source_file=DIR_TEST_DATA / "noise_source_homogeneous.h5",
                                wavefield_file=DIR_TEST_DATA / "wavefield_BND_green.h5",
                                output_file=DIR_TEST_TMP / "test_homogeneous_main.h5")

    comm.Barrier()
    with h5py.File(str(DIR_TEST_TMP / "test_homogeneous_main.h5"), 'r') as hdf5:
        assert np.sum(hdf5["/ELASTIC_BND/data"]) == pytest.approx(3.44259e-3, rel=1e-5), \
            "check if parallel support for hdf5 and h5py is installed"


def test_make_correlation_source_gaussian():
    config = scorr_extensions.load_configuration(helpers.TEST_config, type="scorr")
    config["noise_source"]["filename"] = DIR_TEST_DATA / "noise_source_gaussian.h5"
    scorr.addons.write_json_file(DIR_TEST_TMP / "test_config_gaussian.json", configuration=config)

    api.make_correlation_source(json_file=DIR_TEST_TMP / "test_config_gaussian.json",
                                noise_source_file=DIR_TEST_DATA / "noise_source_gaussian.h5",
                                wavefield_file=DIR_TEST_DATA / "wavefield_BND_green.h5",
                                output_file=DIR_TEST_TMP / "test_gaussian_main.h5")

    comm.Barrier()
    with h5py.File(str(DIR_TEST_TMP / "test_gaussian_main.h5"), 'r') as hdf5:
        assert np.sum(hdf5["/ELASTIC_BND/data"]) == pytest.approx(3.44918e-3, rel=1e-5), \
            "check if parallel support for hdf5 and h5py is installed"


def test_build_source_kernel():
    """
    TODO
    :return:
    """
    pass
