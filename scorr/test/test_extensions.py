#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import io
import json
import os
from copy import deepcopy

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import toml

import scorr.addons
from scorr.extensions import salvus_extensions, scorr_extensions
from scorr.test.helpers import DIR_TEST_TMP, TEST_config, TEST_site


def check_taper():
    """
    todo: test plot
    :return:
    """
    taper = salvus_extensions.design_taper(full_length=1001, taper_length_in_percent=10)

    plt.figure()
    plt.plot(taper)
    plt.title("taper")
    plt.show()


def test_make_ricker_stf():
    """
    test for ricker shape is missing
    for now only visual inspection
    :return:
    """
    dt = 0.01
    nt = 51
    time, stf = salvus_extensions.make_ricker_stf(starttime=0.0, dt=dt, nt=nt, amplitude=1.0, center_freq=0.006,
                                                  time_shift=0.0, id_component=0)
    assert time[0] == 0.0
    assert time[-1] == 0.0 + (nt - 1) * dt
    assert time.shape[0] == stf.shape[0]
    assert np.sum(np.abs(stf[:, 1:])) == 0.0

    dt = 0.20
    nt = 6001
    time, stf = salvus_extensions.make_ricker_stf(starttime=-600.0, dt=dt, nt=nt, amplitude=5e10, center_freq=0.005,
                                                  time_shift=0.0, id_component=2)
    assert time[0] == -600.0
    assert time[-1] == -600.0 + (nt - 1) * dt
    assert time.shape[0] == stf.shape[0]
    assert np.sum(np.abs(stf[:, :2])) == 0.0

    # plot stf
    plt.figure()
    plt.plot(time, stf[:, 2])
    plt.title("Ricker: starttime: -60.0 s, center ferquency = 0.005, time shift: 0.0 s")
    plt.show()


def test_make_gaussian_stf():
    """
    test for gaussian shape is missing
    for now only visual inspection
    :return:
    """
    dt = 0.01
    nt = 51
    time, stf = salvus_extensions.make_gaussian_stf(starttime=0.0, dt=dt, nt=nt, amplitude=1.0, time_shift=0.0,
                                                    width=1.0, id_component=0)
    assert time[0] == 0.0
    assert time[-1] == 0.0 + (nt - 1) * dt
    assert time.shape[0] == stf.shape[0]
    assert np.sum(np.abs(stf[:, 1:])) == 0.0

    dt = 0.01
    nt = 1001
    time, stf = salvus_extensions.make_gaussian_stf(starttime=-5.0, dt=dt, nt=nt, amplitude=5e10, time_shift=0,
                                                    width=1.0, id_component=2)
    assert time[0] == -5.0
    assert time[-1] == -5.0 + (nt - 1) * dt
    assert time.shape[0] == stf.shape[0]
    assert np.sum(np.abs(stf[:, :2])) == 0.0

    # plot stf
    plt.figure()
    plt.plot(time, stf[:, 2])
    plt.title("Gaussian: starttime: -5.0 s, time shift: 0.0 s")
    plt.show()


def test_make_spike_stf():
    dt = 0.01
    nt = 51
    time, stf = salvus_extensions.make_spike_stf(starttime=0.0, dt=dt, nt=nt, amplitude=1.0e10, time_shift=0.0,
                                                 id_component=0)
    assert time[0] == 0.0
    assert time[-1] == 0.0 + (nt - 1) * dt
    assert np.max(stf) == 1.0e10
    assert np.sum(stf) == 1.0e10
    assert stf[0, 0] == 1.0e10

    dt = 0.1
    nt = 150
    time, stf = salvus_extensions.make_spike_stf(starttime=1.0, dt=dt, nt=nt, amplitude=2.0e10, time_shift=20 * dt,
                                                 id_component=1)
    assert time[0] == 1.0
    assert time[-1] == 1.0 + (nt - 1) * dt
    assert np.max(stf) == 2.0e10
    assert np.sum(stf) == 2.0e10
    assert stf[10, 1] == 2.0e10

    # plot stf
    plt.figure()
    plt.plot(time, stf[:, 1])
    plt.title("Delta-function: starttime: 1.0 s, time shift: 2.0 s")
    plt.show()


def test_write_stf_plus_toml():
    filename_hdf5 = os.path.join(DIR_TEST_TMP, "source_spike.h5")
    filename_toml = os.path.join(DIR_TEST_TMP, "source_spike.toml")

    # one source
    location = [[1.0, 2.0, 3.0]]
    dt = 0.01
    nt = 51
    starttime = 0.0

    _, stf = salvus_extensions.make_spike_stf(starttime=starttime, dt=dt, nt=nt, amplitude=1.0e10, time_shift=0.0,
                                              id_component=0)

    salvus_extensions.write_stf(filename_hdf5=filename_hdf5, identifier=["syn_0"],
                                dt=dt, starttime_in_s=starttime, location=location, data=[stf])
    with h5py.File(filename_hdf5, 'r') as fh:
        assert dt == fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["dt"]
        assert starttime == fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["starttime"]
        assert b"vector" == fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["spatial-type"]
        assert np.array_equal(location[0], fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["location"])

        assert np.array_equal(stf, fh["/AuxiliaryData/AdjointSources/syn_0"][:])

        with pytest.raises(KeyError):
            fh["/AuxiliaryData/AdjointSources/syn_1"][:]

    with open(filename_toml, 'r') as fh:
        dictionary = toml.loads(fh.read())
    assert dictionary == {"source": [{"name": "syn_0", "dataset_name": "/AuxiliaryData/AdjointSources/syn_0"}],
                          "source_input_file": filename_hdf5}

    # two sources
    location = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    dt = 0.01
    nt = 150
    starttime = 0.0

    _, stf_0 = salvus_extensions.make_gaussian_stf(starttime=starttime, dt=dt, nt=nt, amplitude=1.0e10,
                                                   time_shift=5 * dt,
                                                   width=1.0, id_component=1)

    _, stf_1 = salvus_extensions.make_gaussian_stf(starttime=starttime, dt=dt, nt=nt, amplitude=1.0e10,
                                                   time_shift=10 * dt,
                                                   width=1.0, id_component=1)

    salvus_extensions.write_stf(filename_hdf5=filename_hdf5, identifier=["syn_0", "syn_1"],
                                dt=dt, starttime_in_s=starttime, location=location, data=[stf_0, stf_1])
    with h5py.File(filename_hdf5, 'r') as fh:
        assert dt == fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["dt"]
        assert starttime == fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["starttime"]
        assert b"vector" == fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["spatial-type"]
        assert np.array_equal(location[0], fh["/AuxiliaryData/AdjointSources/syn_0"].attrs["location"])
        assert np.array_equal(stf_0, fh["/AuxiliaryData/AdjointSources/syn_0"][:])

        assert dt == fh["/AuxiliaryData/AdjointSources/syn_1"].attrs["dt"]
        assert starttime == fh["/AuxiliaryData/AdjointSources/syn_1"].attrs["starttime"]
        assert b"vector" == fh["/AuxiliaryData/AdjointSources/syn_1"].attrs["spatial-type"]
        assert np.array_equal(location[1], fh["/AuxiliaryData/AdjointSources/syn_1"].attrs["location"])
        assert np.array_equal(stf_1, fh["/AuxiliaryData/AdjointSources/syn_1"][:])

        with pytest.raises(KeyError):
            fh["/AuxiliaryData/AdjointSources/syn_3"][:]

    with open(filename_toml, 'r') as fh:
        dictionary = toml.loads(fh.read())
    assert dictionary == {"source": [{"name": "syn_0", "dataset_name": "/AuxiliaryData/AdjointSources/syn_0"},
                                     {"name": "syn_1", "dataset_name": "/AuxiliaryData/AdjointSources/syn_1"}],
                          "source_input_file": filename_hdf5}


def test_write_receiver_toml():
    filename_toml = os.path.join(DIR_TEST_TMP, "test_receiver.toml")

    # one receiver
    location = [[100.0, 200.0, 300.0]]
    salvus_extensions.write_receiver_toml(filename_toml, location)
    with open(filename_toml, 'r') as fh:
        rec_toml_content = toml.loads(fh.read())
    assert rec_toml_content == {'receiver': [{'network': 'AA', 'station': 'rec0',
                                              'medium': 'solid',
                                              'location': '',
                                              'salvus_coordinates': location[0]}]}

    # two receiver
    location = [[100.0, 200.0, 300.0], [1, 2, 3]]
    salvus_extensions.write_receiver_toml(filename_toml, location)
    with open(filename_toml, 'r') as fh:
        rec_toml_content = toml.loads(fh.read())
    assert rec_toml_content == {'receiver': [{'network': 'AA', 'station': 'rec0',
                                              'medium': 'solid',
                                              'location': '',
                                              'salvus_coordinates': location[0]},
                                             {'network': 'AA', 'station': 'rec1',
                                              'medium': 'solid',
                                              'location': '',
                                              'salvus_coordinates': location[1]}]}


def test_load_configuration():
    # test config
    config_orig = scorr_extensions.load_configuration(TEST_config, type="scorr", verbose=True)

    config = deepcopy(config_orig)
    config["simulation"].pop("green_starttime")
    scorr.addons.write_json_file(DIR_TEST_TMP / "new_config.json", configuration=config)
    with pytest.raises(KeyError):
        scorr_extensions.load_configuration(DIR_TEST_TMP / "new_config.json", type="scorr", verbose=True)

    config = deepcopy(config_orig)
    config["noise_source"]["component_wavefield"] = "1"
    scorr.addons.write_json_file(DIR_TEST_TMP / "new_config.json", configuration=config)
    with pytest.raises(TypeError):
        scorr_extensions.load_configuration(DIR_TEST_TMP / "new_config.json", type="scorr", verbose=True)

    config = deepcopy(config_orig)
    config["simulation"]["corr_max_lag"] = -100
    scorr.addons.write_json_file(DIR_TEST_TMP / "new_config.json", configuration=config)
    with pytest.raises(ValueError):
        scorr_extensions.load_configuration(DIR_TEST_TMP / "new_config.json", type="scorr", verbose=True)

    # test site
    site_orig = scorr_extensions.load_configuration(TEST_site, type="site", verbose=True)

    site = deepcopy(site_orig)
    site.pop("site")
    scorr.addons.write_json_file(DIR_TEST_TMP / "new_config.json", configuration=site)
    with pytest.raises(KeyError):
        scorr_extensions.load_configuration(DIR_TEST_TMP / "new_config.json", type="site", verbose=True)

    site = deepcopy(site_orig)
    site["ranks_salvus"] = "2"
    scorr.addons.write_json_file(DIR_TEST_TMP / "new_config.json", configuration=site)
    with pytest.raises(TypeError):
        scorr_extensions.load_configuration(DIR_TEST_TMP / "new_config.json", type="site", verbose=True)


def test_write_configuration_file():
    config_orig = scorr_extensions.load_configuration(TEST_config, type="scorr")
    scorr.addons.write_json_file(filename=os.path.join(DIR_TEST_TMP, "test_config.json"),
                                 configuration=config_orig)

    with io.open(os.path.join(DIR_TEST_TMP, "test_config.json"), 'r') as fh:
        config_written = json.load(fh, object_hook=scorr.addons.from_json)
    assert config_written == config_orig


if __name__ == "__main__":
    check_taper()
    test_make_ricker_stf()
    test_make_gaussian_stf()
    test_make_spike_stf()
