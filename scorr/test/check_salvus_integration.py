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
from pathlib import Path

import numpy as np
import pyasdf
import pytest

import scorr.addons
import scorr.api
from scorr.extensions import scorr_extensions, job_tracker
from scorr.noise_source.noise_source_setup import setup_noise_source
from scorr.tasks import preparation

# specify where the tests should run
DIR_PROJECT = Path.home() / "Desktop" / "SWP_past" / \
    "correlations_3d" / "scorr_test"

# specify mesh
mesh_name = "Hex_IsotropicElastic3D_Elemental_2x2x2.e"
# mesh_name = "Globe3D_3_layer_250.e"

# load and edit configuration
config = scorr_extensions.load_configuration(
    DIR_PROJECT / "config" / "scorr.json", type="scorr")
config["simulation"]["reference_stations"] = config["working_dir_local"] / \
    "reference_stations.json"
config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name
config["simulation"]["green_starttime"] = 0.0
config["simulation"]["corr_max_lag"] = 1.0
config["simulation"]["corr_max_lag_causal"] = 1.0
config["simulation"]["dt"] = 0.01

config["simulation"]["attenuation"] = False
config["simulation"]["sampling_rate_boundary"] = 1
config["simulation"]["sampling_rate_volume"] = 1
config["noise_source"]["filename"] = config["working_dir_local"] / \
    "noise_source" / "noise_source.h5"

if "Globe" in str(config["simulation"]["mesh"]):
    config["simulation"]["sideset"] = "r1"
    config["simulation"]["green_component"] = 0
    config["simulation"]["green_amplitude"] = -1.0e10
    config["noise_source"]["component_dist_source"] = 0
    config["noise_source"]["component_wavefield"] = 0
    config["simulation"]["recording"] = "u_ELASTIC"
    config["noise_source"]["filter_spec"] = None
    config["simulation"]["absorbing"]["boundaries"] = None
    config["simulation"]["absorbing"]["axis-aligned"] = False
    config["simulation"]["spherical"] = True
    loc_sources = [[0.0, 0.0, 1.0]]
    loc_receivers = [[[0.0, 0.1, 0.0]]]
else:
    config["simulation"]["sideset"] = "z0"
    config["simulation"]["green_component"] = 2
    config["simulation"]["green_amplitude"] = 1.0e10
    config["noise_source"]["component_dist_source"] = 2
    config["noise_source"]["component_wavefield"] = 2
    config["simulation"]["recording"] = "u_ELASTIC"
    config["noise_source"]["filter_spec"] = None
    config["simulation"]["absorbing"]["boundaries"] = None
    config["simulation"]["absorbing"]["axis-aligned"] = False
    config["simulation"]["spherical"] = False
    loc_sources = [[0.0, 49950.0, 49950.0], [0.0, 50050.0, 49950.0]]
    loc_receivers = [[[0.0, 50050.0, 50050.0], [0.0, 49950.0, 50050.0]],
                     [[0.0, 50150.0, 50150.0], [0.0, 49950.0, 50150.0]]]

# some safety measures, stf generation is not yet general enough
nt_corr = abs(config["simulation"]["corr_max_lag"]) / \
    config["simulation"]["dt"]
nt_green = abs(config["simulation"]["green_starttime"]) / \
    config["simulation"]["dt"]

assert np.mod(nt_corr, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_green, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr, config["simulation"]
              ["sampling_rate_boundary"]) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_green, config["simulation"]
              ["sampling_rate_boundary"]) == pytest.approx(0, abs=1.0e-8)

# save configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "scorr.json",
                                    config=config, type="scorr", verbose=config["verbose"])

# create site config
site = {"site": "local",
        "ranks_salvus": 2,
        "ranks_scorr": 2,
        "ping_interval_in_seconds": 1,
        "wall_time_in_seconds_salvus": 30,
        "wall_time_in_seconds_scorr": 30}

# save configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "site.json",
                                    config=site, type="site", verbose=config["verbose"])


def check_linearity():
    # set inversion type such that no forward wavefield is saved
    config["inversion"] = ""

    # load noise source configuration
    config_noise_source = scorr_extensions.load_configuration(DIR_PROJECT / "config" / "noise_source.json",
                                                              type="noise_source")
    config_noise_source["type"] = "homogeneous"

    # compute correlations
    ref_station_list = scorr.addons.load_json_file(
        config["working_dir_local"] / "reference_stations.json")

    for identifier in ref_station_list.keys():

        for i in range(1, 3):
            config_noise_source["homog_magnitude"] = i
            setup_noise_source(config=config, site=site,
                               config_noise_source=config_noise_source)

            identifier_homog = "homog_" + \
                identifier.split("_")[1] + "_" + str(i)
            job_tracker.remove_reference_station(ref_station=identifier_homog)
            scorr.api.compute_correlations(site=site, config=config,
                                           ref_identifier=identifier_homog,
                                           src_toml=ref_station_list[identifier]["src_toml"],
                                           rec_toml=ref_station_list[identifier]["rec_toml"],
                                           output_folder=config["working_dir_local"] / "correlations" / (
                                               "homog_" + identifier.split("_")[1] + "_" + str(i)))

            # load correlations
            if i == 1:
                with pyasdf.ASDFDataSet(
                        str(config["working_dir_local"] / "correlations" / identifier_homog / "receivers.h5")) as fh:
                    correlation = 2 * \
                        fh.waveforms["AA.rec0"].displacement[2].data
                    assert np.sum(abs(correlation)) > 0.0
            if i == 2:
                with pyasdf.ASDFDataSet(
                        str(config["working_dir_local"] / "correlations" / identifier_homog / "receivers.h5")) as fh:
                    correlation -= fh.waveforms["AA.rec0"].displacement[2].data

    assert np.sum(correlation) == 0.0


preparation.prepare_source_and_receiver(config=config, identifier_prefix="syn",
                                        loc_sources=loc_sources, loc_receivers=loc_receivers)
check_linearity()
