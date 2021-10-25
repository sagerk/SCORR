#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
from pathlib import Path

import numpy as np
import pytest
from scorr.extensions import scorr_extensions
from scorr.noise_source.noise_source_setup import setup_noise_source
from scorr.tasks import preparation

# specify where the tests should run
DIR_PROJECT = Path.home() / "scorr_inversion_synthetic"

# specify mesh
mesh_name = "Globe3D_prem_iso_one_crust_100.e"
# mesh_name = "Globe3D_prem_iso_one_crust_100_with_s20.e"

# load and edit configuration
config = scorr_extensions.load_configuration(DIR_PROJECT / "config" / "scorr.json", type="scorr")

config["working_dir_local"] = DIR_PROJECT
config["simulation"]["reference_stations"] = config["working_dir_local"] / "reference_stations.json"
config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name

config["simulation"]["green_starttime"] = -400.0
config["simulation"]["corr_max_lag"] = 21600.0
config["simulation"]["corr_max_lag_causal"] = 7200
config["simulation"]["dt"] = 0.5

config["simulation"]["kernel-fields"] = "VP,VS,RHO"
config["simulation"]["anisotropy"] = False
config["simulation"]["attenuation"] = True
config["simulation"]["sampling_rate_boundary"] = 20
config["simulation"]["sampling_rate_volume"] = 20
config["noise_source"]["filename"] = config["working_dir_local"] / "noise_source" / "noise_source.h5"
# config["noise_source"]["filename"] = config["working_dir_local"] / "noise_source" / "noise_source_obs.h5"

config["simulation"]["sideset"] = "r1"
config["simulation"]["green_component"] = 0
config["simulation"]["green_amplitude"] = -1.0e10
config["noise_source"]["component_dist_source"] = 0
config["noise_source"]["component_wavefield"] = 0
config["simulation"]["recording"] = "u_ELASTIC"
config["noise_source"]["filter_spec"] = [0.0033333, 0.01]
config["simulation"]["absorbing"]["boundaries"] = None
config["simulation"]["absorbing"]["axis-aligned"] = False
config["simulation"]["spherical"] = True

# loc_sources = [[0.0, 0.0, 100.0]]
# loc_receivers = [[[0.0, 0.0, 100.0], [0.0, 10.0, 100.0], [0.0, 20.0, 100.0],
# [0.0, 30.0, 100.0], [0.0, 40.0, 100.0], [0.0, 50.0, 100.0],
# [0.0, 60.0, 100.0], [0.0, 70.0, 100.0], [0.0, 80.0, 100.0],
# [0.0, 90.0, 100.0], [0.0, 100.0, 100.0], [0.0, 110.0, 100.0],
# [0.0, 121.0, 100.0], [0.0, 130.0, 100.0], [0.0, 140.0, 100.0],
# [0.0, 150.0, 100.0], [0.0, 160.0, 100.0], [0.0, 170.0, 100.0],
# [0.0, 180.0, 100.0], [0.0, -10.0, 100.0], [0.0, -20.0, 100.0],
# [0.0, -29.0, 100.0], [0.0, -40.0, 100.0], [0.0, -50.0, 100.0],
# [0.0, -60.0, 100.0], [0.0, -70.0, 100.0], [0.0, -79.0, 100.0],
# [0.0, -90.0, 100.0], [0.0, -100.0, 100.0], [0.0, -110.0, 100.0],
# [0.0, -120.0, 100.0], [0.0, -130.0, 100.0], [0.0, -141.0, 100.0],
# [0.0, -150.0, 100.0], [0.0, -160.0, 100.0], [0.0, -170.0, 100.0],
# [10.0, 0.0, 100.0], [20.0, 0.0, 100.0],
# [30.0, 0.0, 100.0], [40.0, 0.0, 100.0], [50.0, 0.0, 100.0],
# [60.0, 0.0, 100.0], [70.0, 0.0, 100.0], [80.0, 0.0, 100.0],
# [90.0, 0.0, 100.0], [-10.0, 0.0, 100.0], [-20.0, 0.0, 100.0],
# [-30.0, 0.0, 100.0], [-40.0, 0.0, 100.0], [-50.0, 0.0, 100.0],
# [-60.0, 0.0, 100.0], [-70.0, 0.0, 100.0], [-80.0, 0.0, 100.0],
# [-90.0, 0.0, 100.0],
# [45.0, 45.0, 100.0], [45.0, -45.0, 100.0], [-45.0, -45.0, 100.0], [-45.0, 45.0, 100.0],
# [33.051491, -114.827057, 100.0]
# ]]

loc_receivers = []
# n_sources = 1
n_sources = 15
with open(DIR_PROJECT / "_sts-1_filtered.txt", mode="r") as fh:
    for i_src in range(n_sources):
        loc_receivers.append([])
        for line in fh:
            station = [float(item) for item in line.strip().split(",")]
            loc_receivers[i_src].append(station)
        fh.seek(0)

loc_sources = []
# source_list = [3]
source_list = [3, 4, 12, 14, 15, 19, 22, 30, 37, 42, 46, 51, 114, 115, 120]
assert len(source_list) == n_sources
for i_src in range(n_sources):
    loc_sources.append(loc_receivers[i_src].pop(source_list[i_src]))

# some safety measures, stf generation is not yet general enough
nt_corr = abs(config["simulation"]["corr_max_lag"]) / config["simulation"]["dt"]
nt_green = abs(config["simulation"]["green_starttime"]) / config["simulation"]["dt"]

assert np.mod(nt_corr, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_green, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr, config["simulation"]["sampling_rate_boundary"]) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_green, config["simulation"]["sampling_rate_boundary"]) == pytest.approx(0, abs=1.0e-8)

# save configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "scorr.json",
                                    config=config, type="scorr")

#####################################
##########  NOISE SOURCE   ##########
#####################################

# save load source configuration
config_noise_source = scorr_extensions.load_configuration(filename=config["working_dir_local"] / "config" /
                                                                   "noise_source.json", type="noise_source")
config_noise_source["type"] = "homogeneous"
# config_noise_source["type"] = "gaussian"
# config_noise_source["spectrum"]["f_peak"] = 0.005

# save noise source configuration
scorr_extensions.save_configuration(
    filename=config["working_dir_local"] / "config" / "noise_source.json", config=config_noise_source,
    type="noise_source")

#####################################
##########      SITE       ##########
#####################################

# load site configuration
# site = scorr_extensions.load_configuration(DIR_PROJECT / "config" / "site.json", type="site")

# change site configuration
site = {"site": "daint",
        "ranks_salvus": 192,
        "ranks_scorr": 24,
        "ping_interval_in_seconds": 60,
        "wall_time_in_seconds_salvus": 18000,
        "wall_time_in_seconds_scorr": 7200}

# save site configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "site.json",
                                    config=site, type="site")

#####################################
##########   MEASUREMENT   ##########
#####################################

# load measurement configuration
config_measurement = scorr_extensions.load_configuration(DIR_PROJECT / "config" / "measurement.json",
                                                         type="measurement")

# change measurement configuration
# config_measurement["type"] = "waveform_differences"
# config_measurement["type"] = "log_amplitude_ratio"
config_measurement["type"] = "cc_time_shift"
config_measurement["component_recording"] = 2
config_measurement["pick_window"] = True
config_measurement["pick_manual"] = False
config_measurement["min_period_in_s"] = 200.0
config_measurement["scale"] = 1e10
config_measurement["snr"] = None
config_measurement["surface_wave_velocity_in_mps"] = 3700.0
# config_measurement["surface_wave_velocity_in_mps"] = 4000.0
config_measurement["window_halfwidth_in_sec"] = 600.0
config_measurement["number_of_stacked_windows_min"] = None
config_measurement["station_list"] = None

# save measurement configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "measurement.json",
                                    config=config_measurement, type="measurement")

#####################################
########   RUN PREPARATION   ########
#####################################

setup_noise_source(config=config, site=site, config_noise_source=config_noise_source)
preparation.prepare_source_and_receiver(config=config, identifier_prefix="syn",
                                        loc_sources=loc_sources, loc_receivers=loc_receivers)
