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
# DIR_PROJECT = Path.home() / "Desktop" / "CurrentLiterature&Projects" / \
#     "Grenoble" / "scorr_test_grenoble"
# DIR_PROJECT = Path("/scratch/snx3000/sagerk/scorr_test_grenoble")
DIR_PROJECT = Path("/home/ksager/scorr_test_grenobleX")

# specify mesh
mesh_name = "mesh/reference_0.3s.e"
# mesh_name = "mesh/reference_0.3s_decrease_0.01_top_layer.e"
# mesh_name = "mesh/reference_0.3s_decrease_0.01rel_constant.e"
# mesh_name = "mesh/reference_0.3s_moho.e"

# load and edit configuration
config = scorr_extensions.load_configuration(
    DIR_PROJECT / "config" / "scorr.json", type="scorr")

config["simulation"]["reference_stations"] = config["working_dir_local"] / \
    "reference_stations.json"
config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name

config["working_dir_local"] = DIR_PROJECT
config["simulation"]["green_starttime"] = -1.0
config["simulation"]["corr_max_lag"] = 7.0
config["simulation"]["corr_max_lag_causal"] = 7.0
config["simulation"]["dt"] = 0.005

config["simulation"]["attenuation"] = False
config["simulation"]["sampling_rate_boundary"] = 10
config["simulation"]["sampling_rate_volume"] = 20
config["noise_source"]["filename"] = config["working_dir_local"] / \
    "noise_source" / "noise_source.h5"

config["simulation"]["sideset"] = "r1"
config["simulation"]["green_component"] = 0
config["simulation"]["green_amplitude"] = 1.0e10
config["noise_source"]["component_dist_source"] = 0
config["noise_source"]["component_wavefield"] = 0
config["simulation"]["recording"] = "u_ELASTIC"
config["noise_source"]["filter_spec"] = [2.0, 4.0]
config["simulation"]["absorbing"]["boundaries"] = None  # "r0,p0,p1,t0,t1"
config["simulation"]["absorbing"]["axis-aligned"] = False
config["simulation"]["spherical"] = True

loc_receivers = []
n_sources = 1
with open(DIR_PROJECT / "arrays.csv", mode="r") as fh:
    for i_src in range(n_sources):
        loc_receivers.append([])
        for line in fh:
            station_part = [float(item)
                            for item in line.strip().split(",")[0:2]]
            station = [station_part[1], station_part[0], 0.0]
            loc_receivers[i_src].append(station)
        fh.seek(0)

loc_sources = []
source_list = [0]
assert len(source_list) == n_sources
for i_src in range(n_sources):
    loc_sources.append(loc_receivers[i_src].pop(source_list[i_src]))

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
                                    config=config, type="scorr")

#####################################
##########  NOISE SOURCE   ##########
#####################################

# save load source configuration
config_noise_source = scorr_extensions.load_configuration(filename=config["working_dir_local"] / "config" /
                                                          "noise_source.json", type="noise_source")
# config_noise_source["type"] = "homogeneous"
config_noise_source["type"] = "gaussian"

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
site = {"site": "supermuc",
        "ranks_salvus": 192,
        "ranks_scorr": 192,
        "ping_interval_in_seconds": 60,
        "wall_time_in_seconds_salvus": 3600,
        "wall_time_in_seconds_scorr": 1800}

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
config_measurement["min_period_in_s"] = 100.0
config_measurement["scale"] = 1e10
config_measurement["snr"] = None
config_measurement["surface_wave_velocity_in_mps"] = 3700.0
# config_measurement["surface_wave_velocity_in_mps"] = 4000.0
# config_measurement["window_halfwidth_in_sec"] = 600.0
config_measurement["window_halfwidth_in_sec"] = 15.0
config_measurement["number_of_stacked_windows_min"] = 320
config_measurement["station_list"] = None # ["AA.r98"]

# save measurement configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "measurement.json",
                                    config=config_measurement, type="measurement")

#####################################
########   RUN PREPARATION   ########
#####################################

setup_noise_source(config=config, site=site,
                   config_noise_source=config_noise_source)
preparation.prepare_source_and_receiver(config=config, identifier_prefix="syn",
                                        loc_sources=loc_sources, loc_receivers=loc_receivers,
                                        stf_type="gaussian")
