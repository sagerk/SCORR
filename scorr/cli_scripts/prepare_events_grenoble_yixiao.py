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
DIR_PROJECT = Path("/scratch/snx3000/sagerk/scorr_test_grenoble")
# DIR_PROJECT = Path("/home/ksager/scorr_test_grenobleX")

# specify mesh
mesh_name = "mesh/reference_0.167s.e"
# mesh_name = "mesh/perturbed_0.167s.e"
# mesh_name = "mesh/reference_0.167s_decrease_constant.e"
# mesh_name = "mesh/reference_0.167s_decrease_between.e"
# mesh_name = "mesh/reference_0.167s_decrease_1gaussian.e"
# mesh_name = "mesh/reference_0.167s_decrease_1gaussian_atsource2.e"

# load and edit configuration
config = scorr_extensions.load_configuration(
    DIR_PROJECT / "config" / "scorr.json", type="scorr")

config["simulation"]["reference_stations"] = config["working_dir_local"] / \
    "reference_stations.json"
config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name

config["working_dir_local"] = DIR_PROJECT
config["simulation"]["green_starttime"] = -1.0
config["simulation"]["corr_max_lag"] = 6.0
config["simulation"]["corr_max_lag_causal"] = 6.0
config["simulation"]["dt"] = 0.001

config["simulation"]["attenuation"] = False
config["simulation"]["sampling_rate_boundary"] = 10
config["simulation"]["sampling_rate_volume"] = 15
config["noise_source"]["filename"] = config["working_dir_local"] / \
    "noise_source" / "noise_source_shifted2.h5"

config["simulation"]["sideset"] = "r1"
config["simulation"]["green_component"] = 0
config["simulation"]["green_amplitude"] = 1.0e10
config["noise_source"]["component_dist_source"] = 0
config["noise_source"]["component_wavefield"] = 0
config["simulation"]["recording"] = "u_ELASTIC"
config["noise_source"]["filter_spec"] = [4.0, 6.0]
config["simulation"]["absorbing"]["boundaries"] = None # "r0,p0,p1,t0,t1"
config["simulation"]["absorbing"]["axis-aligned"] = False
config["simulation"]["spherical"] = True

loc_sources = [[33.6117, -116.4594, 1.0]]
loc_receivers = [[
#    [33.79678, -116.22152, 1.0],
    [33.6107, -116.4555, 1.0],
    [33.605999, -116.454399, 1.0],
    [33.5575, -116.531, 1.0],
    [33.494701, -116.602203, 1.0],
    [33.4722, -116.645, 1.0],
#    [33.35361, -116.86265, 1.0]

# TRAIN LOCATION
#     [33.749785, -116.274012, 1.0],
#
#     [33.751789, -116.278845, 1.0],
#     [33.753730, -116.283776, 1.0],
#     [33.755490, -116.288696, 1.0],
#     [33.757017, -116.293744, 1.0],
#     [33.758549, -116.298860, 1.0],
#     [33.760083, -116.303951, 1.0],
#     [33.761632, -116.309064, 1.0],
#     [33.763165, -116.314172, 1.0],
#     [33.764811, -116.319207, 1.0],
# #     [33.767117, -116.323895, 1.0],
# #     [33.769729, -116.328320, 1.0],
# #     [33.772315, -116.332675, 1.0],
#
#     [33.747784, -116.269271, 1.0],
#     [33.745487, -116.264674, 1.0],
#     [33.743119, -116.260057, 1.0],
#     [33.740817, -116.255445, 1.0],
#     [33.738507, -116.250842, 1.0],
#     [33.736201, -116.246231, 1.0],
#     [33.733891, -116.241603, 1.0],
#     [33.731565, -116.236931, 1.0],
#     [33.729258, -116.232274, 1.0],
# #     [33.726920, -116.227638, 1.0],
# #     [33.724629, -116.223031, 1.0],
# #     [33.722297, -116.218390, 1.0],

# DEPTH
#     [33.6117, -116.4594, 5000.0],
#     [33.6117, -116.4594, 10000.0],
#     [33.6117, -116.4594, 15000.0],
#     [33.6117, -116.4594, 20000.0],
#     [33.6117, -116.4594, 25000.0],

# FAULT
#     [33.521614, -116.570460, 1.0],
#     [33.528332, -116.562708, 1.0],
#     [33.534346, -116.554490, 1.0],
    ]]

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
site = {"site": "daint",
        "ranks_salvus": 192,
        "ranks_scorr": 96,
        "ping_interval_in_seconds": 60,
        "wall_time_in_seconds_salvus": 7200,
        "wall_time_in_seconds_scorr": 900}

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
config_measurement["min_period_in_s"] = None #100.0
config_measurement["scale"] = 1e10
config_measurement["snr"] = None
config_measurement["surface_wave_velocity_in_mps"] = None # 3700.0
# config_measurement["surface_wave_velocity_in_mps"] = 4000.0
# config_measurement["window_halfwidth_in_sec"] = 600.0
config_measurement["window_halfwidth_in_sec"] = None # 15.0
config_measurement["number_of_stacked_windows_min"] = None # 320
config_measurement["station_list"] = ["AA.r3", "AA.r4"]

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
                                        # stf_type="ricker")
                                        stf_type="gaussian")

