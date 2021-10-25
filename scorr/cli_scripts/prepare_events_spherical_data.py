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
from copy import deepcopy

import numpy as np
import pytest
from scorr.extensions import scorr_extensions
from scorr.noise_source.noise_source_setup import setup_noise_source
from scorr.tasks import preparation

# specify where the tests should run
DIR_PROJECT = Path.home() / "scorr_inversion_data_50_final_hessian"

# specify mesh
# mesh_name = "Globe3D_prem_ani_one_crust_100.e"
# mesh_name = "it0000_model_TrRadius_1.788759.e"
# mesh_name = "it0001_model_TrRadius_1.788759.e"
# mesh_name = "it0002_model_TrRadius_1.788759.e"
# mesh_name = "it0003_model_TrRadius_1.788759.e"
# mesh_name = "it0004_model_TrRadius_1.788759.e"
# mesh_name = "it0005_model_TrRadius_1.788759.e"
# mesh_name = "it0006_model_TrRadius_1.788759.e"
mesh_name = "it0007_model_TrRadius_0.700156.e"
# mesh_name = "it0008_model_TrRadius_0.700156.e"

# specify noise source
# noise_source_name = "noise_source_homog.h5"
# noise_source_name = "it0000_model_TrRadius_261.074642.h5"
# noise_source_name = "it0001_model_TrRadius_261.074642.h5"
# noise_source_name = "it0002_model_TrRadius_261.074642.h5"
# noise_source_name = "it0003_model_TrRadius_261.074642.h5"
# noise_source_name = "it0004_model_TrRadius_261.074642.h5"
noise_source_name = "it0005_model_TrRadius_261.074642.h5"


# load and edit configuration
config = scorr_extensions.load_configuration(
    DIR_PROJECT / "config" / "scorr.json", type="scorr")

config["working_dir_local"] = DIR_PROJECT
config["simulation"]["reference_stations"] = config["working_dir_local"] / \
    "reference_stations.json"

config["simulation"]["mesh"] = config["working_dir_local"] / "mesh" / mesh_name
config["noise_source"]["filename"] = config["working_dir_local"] / \
    "noise_source" / noise_source_name

config["simulation"]["green_starttime"] = -400.0
config["simulation"]["corr_max_lag"] = 21600.0
config["simulation"]["corr_max_lag_causal"] = 7000.0
config["simulation"]["dt"] = 0.5

config["simulation"]["kernel-fields"] = "TTI"
config["simulation"]["anisotropy"] = True
config["simulation"]["attenuation"] = True
config["simulation"]["sampling_rate_boundary"] = 20
config["simulation"]["sampling_rate_volume"] = 20

config["simulation"]["sideset"] = "r1"
config["simulation"]["green_component"] = 0
config["simulation"]["green_amplitude"] = 1.0e10
config["noise_source"]["component_dist_source"] = 0
config["noise_source"]["component_wavefield"] = 0
config["simulation"]["recording"] = "u_ELASTIC"
config["noise_source"]["filter_spec"] = [0.0033333, 0.01]
config["simulation"]["absorbing"]["boundaries"] = None
config["simulation"]["absorbing"]["axis-aligned"] = False
config["simulation"]["spherical"] = True

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
        "ranks_scorr": 48,
        "ping_interval_in_seconds": 60,
        "wall_time_in_seconds_salvus": 10800,
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
# config_measurement["type"] = "log_amplitude_ratio"
config_measurement["type"] = "cc_time_shift"
# config_measurement["type"] = "cc_time_asymmetry"
config_measurement["component_recording"] = 2
config_measurement["pick_window"] = True
config_measurement["pick_manual"] = False
config_measurement["min_period_in_s"] = 200.0
config_measurement["scale"] = 1e10
config_measurement["snr"] = 2.0
config_measurement["correlation_coeff"] = 0.6
config_measurement["surface_wave_velocity_in_mps"] = 3700.0
config_measurement["window_halfwidth_in_sec"] = 600.0
config_measurement["number_of_stacked_windows_min"] = 200
config_measurement["station_list"] = None

# save measurement configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "measurement.json",
                                    config=config_measurement, type="measurement")

#####################################
########   RUN PREPARATION   ########
#####################################

setup_noise_source(config=config, site=site,
                   config_noise_source=config_noise_source)
preparation.take_source_receiver_from_lasif_project(
    config=config, identifier_prefix="syn")

# perturbation of source distribution
config_pert = deepcopy(config)
config_pert["noise_source"]["filename"] = config["working_dir_local"] / \
    "noise_source" / "noise_source_pert.h5"
config_noise_source_pert = deepcopy(config_noise_source)
config_noise_source_pert["type"] = "gaussian"
setup_noise_source(config=config_pert, site=site,
                   config_noise_source=config_noise_source_pert)
