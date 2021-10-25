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
import pickle
from copy import deepcopy
from pathlib import Path
from shutil import copy2
import scipy.sparse.linalg as la

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scorr.addons
import scorr.api
from salvus_model.model import Parameter
from salvus_model.model import SalvusModel
from scorr.addons import group_name
from scorr.extensions import job_tracker, job_tracker_hessian, scorr_extensions
from scorr.noise_source.noise_source_setup import setup_noise_source
from scorr.tasks import preparation
from scorr.tasks import sum_misfits_kernels

# specify where the tests should run
DIR_PROJECT = Path.home() / "Desktop" / "SWP_past" / "correlations_3d" / "scorr_test"

# specify mesh
mesh_name = "Hex_IsotropicElastic3D_Elemental_2x2x2.e"
# mesh_name = "Globe3D_3_layer_250.e"

# load and edit configuration
config = scorr_extensions.load_configuration(DIR_PROJECT / "config" / "scorr.json", type="scorr")
config["simulation"]["reference_stations"] = config["working_dir_local"] / "reference_stations.json"
config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name

config["simulation"]["green_starttime"] = -1.0
config["simulation"]["corr_max_lag"] = 4.0
config["simulation"]["corr_max_lag_causal"] = 2.0
config["simulation"]["dt"] = 0.01

config["simulation"]["attenuation"] = False
config["simulation"]["sampling_rate_boundary"] = 1
config["simulation"]["sampling_rate_volume"] = 1
config["noise_source"]["filename"] = config["working_dir_local"] / "noise_source" / "noise_source.h5"

if "Globe" in str(config["simulation"]["mesh"]):
    config["simulation"]["sideset"] = "r1"
    config["simulation"]["green_component"] = 0
    config["simulation"]["green_amplitude"] = 1.0e10
    config["noise_source"]["component_dist_source"] = 0
    config["noise_source"]["component_wavefield"] = 0
    config["simulation"]["recording"] = "u_ELASTIC"
    config["noise_source"]["filter_spec"] = None
    config["simulation"]["absorbing"]["boundaries"] = None
    config["simulation"]["absorbing"]["axis-aligned"] = False
    config["simulation"]["spherical"] = True

    loc_sources = [[0.0, 0.0, 1]]  # , [0.0, 0.0, 2]]
    loc_receivers = [[[0.0, 0.1, 0.0]]]  # , [0.0, 0.1, 1.0]], [[0.1, 0.1, 0.0], [0.1, 0.1, 1.0]]]
else:
    config["simulation"]["sideset"] = "z0"
    config["simulation"]["green_component"] = 2
    config["simulation"]["green_amplitude"] = 1.0e10
    config["noise_source"]["component_dist_source"] = 2
    config["noise_source"]["component_wavefield"] = 2
    config["simulation"]["recording"] = "u_ELASTIC"
    # config["simulation"]["recording"] = "grad"
    config["noise_source"]["filter_spec"] = None
    config["simulation"]["absorbing"]["boundaries"] = None
    config["simulation"]["absorbing"]["axis-aligned"] = False
    config["simulation"]["spherical"] = False

    loc_sources = [[48000.0, 48000.0, 0.0]]
    loc_receivers = [[[52000.0, 52000.0, 0.0]]]
    # loc_sources = [[0.0, 49950.0, 49950.0], [0.0, 50050.0, 49950.0]]
    # loc_receivers = [[[0.0, 50050.0, 50050.0], [0.0, 49950.0, 50050.0]],
    #                  [[0.0, 50150.0, 50150.0], [0.0, 49950.0, 50150.0]]]

# some safety measures, stf generation is not yet general enough
nt_corr = (abs(config["simulation"]["corr_max_lag"]) + abs(config["simulation"]["corr_max_lag_causal"])) / \
          config["simulation"]["dt"]
nt_corr_full = 2 * abs(config["simulation"]["corr_max_lag"]) / config["simulation"]["dt"]
nt_green = abs(config["simulation"]["green_starttime"]) / config["simulation"]["dt"]

# assert np.mod(nt_corr, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr_full, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_green, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr, config["simulation"]["sampling_rate_boundary"]) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr_full, config["simulation"]["sampling_rate_boundary"]) == pytest.approx(0, abs=1.0e-8)
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
site = {"site": "swp",
        "ranks_salvus": 2,
        "ranks_scorr": 2,
        "ping_interval_in_seconds": 1,
        "wall_time_in_seconds_salvus": 1,
        "wall_time_in_seconds_scorr": 1}

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
config_measurement["type"] = "waveform_differences"
# config_measurement["type"] = "log_amplitude_ratio"
config_measurement["component_recording"] = 2
config_measurement["pick_window"] = False
config_measurement["window_halfwidth_in_sec"] = 0.5
config_measurement["pick_manual"] = False
config_measurement["scale"] = 1.0e10
config_measurement["surface_wave_velocity_in_mps"] = 3000
config_measurement["station_list"] = None

# save measurement configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "measurement.json",
                                    config=config_measurement, type="measurement")

os.makedirs(config["working_dir_local"] / "test_files", exist_ok=True)

dm1_structure = 1e10
i_counter_eig = 0


def compute_observed_correlations():
    # reset inversion type such that no forward wavefield is saved
    inversion_type_backup = config["inversion"]
    config["inversion"] = ""

    # change source distribution for observations
    filename_noise_source_backup = config["noise_source"]["filename"]
    filename_noise_source = config["noise_source"]["filename"].parent / \
                            (config["noise_source"]["filename"].stem + "_obs.h5")
    copy2(src=config["noise_source"]["filename"], dst=filename_noise_source)
    config["noise_source"]["filename"] = filename_noise_source

    with h5py.File(str(filename_noise_source), "a") as hdf5:
        distribution = hdf5[group_name + "distribution"]
        distribution[:] += 1.0e10 * (np.random.rand(distribution.shape[0], distribution.shape[1]) - 0.5)

    # change mesh for observations
    model_mesh = SalvusModel.parse(str(config["simulation"]["mesh"]))
    model_mesh.blocks[1].element_parameters[Parameter.ρ] += 111
    mesh_name_obs = config["simulation"]["mesh"].parent / (config["simulation"]["mesh"].stem + "_obs.e")
    model_mesh.write(str(mesh_name_obs))
    config["simulation"]["mesh"] = mesh_name_obs

    # compute observations
    ref_station_list = scorr_extensions.load_reference_stations(config["working_dir_local"] / "reference_stations.json")

    for identifier in ref_station_list.keys():
        # identifier is supposed to be for synthetics, change it here in order to deal with observations
        identifier_obs = "obs_test_" + identifier

        # compute correlations
        job_tracker.remove_reference_station(ref_station=identifier_obs)
        scorr.api.compute_correlations(site=site, config=config,
                                       ref_identifier=identifier_obs,
                                       src_toml=ref_station_list[identifier]["src_toml"],
                                       rec_toml=ref_station_list[identifier]["rec_toml"],
                                       output_folder=config["working_dir_local"] / "correlations" / "observations" /
                                                     identifier_obs)

    # prepare config for Hessian vector product
    config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name
    config["noise_source"]["filename"] = filename_noise_source_backup
    config["inversion"] = inversion_type_backup


def evaluate_hessian_vector_product(dS):
    global i_counter_eig
    print("WUUUHHHA " + str(i_counter_eig))
    i_counter_eig = i_counter_eig + 1

    # lame id
    id = i_counter_eig

    config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name
    ref_station_dict = scorr_extensions.load_reference_stations(config["working_dir_local"] / "reference_stations.json")

    # write dS to file
    with h5py.File(str(config["noise_source"]["filename"]), "a") as hdf5:
        hdf5[group_name + "distribution"][:, :] = np.reshape(dS, (4, 25))

    for identifier in ref_station_dict.keys():
        file_observations = config["working_dir_local"] / "correlations" / "observations" / (
                "obs_test_" + identifier) / "receivers.h5"

        identifier_id = "hessian_" + identifier + "_" + str(id)
        job_tracker.remove_reference_station(ref_station=identifier_id)
        job_tracker.remove_reference_station(ref_station=identifier_id + "_pert")
        job_tracker_hessian.remove_reference_station(ref_station=identifier_id)
        scorr.api.evaluate_hessian_vector_product(site=site, config=config, config_measurement=config_measurement,
                                                  ref_identifier=identifier_id,
                                                  src_toml=ref_station_dict[identifier]["src_toml"],
                                                  rec_toml=ref_station_dict[identifier]["rec_toml"],
                                                  file_observations=file_observations,
                                                  data_lasif=False)

    sum_misfits_kernels.sum_kernels(config=config, id=id, type="hessian", identifier_prefix="hessian_")

    with h5py.File(str(config["working_dir_local"] / "hessian" / ("source_kernel_" + str(id) + ".h5")), 'r') as fkernel:
        Hdm1 = fkernel[group_name + "distribution"][:]

    print(Hdm1.shape)
    return np.reshape(Hdm1, (100, 1))



setup_noise_source(config=config, site=site, config_noise_source=config_noise_source)
# preparation.prepare_source_and_receiver(config=config, identifier_prefix="syn_test",
#                                         loc_sources=loc_sources, loc_receivers=loc_receivers)
# compute_observed_correlations()
#
# # generate noise source file
# config_pert = deepcopy(config)
# config_pert["noise_source"]["filename"] = config["working_dir_local"] / "noise_source" / "noise_source_pert.h5"
# config_noise_source_pert = deepcopy(config_noise_source)
# config_noise_source_pert["type"] = "random"
# setup_noise_source(config=config_pert, site=site, config_noise_source=config_noise_source_pert)
#
# A = la.LinearOperator( (100, 100),  evaluate_hessian_vector_product)
# results = la.eigsh(A, k=2)
#
# np.save(config["working_dir_local"] / "test.npy", results)