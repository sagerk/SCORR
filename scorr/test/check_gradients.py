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
import subprocess
from pathlib import Path
from shutil import copy2

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from salvus_model.model import Parameter
from salvus_model.model import SalvusModel

import scorr.addons
import scorr.api
from scorr.addons import group_name
from scorr.extensions import job_tracker, scorr_extensions
from scorr.noise_source.noise_source_setup import setup_noise_source
from scorr.tasks import preparation, sum_misfits_kernels
from scorr.tasks.measurement import make_measurement

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
config["simulation"]["corr_max_lag"] = 2.0
config["simulation"]["corr_max_lag_causal"] = 2.0
config["simulation"]["dt"] = 0.01

config["simulation"]["attenuation"] = False
config["simulation"]["sampling_rate_boundary"] = 1
config["simulation"]["sampling_rate_volume"] = 1
config["noise_source"]["filename"] = config["working_dir_local"] / "noise_source" / "noise_source.h5"

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
site = {"site": "dead",
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
config_measurement["scale"] = 1e10
config_measurement["surface_wave_velocity_in_mps"] = 3000
config_measurement["station_list"] = None

# save measurement configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "measurement.json",
                                    config=config_measurement, type="measurement")

os.makedirs(config["working_dir_local"] / "test_files", exist_ok=True)


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

    # prepare config for synthetics
    config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name
    config["noise_source"]["filename"] = filename_noise_source_backup
    config["inversion"] = inversion_type_backup


def evaluate_misfit_and_gradient(id):
    config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name

    ref_station_dict = scorr_extensions.load_reference_stations(config["working_dir_local"] / "reference_stations.json")
    for identifier in ref_station_dict.keys():
        file_observations = config["working_dir_local"] / "correlations" / "observations" / (
                "obs_test_" + identifier) / "receivers.h5"

        identifier_id = identifier + "_" + str(id)
        job_tracker.remove_reference_station(ref_station=identifier_id)
        j = scorr.api.evaluate_misfit_and_gradient(site=site, config=config, config_measurement=config_measurement,
                                                   ref_identifier=identifier_id,
                                                   src_toml=ref_station_dict[identifier]["src_toml"],
                                                   rec_toml=ref_station_dict[identifier]["rec_toml"],
                                                   file_observations=file_observations,
                                                   gradient=True, data_lasif=False)[0]

        os.makedirs((config["working_dir_local"] / "kernels" / identifier_id / "misfit.txt").parent,
                    exist_ok=True)
        with open(config["working_dir_local"] / "kernels" / identifier_id / "misfit.txt", "w") as fh:
            fh.write(str(j))


def test_gradient_and_compute_correlations_with_random_vector(id):
    config["simulation"]["adjoint"] = False

    # load kernel, generate random vector and compute dot product
    pickle_entries = {}
    pickle_entries["djdm"] = 0.0
    if config["inversion"] == "source" or config["inversion"] == "joint":
        with h5py.File(str(config["working_dir_local"] / "kernels" / ("source_kernel_" + str(id) + ".h5")), 'r') as fkernel:
            g_source = fkernel[group_name + "distribution"][:]

        dm_source = 2.0e12 * (np.random.rand(g_source.shape[0], g_source.shape[1]) - 0.5)
        pickle_entries["djdm"] = np.sum(g_source * dm_source)  # / scale)

    if config["inversion"] == "structure" or config["inversion"] == "joint":
        subprocess.call(["nccopy -k netCDF-4 {} {}".format(
            config["working_dir_local"] / "kernels" / ("structure_kernel_" + str(id) + ".e"),
            config["working_dir_local"] / "kernels" / ("structure_kernel_n_" + str(id) + ".e"))], shell=True)

        with h5py.File(str(config["working_dir_local"] / "kernels" / ("structure_kernel_n_" + str(id) + ".e")), 'r') as fkernel:
            g_structure = fkernel["vals_nod_var1"][:]

        dm_structure = 1e6
        pickle_entries["djdm"] += np.sum(g_structure * dm_structure)  # / scale)

    # create correlations with mh = m + 10^h * dm
    inversion_type_backup = config["inversion"]
    filename_noise_source_backup = config["noise_source"]["filename"]
    mesh_name_step = config["simulation"]["mesh"].parent / (config["simulation"]["mesh"].stem + "_step.e")
    pickle_entries["steps"] = np.arange(-12, 0, 2)
    for step in pickle_entries["steps"]:

        if config["inversion"] == "source" or config["inversion"] == "joint":
            filename_noise_source = config["noise_source"]["filename"].parent / \
                                    (config["noise_source"]["filename"].stem + "_" + str(step) + ".h5")
            copy2(src=config["noise_source"]["filename"], dst=filename_noise_source)
            config["noise_source"]["filename"] = filename_noise_source

            with h5py.File(str(filename_noise_source), "a") as hdf5:
                distribution = hdf5[group_name + "distribution"]
                distribution[:] += dm_source * np.power(10.0, step)

        if config["inversion"] == "structure" or config["inversion"] == "joint":
            model_mesh = SalvusModel.parse(str(config["working_dir_local"] / mesh_name))
            model_mesh.blocks[1].element_parameters[Parameter.ρ] += dm_structure * np.power(10.0, step)
            model_mesh.write(str(mesh_name_step))
            config["simulation"]["mesh"] = mesh_name_step

        # set inversion to "" to save IO
        config["inversion"] = ""

        # compute correlation function
        ref_station_list = scorr_extensions.load_reference_stations(
            config["working_dir_local"] / "reference_stations.json")

        for identifier in ref_station_list.keys():
            identifier_step = "step_" + str(step) + "_" + identifier + "_" + str(id)
            job_tracker.remove_reference_station(ref_station=identifier_step)
            scorr.api.compute_correlations(site=site, config=config,
                                           ref_identifier=identifier_step,
                                           src_toml=ref_station_list[identifier]["src_toml"],
                                           rec_toml=ref_station_list[identifier]["rec_toml"],
                                           output_folder=config["working_dir_local"] / "correlations" / "steps" /
                                                         identifier_step)

        # reload inversion configuration for if clauses above and filename of noise source
        config["inversion"] = inversion_type_backup
        config["noise_source"]["filename"] = filename_noise_source_backup

    # save djdm and steps in pickle file
    with open(config["working_dir_local"] / "test_files" / "test_setup.pickle", "wb") as pfile:
        pickle.dump(pickle_entries, pfile)


def plot_hockey_stick(id):
    # make measurement for orginal data
    ref_station_list = scorr_extensions.load_reference_stations(config["working_dir_local"] / "reference_stations.json")

    with open(config["working_dir_local"] / "kernels" / ("misfit_" + str(id) + ".txt"), "r") as fh:
        j = float(fh.readline())

    # load djdm and steps from pickle file
    with open(config["working_dir_local"] / "test_files" / "test_setup.pickle", "rb") as pfile:
        pickle_entries = pickle.load(pfile)

    # finite difference approximation of change of misfit with correlations computed with mh = m + 10^h * dm
    djdmh = []
    jh_list = []
    for step in pickle_entries["steps"]:
        jh = 0.0
        for identifier in ref_station_list.keys():
            identifier_obs = "obs_test_" + identifier
            identifier_step = "step_" + str(step) + "_" + identifier + "_" + str(id)
            jh += make_measurement(
                config=config, config_measurement=config_measurement,
                ref_identifier=identifier,
                src_toml=ref_station_list[identifier]["src_toml"],
                rec_toml=ref_station_list[identifier]["rec_toml"],
                filename_synthetics=config["working_dir_local"] / "correlations" / "steps" /
                                    identifier_step / "receivers.h5",
                filename_observations=config["working_dir_local"] / "correlations" / "observations" /
                                      identifier_obs / "receivers.h5")[0]

        jh_list.append(jh)
        djdmh.append((jh - j) / np.power(10.0, step))  # / scale)

    print(j, jh_list)
    print(pickle_entries["djdm"], djdmh)

    plt.figure()
    plt.semilogy(np.array(pickle_entries["steps"]),
                 abs(pickle_entries["djdm"] - djdmh) / abs(pickle_entries["djdm"]))
    plt.show()


setup_noise_source(config=config, site=site, config_noise_source=config_noise_source)
preparation.prepare_source_and_receiver(config=config, identifier_prefix="syn_test",
                                        loc_sources=loc_sources, loc_receivers=loc_receivers)
compute_observed_correlations()
evaluate_misfit_and_gradient(id=0)
sum_misfits_kernels.sum_misfits(config=config, id=0)
sum_misfits_kernels.sum_kernels(config=config, id=0)
test_gradient_and_compute_correlations_with_random_vector(id=0)
plot_hockey_stick(id=0)
