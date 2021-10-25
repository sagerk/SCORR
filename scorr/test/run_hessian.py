#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import sys
import os
from copy import deepcopy
from pathlib import Path
from shutil import copy2

import h5py
import numpy as np
import pytest
from salvus_model.model import Parameter
from salvus_model.model import SalvusModel

import scorr.addons
import scorr.api
from scorr.addons import group_name
from scorr.extensions import job_tracker, scorr_extensions
from scorr.noise_source.noise_source_setup import setup_noise_source
from scorr.tasks import preparation
from scorr.tasks import sum_misfits_kernels

# specify where the tests should run
# DIR_PROJECT = Path.home() / "Desktop" / "scorr_test_hessian"
DIR_PROJECT = Path.home() / "scorr_test_hessian"

# specify mesh
mesh_name = "homog_80x80x20.e"

# load and edit configuration
config = scorr_extensions.load_configuration(
    DIR_PROJECT / "config" / "scorr.json", type="scorr")
config["working_dir_local"] = DIR_PROJECT
config["simulation"]["reference_stations"] = config["working_dir_local"] / \
    "reference_stations.json"
config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name

config["simulation"]["green_starttime"] = -3.0
config["simulation"]["corr_max_lag"] = 25.0
config["simulation"]["corr_max_lag_causal"] = 25.0
config["simulation"]["dt"] = 0.01

config["simulation"]["attenuation"] = False
config["simulation"]["sampling_rate_boundary"] = 1
config["simulation"]["sampling_rate_volume"] = 20
config["noise_source"]["filename"] = config["working_dir_local"] / \
    "noise_source" / "noise_source.h5"

config["simulation"]["sideset"] = "z0"
config["simulation"]["green_component"] = 2
config["simulation"]["green_amplitude"] = 1.0e10
config["noise_source"]["component_dist_source"] = 2
config["noise_source"]["component_wavefield"] = 2
config["simulation"]["recording"] = "u_ELASTIC"
config["noise_source"]["filter_spec"] = [0.8, 1.4]
config["simulation"]["absorbing"]["boundaries"] = "z1,x0,x1,y0,y1"
config["simulation"]["absorbing"]["axis-aligned"] = True
config["simulation"]["spherical"] = False

loc_sources = [[37000.0, 50000.0, 1.0]]
loc_receivers = [[[63000.0, 50000.0, 1.0]]]

# some safety measures, stf generation is not yet general enough
nt_corr = (abs(config["simulation"]["corr_max_lag"]) + abs(config["simulation"]["corr_max_lag_causal"])) / \
    config["simulation"]["dt"]
nt_corr_full = 2 * abs(config["simulation"]
                       ["corr_max_lag"]) / config["simulation"]["dt"]
nt_green = abs(config["simulation"]["green_starttime"]) / \
    config["simulation"]["dt"]

# assert np.mod(nt_corr, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr_full, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_green, 1) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr, config["simulation"]
              ["sampling_rate_boundary"]) == pytest.approx(0, abs=1.0e-8)
assert np.mod(nt_corr_full, config["simulation"]
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
config_noise_source["type"] = "homogeneous"
# config_noise_source["type"] = "gaussian"
# config_noise_source["type"] = "ring"

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
        "ranks_salvus": 72,
        "ranks_scorr": 24,
        "ping_interval_in_seconds": 60,
        "wall_time_in_seconds_salvus": 1800,
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
config_measurement["type"] = "waveform_differences"
# config_measurement["type"] = "energy_difference"
# config_measurement["type"] = "envelope_difference"
# config_measurement["type"] = "envelope_squared_difference"
# config_measurement["type"] = "log_amplitude_ratio"
# config_measurement["type"] = "cc_time_shift"
# config_measurement["type"] = "cc_time_asymmetry"

config_measurement["component_recording"] = 2
config_measurement["pick_window"] = False
config_measurement["window_halfwidth_in_sec"] = 3.5
config_measurement["pick_manual"] = False
config_measurement["scale"] = 1.0e10
config_measurement["surface_wave_velocity_in_mps"] = 3700
config_measurement["station_list"] = None
config_measurement["snr"] = 0.0
config_measurement["correlation_coeff"] = 0.0

# save measurement configuration
scorr_extensions.save_configuration(filename=config["working_dir_local"] / "config" / "measurement.json",
                                    config=config_measurement, type="measurement")


def compute_observed_correlations():
    # reset inversion type such that no forward wavefield is saved
    inversion_type_backup = config["inversion"]
    config["inversion"] = ""

    # # change source distribution for observations
    filename_noise_source_backup = config["noise_source"]["filename"]
    filename_noise_source_obs = config["noise_source"]["filename"].parent / \
        (config["noise_source"]["filename"].stem + "_obs.h5")
    copy2(src=config["noise_source"]["filename"],
          dst=filename_noise_source_obs)
    config["noise_source"]["filename"] = filename_noise_source_obs

    # write new xdmf file
    xdmf_file_new = copy2(src=str(filename_noise_source_backup.parent / (filename_noise_source_backup.stem + ".xdmf")),
                          dst=str(filename_noise_source_obs.parent / (filename_noise_source_obs.stem + ".xdmf")))
    with open(str(xdmf_file_new), 'r') as file:
        filedata = file.read()
    filedata = filedata.replace(
        str(filename_noise_source_backup.name), str(filename_noise_source_obs.name))
    with open(str(xdmf_file_new), 'w') as file:
        file.write(filedata)

    # create source distribution for observations
    with h5py.File(str(filename_noise_source_obs), "a") as hdf5:
        coordinates = hdf5[group_name + "coordinates"][:]
        distribution = hdf5[group_name + "distribution"]
        # distribution[:] += 1.0e9 * (np.random.rand(distribution.shape[0], distribution.shape[1]) - 0.5)

        for id_element_local in range(coordinates.shape[0]):
            for id_gll in range(coordinates.shape[1]):
                # distribution[id_element_local, id_gll] = 1.0 * \
                #     config_noise_source["homog_magnitude"]
                # gauss_magnitude = config_noise_source["gaussian"]["magnitude"]
                # gauss_x = config_noise_source["gaussian"]["x"]
                # gauss_y = config_noise_source["gaussian"]["y"]
                # gauss_width_x = config_noise_source["gaussian"]["width_x"]
                # gauss_width_y = config_noise_source["gaussian"]["width_y"]

                # if config["simulation"]["spherical"] == False:
                #     index_coord = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
                #     if config["simulation"]["sideset"][0] == "x":
                #         id_component = 0
                #     elif config["simulation"]["sideset"][0] == "y":
                #         id_component = 1
                #     else:
                #         id_component = 2

                #     for _i in range(len(gauss_magnitude)):
                #         distribution[id_element_local, id_gll] += 2.0 * gauss_magnitude[_i] * np.exp(-(
                #             (coordinates[id_element_local, id_gll,
                #                          index_coord[id_component][0]] - gauss_x[_i]) ** 2
                #             / gauss_width_x[_i] ** 2 +
                #             (coordinates[id_element_local, id_gll,
                #                          index_coord[id_component][1]] - gauss_y[_i]) ** 2
                #             / gauss_width_y[_i] ** 2))

                x, y, z = coordinates[id_element_local, id_gll, :]

                # ring parameters are currently hard-coded
                ring_center_x = 0.5e5
                ring_center_y = 0.5e5
                ring_radius = 0.3e5
                ring_thickness = 0.2e5
                ring_radial_smoothing = 0.1e4
                ring_angle_center = -90.0
                ring_angle_coverage = 60.0
                ring_angle_smoothing = 15.0
                # previously: 2.0

                radius = np.sqrt((x-ring_center_x)**2 + (y-ring_center_y)**2)
                angle = np.rad2deg(np.arctan2(
                    x-ring_center_x, y-ring_center_y))
                if np.isnan(angle):
                    continue

                # construct ring
                radial_taper = 2.0e10 * np.exp(-np.abs(radius-ring_radius)
                                               ** 2 / ring_radial_smoothing**2)
                radial_pattern = float(radius > (
                    ring_radius-ring_thickness/2) and radius < (ring_radius+ring_thickness/2))
                distribution[id_element_local,
                             id_gll] = radial_taper * radial_pattern

                # select angle
                angle_taper = np.exp(-(angle-ring_angle_center)
                                     ** 2 / ring_angle_smoothing**2)
                angle_pattern = float(
                    angle > ring_angle_center - ring_angle_coverage and angle < ring_angle_center + ring_angle_coverage)
                distribution[id_element_local,
                             id_gll] *= angle_taper * angle_pattern

                # add homogeneous level
                distribution[id_element_local, id_gll] += 1.0 * \
                    config_noise_source["homog_magnitude"]

    # change mesh for observations
    model_mesh = SalvusModel.parse(str(config["simulation"]["mesh"]))
    # model_mesh.blocks[1].element_parameters[Parameter.ρ][:, 4] += 1.1
    # model_mesh.blocks[1].element_parameters[Parameter.μ][:] *= 1.04
    model_mesh.blocks[1].element_parameters[Parameter.μ][:] *= 1.0201
    model_mesh.blocks[1].element_parameters[Parameter.λ][:] *= 1.0201
    mesh_name_obs = config["simulation"]["mesh"].parent / \
        (config["simulation"]["mesh"].stem + "_obs.e")
    model_mesh.write(str(mesh_name_obs))
    config["simulation"]["mesh"] = mesh_name_obs

    # compute observations
    ref_station_list = scorr_extensions.load_reference_stations(
        config["working_dir_local"] / "reference_stations.json")

    for identifier in ref_station_list.keys():
        # identifier is supposed to be for synthetics, change it here in order to deal with observations
        identifier_obs = "obs_test_" + identifier

        # job_tracker.remove_reference_station(ref_station=identifier_obs)

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


def evaluate_hessian_vector_product(id):
    config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name
    ref_station_dict = scorr_extensions.load_reference_stations(
        config["working_dir_local"] / "reference_stations.json")

    if config["hessian"] == "dS":
        # perturbation of source distribution
        config_pert = deepcopy(config)
        config_pert["noise_source"]["filename"] = config["working_dir_local"] / \
            "noise_source" / "noise_source_pert.h5"
        config_noise_source_pert = deepcopy(config_noise_source)
        config_noise_source_pert["type"] = "ring"
        setup_noise_source(config=config_pert, site=site,
                           config_noise_source=config_noise_source_pert)
    else:
        # perturb mesh
        mesh_name_pert = config["simulation"]["mesh"].parent / \
            (config["simulation"]["mesh"].stem + "_pert.e")
        model_mesh = SalvusModel.parse(
            str(config["working_dir_local"] / mesh_name))

        n_elements_h = 80
        i_h = [39 * n_elements_h + 39, 40 * n_elements_h + 39,
               39 * n_elements_h + 40, 40 * n_elements_h + 40]
        i_v = [0 * n_elements_h ** 2, 1 * n_elements_h ** 2]
        model_mesh.blocks[1].element_parameters[Parameter.μ][:,
                                                             [index + offset for offset in i_v for index in i_h]] *= 1.96

        model_mesh.write(str(mesh_name_pert))

    for identifier in ref_station_dict.keys():
        file_observations = config["working_dir_local"] / "correlations" / "observations" / (
            "obs_test_" + identifier) / "receivers.h5"

        identifier_id = "hessian_" + identifier + "_" + str(id)

        # job_tracker.remove_reference_station(ref_station=identifier_id)
        # job_tracker.remove_reference_station(ref_station=identifier_id + "_pert")
        # job_tracker_hessian.remove_reference_station(ref_station=identifier_id)

        scorr.api.evaluate_hessian_vector_product(site=site, config=config, config_measurement=config_measurement,
                                                  ref_identifier=identifier_id,
                                                  src_toml=ref_station_dict[identifier]["src_toml"],
                                                  rec_toml=ref_station_dict[identifier]["rec_toml"],
                                                  file_observations=file_observations,
                                                  data_lasif=False)

    sum_misfits_kernels.sum_kernels(
        config=config, id=id, type="hessian", identifier_prefix="hessian_")


def evaluate_misfit_and_gradient(id):
    config["simulation"]["mesh"] = config["working_dir_local"] / mesh_name

    ref_station_dict = scorr_extensions.load_reference_stations(
        config["working_dir_local"] / "reference_stations.json")
    for identifier in ref_station_dict.keys():
        file_observations = config["working_dir_local"] / "correlations" / "observations" / (
            "obs_test_" + identifier) / "receivers.h5"

        identifier_id = "gradient_" + identifier + "_" + str(id)

        # job_tracker.remove_reference_station(ref_station=identifier_id)

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

    sum_misfits_kernels.sum_misfits(
        config=config, id=id, identifier_prefix="gradient_")
    sum_misfits_kernels.sum_kernels(
        config=config, id=id, identifier_prefix="gradient_")


setup_noise_source(config=config, site=site,
                   config_noise_source=config_noise_source)
preparation.prepare_source_and_receiver(config=config, identifier_prefix="syn_test3",
                                        loc_sources=loc_sources, loc_receivers=loc_receivers)
compute_observed_correlations()

ref_station_dict = scorr_extensions.load_reference_stations(
    config["working_dir_local"] / "reference_stations.json")
measurement_types = ["waveform_differences", ]
# "log_amplitude_ratio",
# "cc_time_shift",
# "cc_time_asymmetry",
# "energy_difference",
# "envelope_difference",
# "envelope_squared_difference"]

id_start = 200
start = True

id = id_start
for type in measurement_types:
    print("\n#############################################")
    print(f"     START WITH {type}")
    print("#############################################\n")
    if not start:
        for identifier in ref_station_dict.keys():
            job_tracker.copy_reference_station("hessian_" + identifier + "_" + str(id_start),
                                               "hessian_" + identifier + "_" + str(id))
            job_tracker.reset_jobs_of_reference_station("hessian_" + identifier + "_" + str(id),
                                                        job_id_adjoint_1=True)
            job_tracker.copy_reference_station("hessian_" + identifier + "_" + str(id_start) + "_pert",
                                               "hessian_" + identifier + "_" + str(id) + "_pert")

    config_measurement["type"] = type
    evaluate_hessian_vector_product(id=id)

    start = False
    id += 1

# evaluate_misfit_and_gradient(id=100)
