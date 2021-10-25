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
import sys
from pathlib import Path

import scorr.api
from scorr.extensions import scorr_extensions
from scorr.tasks import sum_misfits_kernels

skip_measurement = False


def compute_observed_correlations(ref_station_list):
    """
    interesting for synthetic inversions

    """
    # reset inversion type such that no forward wavefield is saved
    inversion_type_backup = config["inversion"]
    config["inversion"] = ""

    # change source distribution for observations
    filename_noise_source_backup = config["noise_source"]["filename"]
    # possibility here to provide different file etc ...
    # better usage: set up config specifically for "observations"

    # load reference stations
    ref_station_dict = scorr_extensions.load_reference_stations(
        config["working_dir_local"] / "reference_stations.json")
    if not ref_station_list:
        ref_station_list = ref_station_dict.keys()

    # compute observations
    for identifier in ref_station_list:
        # identifier is supposed to be for synthetics, change it here in order to deal with observations
        identifier_obs = "obs_" + identifier

        # compute correlations
        scorr.api.compute_correlations(site=site, config=config,
                                       ref_identifier=identifier_obs,
                                       src_toml=ref_station_dict[identifier]["src_toml"],
                                       rec_toml=ref_station_dict[identifier]["rec_toml"],
                                       output_folder=config["working_dir_local"] / "correlations" / "observations" /
                                       identifier_obs)

    # prepare config for synthetics
    config["noise_source"]["filename"] = filename_noise_source_backup
    config["inversion"] = inversion_type_backup


def evaluate_misfit_and_gradient(id, ref_station_list, compute_gradient):
    ref_station_dict = scorr_extensions.load_reference_stations(
        config["working_dir_local"] / "reference_stations.json")
    if not ref_station_list:
        ref_station_list = ref_station_dict.keys()

    for identifier in ref_station_list:
        # if "synthetic" in DIR_PROJECT.name or "test" in DIR_PROJECT.name:
        #     print("\n\nAssume synthetic data!\n")
        #     file_observations = config["working_dir_local"] / "correlations" / "observations" / (
        #         "obs_" + identifier) / "receivers.h5"
        #     data_lasif = False
        # elif "autocorrelation" in DIR_PROJECT.name:
        #     print("\n\nAssume a data independent run!\n")
        #     file_observations = None
        #     data_lasif = False
        # elif "data" in DIR_PROJECT.name:
        #     print("\n\nAssume actual data!\n")
        #     file_observations = config["working_dir_local"] / "correlations" / "observations" / \
        #         (".".join(item for item in identifier.split(
        #             "_")[1:]) + "_winter.h5")
        #     data_lasif = True
        # else:
        #     raise RuntimeError("\nYou have to specify a data file!\n")

        file_observations = None
        data_lasif = False

        identifier_id = identifier + "_" + str(id)
        j, j_clean = scorr.api.evaluate_misfit_and_gradient(site=site, config=config,
                                                            config_measurement=config_measurement,
                                                            ref_identifier=identifier_id,
                                                            src_toml=ref_station_dict[identifier]["src_toml"],
                                                            rec_toml=ref_station_dict[identifier]["rec_toml"],
                                                            file_observations=file_observations,
                                                            gradient=compute_gradient,
                                                            data_lasif=data_lasif,
                                                            skip_measurement=skip_measurement)

        if not skip_measurement:
            os.makedirs((config["working_dir_local"] / "kernels" / identifier_id / "misfit.txt").parent,
                        exist_ok=True)
            with open(config["working_dir_local"] / "kernels" / identifier_id / "misfit.txt", "w") as fh:
                fh.write(str(j))
            with open(config["working_dir_local"] / "kernels" / identifier_id / "misfit_clean.txt", "w") as fh:
                fh.write(str(j_clean))


def evaluate_hessian_vector_product(id, ref_station_list):
    ref_station_dict = scorr_extensions.load_reference_stations(
        config["working_dir_local"] / "reference_stations.json")

    if not ref_station_list:
        ref_station_list = ref_station_dict.keys()

    if config["hessian"] == "dS":
        if not (config["working_dir_local"] / "noise_source" / "noise_source_pert.h5").exists():
            raise FileNotFoundError(
                "Noise source perturbation file does not exist!")
    else:
        raise NotImplementedError(
            "dm is currently not implemented for large domains!")

    for identifier in ref_station_list:
        if "synthetic" in DIR_PROJECT.name or "test" in DIR_PROJECT.name:
            print("\n\nAssume synthetic data!\n")
            file_observations = config["working_dir_local"] / "correlations" / "observations" / (
                "obs_" + identifier) / "receivers.h5"
            data_lasif = False
        elif "data" in DIR_PROJECT.name:
            print("\n\nAssume actual data!\n")
            file_observations = config["working_dir_local"] / "correlations" / "observations" / \
                (".".join(item for item in identifier.split(
                    "_")[1:]) + "_winter.h5")
            data_lasif = True
        else:
            raise RuntimeError("\nYou have to specify a data file!\n")

        identifier_id = "hessian_" + identifier + "_" + str(id)
        scorr.api.evaluate_hessian_vector_product(site=site, config=config,
                                                  config_measurement=config_measurement,
                                                  ref_identifier=identifier_id,
                                                  src_toml=ref_station_dict[identifier]["src_toml"],
                                                  rec_toml=ref_station_dict[identifier]["rec_toml"],
                                                  file_observations=file_observations,
                                                  data_lasif=data_lasif)


if len(sys.argv) <= 3:
    sys.exit(
        "Three arguments are required! DIR_PROJECT id && [observations, misfit, misfit_and_gradient, hessian, sum_misfits, sum_kernels, sum_kernels_hessian]")

# input arguments decide what has to be computed
DIR_PROJECT = Path(str(sys.argv[1]))
id = str(sys.argv[2])
run_type = str(sys.argv[3])
ref_station_list = sys.argv[4:]

if run_type not in ["observations", "misfit", "misfit_and_gradient", "hessian", "sum_misfits", "sum_kernels", "sum_kernels_hessian"]:
    raise NotImplementedError(
        "Options for the third argument are [observations, misfit, misfit_and_gradient, hessian, sum_misfits, sum_kernels, sum_kernels_hessian]!")

# load configurations
config = scorr_extensions.load_configuration(
    DIR_PROJECT / "config" / "scorr.json", type="scorr")
site = scorr_extensions.load_configuration(
    DIR_PROJECT / "config" / "site.json", type="site")
config_measurement = scorr_extensions.load_configuration(DIR_PROJECT / "config" / "measurement.json",
                                                         type="measurement")

if run_type == "observations":
    compute_observed_correlations(ref_station_list=ref_station_list)

if run_type == "misfit":
    evaluate_misfit_and_gradient(
        id=id, ref_station_list=ref_station_list, compute_gradient=False)
elif run_type == "misfit_and_gradient":
    evaluate_misfit_and_gradient(
        id=id, ref_station_list=ref_station_list, compute_gradient=True)
elif run_type == "hessian":
    evaluate_hessian_vector_product(id=id, ref_station_list=ref_station_list)

elif run_type == "sum_misfits":
    if len(sys.argv[4:]) and str(sys.argv[4:][0]) == "clean":
        sum_misfits_kernels.sum_misfits(
            config=config, id=id, clean=True, ref_station_list=ref_station_list[1:])
    else:
        sum_misfits_kernels.sum_misfits(
            config=config, id=id, clean=False, ref_station_list=ref_station_list)

elif run_type == "sum_kernels":
    sum_misfits_kernels.sum_kernels(
        config=config, id=id, ref_station_list=ref_station_list)
elif run_type == "sum_kernels_hessian":
    sum_misfits_kernels.sum_kernels(
        config=config, id=id, ref_station_list=ref_station_list,
        type="hessian", identifier_prefix="hessian_")
