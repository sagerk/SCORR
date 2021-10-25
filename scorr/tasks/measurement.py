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
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pyasdf
import toml
from obspy.core.trace import Trace
from prettytable import PrettyTable

import scorr.addons
import scorr.extensions.misfits
from scorr.extensions import salvus_extensions, rotation_distance, misfit_helpers


def make_measurement(config: Dict, config_measurement: Dict,
                     ref_identifier: str, src_toml: Union[str, Path], rec_toml: Union[str, Path],
                     filename_synthetics: Union[str, Path], filename_observations: Union[str, Path] = None,
                     data_lasif=False, filename_perturbations: Union[str, Path] = None, order: int = 1, add_suffix=""):
    # load synthetic, observed correlations and perturbations
    ds_synthetic = pyasdf.ASDFDataSet(str(filename_synthetics))
    if filename_observations:
        ds_observed = pyasdf.ASDFDataSet(str(filename_observations))

    if filename_perturbations:
        ds_perturbed = pyasdf.ASDFDataSet(str(filename_perturbations))

    # load receiver toml file
    with open(rec_toml, "r") as fh:
        rec_toml_content = toml.load(fh)

    # load source toml file
    with open(src_toml, "r") as fh:
        src_toml_content = toml.load(fh)

    # get source location
    with pyasdf.ASDFDataSet(src_toml_content["source_input_file"]) as fh_src:
        loc_src = fh_src.auxiliary_data.AdjointSources[
            src_toml_content["source"][0]["dataset_name"].split("/")[-1]].parameters["location"]

    if config["simulation"]["spherical"]:
        src_lat, src_lon, src_r = rotation_distance.from_cartesian_to_latlon_r(
            x=loc_src[0], y=loc_src[1], z=loc_src[2])

    # open log table
    log_table = PrettyTable()
    if config["simulation"]["spherical"]:
        log_table.field_names = ["ref_station", "receiver", "src_lat", "src_lon", "src_r_in_km",
                                 "rec_lat", "rec_long", "rec_r_in_km", "distance_in_km", "misfit", "error_msg"]
    else:
        log_table.field_names = ["ref_station", "receiver", "src_x", "src_y", "src_z",
                                 "rec_x", "rec_y", "rec_z", "distance_in_km", "misfit", "error_msg"]
    log_table.align["ref_station"] = "l"
    log_table.align["receiver"] = "l"
    log_table.align["error_msg"] = "l"

    # track windows and check if it was used before
    ref_station_windows = {}
    ref_station_windows_previous = {}
    for waveform in ds_synthetic.waveforms.list():
        ref_station_windows[waveform] = None
        ref_station_windows_previous[waveform] = None
    id = int(ref_identifier.split("_")[-1])
    if os.path.exists(config["working_dir_local"] / "measurements" / ("_".join(ref_identifier.split("_")[:-1])) / (
            str(id - 1) + ".json")):
        ref_station_windows_previous = scorr.addons.load_json_file(
            config["working_dir_local"] / "measurements" / ("_".join(ref_identifier.split("_")[:-1])) / (
                str(id - 1) + ".json"))

    # loop through receivers
    j_total = 0.0
    j_total_clean = 0.0
    adstf_container = []
    identifier_container = []
    loc_rec_container = []

    # write traveltime misfit to file for Ghetto Inversion
    # fh = open(ref_identifier + ".txt", mode="w")
    fh = tempfile.TemporaryFile(mode="w")

    for waveform in ds_synthetic.waveforms.list():
        # check if it is the autocorrelation
        if waveform == ".".join(ref_identifier.split("_")[2:-1]):
            print(f"Is it an autocorrelation? \n"
                  f"\tWaveform: {waveform} \n"
                  f"\tReference station: {ref_identifier}")
            continue

        # if waveform == "IU.HRV":
        #     print(waveform)
        # else:
        #     continue

        # check if measurement should only be performed for specific stations
        if config_measurement["station_list"]:
            if waveform not in config_measurement["station_list"]:
                log_table.add_row([ref_identifier, waveform, "", "", "", "", "", "", "", "",
                                   "Not in given station_list."])
                continue
            else:
                print(f"measure {waveform}")

        # get receiver location from toml file
        for item in rec_toml_content["receiver"]:
            if item["network"] == waveform.split(".")[0] and item["station"] == waveform.split(".")[1]:
                loc_rec_container.append(item["salvus_coordinates"])
                break
        else:
            print(f"No entry for {waveform} found in toml file!")
            if config["simulation"]["spherical"]:
                log_table.add_row([ref_identifier, waveform, src_lat, src_lon, src_r / 1e3, "", "", "", "", "",
                                   "No entry in toml file."])
            else:
                log_table.add_row([ref_identifier, waveform, loc_src[0], loc_src[1], loc_src[2], "", "", "", "", "",
                                   "No entry in toml file."])
            continue

        # compute distance to source
        distance_in_m = rotation_distance.compute_distance_in_m(loc_src=loc_src, loc_rec=loc_rec_container[-1],
                                                                spherical=config["simulation"]["spherical"])

        if config["simulation"]["spherical"]:
            rec_lat, rec_lon, rec_r = rotation_distance.from_cartesian_to_latlon_r(x=loc_rec_container[-1][0],
                                                                                   y=loc_rec_container[-1][1],
                                                                                   z=loc_rec_container[-1][2])

        # load synthetics
        if "u_ELASTIC" in config["simulation"]["recording"]:
            correlation_synthetic = ds_synthetic.waveforms[waveform].displacement[
                config_measurement["component_recording"]]
        elif config["simulation"]["recording"] == "grad":
            correlation_synthetic = ds_synthetic.waveforms[waveform].gradient[
                config_measurement["component_recording"]]
        else:
            raise ValueError(f"recording type {config['simulation']['recording']} not implemented!\n"
                             f"Available options: [u_ELASTIC, grad]")

        # load observations
        if filename_observations:
            if config["simulation"]["recording"] == "u_ELASTIC":
                if data_lasif:
                    groups = ds_observed.auxiliary_data.CrossCorrelation.list()
                    groups_clean = set([".".join(re.findall(
                        r"""^([a-zA-Z]{1,2})_([a-zA-Z0-9]{1,4})""", group)[0]) for group in groups])

                    # make sure that file contains only one reference station
                    assert len(groups_clean) == 1, \
                        f"File {ds_observed.filename} contains waveforms for other reference stations!"

                    # ref_identifier has to be in groups_clean
                    assert ".".join(ref_identifier.split("_")[2:-1]) in groups_clean, \
                        f"Could not find {'.'.join(ref_identifier.split('_')[2:-1])} in file {ds_observed.filename}!"

                    # loop over groups and search station_id
                    for group in groups:
                        correlations = ds_observed.auxiliary_data.CrossCorrelation[group].list(
                        )
                        correlations_clean = [".".join(re.findall(
                            r"""^([a-zA-Z]{1,2})_([a-zA-Z0-9]{1,4})""", waveform)[0]) for waveform in correlations]

                        if waveform in correlations_clean:
                            index = correlations_clean.index(waveform)
                            break
                    else:
                        print(
                            f"Could not find {waveform} in {ds_observed.filename}.")
                        loc_rec_container.pop()
                        log_table.add_row([ref_identifier, waveform, src_lat, src_lon, src_r / 1e3,
                                           rec_lat, rec_lon, rec_r / 1e3, distance_in_m / 1e3, "",
                                           "No waveform in pyasdf data file."])
                        continue

                    # get waveform data and extract parameters
                    correlation_observed_aux = ds_observed.auxiliary_data.CrossCorrelation[
                        group][correlations[index]]
                    parameters = correlation_observed_aux.parameters

                    # only take correlations with a minimum number of stacked windows
                    if config_measurement["number_of_stacked_windows_min"]:
                        if parameters["number_of_stacked_windows"] < config_measurement[
                                "number_of_stacked_windows_min"]:
                            print(
                                f"Not enough stacked windows {parameters['number_of_stacked_windows']} for {waveform}")
                            loc_rec_container.pop()
                            log_table.add_row(
                                [ref_identifier, waveform, src_lat, src_lon, src_r / 1e3, rec_lat, rec_lon, rec_r / 1e3,
                                 distance_in_m / 1e3, "",
                                 f"Not enough stacked windows: {parameters['number_of_stacked_windows']}"])
                            continue

                    # I think that Laura's definition of the correlation function is time reversed compared to mine
                    correlation_observed = Trace(
                        data=np.flipud(correlation_observed_aux.data[:]))

                    # pad, interpolate and filter trace to match synthetics
                    correlation_observed.stats.starttime = parameters["minlag"]
                    correlation_observed.stats.delta = parameters["dt"]
                    correlation_observed.trim(starttime=correlation_synthetic.stats.starttime,
                                              endtime=correlation_synthetic.stats.endtime,
                                              pad=True, fill_value=0.0)
                    correlation_observed.interpolate(sampling_rate=correlation_synthetic.stats.sampling_rate,
                                                     method="lanczos", window="blackman", a=12)

                    assert config["noise_source"]["filter_spec"], "Need a filter for data!"
                else:
                    correlation_observed = ds_observed.waveforms[waveform].displacement[
                        config_measurement["component_recording"]]

            elif config["simulation"]["recording"] == "grad":
                correlation_observed = ds_observed.waveforms[waveform].gradient[
                    config_measurement["component_recording"]]
            else:
                raise ValueError(f"recording type {config['simulation']['recording']} not implemented!\n"
                                 f"Available options: [u_ELASTIC, grad]")
        else:
            correlation_observed = deepcopy(correlation_synthetic)
            correlation_observed.data = 0.0 * correlation_observed.data

        # load perturbations
        if filename_perturbations:
            correlation_perturbed = ds_perturbed.waveforms[waveform].displacement[
                config_measurement["component_recording"]]
        else:
            correlation_perturbed = deepcopy(correlation_synthetic)
            correlation_perturbed.data = 0.0 * correlation_perturbed.data

        # filter observations, synthetics and perturbations
        if config["noise_source"]["filter_spec"]:
            correlation_observed.filter(type="bandpass",
                                        freqmin=min(
                                            config["noise_source"]["filter_spec"]),
                                        freqmax=max(
                                            config["noise_source"]["filter_spec"]),
                                        zerophase=True, corners=5)
            correlation_synthetic.filter(type="bandpass",
                                         freqmin=min(
                                             config["noise_source"]["filter_spec"]),
                                         freqmax=max(
                                             config["noise_source"]["filter_spec"]),
                                         zerophase=True, corners=5)
            correlation_perturbed.filter(type="bandpass",
                                         freqmin=min(
                                             config["noise_source"]["filter_spec"]),
                                         freqmax=max(
                                             config["noise_source"]["filter_spec"]),
                                         zerophase=True, corners=5)

        if data_lasif:
            # normalize data and synthetics
            correlation_synthetic = correlation_synthetic.normalize()
            correlation_observed = correlation_observed.normalize()

        # compute misfit and adjoint source time function for receiver
        if config["simulation"]["spherical"]:
            fh.write(waveform + " " + str(src_lat) + " " + str(src_lon) +
                     " " + str(rec_lat) + " " + str(rec_lon) + " ")
        j, adstf, message_measurement = scorr.extensions.misfits.make_measurement(config_measurement=config_measurement,
                                                                                  u=correlation_synthetic.data,
                                                                                  u_0=correlation_observed.data,
                                                                                  du=correlation_perturbed.data,
                                                                                  starttime=-
                                                                                  config["simulation"]["corr_max_lag"],
                                                                                  dt=config["simulation"]["dt"],
                                                                                  distance_in_m=distance_in_m,
                                                                                  order=order, fh=fh)
        fh.write("\n")

        # if j and adstf are None, the measurement was rejected
        if j is None and adstf is None:
            print(f"No measurement for {waveform}! " + message_measurement)
            loc_rec_container.pop()
            if ref_station_windows_previous[waveform] is not None:
                j_total_clean += ref_station_windows_previous[waveform]
                message = "Previously possible. " + message_measurement
            else:
                message = message_measurement

            if config["simulation"]["spherical"]:
                log_table.add_row(
                    [ref_identifier, waveform, src_lat, src_lon, src_r / 1e3, rec_lat, rec_lon, rec_r / 1e3,
                     distance_in_m / 1e3, "", message])
            else:
                log_table.add_row([ref_identifier, waveform, loc_src[0], loc_src[1], loc_src[2],
                                   loc_rec_container[-1][0], loc_rec_container[-1][1], loc_rec_container[-1][2],
                                   distance_in_m / 1e3, "", message])

            continue

        # write log file
        if ref_station_windows_previous[waveform] is not None:
            message = message_measurement
        else:
            message = "New measurement. " + message_measurement

        if config["simulation"]["spherical"]:
            log_table.add_row([ref_identifier, waveform, src_lat, src_lon, src_r / 1e3, rec_lat, rec_lon, rec_r / 1e3,
                               distance_in_m / 1e3, j, message])
        else:
            log_table.add_row([ref_identifier, waveform, loc_src[0], loc_src[1], loc_src[2],
                               loc_rec_container[-1][0], loc_rec_container[-1][1], loc_rec_container[-1][2],
                               distance_in_m / 1e3, j, message])

        # sum up total misfit (can be None for second derivatives)
        if j:
            ref_station_windows[waveform] = j
            j_total += j

            if ref_station_windows_previous[waveform] is not None:
                j_total_clean += j

        # assemble 3 component adjoint source time function
        adstf_3C = np.zeros((adstf.shape[0], 3))
        adstf_3C[:, config["simulation"]["green_component"]] = adstf

        # rotate adjoint source time function if necessary
        if config["simulation"]["spherical"]:
            M_inv = rotation_distance.get_transformation_matrix_3d(
                latitude=rec_lat, longitude=rec_lon, inverse=True)
            adstf_3C = rotation_distance.rotate_vector(M_inv, adstf_3C)

        # add adstf to container
        adstf_container.append(adstf_3C)

        # add identifier for each adjoint source (is not used in workflow yet)
        identifier_container.append("adj_" + str(waveform).replace(".", "_"))

    # write log file
    os.makedirs(config["working_dir_local"] / "measurements", exist_ok=True)
    with open(config["working_dir_local"] / "measurements" / (
            "measurement_" + ref_identifier + "_order_" + str(order) + ".txt"),
            "w") as file:
        file.write(log_table.get_string())

    # write file for measurement tracker
    os.makedirs(config["working_dir_local"] / "measurements" / ("_".join(ref_identifier.split("_")[:-1])),
                exist_ok=True)
    scorr.addons.write_json_file(
        config["working_dir_local"] / "measurements" /
        ("_".join(ref_identifier.split("_")[:-1])) / (str(id) + ".json"),
        ref_station_windows)

    # make sure that measurements were taken and that number of adjoint source time functions and receivers match
    assert len(loc_rec_container) > 0, f"No measurements for {ref_identifier}?"
    assert len(loc_rec_container) == len(
        adstf_container), "Number of receiver locations and adjoint source time functions has to be the same!"

    # write adjoint source time functions to file
    adsrc_toml = salvus_extensions.write_stf(
        filename_hdf5=config["working_dir_local"] / "adjoint_stf" / ref_identifier / (
            "adjoint_stf_" + str(order) + "nd" + add_suffix + ".h5"),
        identifier=identifier_container,
        dt=config["simulation"]["dt"],
        starttime_in_s=-config["simulation"]["corr_max_lag"],
        location=loc_rec_container, data=adstf_container)

    fh.close()

    # return configuration
    return j_total, adsrc_toml, j_total_clean
