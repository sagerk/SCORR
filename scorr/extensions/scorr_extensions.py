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

import toml

import scorr.addons
from scorr.addons import load_json_file, rank

REFERENCE_scorr = {
    "working_dir_local": Path,
    "simulation": {
        "green_amplitude": (int, float), "green_component": int, "green_starttime": (int, float),
        "corr_max_lag": (int, float), "corr_max_lag_causal": (int, float, type(None)),
        "dt": (int, float),
        "mesh": Path, "sideset": str, "spherical": bool, "anisotropy": bool,
        "recording": str,
        "reference_stations": Path,
        "precision": str,
        "adjoint": bool, "attenuation": bool, "kernel-fields": (str, type(None)),
        "sampling_rate_volume": int, "sampling_rate_boundary": int,
        "absorbing": {"boundaries": (str, type(None)), "axis-aligned": bool}
    },
    "inversion": str,
    "hessian": (str, type(None)),
    "noise_source": {
        "filename": Path,
        "filter_spec": (list, type(None)),
        "component_dist_source": int,
        "component_wavefield": int,
        "shortcut": bool,
        "taper": bool,
        "taper_length_in_percent": (int, float)
    },
    "verbose": bool
}
REFERENCE_site = {
    "site": str,
    "ranks_salvus": int,
    "ranks_scorr": int,
    "wall_time_in_seconds_salvus": int,
    "wall_time_in_seconds_scorr": int,
    "ping_interval_in_seconds": int
}
REFERENCE_noise_source = {
    "homog_magnitude": (int, float),
    "gaussian": {"magnitude": list, "x": list, "y": list, "width_x": list, "width_y": list},
    "type": str, "filename": Path,
    "ocean_only": bool,
    "spectrum": {"f_peak": (int, float), "bandwidth": (int, float), "strength": (int, float)},
}
REFERENCE_measurement = {
    "type": str,
    "component_recording": int,
    "number_of_stacked_windows_min": (int, type(None)),
    "pick_window": bool,
    "window_halfwidth_in_sec": (int, float, type(None)),
    "min_period_in_s": (int, float, type(None)),
    "pick_manual": bool,
    "surface_wave_velocity_in_mps": (int, float, type(None)),
    "scale": (int, float, type(None)),
    "station_list": (list, type(None)),
    "snr": (int, float, type(None)),
    "correlation_coeff": (int, float, type(None))
}


def save_configuration(filename, config, type, verbose=False):
    check_configuration(config_input=config, type=type, verbose=verbose)
    scorr.addons.write_json_file(filename=filename, configuration=config)


def load_configuration(filename, type, verbose=False):
    config = load_json_file(filename=filename)

    check_configuration(config_input=config, type=type, verbose=verbose)
    # if type == "scorr":
    #     assert config["working_dir_local"] == filename.parent.parent, \
    # "Specified project directory and working directory in config file do not match."

    return config


def check_configuration(config_input, type, verbose=False):
    missing = []
    wrong_type = []

    def check_dictionary(reference, input):
        for item in reference.keys():
            if item in input.keys():
                if isinstance(reference[item], dict):
                    check_dictionary(reference[item], input[item])
                else:
                    if not isinstance(input[item], reference[item]):
                        wrong_type.append(item)
            else:
                missing.append(item)

    if type == "scorr":
        check_dictionary(REFERENCE_scorr, config_input)
    elif type == "site":
        check_dictionary(REFERENCE_site, config_input)
    elif type == "noise_source":
        check_dictionary(REFERENCE_noise_source, config_input)
    elif type == "measurement":
        check_dictionary(REFERENCE_measurement, config_input)
    else:
        raise ValueError("type not known!")

    if missing:
        raise KeyError(f"missing {missing}")
    if wrong_type:
        raise TypeError(f"wrong_type: {wrong_type}")

    if type == "scorr":
        check_range(configuration=config_input, type=type, verbose=verbose)

    if rank == 0 and verbose:
        print("correct input!")


def check_range(configuration, type, verbose=False):
    if type == "scorr":
        if configuration["simulation"]["precision"] not in ["single", "double"]:
            raise ValueError("precision has to be either 'single' or 'double'")
        if configuration["simulation"]["recording"] not in ["u_ELASTIC", "grad", "u_ELASTIC,grad"]:
            raise ValueError(
                "recording has to be either 'u_ELASTIC' or 'grad'")
        if configuration["simulation"]["green_starttime"] > 0.0:
            raise ValueError("starttime cannot be larger than 0.0")
        if configuration["simulation"]["corr_max_lag"] < 0.0:
            raise ValueError("endtime must be larger than 0.0")
        if configuration["simulation"]["corr_max_lag_causal"]:
            if configuration["simulation"]["corr_max_lag"] < configuration["simulation"]["corr_max_lag_causal"]:
                raise ValueError(
                    "corr_max_lag must be larger than corr_max_lag_causal")
        if configuration["noise_source"]["taper"]:
            if configuration["noise_source"]["taper_length_in_percent"] < 0 or configuration["noise_source"][
                    "taper_length_in_percent"] > 100:
                raise ValueError(
                    "taper_length_in_percent has to be between 0 and 100")
    elif type == "noise_source":
        dict_gaussian = configuration["noise_source"]["gaussian"]
        assert len(dict_gaussian["magnitude"]) == len(dict_gaussian["x"]) == \
            len(dict_gaussian["y"]) == len(dict_gaussian["width"])

    if rank == 0 and verbose:
        print("correct ranges of input values!")


def load_reference_stations(ref_station_json):
    # load reference stations
    ref_station_list = scorr.addons.load_json_file(ref_station_json)

    # CHECK PROBLEMATIC FOR RUN WITH MULTIPLE REFERENCE STATIONS AT THE SAME TIME
    # IDENTIFIER MIGHT HAVE BEEN CHANGED BEFORE SOME LOAD THE REFERENCE STATION FILE
    # UPDATE: identifier of reference station is not changed in src toml file anymore

    # loop through reference stations and check if identifiers are consistent with the respective name in src_toml files
    for identifier in ref_station_list.keys():
        with open(ref_station_list[identifier]["src_toml"], 'r') as fh:
            src_toml_name = toml.load(fh)["source"][0]["name"]
            if src_toml_name != identifier and "_".join(src_toml_name.split("_")[:-1]) != identifier:
                raise ValueError(
                    f"Identifier {identifier} not consistent with 'name' {src_toml_name} in src_toml file.")

    return ref_station_list


def add_reference_station_to_json_file(ref_station_json, identifier, src_toml, rec_toml):
    # open file with reference stations
    try:
        ref_station = scorr.addons.load_json_file(ref_station_json)
    except FileNotFoundError:
        ref_station = {}

    # check if src_toml and rec_toml file exist
    if not os.path.exists(src_toml) or not os.path.exists(rec_toml):
        raise FileNotFoundError

    # check if identifier is consistent with name in src_toml file
    with open(src_toml, 'r') as fh:
        src_toml_name = toml.load(fh)["source"][0]["name"]
        if src_toml_name != identifier:
            raise ValueError(
                f"Identifier {identifier} not consistent with 'name' {src_toml_name} in src_toml file.")

    # update list
    ref_station.update(
        {str(identifier): {
            "src_toml": src_toml,
            "rec_toml": rec_toml}})

    # write new reference station to file
    scorr.addons.write_json_file(ref_station_json, configuration=ref_station)


def get_reference_station_identifier(src_toml):
    if src_toml is "noise_source_setup":
        return "noise_source"

    with open(src_toml, 'r') as fh:
        return toml.load(fh)["source"][0]["name"]
