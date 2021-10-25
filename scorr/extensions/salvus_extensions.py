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
from typing import Union, Optional

import numpy as np
import pyasdf
import toml

from scorr.addons import comm, rank


###########################################
##########     MISCELLANEOUS     ##########
###########################################

def design_taper(full_length: int, taper_length_in_percent: int = 10,
                 zero_padding: bool = True, padding_length_percent: int = 1):
    # taper_length = int(round(taper_length_in_percent / 100 * full_length))
    taper_length = 21

    # need uneven number for taper design
    if np.mod(taper_length, 2) == 0:
        taper_length += 1

    # taper is the first part of a hanning window
    taper = np.hanning(taper_length)[:int(taper_length / 2) + 1]

    # increase taper with zero padding if requested
    if zero_padding:
        # taper = np.pad(taper, (int(padding_length_percent / 100 *
        #                            taper_length), 0), 'constant', constant_values=(0, 0))
        taper = np.pad(taper, (2, 0), 'constant', constant_values=(0, 0))

    return taper


def extract_starttime_in_s_from_src_toml(src_toml, ref_identifier):
    if src_toml is "noise_source_setup":
        print("Is it a noise source setup run? Set starttime to 0.0 seconds.")
        return 0.0

    with open(src_toml, "r") as fh:
        src_toml_content = toml.load(fh)

    for item in src_toml_content["source"]:
        if item["name"] == ref_identifier:
            dataset_name = item["dataset_name"]
            break
    else:
        raise ValueError("No corresponding entry found in toml file!")

    # get starttime
    with pyasdf.ASDFDataSet(src_toml_content["source_input_file"]) as ds:
        return ds.auxiliary_data.AdjointSources[dataset_name.split("/")[-1]].data.attrs.get("starttime") / 1.0e9


def extract_starttime_in_s_from_src_toml_simple(src_toml):
    if src_toml is "noise_source_setup":
        print("Is it a noise source setup run? Set starttime to 0.0 seconds.")
        return 0.0

    with open(src_toml, "r") as fh:
        src_toml_content = toml.load(fh)

    # get first entry under the assumption that all datasets referenced in src_toml file have the same starttime
    dataset_name = src_toml_content["source"][0]["dataset_name"]

    # get starttime in
    with pyasdf.ASDFDataSet(src_toml_content["source_input_file"]) as ds:
        return ds.auxiliary_data.AdjointSources[dataset_name.split("/")[-1]].data.attrs.get("starttime") / 1.0e9


def write_error_message_and_exit(filename, job_id):
    print(f"Job {job_id} didn't finish as expected!")
    with open(filename, "w") as fh_error:
        fh_error.write(f"Job {job_id} didn't finish as expected!")
    sys.exit()


###########################################
########## SOURCE TIME FUNCTIONS ##########
###########################################

def make_ricker_stf(starttime, dt, nt, amplitude, center_freq, time_shift, id_component):
    time = np.linspace(starttime, starttime + (nt - 1) * dt, nt)

    stf = np.zeros((time.shape[0], 3))
    factor = np.pi ** 2 * center_freq ** 2 * (time - time_shift) ** 2
    stf[:, id_component] = amplitude * (1 - 2 * factor) * np.exp(-1 * factor)

    print(
        f"Suggested time delay for simulation with ricker wavelet: {-2 * np.sqrt(6) / (np.pi * center_freq)}")

    return time, stf


def make_gaussian_stf(starttime, dt, nt, amplitude, time_shift, width, id_component):
    time = np.linspace(starttime, starttime + (nt - 1) * dt, nt)

    stf = np.zeros((time.shape[0], 3))
    stf[:, id_component] = amplitude * \
        np.exp(-(time - time_shift) ** 2 / (2 * width) ** 2)

    return time, stf


def make_spike_stf(starttime, dt, nt, amplitude, time_shift, id_component):
    def find_nearest(array, value):
        return (np.abs(array - value)).argmin()

    assert starttime <= time_shift
    time = np.linspace(starttime, starttime + (nt - 1) * dt, nt)

    stf = np.zeros((time.shape[0], 3))
    stf[find_nearest(time, time_shift), id_component] = amplitude

    return time, stf


def write_stf(filename_hdf5, identifier, dt, starttime_in_s, location, data):
    assert isinstance(identifier, list)
    assert isinstance(data, list)
    assert isinstance(location[0], list)

    if rank == 0:
        os.makedirs(os.path.dirname(filename_hdf5), exist_ok=True)
    comm.Barrier()

    with pyasdf.ASDFDataSet(str(filename_hdf5), mode='w') as ds_out:
        sources = []
        n_sources = np.size(location, 0)
        for i in range(n_sources):
            ds_out.add_auxiliary_data(
                data=data[i][:],
                data_type="AdjointSources",
                path=identifier[i],
                parameters={
                    "dt": dt,
                    "starttime": 1.0e9 * starttime_in_s,
                    "location": location[i],
                    "spatial-type": np.string_(("vector\x00").encode())})

            dataset_name = f"/AuxiliaryData/AdjointSources/{identifier[i]}"
            sources.append({
                "name": identifier[i],
                "dataset_name": dataset_name})

    if rank == 0:
        with open(os.path.splitext(str(filename_hdf5))[0] + ".toml", "wt") as fh:
            toml.dump({"source_input_file": str(
                filename_hdf5), "source": sources}, fh)
    comm.Barrier()

    return Path(os.path.splitext(str(filename_hdf5))[0] + ".toml")


def write_receiver_toml(filename_toml, location):
    dictionary = {}
    dictionary["receiver"] = []

    for i in range(np.size(location, 0)):
        dictionary['receiver'].append(
            {'network': 'AA',
             'station': 'rec' + str(i),
             'medium': 'solid',
             'location': '',
             'salvus_coordinates': location[i]})
        # 'transform_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]})

    if rank == 0:
        os.makedirs(os.path.dirname(filename_toml), exist_ok=True)
        with open(filename_toml, 'w') as fh:
            toml.dump(dictionary, fh)
    comm.Barrier()


###########################################
##########       BACKBONE        ##########
###########################################

def write_salvus_command(filename: str, type: str, mesh: str,
                         starttime: Union[int, float], endtime: Union[int, float], dt: Union[int, float],
                         sideset: str, attenuation: bool, recording: str,
                         src_toml: Optional[str] = None, rec_toml: Optional[str] = None,
                         sampling_rate_volume: Optional[int] = 10, sampling_rate_boundary: Optional[int] = 10,
                         absorbing: Optional[str] = None, axis_aligned: Optional[bool] = False,
                         anisotropy: Optional[bool] = False, kernel_fields: Optional[str] = None,
                         volumetric_source: Optional[str] = None):
    if not type in {"green", "green_structure", "correlation", "correlation_structure",
                    "adjoint_source", "adjoint_structure_1", "adjoint_structure_2"}:
        raise ValueError(f"Salvus run {type} is not implemented!")

    # check if given information is consistent
    assert sampling_rate_volume >= 1 and sampling_rate_boundary >= 1, "sampling rates must be >= 1"
    if "green" in type:
        assert src_toml is not None, "need a source toml file for Green function run"
    if "correlation" in type:
        assert rec_toml is not None, "need a receiver toml file for correlation run"

    # prepare small command snippets for mesh, timing, source, receiver and adjoint run
    mesh_model = f"--mesh-file {mesh} --model-file {mesh} --dimension 3 --polynomial-order 4 " \
                 f"--with-attenuation {attenuation} "
    with_anisotropy = f"--with-anisotropy "
    timing = f"--start-time {starttime} --end-time {endtime} --time-step {dt} "

    source = f"--source-toml {src_toml} "
    receiver = f"--receiver-toml {rec_toml} --receiver-fields {recording} --receiver-file-name dummy.h5 "
    adjoint = f"--adjoint true "

    # salvus-flow uses standardized output names for wavefield and boundary files
    if src_toml == "noise_source_setup":
        save_boundaries = f"--save-boundaries {sideset} --save-boundary-fields u_ELASTIC,mm_ELASTIC --save-boundaries-file dummy.h5 "
    elif type == "adjoint_source":
        save_boundaries = f"--save-boundaries {sideset} --save-boundary-fields u_ELASTIC --save-boundaries-file dummy.h5 "
    else:
        save_boundaries = f"--save-boundaries {sideset} --save-boundary-fields {recording} --save-boundaries-file dummy.h5 "
    save_fields = f"--save-fields adjoint --save-wavefield-file dummy.h5 "

    # wavefield and boundary files are determined by salvus-flow using the job_id
    load_boundaries = f"--load-boundaries {sideset} --load-boundary-fields source --load-boundaries-file dummy.h5 "
    load_fields_source = f"--load-fields source --load-wavefield-file dummy.h5 "
    load_fields_kernel = f"--load-fields adjoint --load-wavefield-file dummy.h5 --kernel-file dummy.e "
    if kernel_fields:
        load_fields_kernel += f"--kernel-fields {kernel_fields} "

    # sampling rates for volume and boundary data
    sampling_rate = f"--io-sampling-rate-volume {sampling_rate_volume} " \
                    f"--io-sampling-rate-boundary {sampling_rate_boundary} " \
                    f"--io-polynomial-degree-boundary 4 " \
                    f"--io-polynomial-degree-volume 4 "
    # f"--io-polynomial-degree-volume 2 "

    # assemble basic salvus command
    command_basic = mesh_model + timing + sampling_rate
    if rec_toml is not None:
        command_basic += receiver
    if anisotropy:
        command_basic += with_anisotropy
    if volumetric_source:
        command_basic += load_fields_source
    if absorbing is not None:
        if axis_aligned:
            command_basic += f"--absorbing-boundaries {absorbing} --num-absorbing-layers 12 --assume-axis-aligned "
        else:
            command_basic += f"--absorbing-boundaries {absorbing} --num-absorbing-layers 12 "

    # forward modelling of correlation functions and adjoint run for source kernel
    if "green" in type or type == "adjoint_source":
        command = command_basic + (src_toml is not "noise_source_setup") * source \
            + save_boundaries + (type == "green_structure") * save_fields
    if "correlation" in type:
        command = command_basic + load_boundaries + \
            (type == "correlation_structure") * save_fields + save_boundaries

    # adjoint runs for structure kernel
    if type == "adjoint_structure_1":
        command = command_basic + source + adjoint + \
            save_boundaries + load_fields_kernel
    if type == "adjoint_structure_2":
        command = command_basic + adjoint + load_boundaries + load_fields_kernel

    # write salvus command to file
    if rank == 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as fh:
            fh.write(command)
    comm.Barrier()
