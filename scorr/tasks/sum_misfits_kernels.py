#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import shutil
from collections import Counter

import h5py
import numpy as np
import pyexodus

from scorr.extensions import scorr_extensions
from scorr.kernel.source_kernel import SourceKernel


def sum_misfits(config, id, clean=False, identifier_prefix="", ref_station_list=None):
    ref_station_dict = scorr_extensions.load_reference_stations(config["working_dir_local"] / "reference_stations.json")
    if not ref_station_list:
        ref_station_list = ref_station_dict.keys()

    j = 0.0
    for identifier in ref_station_list:
        with open(config["working_dir_local"] / "kernels" / (identifier_prefix + identifier + "_" + str(id)) / (
                "misfit" + (clean * "_clean") + ".txt"), "r") as fh:
            j += float(fh.readline())

    with open(config["working_dir_local"] / "kernels" / ("misfit_" + str(id) + (clean * "_clean") + ".txt"), "w") as fh:
        fh.write(str(j))

    print("Total misfit " + (clean * "(clean)") + f": {j}")


def sum_kernels(config, id, type="gradient", identifier_prefix="", ref_station_list=None):
    ref_station_dict = scorr_extensions.load_reference_stations(config["working_dir_local"] / "reference_stations.json")
    if not ref_station_list:
        ref_station_list = ref_station_dict.keys()

    # define folder where kernels are located
    if type == "gradient":
        folder_kernel = config["working_dir_local"] / "kernels"
    elif type == "hessian":
        folder_kernel = config["working_dir_local"] / "hessian"
    else:
        raise ValueError(f"Only two options: ['gradient', 'hessian']")

    # initialize counter for source and structure kernels
    i_source = 0
    i_structure = 0

    # sum source kernels
    if config["inversion"] == "source" or config["inversion"] == "joint":
        source_kernel_sum = SourceKernel()

        for identifier in ref_station_list:
            for file_kernel in (folder_kernel / (identifier_prefix + identifier + "_" + str(id))).glob("source*/*.h5"):
                print(file_kernel)
                source_kernel_sum += SourceKernel.init_with_kernel_file(filename_kernel_h5=file_kernel,
                                                                        verbose=config["verbose"])
                i_source += 1

        source_kernel_sum.write_kernel_to_file(
            filename_kernel_h5=folder_kernel / ("source_kernel_" + str(id) + ".h5"),
            precision=config["simulation"]["precision"], verbose=config["verbose"])

    # sum structure kernels
    if config["inversion"] == "structure" or config["inversion"] == "joint" or config["inversion"] == "source":

        if config["simulation"]["kernel-fields"] is None:
            if config["simulation"]["anisotropy"]:
                fields = ["RHO",
                          "C11", "C12", "C13", "C14", "C15", "C16",
                          "C22", "C23", "C24", "C25", "C26",
                          "C33", "C34", "C35", "C36",
                          "C44", "C45", "C46",
                          "C55", "C56",
                          "C66"]
            else:
                fields = ["RHO", "LAMBDA", "MU"]

        elif config["simulation"]["kernel-fields"] == "VP,VS,RHO":
            fields = ["VP", "VS", "RHO"]
        elif config["simulation"]["kernel-fields"] == "TTI":
            fields = ["VPV", "VPH", "VSV", "VSH", "RHO"]
        else:
            raise ValueError("New kernel-fields given! Need help!")

        filename_list = folder_kernel / ("list_structure_kernels_" + str(id) + ".txt")
        list_gradients = []
        with open(filename_list, "w") as fh:
            for identifier in ref_station_list:
                for file_kernel in (folder_kernel / (identifier_prefix + identifier + "_" + str(id))).glob(
                        "structure*/kernel.e"):
                    print(file_kernel)
                    fh.write(str(file_kernel) + "\n")
                    list_gradients.append(file_kernel)
                    i_structure += 1
        sum_fields_exodus(output_filename=folder_kernel / ('structure_kernel_' + str(id) + '.e'),
                          input_gradients=list_gradients, fields=fields)

        list_gradients = []
        with open(filename_list, "a") as fh:
            for identifier in ref_station_list:
                for file_kernel in (folder_kernel / (identifier_prefix + identifier + "_" + str(id))).glob(
                        "structure*/*.h5"):
                    print(file_kernel)
                    fh.write(str(file_kernel) + "\n")
                    list_gradients.append(file_kernel)
                    i_structure += 1
        sum_fields_hdf5(output_filename=folder_kernel / ('structure_kernel_' + str(id) + '.h5'),
                        input_gradients=list_gradients, fields=fields)

    # print summary
    print("\nSummed " + str(i_source) + " source kernel" + (i_source > 1) * "s" + " and "
          + str(i_structure) + " structure kernel" + (i_structure > 1) * "s" + "!" + "\n")


def sum_fields_exodus(output_filename, input_gradients, fields):
    """
    very simple function summing exodus gradients.
    <3<3<3  LION  <3<3<3
    """
    # ghetto initialization by just copying the first input file.
    if input_gradients:
        output_file = shutil.copy(src=input_gradients[0], dst=output_filename)
        output = pyexodus.exodus(str(output_file), mode="a")
    else:
        print("There are no exodus gradients to be summed.\n")
        return

    summed_fields = {}

    # zero the fields of interest.
    for field in fields:
        print(field)
        summed_fields[field] = \
            np.zeros_like(output.get_node_variable_values(field, step=1))

    # now loop over all files and just add the values.
    for g in input_gradients:
        with pyexodus.exodus(str(g), mode="r") as e_in:
            for field in fields:
                summed_fields[field] += \
                    e_in.get_node_variable_values(field, step=1)

    # finally write to file.
    for k, v in summed_fields.items():
        output.put_node_variable_values(k, step=1, values=v)

    # also zero all the other fields, but the mass matrix.
    ignore_fields = ["massmatrix"]
    ignore_fields.extend(fields)
    for field in output.get_node_variable_names():
        if field in ignore_fields:
            continue
        print(f"zeroing ignored field {field} in final gradient.")
        output.put_node_variable_values(
            field, step=1, values=np.zeros_like(
                output.get_node_variable_values(field, step=1)))


def sum_fields_hdf5(output_filename, input_gradients, fields):
    """
    very simple function summing hdf5 gradients.
    """
    # ghetto initialization by just copying the first input file.
    if input_gradients:
        output_file = shutil.copy(src=input_gradients[0], dst=output_filename)
        output = h5py.File(str(output_file), mode="a")
    else:
        print("There are no hdf5 gradients to be summed.\n")
        return

    summed_fields = {}
    field_index = {"VPV": 0,
                   "VPH": 1,
                   "VSV": 2,
                   "VSH": 3,
                   "RHO": 4}

    # zero the fields of interest.
    for field in fields:
        print(field)
        summed_fields[field] = np.zeros(output["/ELASTIC/data"][0, :, field_index[field], :].shape)

    # now loop over all files and just add the values.
    for g in input_gradients:
        with h5py.File(str(g), mode="r") as h5_in:
            for field in fields:
                summed_fields[field] += h5_in["/ELASTIC/data"][0, :, field_index[field], :]

    # finally write to file.
    for k, v in summed_fields.items():
        output["/ELASTIC/data"][0, :, field_index[k], :] = v
    output.close()

    # write new xdmf file
    xdmf_file = shutil.copy(src=".".join(str(input_gradients[0]).split(".")[:-1]) + "_ELASTIC.xdmf",
                            dst=".".join(str(output_filename).split(".")[:-1]) + ".xdmf" )
    # Read in the file
    with open(str(xdmf_file), 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('kernel.h5', str(output_filename.name))

    # Write the file out again
    with open(str(xdmf_file), 'w') as file:
        file.write(filedata)
