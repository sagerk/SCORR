#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import json

import salvus_seismo
import toml
import numpy as np

from obspy.signal.filter import bandpass
from scorr.addons import rank, comm
from scorr.extensions import scorr_extensions, salvus_extensions, rotation_distance


def prepare_source_and_receiver(config, identifier_prefix, loc_sources, loc_receivers, stf_type="gaussian"):
    # reset reference station list
    with open(config["simulation"]["reference_stations"], "w") as fh:
        json.dump({}, fh)

    # make source time function
    nt_green = int((abs(config["simulation"]["green_starttime"]) + abs(config["simulation"]["corr_max_lag"]))
                   / config["simulation"]["dt"] + 1)

    if stf_type == "ricker":
        config_noise_source = scorr_extensions.load_configuration(filename=config["working_dir_local"] / "config" /
                                                                  "noise_source.json", type="noise_source")
        _, stf = salvus_extensions.make_ricker_stf(starttime=config["simulation"]["green_starttime"],
                                                   dt=config["simulation"]["dt"], nt=nt_green,
                                                   amplitude=config["simulation"]["green_amplitude"],
                                                   center_freq=config_noise_source["spectrum"]["f_peak"],
                                                   time_shift=0.0, id_component=config["simulation"]["green_component"])
    elif stf_type == "gaussian":
        _, stf = salvus_extensions.make_gaussian_stf(starttime=config["simulation"]["green_starttime"],
                                                     dt=config["simulation"]["dt"], nt=nt_green,
                                                     amplitude=config["simulation"]["green_amplitude"],
                                                     time_shift=0.0, width=10 * config["simulation"]["dt"],
                                                     id_component=config["simulation"]["green_component"])
        # stf = bandpass(data=stf, freqmin=np.min(config["noise_source"]["filter_spec"]), freqmax=np.max(config["noise_source"]["filter_spec"]), df=1.0/config["simulation"]["dt"], zerophase=True, corners=5)
    else:
        raise NotImplementedError(
            f"Source time function type {stf_type} is not implemented!")

    for i_src in range(len(loc_sources)):
        if config["simulation"]["spherical"]:
            # set up salvus_seismo configuration with sources and receivers
            src = salvus_seismo.Source(latitude=loc_sources[i_src][0], longitude=loc_sources[i_src][1],
                                       depth_in_m=loc_sources[i_src][2], center_frequency=0.4)

            rec_toml_content_all = {'receiver': []}
            rec_paraview_all = []
            src_paraview_all = []
            for i_rec in range(len(loc_receivers[i_src])):
                rec = salvus_seismo.Receiver(latitude=loc_receivers[i_src][i_rec][0],
                                             longitude=loc_receivers[i_src][i_rec][1],
                                             depth_in_m=loc_receivers[i_src][i_rec][2],
                                             network="AA", station="r" + str(i_rec))
                seismo_config = salvus_seismo.Config(mesh_file=config["simulation"]["mesh"], end_time=1000,
                                                     salvus_call="yala", polynomial_order=4, verbose=True, dimensions=3)

                # generate command line call with salvus_seismo
                if rank == 0:
                    salvus_seismo.generate_cli_call(source=src, receivers=[rec], config=seismo_config,
                                                    output_folder=config["working_dir_local"] / "seismo",
                                                    exodus_file=str(config["simulation"]["mesh"]))

                    # collect receiver toml files
                    with open(config["working_dir_local"] / "seismo" / "receivers.toml", "r") as fh:
                        rec_toml_content_one = toml.load(fh)
                    try:
                        rec_toml_content_all['receiver'].append(
                            rec_toml_content_one['receiver'][0])
                    except KeyError:
                        print("ERROR")
                        continue

                    # collect paraview files
                    with open(config["working_dir_local"] / "seismo" / "receivers_paraview.csv", "r") as fh:
                        if i_rec == 0:
                            for line in fh.readlines():
                                rec_paraview_all.append(line)
                        else:
                            rec_paraview_all.append(fh.readlines()[-1])
                comm.Barrier()

            # write receiver toml file and paraview files
            filename_rec_toml = config["working_dir_local"] / \
                "seismo" / ("receivers_" + str(i_src) + ".toml")
            if rank == 0:
                with open(filename_rec_toml, 'w') as fh:
                    toml.dump(rec_toml_content_all, fh)

                # write receiver paraview file
                with open(config["working_dir_local"] / "seismo" / ("receivers_paraview_" + str(i_src) + ".csv"),
                          "w") as fh:
                    fh.writelines(rec_paraview_all)

                # collect and write source paraview file
                with open(config["working_dir_local"] / "seismo" / "source_paraview.csv", "r") as fh:
                    src_paraview_all.append(fh.readlines()[-1])

                    with open(config["working_dir_local"] / "seismo" / ("source_paraview_" + str(i_src) + ".csv"),
                              "w") as fh:
                        fh.writelines("x_coord, y_coord, z_coord, station\n")
                        fh.writelines(src_paraview_all)
            comm.Barrier()

            # read in actual source location for rotation matrix
            loc_source_actual = [[0, 0, 0]]
            with open(config["working_dir_local"] / "seismo" / "run_salvus.sh", "r") as fh:
                for option in fh.readlines()[0].split('--'):
                    if 'source-location' in option:
                        if '-x' in option:
                            loc_source_actual[0][0] = float(
                                option.strip().split(' ')[1])
                        if '-y' in option:
                            loc_source_actual[0][1] = float(
                                option.strip().split(' ')[1])
                        if '-z' in option:
                            loc_source_actual[0][2] = float(
                                option.strip().split(' ')[1])

            # build rotation matrix
            lat, lon = rotation_distance.from_cartesian_to_latlon(x=loc_source_actual[0][0], y=loc_source_actual[0][1],
                                                                  z=loc_source_actual[0][2])
            M_inv = rotation_distance.get_transformation_matrix_3d(
                latitude=lat, longitude=lon, inverse=True)

            # rotate source and save in source work list
            source_work = [rotation_distance.rotate_vector(M_inv, stf)]

        else:
            # write receiver toml file
            filename_rec_toml = config["working_dir_local"] / \
                "seismo" / ("receivers_" + str(i_src) + ".toml")
            salvus_extensions.write_receiver_toml(
                filename_rec_toml, loc_receivers[i_src])

            if rank == 0:
                # write receiver paraview file
                with open(config["working_dir_local"] / "seismo" / ("receivers_paraview_" + str(i_src) + ".csv"),
                          "w") as fh:
                    fh.writelines("x_coord, y_coord, z_coord, station\n")
                    for i_rec in range(len(loc_receivers[i_src])):
                        fh.write(f"{loc_receivers[i_src][i_rec][0]: f}, "
                                 f"{loc_receivers[i_src][i_rec][1]: f}, "
                                 f"{loc_receivers[i_src][i_rec][2]: f}, "
                                 f"receiver\n")

                # write source paraview file
                with open(config["working_dir_local"] / "seismo" / ("source_paraview_" + str(i_src) + ".csv"),
                          "w") as fh:
                    fh.writelines("x_coord, y_coord, z_coord, station\n")
                    fh.write(f"{loc_sources[i_src][0]: f}, "
                             f"{loc_sources[i_src][1]: f}, "
                             f"{loc_sources[i_src][2]: f}, "
                             f"source\n")

            comm.Barrier()

            # save stf in source work list and save source location
            loc_source_actual = [loc_sources[i_src]]
            source_work = [stf]

        # write source time function for synthetics
        identifier = [identifier_prefix + "_" + str(i_src)]
        salvus_extensions.write_stf(
            filename_hdf5=config["working_dir_local"] /
            "seismo" / ("source_" + str(i_src) + ".h5"),
            identifier=identifier,
            dt=config["simulation"]["dt"],
            starttime_in_s=config["simulation"]["green_starttime"],
            location=loc_source_actual, data=source_work)

        # add source and receiver toml files to reference station list
        scorr_extensions.add_reference_station_to_json_file(
            ref_station_json=config["simulation"]["reference_stations"],
            identifier=identifier[0],
            src_toml=config["working_dir_local"] /
            "seismo" / ("source_" + str(i_src) + ".toml"),
            rec_toml=filename_rec_toml)


def take_source_receiver_from_lasif_project(config, identifier_prefix, stf_type="gaussian"):
    # reset reference station list
    with open(config["simulation"]["reference_stations"], "w") as fh:
        json.dump({}, fh)

    # make source time function
    nt_green = int((abs(config["simulation"]["green_starttime"]) + abs(config["simulation"]["corr_max_lag"]))
                   / config["simulation"]["dt"] + 1)

    if stf_type == "ricker":
        config_noise_source = scorr_extensions.load_configuration(filename=config["working_dir_local"] / "config" /
                                                                  "noise_source.json", type="noise_source")
        _, stf = salvus_extensions.make_ricker_stf(starttime=config["simulation"]["green_starttime"],
                                                   dt=config["simulation"]["dt"], nt=nt_green,
                                                   amplitude=config["simulation"]["green_amplitude"],
                                                   center_freq=config_noise_source["spectrum"]["f_peak"],
                                                   time_shift=0.0, id_component=config["simulation"]["green_component"])
    elif stf_type == "gaussian":
        _, stf = salvus_extensions.make_gaussian_stf(starttime=config["simulation"]["green_starttime"],
                                                     dt=config["simulation"]["dt"], nt=nt_green,
                                                     amplitude=config["simulation"]["green_amplitude"],
                                                     time_shift=0.0, width=10 * config["simulation"]["dt"],
                                                     id_component=config["simulation"]["green_component"])
    else:
        raise NotImplementedError(
            f"Source time function type {stf_type} is not implemented!")

    lasif_project = config["working_dir_local"] / "seismo"
    lasif_folders = [x for x in lasif_project.iterdir() if (
        x.is_dir() and len(x.name.split(".")) == 2)]

    for lasif_folder in lasif_folders:
        filename_rec_toml = lasif_folder / "receivers.toml"

        # read in actual source location for rotation matrix
        loc_source_actual = [[0, 0, 0]]
        with open(lasif_folder / "run_salvus.sh", "r") as fh:
            for option in fh.readlines()[0].split('--'):
                if 'source-location' in option:
                    if '-x' in option:
                        loc_source_actual[0][0] = float(
                            option.strip().split(' ')[1])
                    if '-y' in option:
                        loc_source_actual[0][1] = float(
                            option.strip().split(' ')[1])
                    if '-z' in option:
                        loc_source_actual[0][2] = float(
                            option.strip().split(' ')[1])

        if config["simulation"]["spherical"]:
            # build rotation matrix
            lat, lon = rotation_distance.from_cartesian_to_latlon(x=loc_source_actual[0][0], y=loc_source_actual[0][1],
                                                                  z=loc_source_actual[0][2])
            M_inv = rotation_distance.get_transformation_matrix_3d(
                latitude=lat, longitude=lon, inverse=True)

            # rotate source and save in source work list
            source_work = [rotation_distance.rotate_vector(M_inv, stf)]

        else:
            source_work = [stf]

        # write source time function for synthetics
        identifier = [identifier_prefix + "_" +
                      lasif_folder.name.replace(".", "_")]
        salvus_extensions.write_stf(
            filename_hdf5=config["working_dir_local"] / "seismo" / (
                "source_" + lasif_folder.name.replace(".", "_") + ".h5"),
            identifier=identifier,
            dt=config["simulation"]["dt"],
            starttime_in_s=config["simulation"]["green_starttime"],
            location=loc_source_actual, data=source_work)

        # add source and receiver toml files to reference station list
        scorr_extensions.add_reference_station_to_json_file(
            ref_station_json=config["simulation"]["reference_stations"],
            identifier=identifier[0],
            src_toml=config["working_dir_local"] / "seismo" / (
                "source_" + lasif_folder.name.replace(".", "_") + ".toml"),
            rec_toml=filename_rec_toml)
