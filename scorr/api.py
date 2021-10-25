#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import datetime
import os
import shutil
import sqlite3

import h5py
import numpy as np
import salvus_flow.api

from scorr.addons import rank, comm
from scorr.distributed_source.correlation_source import CorrelationSource
from scorr.extensions import job_tracker, job_tracker_hessian, job_tracker_ghetto, scorr_extensions, salvus_extensions
from scorr.kernel.source_kernel import SourceKernel
from scorr.noise_source.distribution import Distribution
from scorr.noise_source.noise_source import NoiseSource
from scorr.noise_source.spectrum import Spectrum
from scorr.tasks import simulations, measurement
from scorr.wavefield.adjoint_wavefield import AdjointWavefield
from scorr.wavefield.green_function import GreenFunction


def make_correlation_source(json_file, noise_source_file, wavefield_file, output_file):
    """
    Combination of Green function with power-spectral density distribution

    :return:
    """
    # load configuration
    config = scorr_extensions.load_configuration(
        filename=json_file, type="scorr")

    # set up noise source with spectrum and distribution
    spectrum = Spectrum.init_spectrum_with_file(
        filename=noise_source_file, verbose=config["verbose"])
    distribution = Distribution.init_distribution_with_file(
        filename=noise_source_file, verbose=config["verbose"])
    noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution,
                                                                                verbose=config["verbose"])

    # open wavefield
    if config["simulation"]["adjoint"]:
        if config["simulation"]["corr_max_lag_causal"]:
            endtime = config["simulation"]["corr_max_lag_causal"]
        else:
            endtime = config["simulation"]["corr_max_lag"]

        wavefield = AdjointWavefield(filename=wavefield_file,
                                     starttime=-
                                     config["simulation"]["corr_max_lag"],
                                     endtime=endtime,
                                     rotate=config["simulation"]["spherical"],
                                     verbose=config["verbose"])
    else:
        wavefield = GreenFunction(filename=wavefield_file,
                                  starttime=config["simulation"]["green_starttime"],
                                  endtime=config["simulation"]["corr_max_lag"],
                                  rotate=config["simulation"]["spherical"],
                                  verbose=config["verbose"])

    # initialize correlation source
    corr_source = CorrelationSource(
        noise_source, wavefield, verbose=config["verbose"])

    # convolve green function with time domain equivalent of power-spectral density distribution
    corr_source.convolve_wavefield(id_component_wavefield=config["noise_source"]["component_wavefield"],
                                   id_component_dist_source=config["noise_source"]["component_dist_source"],
                                   short_cut=config["noise_source"]["shortcut"],
                                   tapering=config["noise_source"]["taper"],
                                   taper_length_in_percent=config["noise_source"]["taper_length_in_percent"],
                                   filter_spec=config["noise_source"]["filter_spec"], verbose=config["verbose"])

    # save correlation source to file
    if config["simulation"]["adjoint"]:
        cutting_bonus = int(np.round(abs(config["simulation"]["green_starttime"]) / config["simulation"]["dt"] /
                                     config["simulation"]["sampling_rate_boundary"], 0))

        corr_source.write_distributed_source_to_file(filename_hdf5=output_file,
                                                     precision=config["simulation"]["precision"],
                                                     adjoint=True, cutting_bonus=cutting_bonus,
                                                     verbose=config["verbose"])
    else:
        if config["simulation"]["corr_max_lag_causal"]:
            cutting_bonus = -int(np.round(abs(
                config["simulation"]["corr_max_lag"] - config["simulation"]["corr_max_lag_causal"]) /
                config["simulation"]["dt"] /
                config["simulation"]["sampling_rate_boundary"], 0))
        else:
            cutting_bonus = int(0)

        corr_source.write_distributed_source_to_file(filename_hdf5=output_file,
                                                     precision=config["simulation"]["precision"],
                                                     adjoint=False, cutting_bonus=cutting_bonus,
                                                     verbose=config["verbose"])

    # print summary
    if rank == 0 and config["verbose"]:
        print("done!\n")
        print(wavefield)
        print(corr_source)
        print("\n")


def build_source_kernel(json_file, green_file, adjoint_file, spectrum, output_file):
    """
    Build source kernel from Green function and adjoint wavefield.

    :return:
    """
    # load configuration
    config = scorr_extensions.load_configuration(
        filename=json_file, type="scorr")

    # create spectrum
    # spectrum = Spectrum.init_spectrum_with_file(filename=config["noise_source"]["filename"],
    #                                             verbose=config["verbose"])
    spectrum = Spectrum(f_peak=spectrum[0], bandwidth=spectrum[1],
                        strength=spectrum[2], verbose=config["verbose"])

    # open Green function
    green_function = GreenFunction.init_green_function_with_dict(
        filename=green_file, config=config)
    # green_function.write_wavefield(Path(output_file).parent / "green.h5")

    # open adjoint wavefield
    adjoint_wavefield = AdjointWavefield.init_adjoint_wavefield_function_with_dict(
        filename=adjoint_file, config=config)
    # adjoint_wavefield.write_wavefield(Path(output_file).parent / "adjoint.h5")

    # build source kernel
    g = SourceKernel.init_with_wavefields(wavefield_fwd=green_function, wavefield_adj=adjoint_wavefield,
                                          spectrum=spectrum,
                                          id_component_wavefield=config["noise_source"]["component_wavefield"],
                                          shortcut=config["noise_source"]["shortcut"], verbose=config["verbose"])
    g.write_kernel_to_file(filename_kernel_h5=output_file,
                           precision=config["simulation"]["precision"])


def wavefield_diff(ref_file, pert_file, minus):
    """
    Compute wavefield difference.

    :return:
    """
    # define group name
    # if "/ELASTIC_BND" in ds_ref:
    group_name = "/ELASTIC_BND"
    # else:
    #     group_name = "/ELASTIC"

    # wavefield files
    if rank == 0:
        shutil.copy2(src=str(pert_file), dst=str(
            ".".join(pert_file.split(".")[:-1]) + "_copy.h5"))

        with h5py.File(str(ref_file), mode="r") as ds_ref:
            with h5py.File(str(pert_file), mode="a") as ds_pert:
                assert group_name in ds_pert, f"Group names do not agree."

                # compute difference
                if minus:
                    ds_pert[group_name + "/data"][:] = ds_pert[group_name + "/data"][:] - ds_ref[group_name + "/data"][
                        :]
                else:
                    ds_pert[group_name + "/data"][:] = ds_pert[group_name + "/data"][:] + ds_ref[group_name + "/data"][
                        :]

    comm.Barrier()


def compute_correlations(site, config, ref_identifier, src_toml, rec_toml, output_folder=None, skip_measurement=False):
    # computing correlations only makes sense for adjoint=False
    if config["simulation"]["adjoint"]:
        print(
            "Warning: config['simulation']['adjoint'] was set to 'true', changed it to 'false'!")
        config["simulation"]["adjoint"] = False

    # add reference station to database
    try:
        ref_info = job_tracker.add_reference_station(
            ref_station=ref_identifier, inversion_type=config["inversion"])
    except sqlite3.IntegrityError:
        print("\nReference station exists already! Try to use available simulations.")
        ref_info = job_tracker.get_jobs_of_reference_station(
            ref_station=ref_identifier)
    finally:
        '''
        for a structure kernel, the forward modelling has to be repeated if the previous runs in the database were meant
        to be for a source kernel
        have to do that in the finally section, since the error locks the database
        '''
        if not (config["inversion"] == ""):
            if (config["inversion"] == "structure" or config["inversion"] == "joint"):
                if ref_info.inversion_type == None or ref_info.inversion_type == "" or ref_info.inversion_type == "source":
                    ref_info = job_tracker.reset_reference_station(ref_station=ref_identifier,
                                                                   inversion_type=config["inversion"])

            # update inversion type if current inversion type is "joint" and the previous run was for "structure"
            if config["inversion"] == "joint" and ref_info.inversion_type == "structure":
                ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                                inversion_type=config["inversion"])

            # source and structure runs combined lead to a joint inversion setup
            if config["inversion"] == "source" and ref_info.inversion_type == "structure" or \
                    config["inversion"] == "structure" and ref_info.inversion_type == "source":
                ref_info = job_tracker.update_reference_station(
                    ref_station=ref_identifier, inversion_type="joint")

    # if "step" in ref_identifier:
    #     ref_info_syn = job_tracker.get_jobs_of_reference_station(ref_station="syn_test_0_0")
    #     ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
    #                                                     job_id_green=ref_info_syn.job_id_green,
    #                                                     job_id_corr_source=ref_info_syn.job_id_corr_source)

    # compute green function
    if not (ref_info.job_id_correlation and config["inversion"] == ""):
        if not ref_info.job_id_green:
            print("\nComputing Green function!\n")
            job_id_green = simulations.compute_wavefield_point_source(site=site, config=config, src_toml=src_toml,
                                                                      rec_toml=rec_toml)
            ref_info = job_tracker.update_reference_station(
                ref_station=ref_identifier, job_id_green=job_id_green)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info.job_id_green,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_green)

    # import sys; sys.exit()

    # compute correlation source
    if not ref_info.job_id_correlation:
        if not ref_info.job_id_corr_source:
            print("\nSetting up correlation source!\n")
            job_id_corr_source = simulations.compute_distributed_source(site=site, config=config,
                                                                        job_id_wavefield=ref_info.job_id_green)
            ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                            job_id_corr_source=job_id_corr_source)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info.job_id_corr_source,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_corr_source)

    # import sys; sys.exit()

    # compute synthetic correlations
    if not ref_info.job_id_correlation:
        # run simulation
        print("\nComputing correlation function!\n")
        job_id_correlation = simulations.compute_wavefield_distributed_source(site=site, config=config,
                                                                              job_id_distsource=ref_info.job_id_corr_source,
                                                                              rec_toml=rec_toml)
        ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                        job_id_correlation=job_id_correlation)

    # wait for job to finish
    status = salvus_flow.api.wait_for_job(job=ref_info.job_id_correlation,
                                          ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                          quiet=not config["verbose"])
    if status.value != 2:
        salvus_extensions.write_error_message_and_exit(
            config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_correlation)

    # define folder containing correlation functions and download them
    if not skip_measurement:
        if output_folder is None:
            output_folder = config["working_dir_local"] / \
                "correlations" / ref_identifier
        salvus_flow.api.get_output(job=ref_info.job_id_correlation, destination=output_folder,
                                   get_all=False, force=True,
                                   quiet=not config["verbose"])

    return ref_info


def evaluate_misfit_and_gradient(site, config, config_measurement, ref_identifier, src_toml, rec_toml,
                                 file_observations=None, gradient=False, data_lasif=False, skip_measurement=False):
    if config["inversion"] == "":
        raise ValueError("Error in config: what should be inverted for?")

    # compute correlations
    config["simulation"]["adjoint"] = False
    folder_synthetics = config["working_dir_local"] / \
        "correlations" / "synthetics" / ref_identifier
    ref_info = compute_correlations(site=site, config=config,
                                    ref_identifier=ref_identifier,
                                    src_toml=src_toml, rec_toml=rec_toml,
                                    output_folder=folder_synthetics,
                                    skip_measurement=skip_measurement)

    # make measurement and return toml for adjoint source time function
    if not skip_measurement:
        j, adsrc_toml, j_clean = measurement.make_measurement(config=config, config_measurement=config_measurement,
                                                              ref_identifier=ref_identifier,
                                                              src_toml=src_toml, rec_toml=rec_toml,
                                                              filename_synthetics=folder_synthetics / "receivers.h5",
                                                              filename_observations=file_observations,
                                                              data_lasif=data_lasif)

        os.makedirs((config["working_dir_local"] / "kernels" / ref_identifier / "misfit.txt").parent,
                    exist_ok=True)
        with open(config["working_dir_local"] / "kernels" / ref_identifier / "misfit.txt", "w") as fh:
            fh.write(str(j))
        with open(config["working_dir_local"] / "kernels" / ref_identifier / "misfit_clean.txt", "w") as fh:
            fh.write(str(j_clean))
    else:
        j = 0.0
        j_clean = 0.0
        adsrc_toml = config["working_dir_local"] / \
            "adjoint_stf" / ref_identifier / ("adjoint_stf_1nd.toml")

    # import sys; sys.exit()

    # compute gradient
    if gradient:
        # run adjoint simulations
        config["simulation"]["adjoint"] = True

        # run first adjoint simulation - interacts for structure kernel with correlation wavefield
        if not ref_info.job_id_adjoint_1:
            # run first adjoint simulation
            print("\nRunning first adjoint run!\n")
            job_id_adjoint_1 = simulations.compute_wavefield_point_source(site=site, config=config, src_toml=adsrc_toml,
                                                                          job_id_fwd_wavefield=ref_info.job_id_correlation)
            ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                            job_id_adjoint_1=job_id_adjoint_1)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info.job_id_adjoint_1,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_adjoint_1)

        # download structure kernel
        if config["inversion"] == "structure" or config["inversion"] == "joint":
            output_folder_kernel = config["working_dir_local"] / \
                "kernels" / ref_identifier / "structure_0"
            salvus_flow.api.get_output(job=ref_info.job_id_adjoint_1, destination=output_folder_kernel,
                                       get_all=False, force=True, quiet=not config["verbose"])

        # build source kernel
        if config["inversion"] == "source" or config["inversion"] == "joint":
            if not ref_info.job_id_source_kernel:
                # compute source kernel and update reference station
                print("\nAssembling source kernel!\n")
                job_id_source_kernel = simulations.build_source_kernel(site=site, config=config,
                                                                       job_id_green=ref_info.job_id_green,
                                                                       job_id_adjoint=ref_info.job_id_adjoint_1)
                ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                                job_id_source_kernel=job_id_source_kernel)

            # wait for job to finish
            status = salvus_flow.api.wait_for_job(job=ref_info.job_id_source_kernel,
                                                  ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                                  quiet=not config["verbose"])
            if status.value != 2:
                salvus_extensions.write_error_message_and_exit(
                    config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_source_kernel)

            # download source kernel
            output_folder_kernel = config["working_dir_local"] / \
                "kernels" / ref_identifier / "source"
            salvus_flow.api.get_output(job=ref_info.job_id_source_kernel, destination=output_folder_kernel,
                                       get_all=True, force=True, quiet=not config["verbose"])

        if config["inversion"] == "structure" or config["inversion"] == "joint":
            # prepare second adjoint simulation
            if not ref_info.job_id_adjoint_2:
                if not ref_info.job_id_dist_adjstf and not ref_info.job_id_adjoint_2:
                    print("\nSetting up distributed source for second adjoint run!\n")
                    job_id_dist_adjstf = simulations.compute_distributed_source(site=site, config=config,
                                                                                job_id_wavefield=ref_info.job_id_adjoint_1)
                    ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                                    job_id_dist_adjstf=job_id_dist_adjstf)
                # wait for job to finish
                status = salvus_flow.api.wait_for_job(job=ref_info.job_id_dist_adjstf,
                                                      ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                                      quiet=not config["verbose"])
                if status.value != 2:
                    salvus_extensions.write_error_message_and_exit(
                        config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_dist_adjstf)

            # run second adjoint simulation - interacts for structure kernel with green function wavefield
            if not ref_info.job_id_adjoint_2:
                # compute second structure kernel
                print("\nRunning second adjoint run!\n")
                job_id_adjoint_2 = simulations.compute_wavefield_distributed_source(site=site, config=config,
                                                                                    job_id_distsource=ref_info.job_id_dist_adjstf,
                                                                                    job_id_fwd_wavefield=ref_info.job_id_green)
                ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                                job_id_adjoint_2=job_id_adjoint_2)

            # wait for job to finish
            status = salvus_flow.api.wait_for_job(job=ref_info.job_id_adjoint_2,
                                                  ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                                  quiet=not config["verbose"])
            if status.value != 2:
                salvus_extensions.write_error_message_and_exit(
                    config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_adjoint_2)

            # define folder containing correlation functions after computations
            output_folder_kernel = config["working_dir_local"] / \
                "kernels" / ref_identifier / "structure_1"
            salvus_flow.api.get_output(job=ref_info.job_id_adjoint_2, destination=output_folder_kernel,
                                       get_all=False, force=True, quiet=not config["verbose"])

    # return misfit
    return j, j_clean


def evaluate_hessian_vector_product(site, config, config_measurement, ref_identifier, src_toml, rec_toml,
                                    file_observations=None, data_lasif=False):
    # add reference station to database
    try:
        ref_info_hessian = job_tracker_hessian.add_reference_station(ref_station=ref_identifier,
                                                                     inversion_type=config["inversion"])
    except sqlite3.IntegrityError:
        print("\nReference station exists already! Try to use available simulations.")
        ref_info_hessian = job_tracker_hessian.get_jobs_of_reference_station(
            ref_station=ref_identifier)

    # ghetto style implementation of changes of wavefields
    try:
        ref_info_ghetto = job_tracker_ghetto.add_reference_station(ref_station=ref_identifier,
                                                                   inversion_type=config["inversion"])
    except sqlite3.IntegrityError:
        print("\nReference station exists already! Try to use available simulations.")
        ref_info_ghetto = job_tracker_ghetto.get_jobs_of_reference_station(
            ref_station=ref_identifier)

    ##########################################################
    #                     C WAVEFIELD                        #
    ##########################################################
    # compute correlations
    config["simulation"]["adjoint"] = False
    folder_synthetics = config["working_dir_local"] / \
        "correlations" / "synthetics" / ref_identifier
    ref_info = compute_correlations(site=site, config=config,
                                    ref_identifier=ref_identifier,
                                    src_toml=src_toml, rec_toml=rec_toml,
                                    output_folder=folder_synthetics)

    ##########################################################
    #         PERTURBATIONS FORWARD WAVEFIELDS               #
    ##########################################################
    folder_perturbation = config["working_dir_local"] / \
        "correlations" / "perturbation" / ref_identifier

    if config["hessian"] == "dS":
        ####################################
        #########    SIMULATION    #########
        ####################################
        filename_noise_source_backup = config["noise_source"]["filename"]
        config["noise_source"]["filename"] = config["noise_source"]["filename"].parent / \
            "noise_source_pert.h5"
        ref_info_pert = compute_correlations(site=site, config=config,
                                             ref_identifier=ref_identifier + "_pert",
                                             src_toml=src_toml, rec_toml=rec_toml,
                                             output_folder=folder_perturbation)
        config["noise_source"]["filename"] = filename_noise_source_backup

    else:
        ####################################
        #########    SIMULATION    #########
        ####################################
        filename_mesh_backup = config["simulation"]["mesh"]
        config["simulation"]["mesh"] = config["simulation"]["mesh"].parent / (
            config["simulation"]["mesh"].stem + "_pert.e")
        ref_info_pert = compute_correlations(site=site, config=config,
                                             ref_identifier=ref_identifier + "_pert",
                                             src_toml=src_toml, rec_toml=rec_toml,
                                             output_folder=folder_perturbation)
        config["simulation"]["mesh"] = filename_mesh_backup

        # compute difference for correlation recordings
        # correlation_reference = salvus_flow.api.get_output_files(ref_info.job_id_correlation, get_all=True)
        # correlation_perturbation = salvus_flow.api.get_output_files(ref_info_pert.job_id_correlation, get_all=True)
        # shutil.copy2(src=str(correlation_perturbation["receivers.h5"]), dst="/Users/korbinian/Desktop/copy.h5")

        # with pyasdf.ASDFDataSet(str(folder_synthetics / "receivers.h5"), mode="r") as ds_correlation_reference:
        #     with pyasdf.ASDFDataSet(str(folder_perturbation / "receivers.h5"), mode="a") as ds_correlation_perturbation:
        #         for waveform in ds_correlation_reference.waveforms.list():
        #             for index in range(len(ds_correlation_reference.waveforms[waveform].displacement)):
        #                 ds_correlation_perturbation.waveforms[waveform].displacement[index].data[:] -= \
        #                     ds_correlation_reference.waveforms[waveform].displacement[index].data

        with h5py.File(str(folder_synthetics / "receivers.h5"), mode="r") as ds_correlation_reference:
            with h5py.File(str(folder_perturbation / "receivers.h5"), mode="a") as ds_correlation_perturbation:
                date_ref = datetime.datetime(1970, 1, 1, 0, 0, 0)
                date_start = (date_ref - datetime.timedelta(seconds=config["simulation"]["corr_max_lag"])).strftime(
                    "%Y-%m-%dT%H:%M:%S")
                date_end = (date_ref + datetime.timedelta(
                    seconds=config["simulation"]["corr_max_lag_causal"])).strftime(
                    "%Y-%m-%dT%H:%M:%S")
                time_string = date_start + "__" + date_end

                ds_correlation_perturbation[
                    "/Waveforms/AA.rec0/AA.rec0..XDX__" + time_string + "__displacement"][:] = \
                    ds_correlation_perturbation[
                        "/Waveforms/AA.rec0/AA.rec0..XDX__" + time_string + "__displacement"][:] - \
                    ds_correlation_reference[
                        "/Waveforms/AA.rec0/AA.rec0..XDX__" + time_string + "__displacement"][:]

                ds_correlation_perturbation[
                    "/Waveforms/AA.rec0/AA.rec0..XDY__" + time_string + "__displacement"][:] = \
                    ds_correlation_perturbation[
                        "/Waveforms/AA.rec0/AA.rec0..XDY__" + time_string + "__displacement"][:] - \
                    ds_correlation_reference[
                        "/Waveforms/AA.rec0/AA.rec0..XDY__" + time_string + "__displacement"][:]

                ds_correlation_perturbation[
                    "/Waveforms/AA.rec0/AA.rec0..XDZ__" + time_string + "__displacement"][:] = \
                    ds_correlation_perturbation[
                        "/Waveforms/AA.rec0/AA.rec0..XDZ__" + time_string + "__displacement"][:] - \
                    ds_correlation_reference[
                        "/Waveforms/AA.rec0/AA.rec0..XDZ__" + time_string + "__displacement"][:]

        ####################################
        #########    DIFFERENCE    #########
        ####################################
        # compute wavefield differences for boundary
        if not ref_info_ghetto.job_id_dGm:
            print("\nCompute dGm boundary!\n")
            job_id_dGm = simulations.wavefield_diff(site=site, config=config,
                                                    job_id_ref=ref_info.job_id_green,
                                                    job_id_pert=ref_info_pert.job_id_green,
                                                    boundary=True, minus=True)
            ref_info_ghetto = job_tracker_ghetto.update_reference_station(ref_station=ref_identifier,
                                                                          job_id_dGm=job_id_dGm)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_ghetto.job_id_dGm,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info_ghetto.job_id_dGm)

        # TODO: differences for Green function wavefield
        # TODO: differences for correlation wavefield

    # import sys; sys.exit()

    ##########################################################
    #                    MEASUREMENT                         #
    ##########################################################
    # make measurement and save toml for adjoint source time function in config
    _, adsrc_toml_1nd, _ = measurement.make_measurement(config=config, config_measurement=config_measurement,
                                                        ref_identifier=ref_identifier,
                                                        src_toml=src_toml, rec_toml=rec_toml,
                                                        filename_synthetics=folder_synthetics / "receivers.h5",
                                                        filename_observations=file_observations,
                                                        data_lasif=data_lasif,
                                                        order=1)

    _, adsrc_toml_2nd, _ = measurement.make_measurement(config=config, config_measurement=config_measurement,
                                                        ref_identifier=ref_identifier,
                                                        src_toml=src_toml, rec_toml=rec_toml,
                                                        filename_synthetics=folder_synthetics / "receivers.h5",
                                                        filename_observations=file_observations,
                                                        data_lasif=data_lasif,
                                                        filename_perturbations=folder_perturbation / "receivers.h5",
                                                        order=2)

    # import sys; sys.exit()

    # time for adjoint simulations
    config["simulation"]["adjoint"] = True

    ##########################################################
    #                     U WAVEFIELD                        #
    ##########################################################
    if not (config["hessian"] == "dS" and config["inversion"] == "source"):
        # compute adjoint wavefield udagger
        if not ref_info.job_id_adjoint_1:
            print("\nRunning adjoint run to get udagger!\n")
            job_id_adjoint_1 = simulations.compute_wavefield_point_source(site=site, config=config,
                                                                          src_toml=adsrc_toml_1nd,
                                                                          job_id_fwd_wavefield=ref_info_pert.job_id_correlation)
            ref_info = job_tracker.update_reference_station(ref_station=ref_identifier,
                                                            job_id_adjoint_1=job_id_adjoint_1)
        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info.job_id_adjoint_1,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info.job_id_adjoint_1)

        # define folder for kernel
        if config["inversion"] == "structure":
            output_folder_kernel = config["working_dir_local"] / \
                "hessian" / ref_identifier / "structure_0"
            salvus_flow.api.get_output(job=ref_info.job_id_adjoint_1, destination=output_folder_kernel,
                                       get_all=False, force=True, quiet=not config["verbose"])

    if config["hessian"] == "dm":
        ####################################
        #########    SIMULATION    #########
        ####################################
        # compute adjoint wavefield d(udagger)
        if not ref_info_pert.job_id_adjoint_1:
            filename_mesh_backup = config["simulation"]["mesh"]
            config["simulation"]["mesh"] = config["simulation"]["mesh"].parent / (
                config["simulation"]["mesh"].stem + "_pert.e")

            print("\nRunning adjoint run to get d(udagger)!\n")
            job_id_adjoint_1_pert = simulations.compute_wavefield_point_source(site=site, config=config,
                                                                               src_toml=adsrc_toml_1nd,
                                                                               job_id_fwd_wavefield=ref_info_pert.job_id_correlation)
            ref_info_pert = job_tracker.update_reference_station(ref_station=ref_identifier + "_pert",
                                                                 job_id_adjoint_1=job_id_adjoint_1_pert)
            config["simulation"]["mesh"] = filename_mesh_backup

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_pert.job_id_adjoint_1,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] /
                ("error_" + ref_identifier + "_pert.log"),
                ref_info_pert.job_id_adjoint_1)

        ####################################
        #########    DIFFERENCE    #########
        ####################################
        if not ref_info_ghetto.job_id_dudagger:
            print("\nCompute dudagger!\n")
            job_id_dudagger = simulations.wavefield_diff(site=site, config=config,
                                                         job_id_ref=ref_info.job_id_adjoint_1,
                                                         job_id_pert=ref_info_pert.job_id_adjoint_1,
                                                         boundary=True, minus=True)
            ref_info_ghetto = job_tracker_ghetto.update_reference_station(ref_station=ref_identifier,
                                                                          job_id_dudagger=job_id_dudagger)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_ghetto.job_id_dudagger,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info_ghetto.job_id_dudagger)

    ##########################################################
    #                     P WAVEFIELD                        #
    ##########################################################
    if not ref_info_hessian.job_id_p:
        print("\nRunning adjoint run to get p!\n")
        job_id_p = simulations.compute_wavefield_point_source(site=site, config=config,
                                                              src_toml=adsrc_toml_2nd,
                                                              job_id_fwd_wavefield=ref_info.job_id_correlation)
        ref_info_hessian = job_tracker_hessian.update_reference_station(ref_station=ref_identifier,
                                                                        job_id_p=job_id_p)

    # wait for job to finish
    status = salvus_flow.api.wait_for_job(job=ref_info_hessian.job_id_p,
                                          ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                          quiet=not config["verbose"])
    if status.value != 2:
        salvus_extensions.write_error_message_and_exit(
            config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info_hessian.job_id_p)

    # define folder for kernel
    if config["inversion"] == "structure":
        output_folder_kernel = config["working_dir_local"] / \
            "hessian" / ref_identifier / "structure_1"
        salvus_flow.api.get_output(job=ref_info_hessian.job_id_p, destination=output_folder_kernel,
                                   get_all=False, force=True, quiet=not config["verbose"])

    if config["hessian"] == "dm":
        # TODO: ADD FULL P TO GHETTO JOB TRACKER

        ####################################
        ###########   FULL P    ############
        ####################################
        print("\nCompute full p, i.e. add dudagger!\n")
        job_id_p_full = simulations.wavefield_diff(site=site, config=config,
                                                   job_id_ref=ref_info_pert.job_id_adjoint_1,
                                                   job_id_pert=ref_info_hessian.job_id_p,
                                                   boundary=True, minus=False)
        status = salvus_flow.api.wait_for_job(job=job_id_p_full,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), job_id_p_full)

    ##########################################################
    #                    SOURCE KERNEL                       #
    ##########################################################
    if config["inversion"] == "source":
        ####################################
        ### build source kernel from pPG ###
        ####################################
        if not ref_info_hessian.job_id_source_kernel_pPG:
            print("\nBuild source kernel pPG!\n")
            job_id_source_kernel_pPG = simulations.build_source_kernel(site=site, config=config,
                                                                       job_id_green=ref_info.job_id_green,
                                                                       job_id_adjoint=ref_info_hessian.job_id_p)
            ref_info_hessian = job_tracker_hessian.update_reference_station(ref_station=ref_identifier,
                                                                            job_id_source_kernel_pPG=job_id_source_kernel_pPG)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_hessian.job_id_source_kernel_pPG,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] /
                ("error_" + ref_identifier + ".log"),
                ref_info_hessian.job_id_source_kernel_pPG)

        # download source kernel from pPG
        output_folder_hessian = config["working_dir_local"] / \
            "hessian" / ref_identifier / "source_pPG"
        salvus_flow.api.get_output(job=ref_info_hessian.job_id_source_kernel_pPG,
                                   destination=output_folder_hessian, get_all=True, force=True,
                                   quiet=not config["verbose"])

        if config["hessian"] == "dm":
            #####################################
            ### build source kernel from uPdG ###
            #####################################

            # to get dG, ref_info_pert.job_id_green was changed in place
            # might be good to change that in the future

            if not ref_info_hessian.job_id_source_kernel_uPdG:
                print("\nBuild source kernel uPdG!\n")
                job_id_source_kernel_uPdG = simulations.build_source_kernel(site=site, config=config,
                                                                            job_id_green=ref_info_pert.job_id_green,
                                                                            job_id_adjoint=ref_info.job_id_adjoint_1)
                ref_info_hessian = job_tracker_hessian.update_reference_station(ref_station=ref_identifier,
                                                                                job_id_source_kernel_uPdG=job_id_source_kernel_uPdG)

            # wait for job to finish
            status = salvus_flow.api.wait_for_job(job=ref_info_hessian.job_id_source_kernel_uPdG,
                                                  ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                                  quiet=not config["verbose"])
            if status.value != 2:
                salvus_extensions.write_error_message_and_exit(
                    config["working_dir_local"] /
                    ("error_" + ref_identifier + ".log"),
                    ref_info_hessian.job_id_source_kernel_uPdG)

            # download source kernel from uPdG
            output_folder_hessian = config["working_dir_local"] / \
                "hessian" / ref_identifier / "source_uPdG"
            salvus_flow.api.get_output(job=ref_info_hessian.job_id_source_kernel_uPdG,
                                       destination=output_folder_hessian, get_all=True, force=True,
                                       quiet=not config["verbose"])

    if config["inversion"] == "structure":

        # TODO: C and W wavefields

        ##########################################################
        #                     Z WAVEFIELD                        #
        ##########################################################
        # build distributed source from p and S for z wavefield
        if not ref_info_hessian.job_id_source_z:
            print("\nCompute distributed source for z!\n")
            job_id_source_z = simulations.compute_distributed_source(site=site, config=config,
                                                                     job_id_wavefield=ref_info_hessian.job_id_p)
            ref_info_hessian = job_tracker_hessian.update_reference_station(ref_station=ref_identifier,
                                                                            job_id_source_z=job_id_source_z)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_hessian.job_id_source_z,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info_hessian.job_id_source_z)

        # run adjoint simulation to get z wavefield
        if not ref_info_hessian.job_id_z:
            print("\nRunning adjoint run to get z!\n")
            job_id_z = simulations.compute_wavefield_distributed_source(site=site, config=config,
                                                                        job_id_distsource=ref_info_hessian.job_id_source_z,
                                                                        job_id_fwd_wavefield=ref_info.job_id_green)
            ref_info_hessian = job_tracker_hessian.update_reference_station(ref_station=ref_identifier,
                                                                            job_id_z=job_id_z)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_hessian.job_id_z,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info_hessian.job_id_z)

        # define folder for kernel
        output_folder_kernel = config["working_dir_local"] / \
            "hessian" / ref_identifier / "structure_2"
        salvus_flow.api.get_output(job=ref_info_hessian.job_id_z, destination=output_folder_kernel,
                                   get_all=False, force=True, quiet=not config["verbose"])

        ##########################################################
        #                     Q WAVEFIELD                        #
        ##########################################################
        # build distributed source from u and dS for q wavefield
        if not ref_info_hessian.job_id_source_q:
            print("\nCompute distributed source for q!\n")
            filename_noise_source_backup = config["noise_source"]["filename"]
            config["noise_source"]["filename"] = config["noise_source"]["filename"].parent / \
                ("noise_source_pert.h5")
            job_id_source_q = simulations.compute_distributed_source(site=site, config=config,
                                                                     job_id_wavefield=ref_info.job_id_adjoint_1)
            ref_info_hessian = job_tracker_hessian.update_reference_station(ref_station=ref_identifier,
                                                                            job_id_source_q=job_id_source_q)
            config["noise_source"]["filename"] = filename_noise_source_backup

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_hessian.job_id_source_q,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info_hessian.job_id_source_q)

        # run adjoint simulation to get q wavefield
        if not ref_info_hessian.job_id_q:
            print("\nRunning adjoint run to get q!\n")
            job_id_q = simulations.compute_wavefield_distributed_source(site=site, config=config,
                                                                        job_id_distsource=ref_info_hessian.job_id_source_q,
                                                                        job_id_fwd_wavefield=ref_info.job_id_green)
            ref_info_hessian = job_tracker_hessian.update_reference_station(ref_station=ref_identifier,
                                                                            job_id_q=job_id_q)

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=ref_info_hessian.job_id_q,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        if status.value != 2:
            salvus_extensions.write_error_message_and_exit(
                config["working_dir_local"] / ("error_" + ref_identifier + ".log"), ref_info_hessian.job_id_q)

        # define folder for kernel
        output_folder_kernel = config["working_dir_local"] / \
            "hessian" / ref_identifier / "structure_3"
        salvus_flow.api.get_output(job=ref_info_hessian.job_id_q, destination=output_folder_kernel,
                                   get_all=False, force=True, quiet=not config["verbose"])
