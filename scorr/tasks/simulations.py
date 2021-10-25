#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
from pathlib import Path

import h5py
import salvus_flow.api

import scorr.extensions.salvus_extensions
import scorr.extensions.scorr_extensions
from salvus_flow.sites.job import Job
from scorr.addons import comm, rank, write_json_file, git_hash_and_diff, group_name
from scorr.extensions import salvus_extensions


def determine_simulation_type(config, point_source, job_id_fwd_wavefield):
    if config["inversion"] == "" or config["inversion"] == "source":
        job_id_fwd_wavefield = None
        if point_source:
            if config["simulation"]["adjoint"] == False:
                simulation_type = "green"
            else:
                simulation_type = "adjoint_source"
        else:
            simulation_type = "correlation"

    elif config["inversion"] == "structure" or config["inversion"] == "joint":
        if config["simulation"]["adjoint"] == False:
            job_id_fwd_wavefield = None
            if point_source:
                simulation_type = "green_structure"
            else:
                simulation_type = "correlation_structure"
        else:
            if job_id_fwd_wavefield is None:
                raise ValueError("To compute a structure kernel, "
                                 "the job id of the corresponding forward wavefield is required.")

            if point_source:
                simulation_type = "adjoint_structure_1"
            else:
                simulation_type = "adjoint_structure_2"

    else:
        raise ValueError(f"Inversion type {config['inversion']} does not exist!")

    return simulation_type, job_id_fwd_wavefield


def compute_wavefield_point_source(site, config, src_toml, rec_toml=None,
                                   job_id_fwd_wavefield=None):
    # create site and job
    site_instance = salvus_flow.api.init_site(site["site"])
    job = Job.create_new_job(site=site_instance)

    # get starttime from stf file
    starttime = scorr.extensions.salvus_extensions.extract_starttime_in_s_from_src_toml_simple(src_toml=src_toml)

    # set endtime
    if not config["simulation"]["adjoint"]:
        endtime = config["simulation"]["corr_max_lag"]
    else:
        if config["simulation"]["corr_max_lag_causal"]:
            endtime = config["simulation"]["corr_max_lag_causal"]
        else:
            endtime = config["simulation"]["corr_max_lag"]

    # figure out, what kind of simulation is needed
    simulation_type, job_id_fwd_wavefield = \
        determine_simulation_type(config=config, point_source=True, job_id_fwd_wavefield=job_id_fwd_wavefield)

    # wite salvus command
    working_dir_local = Path(config["working_dir_local"]).absolute()
    cmd_line_file = working_dir_local / "job_tracker" / (str(job.job_name) + "_salvus_command_point.sh")
    salvus_extensions.write_salvus_command(filename=cmd_line_file,
                                           type=simulation_type,
                                           mesh=config["simulation"]["mesh"],
                                           dt=config["simulation"]["dt"],
                                           starttime=starttime,
                                           endtime=endtime,
                                           src_toml=src_toml,
                                           rec_toml=rec_toml,
                                           sideset=config["simulation"]["sideset"],
                                           recording=config["simulation"]["recording"],
                                           attenuation=config["simulation"]["attenuation"],
                                           sampling_rate_boundary=config["simulation"]["sampling_rate_boundary"],
                                           sampling_rate_volume=config["simulation"]["sampling_rate_volume"],
                                           absorbing=config["simulation"]["absorbing"]["boundaries"],
                                           axis_aligned=config["simulation"]["absorbing"]["axis-aligned"],
                                           anisotropy=config["simulation"]["anisotropy"],
                                           kernel_fields=config["simulation"]["kernel-fields"])

    # save scorr status of client
    git_hash_and_diff(working_dir_local / "job_tracker" / (str(job.job_name) + "_scorr_status_client.sh"))

    if rank == 0:
        # launch salvus run
        job_id = salvus_flow.api.run_salvus(cmd_line=cmd_line_file,
                                            job=job,
                                            site=site["site"],
                                            ranks=site["ranks_salvus"],
                                            wall_time_in_seconds=site["wall_time_in_seconds_salvus"],
                                            wavefield_job_name=job_id_fwd_wavefield,
                                            quiet=not config["verbose"])
    else:
        job_id = None

    return comm.bcast(job_id, root=0)


def compute_distributed_source(site, config, job_id_wavefield):
    # retrieve information on wavefield computation
    job_wavefield = salvus_flow.api.get_output_files(job=job_id_wavefield, get_all=True)

    # create site and job
    site_instance = salvus_flow.api.init_site(site["site"] + "_scorr")
    job = Job.create_new_job(site=site_instance)

    # write configuration to local json file
    working_dir_local = Path(config["working_dir_local"]).absolute()
    json_file_local = working_dir_local / "job_tracker" / (str(job.job_name) + "_scorr_setup.json")
    write_json_file(filename=json_file_local, configuration=config)

    # copy local json file to remote site
    json_file_remote = job.input_path / "config_scorr.json"
    site_instance.remote_put(json_file_local, json_file_remote, progressbar=config["verbose"])

    # upload noise source file
    noise_source_file_remote = job.input_path / "noise_source.h5"
    site_instance.remote_put(config["noise_source"]["filename"], noise_source_file_remote,
                             progressbar=config["verbose"])

    # assemble scorr command
    if "local" in site["site"]:
        command = f"/Users/korbinian/Applications/miniconda3/envs/scorr2/bin/scorr "
    elif "wyoming" in site["site"]:
        command = f"/home/korbinian/miniconda3/envs/scorr/bin/scorr "
    elif "swp" in site["site"] or "brown" in site["site"]:
        command = f"/home/ksager/miniconda3/envs/scorr/bin/scorr "
    elif "daint" in site["site"]:
        command = f"/users/sagerk/miniconda3/envs/scorr/bin/scorr "
    elif "supermuc" in site["site"]:
        command = f"/dss/dsshome1/03/gu95lon2/miniconda3/envs/scorr/bin/scorr "
    else:
        raise ValueError("Site not yet implemented!")

    command += f"corr-source " \
               f"--json_file {json_file_remote} " \
               f"--noise_source_file {noise_source_file_remote} " \
               f"--wavefield_file {job_wavefield['wavefield_BND.h5']} " \
               f"--output_file {job.tmp_path / 'wavefield_BND.h5'}"

    # save scorr status of client
    git_hash_and_diff(working_dir_local / "job_tracker" / (str(job.job_name) + "_scorr_status_client.sh"))

    if rank == 0:
        # launch scorr run
        job_id_scorr = salvus_flow.api.run_job(cmd_line=command,
                                               job=job,
                                               site=site["site"] + "_scorr",
                                               ranks=site["ranks_scorr"],
                                               wall_time_in_seconds=site["wall_time_in_seconds_scorr"],
                                               quiet=not config["verbose"])
    else:
        job_id_scorr = None

    return comm.bcast(job_id_scorr, root=0)


def compute_wavefield_distributed_source(site, config, job_id_distsource, rec_toml=None,
                                         job_id_fwd_wavefield=None):
    # create site and job
    site_instance = salvus_flow.api.init_site(site["site"])
    job = Job.create_new_job(site=site_instance)

    # set start- and endtime
    if config["simulation"]["adjoint"]:
        starttime = config["simulation"]["green_starttime"]
        endtime = config["simulation"]["corr_max_lag"]
        rec_toml = None
    else:
        starttime = -config["simulation"]["corr_max_lag"]
        if config["simulation"]["corr_max_lag_causal"]:
            endtime = config["simulation"]["corr_max_lag_causal"]
        else:
            endtime = config["simulation"]["corr_max_lag"]

    # figure out, what kind of simulation is needed
    simulation_type, job_id_fwd_wavefield = \
        determine_simulation_type(config=config, point_source=False, job_id_fwd_wavefield=job_id_fwd_wavefield)

    # wite salvus command
    working_dir_local = Path(config["working_dir_local"]).absolute()
    cmd_line_file = working_dir_local / "job_tracker" / (str(job.job_name) + "_salvus_command_dist.sh")
    salvus_extensions.write_salvus_command(filename=cmd_line_file,
                                           type=simulation_type,
                                           mesh=config["simulation"]["mesh"],
                                           dt=config["simulation"]["dt"],
                                           starttime=starttime,
                                           endtime=endtime,
                                           rec_toml=rec_toml,
                                           sideset=config["simulation"]["sideset"],
                                           recording=config["simulation"]["recording"],
                                           attenuation=config["simulation"]["attenuation"],
                                           sampling_rate_boundary=config["simulation"]["sampling_rate_boundary"],
                                           sampling_rate_volume=config["simulation"]["sampling_rate_volume"],
                                           absorbing=config["simulation"]["absorbing"]["boundaries"],
                                           axis_aligned=config["simulation"]["absorbing"]["axis-aligned"],
                                           anisotropy=config["simulation"]["anisotropy"],
                                           kernel_fields=config["simulation"]["kernel-fields"])

    # save scorr status of client
    git_hash_and_diff(working_dir_local / "job_tracker" / (str(job.job_name) + "_scorr_status_client.sh"))

    if rank == 0:
        # launch salvus run
        job_id_correlation = salvus_flow.api.run_salvus(cmd_line=cmd_line_file,
                                                        job=job,
                                                        site=site["site"],
                                                        ranks=site["ranks_salvus"],
                                                        wall_time_in_seconds=site["wall_time_in_seconds_salvus"],
                                                        wavefield_job_name=job_id_fwd_wavefield,
                                                        boundary_job_name=job_id_distsource,
                                                        quiet=not config["verbose"])
    else:
        job_id_correlation = None

    return comm.bcast(job_id_correlation, root=0)


def build_source_kernel(site, config, job_id_green, job_id_adjoint):
    # retrieve information on Green function and adjoint computation
    job_green = salvus_flow.api.get_output_files(job=job_id_green, get_all=True)
    job_adjoint = salvus_flow.api.get_output_files(job=job_id_adjoint, get_all=True)

    # create site and job
    site_instance = salvus_flow.api.init_site(site["site"] + "_scorr")
    job = Job.create_new_job(site=site_instance)

    # write configuration to local json file
    working_dir_local = Path(config["working_dir_local"]).absolute()
    json_file_local = working_dir_local / "job_tracker" / (str(job.job_name) + "_scorr_setup.json")
    write_json_file(filename=json_file_local, configuration=config)

    # copy local json file to remote site
    json_file_remote = job.input_path / "config_scorr.json"
    site_instance.remote_put(json_file_local, json_file_remote, progressbar=config["verbose"])

    # get spectrum and number of ranks used for salvus
    with h5py.File(str(config["noise_source"]["filename"]), mode="r") as hdf5:
        spectrum = hdf5[group_name + "spectrum"][:]

        # salvus_ranks has to be consistent with site
        # SalvusOpt doesn't copy this value, so a check is not possible during an inversion
        # assert hdf5[group_name + "ranks_salvus"].value == site["ranks_salvus"], "Site config changed! Not allowed!"

    # assemble scorr command
    outputfile = job.output_path / "source_kernel.h5"

    if "local" in site["site"]:
        command = f"/Users/korbinian/Applications/miniconda3/envs/scorr2/bin/scorr "
    elif "wyoming" in site["site"]:
        command = f"/home/korbinian/miniconda3/envs/scorr/bin/scorr "
    elif "swp" in site["site"] or "brown" in site["site"]:
        command = f"/home/ksager/miniconda3/envs/scorr/bin/scorr "
    elif "daint" in site["site"]:
        command = f"/users/sagerk/miniconda3/envs/scorr/bin/scorr "
    elif "supermuc" in site["site"]:
        command = f"/dss/dsshome1/03/gu95lon2/miniconda3/envs/scorr/bin/scorr "
    else:
        raise ValueError("Site not yet implemented!")

    command += f"source-kernel " \
               f"--json_file {json_file_remote} " \
               f"--green_file {job_green['wavefield_BND.h5']} " \
               f"--adjoint_file {job_adjoint['wavefield_BND.h5']} " \
               f"--spectrum {spectrum[0]:f},{spectrum[1]:f},{spectrum[2]:f} " \
               f"--output_file {outputfile}"

    if rank == 0:
        # launch scorr run
        job_id_scorr = salvus_flow.api.run_job(cmd_line=command,
                                               job=job,
                                               site=site["site"] + "_scorr",
                                               ranks=site["ranks_scorr"],
                                               wall_time_in_seconds=site["wall_time_in_seconds_scorr"],
                                               quiet=not config["verbose"])
    else:
        job_id_scorr = None

    return comm.bcast(job_id_scorr, root=0)


def wavefield_diff(site, config, job_id_ref, job_id_pert, boundary=True, minus=True):
    # retrieve information on both simulations
    job_ref = salvus_flow.api.get_output_files(job=job_id_ref, get_all=True)
    job_pert = salvus_flow.api.get_output_files(job=job_id_pert, get_all=True)

    # create site and job
    site_instance = salvus_flow.api.init_site(site["site"] + "_scorr")
    job = Job.create_new_job(site=site_instance)

    # assemble scorr command
    if "local" in site["site"]:
        command = f"/Users/korbinian/Applications/miniconda3/envs/scorr2/bin/scorr "
    elif "wyoming" in site["site"]:
        command = f"/home/korbinian/miniconda3/envs/scorr/bin/scorr "
    elif "swp" in site["site"] or "brown" in site["site"]:
        command = f"/home/ksager/miniconda3/envs/scorr/bin/scorr "
    elif "daint" in site["site"]:
        command = f"/users/sagerk/miniconda3/envs/scorr/bin/scorr "
    elif "supermuc" in site["site"]:
        command = f"/dss/dsshome1/03/gu95lon2/miniconda3/envs/scorr/bin/scorr "
    else:
        raise ValueError("Site not yet implemented!")

    command += f"wavefield-diff " \
        f"--minus {minus} "

    if boundary:
        command += f"--ref_file {job_ref['wavefield_BND.h5']} " \
            f"--pert_file {job_pert['wavefield_BND.h5']} "
    else:
        command += f"--ref_file {job_ref['wavefield.h5']} " \
            f"--pert_file {job_pert['wavefield.h5']} "

    if rank == 0:
        # launch scorr run
        job_id_scorr = salvus_flow.api.run_job(cmd_line=command,
                                               job=job,
                                               site=site["site"] + "_scorr",
                                               ranks=site["ranks_scorr"],
                                               wall_time_in_seconds=site["wall_time_in_seconds_scorr"],
                                               quiet=not config["verbose"])
    else:
        job_id_scorr = None

    return comm.bcast(job_id_scorr, root=0)
