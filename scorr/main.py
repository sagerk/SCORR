#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import inspect
import os
from pathlib import Path
from shutil import copy2, copytree, rmtree

import click
import numpy as np

import scorr.addons
import scorr.api
from scorr.addons import comm, rank
from scorr.extensions import job_tracker, job_tracker_hessian, job_tracker_ghetto, scorr_extensions


@click.group()
def main():
    """
    SCOOOOOOOOOOOOOOORR
    """
    pass


@main.command(name="init-project")
@click.argument("project_name")
def init_project(project_name):
    """
    Initialize project.
    """
    project_name = Path(project_name).absolute()

    if project_name.exists():
        print("Warning! Project folder exists already!")

    if rank == 0:
        os.makedirs(project_name, exist_ok=True)
        os.makedirs(project_name / "config", exist_ok=True)
        os.makedirs(project_name / "job_tracker", exist_ok=True)
        os.makedirs(project_name / "job_tracker" / "logs", exist_ok=True)
        os.makedirs(project_name / "noise_source", exist_ok=True)
    comm.Barrier()

    # load default configuration for scorr and modifiy "working_dir_local" accordingly
    config_dir = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) / "config"
    config = scorr_extensions.load_configuration(filename=config_dir / "scorr_template.json", type="scorr")
    config["working_dir_local"] = project_name

    # save scorr configuration to project folder
    scorr.addons.write_json_file(filename=project_name / "config" / "scorr.json", configuration=config)

    # copy default site and noise source configuration to project folder
    if rank == 0:
        copy2(config_dir / "site_template.json", project_name / "config" / "site.json")
        copy2(config_dir / "noise_source_template.json", project_name / "config" / "noise_source.json")
        copy2(config_dir / "measurement_template.json", project_name / "config" / "measurement.json")
    comm.Barrier()


@main.command(name="copy-project")
@click.argument("src_project_name")
@click.argument("dst_project_name")
def copy_project(src_project_name, dst_project_name):
    """
    Copy project.
    """
    src_project_name = Path(src_project_name).absolute()
    dst_project_name = Path(dst_project_name).absolute()

    if dst_project_name.exists():
        print("Destination folder exists already!")
        return

    if rank == 0:
        copytree(src=src_project_name, dst=dst_project_name)
    comm.Barrier()

    # load default configuration for scorr and modifiy "working_dir_local" accordingly
    config = scorr.addons.load_json_file(filename=dst_project_name / "config" / "scorr.json")
    config["working_dir_local"] = dst_project_name

    # save scorr configuration to project folder
    scorr.addons.write_json_file(filename=dst_project_name / "config" / "scorr.json", configuration=config)


@main.command(name="cleanup-project")
@click.argument("project_name")
def cleanup_project(project_name):
    """
    Clean up project.
    """
    project_name = Path(project_name).absolute()

    if not project_name.exists():
        print("Project does not exist!")
        return

    if rank == 0:
        rmtree(project_name / "adjoint_stf", ignore_errors=True)
        rmtree(project_name / "correlations", ignore_errors=True)
        rmtree(project_name / "kernels", ignore_errors=True)
        rmtree(project_name / "seismo", ignore_errors=True)
        rmtree(project_name / "test_files", ignore_errors=True)

        for file in (project_name / "job_tracker").glob("*"):
            os.remove(file)
        for file in (project_name).glob("*_obs.e"):
            os.remove(file)
        for file in (project_name).glob("*_step.e"):
            os.remove(file)


@main.command(name="corr-source")
@click.option("--json_file", type=str, required=True,
              help="JSON file for scorr configuration.")
@click.option("--noise_source_file", type=str, required=True,
              help="Filename of noise source.")
@click.option("--wavefield_file", type=str, required=True,
              help="Filename of Green function.")
@click.option("--output_file", type=str, required=True,
              help="Output filename for correlation source.")
def make_correlation_source(json_file, noise_source_file, wavefield_file, output_file):
    """
    Convolve Green function with PSD distribution.
    """
    scorr.api.make_correlation_source(json_file=json_file,
                                      noise_source_file=noise_source_file,
                                      wavefield_file=wavefield_file,
                                      output_file=output_file)


@main.command(name="source-kernel")
@click.option("--json_file", type=str, required=True,
              help="JSON file for scorr configuration.")
@click.option("--green_file", type=str, required=True,
              help="Filename of Green function.")
@click.option("--adjoint_file", type=str, required=True,
              help="Filename of adjoint wavefield.")
@click.option("--spectrum", type=str, required=True,
              help="Spectrum f_peak,bandwidth,strength.")
@click.option("--output_file", type=str, required=True,
              help="Output filename for correlation source.")
def build_source_kernel(json_file, green_file, adjoint_file, spectrum, output_file):
    """
    Build kernel for the distribution of PSD.
    """
    # convert spectrum to list and check for correct size
    spectrum = [float(i) for i in spectrum.split(",")]
    assert np.size(spectrum) == 3

    # build source kernel
    scorr.api.build_source_kernel(json_file=json_file,
                                  green_file=green_file,
                                  adjoint_file=adjoint_file,
                                  spectrum=spectrum,
                                  output_file=output_file)

@main.command(name="wavefield-diff")
@click.option("--ref_file", type=str, required=True,
              help="Filename of reference wavefield.")
@click.option("--pert_file", type=str, required=True,
              help="Filename of perturbation wavefield.")
@click.option("--minus", type=bool, required=True,
              help="True for minus, false for plus.")
def wavefield_diff(ref_file, pert_file, minus):
    """
    Compute wavefield differences.
    """
    scorr.api.wavefield_diff(ref_file=ref_file,
                             pert_file=pert_file,
                             minus=minus)


@main.command()
def status():
    """
    Get the status of all jobs.
    """

    ref_stations = job_tracker.get_all_reference_stations()

    if not ref_stations:
        # print("No entries in database!")
        return

    fstr = lambda s: str(s).split("@")[0] or "None   "
    cstr = lambda s: "yellow" if (s is not None and "local" in str(s).split("@")[1]) \
        else "red" if (s is None) else "cyan" if ("lion" in str(s).split("@")[1]) else "green"

    click.echo("\n  {:20s} {:22s} {:22s} {:22s} {:22s} {:22s} {:22s} {:22s} {:22s}".format(
        "Reference station", "Green function", "Correlation source", "Correlation function",
        "Adjoint run 1", "Source kernel", "Dist. adjoint stf", "Adjoint run 2", "Inversion type"))
    for ref_station in ref_stations:
        info = f"last updated: {ref_station.last_updated.humanize()}"

        click.echo(
            "* " +
            click.style(f"{ref_station.ref_station:20s} ", fg="magenta") +
            click.style(f"{fstr(ref_station.job_id_green):22s} ",
                        fg=cstr(ref_station.job_id_green)) +
            click.style(f"{fstr(ref_station.job_id_corr_source):22s} ",
                        fg=cstr(ref_station.job_id_corr_source)) +
            click.style(f"{fstr(ref_station.job_id_correlation):22s} ",
                        fg=cstr(ref_station.job_id_correlation)) +
            click.style(f"{fstr(ref_station.job_id_adjoint_1):22s} ",
                        fg=cstr(ref_station.job_id_adjoint_1)) +
            click.style(f"{fstr(ref_station.job_id_source_kernel):22s} ",
                        fg=cstr(ref_station.job_id_source_kernel)) +
            click.style(f"{fstr(ref_station.job_id_dist_adjstf):22s} ",
                        fg=cstr(ref_station.job_id_dist_adjstf)) +
            click.style(f"{fstr(ref_station.job_id_adjoint_2):22s} ",
                        fg=cstr(ref_station.job_id_adjoint_2)) +
            click.style(f"{fstr(ref_station.inversion_type):22s} ",
                        fg="blue") +
            click.style(f"({info})", fg="white")
        )

    print("")


@main.command()
def status_hessian():
    """
    Get the status of all jobs.
    """

    ref_stations = job_tracker_hessian.get_all_reference_stations()

    if not ref_stations:
        # print("No entries in database!")
        return

    fstr = lambda s: str(s).split("@")[0] or "None   "
    cstr = lambda s: "yellow" if (s is not None and "local" in str(s).split("@")[1]) \
        else "red" if (s is None) else "cyan" if ("lion" in str(s).split("@")[1]) else "green"

    click.echo("\n  {:20s} {:22s} {:22s} {:22s} {:22s} {:22s} {:22s} {:22s} {:22s}".format(
        "Reference station", "p", "source_z", "z", "source_q", "q", "w", "source_kernel_pPG", "source_kernel_uPdG",
        "Inversion type"))
    for ref_station in ref_stations:
        info = f"last updated: {ref_station.last_updated.humanize()}"

        click.echo(
            "* " +
            click.style(f"{ref_station.ref_station:20s} ", fg="magenta") +
            click.style(f"{fstr(ref_station.job_id_p):22s} ",
                        fg=cstr(ref_station.job_id_p)) +
            click.style(f"{fstr(ref_station.job_id_source_z):22s} ",
                        fg=cstr(ref_station.job_id_source_z)) +
            click.style(f"{fstr(ref_station.job_id_z):22s} ",
                        fg=cstr(ref_station.job_id_z)) +
            click.style(f"{fstr(ref_station.job_id_source_q):22s} ",
                        fg=cstr(ref_station.job_id_source_q)) +
            click.style(f"{fstr(ref_station.job_id_q):22s} ",
                        fg=cstr(ref_station.job_id_q)) +
            click.style(f"{fstr(ref_station.job_id_w):22s} ",
                        fg=cstr(ref_station.job_id_w)) +
            click.style(f"{fstr(ref_station.job_id_source_kernel_pPG):22s} ",
                        fg=cstr(ref_station.job_id_source_kernel_pPG)) +
            click.style(f"{fstr(ref_station.job_id_source_kernel_uPdG):22s} ",
                        fg=cstr(ref_station.job_id_source_kernel_uPdG)) +
            click.style(f"{fstr(ref_station.inversion_type):22s} ",
                        fg="blue") +
            click.style(f"({info})", fg="white")
        )

    print("")


@main.command()
def status_ghetto():
    """
    Get the status of all jobs.
    """

    ref_stations = job_tracker_ghetto.get_all_reference_stations()

    if not ref_stations:
        # print("No entries in database!")
        return

    fstr = lambda s: str(s).split("@")[0] or "None   "
    cstr = lambda s: "yellow" if (s is not None and "local" in str(s).split("@")[1]) \
        else "red" if (s is None) else "cyan" if ("lion" in str(s).split("@")[1]) else "green"

    click.echo("\n  {:20s} {:22s} {:22s} {:22s}".format(
        "Reference station", "job_id_dGm", "job_id_dudagger",
        "Inversion type"))
    for ref_station in ref_stations:
        info = f"last updated: {ref_station.last_updated.humanize()}"

        click.echo(
            "* " +
            click.style(f"{ref_station.ref_station:20s} ", fg="magenta") +
            click.style(f"{fstr(ref_station.job_id_dGm):22s} ",
                        fg=cstr(ref_station.job_id_dGm)) +
            click.style(f"{fstr(ref_station.job_id_dudagger):22s} ",
                        fg=cstr(ref_station.job_id_dudagger)) +
            click.style(f"{fstr(ref_station.inversion_type):22s} ",
                        fg="blue") +
            click.style(f"({info})", fg="white")
        )

    print("")


@main.command(name="remove-reference-station")
@click.argument("ref_station", type=str)
def remove_reference_station(ref_station):
    """
    Delete reference station from database.
    """
    job_tracker.remove_reference_station(ref_station=ref_station)


@main.command(name="delete-database")
def cleanup_database():
    """
    Delete all reference stations from database.
    """
    # while True:
    #     choice = input("Ary you sure? [yn] ")
    #     if choice in "Yy":
    #         break
    #     elif choice in "Nn":
    #         return

    filename = Path.home() / Path(".scorr-ref_station-tracker.sqlite")
    if filename.exists():
        os.remove(filename)

    filename = Path.home() / Path(".scorr-ref_station-tracker-hessian.sqlite")
    if filename.exists():
        os.remove(filename)

    filename = Path.home() / Path(".scorr-ref_station-tracker-ghetto.sqlite")
    if filename.exists():
        os.remove(filename)
