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
import warnings
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
import salvus_flow.api
# from mpl_toolkits.basemap import Basemap
from scorr.addons import rank, rank_element_dictionary, group_name
from scorr.extensions import rotation_distance, kdtree_spherical
from scorr.tasks import simulations
from tqdm import tqdm


def setup_noise_source(config, site, config_noise_source):
    # get file to generate noise source
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        get_NoiseSource_file(config=config, site=site)

    # change distribution according to a json file in project/config
    with h5py.File(str(config["noise_source"]["filename"]), "r") as hdf5:
        n_elements_global = hdf5[group_name + "coordinates"].shape[0]
        rank_elements = rank_element_dictionary(n_elements_global)

        coordinates = hdf5[group_name + "coordinates"][rank_elements[rank]
                                                       [0]:rank_elements[rank][1], :, :]
        distribution = hdf5[group_name +
                            "distribution"][rank_elements[rank][0]:rank_elements[rank][1], :]

    if config_noise_source["type"] == "laura" or config_noise_source["type"] == "file":
        with h5py.File(str(config_noise_source["filename"]), "r") as hdf5_laura:
            lon_laura, lat_laura = hdf5_laura["/coordinates"][:]
            coord_laura = pd.DataFrame(
                np.empty((len(lat_laura), 2)), columns=["lat", "lon"])
            coord_laura["lon"] = lon_laura
            coord_laura["lat"] = lat_laura

            # create kd tree
            snn = kdtree_spherical.SphericalNearestNeighbour(coord_laura)

            # get laura's inversion result
            dist_laura = hdf5_laura["/distr_basis"][0, :]

    # set up distribution
    # m = Basemap(projection='hammer', lon_0=0.0, resolution='c')

    for id_element_local in tqdm(range(coordinates.shape[0])):
        for id_gll in range(coordinates.shape[1]):
            x, y, z = coordinates[id_element_local, id_gll, :]

            # if config_noise_source["ocean_only"]:
            #     lat, lon = rotation_distance.from_cartesian_to_latlon(x, y, z)
            #     x_map, y_map = m(lon, lat)
            #     if m.is_land(x_map, y_map):
            #         distribution[id_element_local, id_gll] = 0.0
            #         continue

            if config_noise_source["type"] == "homogeneous":
                # if y > 15e3 and y < 45e3:
                distribution[id_element_local, id_gll] = config_noise_source["homog_magnitude"]

            elif config_noise_source["type"] == "random":
                distribution[id_element_local, id_gll] = config_noise_source["homog_magnitude"] * \
                    (2.0 * (np.random.rand(1) - 0.5))[0]

            elif config_noise_source["type"] == "gaussian":
                # distribution[id_element_local, id_gll] = config_noise_source["homog_magnitude"]
                distribution[id_element_local, id_gll] = 0.0 * \
                    config_noise_source["homog_magnitude"]
                gauss_magnitude = config_noise_source["gaussian"]["magnitude"]
                gauss_x = config_noise_source["gaussian"]["x"]
                gauss_y = config_noise_source["gaussian"]["y"]
                gauss_width_x = config_noise_source["gaussian"]["width_x"]
                gauss_width_y = config_noise_source["gaussian"]["width_y"]

                if config["simulation"]["spherical"] == False:
                    index_coord = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
                    if config["simulation"]["sideset"][0] == "x":
                        id_component = 0
                    elif config["simulation"]["sideset"][0] == "y":
                        id_component = 1
                    else:
                        id_component = 2

                    for _i in range(len(gauss_magnitude)):
                        distribution[id_element_local, id_gll] += gauss_magnitude[_i] * np.exp(-(
                            (coordinates[id_element_local, id_gll,
                                         index_coord[id_component][0]] - gauss_x[_i]) ** 14
                            / gauss_width_x[_i] ** 14 +
                            (coordinates[id_element_local, id_gll,
                                         index_coord[id_component][1]] - gauss_y[_i]) ** 14
                            / gauss_width_y[_i] ** 14))


                else:
                    lat, lon = rotation_distance.from_cartesian_to_latlon(
                        x, y, z)

                    for _i in range(len(gauss_magnitude)):
                        theta = np.deg2rad(67.0)
                        x_true = lon * np.cos(theta) + lat * np.sin(theta)
                        y_true = -lon * np.sin(theta) + lat * np.cos(theta) + 0.01625 # + 0.0325 # - 0.009
                        gauss_x_true = gauss_x[_i] * np.cos(theta) + gauss_y[_i] * np.sin(theta) + 0.01
                        gauss_y_true = -gauss_x[_i] * np.sin(theta) + gauss_y[_i] * np.cos(theta) + 0.01

                        distribution[id_element_local, id_gll] += gauss_magnitude[_i] * np.exp(-((x_true - gauss_x_true) ** 8 / gauss_width_x[_i] ** 8 + (y_true - gauss_y_true) ** 8 / gauss_width_y[_i] ** 8))

                        # distribution[id_element_local, id_gll] += gauss_magnitude[_i] * np.exp(-(
                        #     (lon - gauss_x[_i]) ** 2 / gauss_width_x[_i] ** 2 +
                        #     (lat - gauss_y[_i]) ** 2 / gauss_width_y[_i] ** 2))

            elif config_noise_source["type"] == "ring":

                distribution[id_element_local, id_gll] = 0.0 * \
                    config_noise_source["homog_magnitude"]

                # ring parameters are currently hard-coded
                ring_center_x = 0.0
                ring_center_y = 0.0
                ring_radius = 12.5e3
                ring_thickness = 2.0e3
                ring_radial_smoothing = 600.0
                ring_angle_center = -90.0
                ring_angle_coverage = 90.0
                ring_angle_smoothing = 15.0

                radius = np.sqrt((x-ring_center_x)**2 + (y-ring_center_y)**2)
                angle = np.rad2deg(np.arctan2(
                    x-ring_center_x, y-ring_center_y))
                if np.isnan(angle):
                    continue

                # construct ring
                radial_taper = 1.0e10 * np.exp(-np.abs(radius-ring_radius)
                                               ** 2 / ring_radial_smoothing**2)
                radial_pattern = float(radius > (
                    ring_radius-ring_thickness/2) and radius < (ring_radius+ring_thickness/2))
                distribution[id_element_local,
                             id_gll] = radial_taper * radial_pattern

                # # select angle
                # angle_taper = np.exp(-(angle-ring_angle_center)
                #                      ** 2 / ring_angle_smoothing**2)
                # angle_pattern = float(
                #     angle > ring_angle_center - ring_angle_coverage and angle < ring_angle_center + ring_angle_coverage)
                # distribution[id_element_local,
                #              id_gll] *= angle_taper * angle_pattern

            elif config_noise_source["type"] == "patch":
                if config["simulation"]["spherical"] == True:
                    raise NotImplementedError()

                hsize = 3e3
                center = 30e3
                if x < 3*hsize and x >= hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10
                if x < hsize and x >= -hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10
                if x < -hsize and x >= -3*hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10

                center = 30e3 + 2*hsize
                if x < 3*hsize and x >= hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10
                if x < hsize and x >= -hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10
                if x < -hsize and x >= -3*hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10

                center = 30e3 - 2*hsize
                if x < 3*hsize and x >= hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10
                if x < hsize and x >= -hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10
                if x < -hsize and x >= -3*hsize:
                    if y < center + hsize and y >= center - hsize:
                        distribution[id_element_local, id_gll] = 1.0e10


            elif config_noise_source["type"] == "laura" or config_noise_source["type"] == "file":
                lat_salvus, lon_salvus = rotation_distance.from_cartesian_to_latlon(
                    x=x, y=y, z=z)
                _, nearest = snn.query(pd.DataFrame(
                    {"lat": [lat_salvus], "lon": [lon_salvus]}), k=1)
                distribution[id_element_local, id_gll] = 1.0e10 * \
                    dist_laura[nearest[0]]

    # write distribution and spectrum to file
    with h5py.File(str(config["noise_source"]["filename"]), "a") as hdf5:
        hdf5[group_name + "distribution"][rank_elements[rank]
                                          [0]:rank_elements[rank][1], :] = distribution
        hdf5[group_name + "spectrum"][0] = config_noise_source["spectrum"]["f_peak"],
        hdf5[group_name + "spectrum"][1] = config_noise_source["spectrum"]["bandwidth"]
        hdf5[group_name + "spectrum"][2] = config_noise_source["spectrum"]["strength"]


def get_NoiseSource_file(config, site):
    # change config to run 1 time step efficiently (without saving wavefields, etc.)
    config_tmp = deepcopy(config)
    config_tmp["simulation"]["green_starttime"] = 0.0
    config_tmp["simulation"]["corr_max_lag"] = 0.0
    config_tmp["simulation"]["dt"] = 0.01
    config_tmp["inversion"] = ""
    config_tmp["simulation"]["adjoint"] = False
    config_tmp["simulation"]["recording"] = "u_ELASTIC"
    config_tmp["simulation"]["absorbing"]["boundaries"] = None
    config_tmp["verbose"] = True

    # change site to have a small wall time, should start faster then
    site_tmp = deepcopy(site)
    site_tmp["wall_time_in_seconds_salvus"] = int(1800)

    # compute 1 time step
    if not (config_tmp["noise_source"]["filename"].parent / "wavefield_BND.h5").exists():
        job_id = simulations.compute_wavefield_point_source(site=site_tmp, config=config_tmp,
                                                            src_toml="noise_source_setup")

        # wait for job to finish
        status = salvus_flow.api.wait_for_job(job=job_id,
                                              ping_interval_in_seconds=site["ping_interval_in_seconds"],
                                              quiet=not config["verbose"])
        assert status.value == 2, f"Job {job_id} didn't finish as expected!"

        # get pathes of files of dummy run
        salvus_flow.api.get_output(job=job_id, destination=config_tmp["noise_source"]["filename"].parent,
                                   get_all=True, force=True, quiet=not config_tmp["verbose"])

    # set dtype depending on precision
    if config["simulation"]["precision"] == "single":
        dtype = "32"
    else:
        dtype = "64"

    # write noise source file with information from wavefield file
    with h5py.File(str(config_tmp["noise_source"]["filename"].parent / "wavefield_BND.h5"), 'r') as hdf5_input:
        # noise source
        with h5py.File(str(config_tmp["noise_source"]["filename"]), 'w') as hdf5_output1:
            # copy connectivity, globalElementIds and coordinates
            hdf5_output1[group_name +
                         "connectivity"] = hdf5_input["/ELASTIC_BND/connectivity"][:]
            hdf5_output1[group_name +
                         "globalElementIds"] = hdf5_input["/ELASTIC_BND/globalElementIds"][:]
            hdf5_output1[group_name +
                         "coordinates"] = hdf5_input["/ELASTIC_BND/coordinates"][:]

            # get information about sizes
            n_connectivity = hdf5_output1[group_name + "connectivity"].shape[0]
            n_elements = hdf5_output1[group_name + "coordinates"].shape[0]
            n_gll = hdf5_output1[group_name + "coordinates"].shape[1]

            # create DISTC_BND/distribution filled with zeros
            hdf5_output1.create_dataset(name=group_name + "distribution",
                                        shape=hdf5_output1[group_name +
                                                           "coordinates"].shape[:-1],
                                        dtype='float' + dtype)

            # create DISTC_BND/spectrum filled with zeros
            hdf5_output1.create_dataset(
                name=group_name + "spectrum", shape=(3,), dtype='float' + dtype)
            hdf5_output1[group_name +
                         "spectrum"].dims[0].label = '[ f_peak | bandwidth | strength ]'

            # safe salvus ranks in file
            hdf5_output1[group_name + "ranks_salvus"] = site["ranks_salvus"]

    if rank == 0:
        # write corresponding xdmf file
        with open(config_tmp["noise_source"]["filename"].parent / (
                config_tmp["noise_source"]["filename"].stem + ".xdmf"),
                "w") as file:
            file.write(f'<Xdmf Version="2.0">\n'
                       f'<Domain>\n'
                       f'   <Topology TopologyType="Quadrilateral" NumberOfElements="{n_connectivity}">\n'
                       f'       <DataItem Format="HDF" DataType="Int" Dimensions="{n_connectivity} 4">\n'
                       f'           {config["noise_source"]["filename"].name}:{group_name}connectivity\n'
                       f'       </DataItem>\n'
                       f'   </Topology>\n'
                       f'    <Geometry GeometryType="XYZ">\n'
                       f'        <DataItem Dimensions="{n_elements} {n_gll} 3" Format="HDF">\n'
                       f'            {config["noise_source"]["filename"].name}:{group_name}coordinates\n'
                       f'        </DataItem>\n'
                       f'    </Geometry>\n'
                       f'    <Grid Name="noise_source">\n'
                       f'        <Time Value="0" />\n'
                       f'        <Topology Reference="/Xdmf/Domain/Topology[1]" />\n'
                       f'        <Geometry Reference="/Xdmf/Domain/Geometry[1]" />\n'
                       f'        <Attribute AttributeType="Scalar" Center="Node" Name="PSD_ZZ">\n'
                       f'            <DataItem Dimensions="{n_elements} {n_gll} " ItemType="HyperSlab" Type="HyperSlab">\n'
                       f'                <DataItem Dimensions="3 2" Format="XML">\n'
                       f'                    0 0\n'
                       f'                    1 1\n'
                       f'                    {n_elements} {n_gll}\n'
                       f'                </DataItem>\n'
                       f'                <DataItem Dimensions="{n_elements} {n_gll}" Format="HDF" Name="Points">\n'
                       f'                    {config["noise_source"]["filename"].name}:{group_name}distribution\n'
                       f'                </DataItem>\n'
                       f'            </DataItem>\n'
                       f'        </Attribute>\n'
                       f'    </Grid>\n'
                       f'</Domain>\n'
                       f'</Xdmf>')

        # delete salvus flow job and source files
        # salvus_flow.api.delete_job(job=job_id, quiet=True)
        # os.remove(config_tmp["noise_source"]["filename"].parent / "wavefield_BND.h5")
        # os.remove(config_tmp["noise_source"]["filename"].parent / "wavefield_BND_ELASTIC_BND.xdmf")
        # try:
        #     os.remove(config_tmp["noise_source"]["filename"].parent / "stdout")
        #     os.remove(config_tmp["noise_source"]["filename"].parent / "stderr")
        # except OSError:
        #     pass
