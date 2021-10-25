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
from pathlib import Path

import h5py
# import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

from scorr import addons
from scorr.addons import comm, rank, size
from scorr.extensions import rotation_distance


class Wavefield():
    @addons.verbose_mode
    def __init__(self, filename, starttime, endtime, sampling_rate=1, rotate=False, verbose=False):
        """
        initialize wavefield

        :param filename:
        :param starttime:
        :param endtime:
        :param verbose:
        """
        # assume here that we only have wavefields with endtime > 0.0
        # important to get length of causal branch
        assert endtime > 0.0, "endtime has to be > 0"

        dataset = "/ELASTIC_BND/"
        with h5py.File(str(filename), 'r') as file:
            self.connectivity = file[dataset + "connectivity"][:]
            self.globalElementIds = file[dataset + "globalElementIds"][:]

            self.n_elements_global = file[dataset + "coordinates"].shape[0]
            rank_elements = addons.rank_element_dictionary(
                self.n_elements_global)
            self.coordinates = file[dataset + "coordinates"][rank_elements[rank]
                                                             [0]:rank_elements[rank][1], :, :]

            self.n_elements_local = self.coordinates.shape[0]
            self.n_gll = self.coordinates.shape[1]
            self.n_components = self.coordinates.shape[2]

            # sampling rate used in salvus for storing boundary/volume data
            self.sampling_rate = sampling_rate

            # get number of simulated time steps and infer time step
            self.nt_stored = file[dataset + "data"].shape[0]
            self.dt = (abs(starttime) + abs(endtime)) / (self.nt_stored - 1)
            self.fs = 1 / self.dt
            self.starttime = starttime
            self.endtime = endtime
            self.time = file[dataset + "time"][:]

            # compute number of time steps for causal branch and full correlation
            if abs(starttime) > abs(endtime):
                self.nt_longest_branch = int(
                    np.round(abs(starttime) / self.dt + 1, 0))
                self.nt_full = 2 * self.nt_longest_branch - 1
            else:
                self.nt_longest_branch = int(
                    np.round(endtime / self.dt + 1, 0))
                self.nt_full = 2 * self.nt_longest_branch - 1

            # read in wavefield
            self.wavefield = file["/ELASTIC_BND/data"][:,
                                                       rank_elements[rank][0]:rank_elements[rank][1], :, :]

            # rotate wavefield
            self.is_rotated = False
            if rotate:
                self.rotate_wavefield(verbose=verbose)
                self.is_rotated = True

    def __str__(self):
        return f"dt of wavefield: {self.dt}"

    @addons.verbose_mode
    def rotate_wavefield(self, verbose=False):
        """
        rotate wavefield

        :param verbose:
        :return:
        """
        if self.is_rotated:
            warnings.warn(
                "wavefield is already rotated - set is_rotated to False if another rotation is actually wanted")
            return

        for id_element_local in range(self.n_elements_local):
            for id_gll in range(self.n_gll):
                x, y, z = self.coordinates[id_element_local, id_gll, :]
                lat, lon = rotation_distance.from_cartesian_to_latlon(x, y, z)
                R = rotation_distance.get_transformation_matrix_3d(
                    latitude=lat, longitude=lon)

                self.wavefield[:, id_element_local, :, id_gll] = np.einsum(
                    'ij,kj->ki', R, self.wavefield[:, id_element_local, :, id_gll], optimize=True)

    def get_coordinates_of_gll(self, id_element, id_gll):
        """
        get coordinates of gll point

        :param id_element:
        :param id_gll:
        :return:
        """
        return self.coordinates[id_element, id_gll, :]

    def get_wavefield_comp_at_gll(self, id_element, id_gll, id_component_wavefield):
        """
        get one wavefield component of one gll point

        :param id_element:
        :param id_gll:
        :param id_component_wavefield:
        :return:
        """
        def find_nearest(array, value):
            return (np.abs(array - value)).argmin()

        # spatial filter for body waves
        v_min = 4.7e3
        v_taper = 1.0e3

        # x_src = 0.0
        # y_src = 0.0
        # y_src = -30.0e3
        # z_src = 6370998.0

        # x_src = -2.36408e+06
        # y_src = -4.75004e+06
        # z_src = 3.52674e+06

        x_src = 8348.925758
        y_src = 9978.470580
        z_src = 6370999.0

        distance = np.sqrt((self.coordinates[id_element, id_gll, 0] - x_src)**2 +
                (self.coordinates[id_element, id_gll, 1] - y_src) ** 2 +
                (self.coordinates[id_element, id_gll, 2] - z_src) ** 2)

        t_min = distance / v_min
        t_max = distance / (v_min - v_taper)

        # if rank == 0:
        #     print(self.time)
        # time = self.time + 9.4

        # time = self.time + 10.49
        # time = self.time + 9.79

        time = self.time + 0.0
        # if rank == 0:
        #     print(time)

        taper = np.ones_like(time)
        taper[time >= t_max] = 0.0
        index_t_min = find_nearest(time, t_min)
        index_t_max = find_nearest(time, t_max)

        taper_length = 2*(index_t_max - index_t_min) + 1
        taper_han = np.hanning(taper_length)[int(taper_length/2):]
        taper[index_t_min:index_t_max+1] *= taper_han

        return self.wavefield[:, id_element, id_component_wavefield, id_gll], taper
        # return self.wavefield[:, id_element, id_component_wavefield, id_gll]

    def get_pad_wavefield_comp_at_gll(self, id_element, id_gll, id_component_wavefield):
        """
        get one wavefield component of one gll point, padded with zeros to full length

        :param id_element:
        :param id_gll:
        :param id_component_wavefield:
        :return:
        """
        time_series, taper = self.get_wavefield_comp_at_gll(
            id_element, id_gll, id_component_wavefield)
        add_n = self.nt_full - self.nt_stored

        time_series_padded = np.pad(
            time_series, (add_n, 0), 'constant', constant_values=(0, 0))
        taper_padded = np.pad(taper, (add_n, 0),
                              'constant', constant_values=(0, 0))

        return time_series_padded, taper_padded

    def get_pad_rev_wavefield_comp_at_gll(self, id_element, id_gll, id_component_wavefield):
        """
        get one wavefield component of one gll point, padded with zeros to full length and time reversed

        :param id_element:
        :param id_gll:
        :param id_component_wavefield:
        :return:
        """
        wavefield, taper = self.get_pad_wavefield_comp_at_gll(
            id_element, id_gll, id_component_wavefield)
        return np.flipud(wavefield), np.flipud(taper)

    def write_wavefield(self, filename):
        rank_elements = addons.rank_element_dictionary(self.n_elements_global)

        with h5py.File(str(filename), mode='w') as hdf5:
            # coordinates
            shape_coord = (self.n_elements_global,
                           self.n_gll, self.n_components)
            dset_coord = hdf5.create_dataset(
                "/ELASTIC_BND/coordinates", shape_coord, dtype='float32')
            dset_coord[rank_elements[rank][0]                       :rank_elements[rank][1], :, :] = self.coordinates

            # connect
            hdf5.create_dataset("/ELASTIC_BND/connectivity", shape=self.connectivity.shape,
                                dtype='int32', data=self.connectivity)

            # id
            hdf5.create_dataset("/ELASTIC_BND/globalElementIds", shape=self.globalElementIds.shape,
                                dtype='int32', data=self.globalElementIds)

            # time - begins with 0 here for simplicity
            time = np.linspace(-1.0, 7.0, self.nt_stored)
            hdf5.create_dataset("/ELASTIC_BND/time",
                                (self.nt_stored,), dtype='float32', data=time)

            # wavefield
            shape_data = (self.nt_stored, self.n_elements_global, self.n_gll)
            dset_data = hdf5.create_dataset("/ELASTIC_BND/data", shape_data, dtype='float32',
                                            chunks=(shape_data[0], 1, shape_data[2]))
            dset_data[:, rank_elements[rank][0]:rank_elements[rank]
                      [1], :] = self.wavefield[:, :, 0, :]

        self.write_xdmf_file(filename_xdmf=Path(filename).absolute().parent / (Path(filename).stem + ".xdmf"),
                             filename_hdf5=filename)

    def write_xdmf_file(self, filename_xdmf, filename_hdf5, verbose=False):
        """
        write xdmf file for wavefield

        :param filename_xdmf:
        :param filename_hdf5:
        :param verbose:
        :return:
        """

        def write_xdmf_header(file, filename_hdf5):
            file.write(f'<?xml version="1.0" ?>\n'
                       f'<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
                       f'<Xdmf Version="2.0">\n'
                       f'\t<Domain>\n'
                       f'\t\t<Topology TopologyType="Quadrilateral" NumberOfElements="{self.connectivity.shape[0]}">\n'
                       f'\t\t\t<DataItem Format="HDF" DataType="Int" Dimensions="{self.connectivity.shape[0]} 4">\n'
                       f'\t\t\t\t{filename_hdf5}:/ELASTIC_BND/connectivity\n'
                       f'\t\t\t</DataItem>\n'
                       f'\t\t</Topology>\n'
                       f'\t\t<Geometry GeometryType="XYZ">\n'
                       f'\t\t\t<DataItem Format="HDF" Dimensions="{self.n_elements_global} {self.coordinates.shape[1]} 3">\n'
                       f'\t\t\t\t{filename_hdf5}:/ELASTIC_BND/coordinates</DataItem>\n'
                       f'\t\t</Geometry>\n'
                       f'\t\t<Grid Name="Salvus Wavefield Data" GridType="Collection" CollectionType="Temporal">\n')

        def write_xdmf_footer(file):
            file.write(f'\t\t</Grid>\n'
                       f'\t</Domain>\n'
                       f'</Xdmf>\n')

        def write_xdmf_timestep(file, filename_hdf5, id_time, time):
            file.write(f'\t\t\t<Grid Name="Wavefield_{time} ">\n'
                       f'\t\t\t<Time Value="{time}" />\n'
                       f'\t\t\t<Topology Reference="/Xdmf/Domain/Topology[1]"/>\n'
                       f'\t\t\t<Geometry Reference="/Xdmf/Domain/Geometry[1]"/>\n')

            write_xdmf_field(file, filename_hdf5, 0, id_time)
            file.write(f'\t\t\t</Grid>\n')

        def write_xdmf_field(file, filename_hdf5, id_component_wavefield, id_time):
            component_name = "wavefield" + "_" + str(id_component_wavefield)

            file.write(f'\t\t\t\t<Attribute Name="{component_name}" AttributeType="Scalar" Center="Node">\n'
                       f'\t\t\t\t<DataItem ItemType="HyperSlab" Dimensions="1 {self.n_elements_global} {self.n_gll} " Type="HyperSlab">\n'
                       f'\t\t\t\t\t<DataItem Dimensions="3 3" Format="XML">\n'
                       f'\t\t\t\t\t\t{id_time} 0 0\n'
                       f'\t\t\t\t\t\t1 1 1\n'
                       f'\t\t\t\t\t\t1 {self.n_elements_global} {self.n_gll}\n'
                       f'\t\t\t\t\t</DataItem>\n'
                       f'\t\t\t\t\t<DataItem Name="Points" Dimensions="{self.nt_full} {self.n_elements_global} {self.n_gll}" Format="HDF">\n'
                       f'\t\t\t\t\t\t{filename_hdf5}:/ELASTIC_BND/data\n'
                       f'\t\t\t\t\t</DataItem>\n'
                       f'\t\t\t\t</DataItem>\n'
                       f'\t\t\t\t</Attribute>\n')

        # write xdmf file
        print(filename_xdmf)
        if rank == 0:
            time = -1.0
            with open(filename_xdmf, 'w') as f:
                filename_hdf5_relative = "./" + os.path.basename(filename_hdf5)
                write_xdmf_header(f, filename_hdf5_relative)
                for id_time in range(self.nt_stored):
                    write_xdmf_timestep(
                        f, filename_hdf5_relative, id_time, time)
                    time = np.round(time + self.dt, 2)
                write_xdmf_footer(f)
        comm.Barrier()

    # def plot_coordinates(self):
    #     if size > 1:
    #         raise NotImplementedError("Plotting of coordinates is not implemented for parallel usage!")

    #     figure = plt.figure()

    #     ax = Axes3D(figure)
    #     ax.scatter3D(self.coordinates[:, :, 0], self.coordinates[:, :, 1], self.coordinates[:, :, 2])

    #     ax.set_xlabel("x [km]")
    #     ax.set_ylabel("y [km]")
    #     ax.set_zlabel("z [km]")
    #     plt.title("Coordinates")
    #     plt.show()

    # def plot_wavefield(self):
    #     if size > 1:
    #         raise NotImplementedError("Plotting of wavefield is not implemented for parallel usage!")

    #     x = self.coordinates[:, :, 0]
    #     y = self.coordinates[:, :, 1]
    #     z = self.coordinates[:, :, 2]

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     wframe = None
    #     for k in range(0, self.wavefield.shape[0]):
    #         if wframe:
    #             ax.collections.remove(wframe)

    #         wframe = ax.scatter3D(x, y, z, c=self.hdf5["/ELASTIC_BND/data"][k, :, 0, :])
    #         plt.pause(.001)
