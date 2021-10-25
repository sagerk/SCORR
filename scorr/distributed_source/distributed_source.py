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
import warnings

import h5py
import numpy as np
from obspy.signal.filter import bandpass, lowpass
from scipy.signal import convolve

from scorr import addons
from scorr.addons import comm, rank, size
from scorr.extensions import rotation_distance, salvus_extensions
from scorr.noise_source.noise_source import NoiseSource
from scorr.wavefield.wavefield import Wavefield


class DistributedSource():
    @addons.verbose_mode
    def __init__(self, noise_source: NoiseSource, wavefield: Wavefield, verbose: bool = False):
        """
        initialize distributed source

        :param noise_source:
        :param wavefield:
        :param verbose:
        """
        self.noise_source = noise_source
        self.wavefield = wavefield

        # set up information on correlation source
        self.nt_longest_branch = self.wavefield.nt_longest_branch
        self.nt_full = self.wavefield.nt_full
        self.dt = self.wavefield.dt
        self.fs = self.wavefield.fs

        # get coordinates, mesh and components
        self.coordinates = self.wavefield.coordinates
        self.n_elements_global = self.wavefield.n_elements_global
        self.n_elements_local = self.wavefield.n_elements_local
        self.n_gll = self.wavefield.n_gll
        self.n_components = 3

        # set up time and frequency vector
        self.time = np.linspace(-(self.nt_longest_branch - 1) * self.dt, (self.nt_longest_branch - 1) * self.dt,
                                self.nt_full)
        assert (self.time[1] - self.time[0]) - self.dt < 1e-8
        self.f = np.linspace(-self.fs / 2, self.fs / 2, self.nt_full)

        # set up noise source array
        self.distributed_source = np.zeros(
            (self.nt_full, self.n_elements_local, self.n_components, self.n_gll))

        # indicate is distributed source is rotated
        self.is_rotated = False

    def __str__(self):
        return f"dt and nt_full of distributed source: {self.dt}, {self.nt_full}"

    def convolve_wavefield(self, id_component_wavefield, id_component_dist_source, short_cut=False, tapering=False,
                           taper_length_in_percent=10, filter_spec=None, verbose=False):
        """
        convolve wavefield

        :param id_component_dist_source:
        :param id_component_wavefield:
        :param tapering:
        :param verbose:
        :return:
        """
        if tapering:
            # currently taper is 13 time steps long (currently hard coded)
            taper = salvus_extensions.design_taper(full_length=self.nt_longest_branch,
                                                   taper_length_in_percent=taper_length_in_percent,
                                                   zero_padding=True, padding_length_percent=1)

        ###########################################
        ##########      CONVOLUTION      ##########
        ###########################################
        for id_element_local in range(self.n_elements_local):
            for id_gll in range(self.n_gll):

                # get wavefield of specific gll point
                wavefield, taper_body = self.wavefield.get_pad_rev_wavefield_comp_at_gll(id_element=id_element_local, id_gll=id_gll,
                                                                                         id_component_wavefield=id_component_wavefield)
                # # filter waveform
                if filter_spec:
                    wavefield = bandpass(data=wavefield, freqmin=min(filter_spec), freqmax=max(filter_spec),
                                         df=self.fs, zerophase=True, corners=5)
                #     wavefield = lowpass(data=wavefield, freq=max(filter_spec),
                #             df=self.fs, zerophase=True, corners=5)

                # short cut if wavefield already has frequency content of noise source
                if short_cut:
                    scale_factor = self.noise_source.get_value_of_distribution_at_gll(id_element=id_element_local,
                                                                                      id_gll=id_gll)
                    self.distributed_source[:, id_element_local, id_component_dist_source, id_gll] = \
                        scale_factor * wavefield * taper_body

                else:
                    stf = self.noise_source.compute_stf_at_gll(
                        id_element=id_element_local, id_gll=id_gll, f=self.f)
                    assert stf.shape == wavefield.shape
                    self.distributed_source[:, id_element_local, id_component_dist_source, id_gll] = \
                        convolve(stf, wavefield, mode="same")

                if tapering:
                    self.distributed_source[:taper.shape[0], id_element_local, id_component_dist_source,
                                            id_gll] *= taper

                if verbose and rank == 0 and (id_element_local % (self.n_elements_global)/100):
                    sys.stdout.write("\r")
                    sys.stdout.write("[%-99s] %d%%" % (
                        "=" * int(id_element_local /
                                  self.n_elements_local * 100),
                        round(id_element_local / self.n_elements_local, 1) * 100))
                    sys.stdout.flush()

        if verbose and rank == 0:
            sys.stdout.write("\n")

        # rotate source if wavefield was rotated
        if self.wavefield.is_rotated:
            self.rotate_distributed_source(verbose=verbose)
            self.is_rotated = True

    @addons.verbose_mode
    def rotate_distributed_source(self, verbose=False):
        """
        rotate distributed source from [Z, N, E] to [x, y, z]

        :return:
        """
        if self.is_rotated:
            warnings.warn(
                "source is already rotated - set is_rotated to False if another rotation is actually wanted")
            return

        for id_element_local in range(self.n_elements_local):
            for id_gll in range(self.n_gll):
                x, y, z = self.coordinates[id_element_local, id_gll, :]
                lat, lon = rotation_distance.from_cartesian_to_latlon(x, y, z)
                R_inv = rotation_distance.get_transformation_matrix_3d(
                    latitude=lat, longitude=lon, inverse=True)

                self.distributed_source[:, id_element_local, :, id_gll] = np.einsum(
                    'ij,kj->ki', R_inv, self.distributed_source[:, id_element_local, :, id_gll], optimize=True)

    @addons.verbose_mode
    def write_distributed_source_to_file(self, filename_hdf5, mode='w', precision="single",
                                         adjoint=False, cutting_bonus=0, verbose=False):
        """
        write distributed source to file and generate xdmf file

        :param filename_hdf5:
        :param mode:
        :param adjoint:
        :param cutting_bonus:
        :param verbose:
        :return:
        """
        # make sure that source and wavefield have the same rotation
        assert self.is_rotated == self.wavefield.is_rotated
        assert self.wavefield.sampling_rate == 1

        # set dtype depending on precision
        if precision == "single":
            dtype = "32"
        else:
            dtype = "64"

        if rank == 0:
            os.makedirs(os.path.dirname(filename_hdf5), exist_ok=True)
        comm.Barrier()

        # each rank writes its own file
        filename_hdf5_rank = ".".join(str(filename_hdf5).split(".")[
                                      :-1]) + "_" + str(rank) + ".h5"
        with h5py.File(filename_hdf5_rank, mode) as hdf5:
            shape_coord = (self.n_elements_local,
                           self.n_gll, self.n_components)
            dset_coord = hdf5.create_dataset(
                "/ELASTIC_BND/coordinates", shape_coord, dtype="float" + dtype)
            dset_coord[:, :, :] = self.coordinates

            if adjoint:
                shape_data = (
                    self.nt_longest_branch + cutting_bonus, self.n_elements_local, self.n_components, self.n_gll)
                dset_data = hdf5.create_dataset(
                    "/ELASTIC_BND/data", shape_data, dtype="float" + dtype)
                dset_data[:, :, :, :] = np.flipud(
                    self.distributed_source[0:(self.nt_longest_branch + cutting_bonus), :, :, :])
            else:
                shape_data = (self.nt_full + cutting_bonus,
                              self.n_elements_local, self.n_components, self.n_gll)
                dset_data = hdf5.create_dataset(
                    "/ELASTIC_BND/data", shape_data, dtype="float" + dtype)
                dset_data[:, :, :,
                          :] = self.distributed_source[:shape_data[0], :, :, :]

        # collect all files
        rank_elements = addons.rank_element_dictionary(self.n_elements_global)
        comm.Barrier()
        if rank == 0:
            with h5py.File(str(filename_hdf5), mode) as hdf5:
                hdf5.create_dataset("/ELASTIC_BND/connectivity", shape=self.wavefield.connectivity.shape,
                                    dtype="int" + dtype, data=self.wavefield.connectivity)
                hdf5.create_dataset("/ELASTIC_BND/globalElementIds", shape=self.wavefield.globalElementIds.shape,
                                    dtype="int" + dtype, data=self.wavefield.globalElementIds)

                shape_coord = (self.n_elements_global,
                               self.n_gll, self.n_components)
                dset_coord = hdf5.create_dataset(
                    "/ELASTIC_BND/coordinates", shape_coord, dtype="float" + dtype)

                if adjoint:
                    hdf5.create_dataset("/ELASTIC_BND/time", (self.nt_longest_branch + cutting_bonus,),
                                        dtype="float" + dtype,
                                        data=self.time[-(self.nt_longest_branch + cutting_bonus):])

                    shape_data = (
                        self.nt_longest_branch + cutting_bonus, self.n_elements_global, self.n_components, self.n_gll)
                    dset_data = hdf5.create_dataset("/ELASTIC_BND/data", shape_data, dtype="float" + dtype,
                                                    chunks=(shape_data[0], 1, shape_data[2], shape_data[3]))
                else:
                    shape_time = (self.nt_full + cutting_bonus,)
                    hdf5.create_dataset("/ELASTIC_BND/time", shape_time,
                                        dtype="float" + dtype, data=self.time[:shape_time[0]])

                    shape_data = (self.nt_full + cutting_bonus,
                                  self.n_elements_global, self.n_components, self.n_gll)
                    dset_data = hdf5.create_dataset("/ELASTIC_BND/data", shape_data, dtype="float" + dtype,
                                                    chunks=(shape_data[0], 1, shape_data[2], shape_data[3]))

                # loop over files
                for i_rank in range(size):
                    filename_hdf5_i_rank = ".".join(str(filename_hdf5).split(".")[
                                                    :-1]) + "_" + str(i_rank) + ".h5"
                    with h5py.File(filename_hdf5_i_rank, mode="r") as hdf5_i_rank:
                        dset_coord[rank_elements[i_rank][0]:rank_elements[i_rank]
                                   [1], :, :] = hdf5_i_rank["/ELASTIC_BND/coordinates"][:]

                        if adjoint:
                            dset_data[:, rank_elements[i_rank][0]:rank_elements[i_rank]
                                      [1], :, :] = hdf5_i_rank["/ELASTIC_BND/data"][:]
                        else:
                            dset_data[:, rank_elements[i_rank][0]:rank_elements[i_rank]
                                      [1], :, :] = hdf5_i_rank["/ELASTIC_BND/data"][:]

                    # remove hdf5 file - can be quite large
                    os.remove(filename_hdf5_i_rank)

                # label hdf5 arrays according to Salvus naming convention
                hdf5["/ELASTIC_BND/data"].dims[0].label = 'time'
                hdf5["/ELASTIC_BND/data"].dims[1].label = 'element'
                hdf5["/ELASTIC_BND/data"].dims[2].label = '[ source_CMP_X | source_CMP_Y | source_CMP_Z ]'
                hdf5["/ELASTIC_BND/data"].dims[3].label = 'point'

        comm.Barrier()
        self.write_xdmf_file(filename_xdmf=os.path.splitext(filename_hdf5)[0] + ".xdmf",
                             filename_hdf5=filename_hdf5,
                             adjoint=adjoint, cutting_bonus=cutting_bonus,
                             verbose=verbose)

    @addons.verbose_mode
    def write_xdmf_file(self, filename_xdmf, filename_hdf5, adjoint=False, cutting_bonus=0, verbose=False):
        """
        write xdmf file for distributed source

        :param filename_xdmf:
        :param filename_hdf5:
        :param adjoint:
        :param cutting_bonus:
        :param verbose:
        :return:
        """

        def write_xdmf_header(file, filename_hdf5):
            file.write(f'<?xml version="1.0" ?>\n'
                       f'<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
                       f'<Xdmf Version="2.0">\n'
                       f'\t<Domain>\n'
                       f'\t\t<Topology TopologyType="Quadrilateral" NumberOfElements="{self.wavefield.connectivity.shape[0]}">\n'
                       f'\t\t\t<DataItem Format="HDF" DataType="Int" Dimensions="{self.wavefield.connectivity.shape[0]} 4">\n'
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

            for id_component in range(self.n_components):
                write_xdmf_field(file, filename_hdf5, id_component, id_time)

            file.write(f'\t\t\t</Grid>\n')

        def write_xdmf_field(file, filename_hdf5, id_component, id_time):
            component_name = "source_CMP_" + str(id_component)

            file.write(f'\t\t\t\t<Attribute Name="{component_name}" AttributeType="Scalar" Center="Node">\n'
                       f'\t\t\t\t<DataItem ItemType="HyperSlab" Dimensions="1 {self.n_elements_global} 1 {self.n_gll} " Type="HyperSlab">\n'
                       f'\t\t\t\t\t<DataItem Dimensions="3 4" Format="XML">\n'
                       f'\t\t\t\t\t\t{id_time} 0 {id_component} 0\n'
                       f'\t\t\t\t\t\t1 1 1 1\n'
                       f'\t\t\t\t\t\t1 {self.n_elements_global} 1 {self.n_gll}\n'
                       f'\t\t\t\t\t</DataItem>\n'
                       f'\t\t\t\t\t<DataItem Name="Points" Dimensions="{self.nt_full} {self.n_elements_global} {self.n_components} {self.n_gll}" Format="HDF">\n'
                       f'\t\t\t\t\t\t{filename_hdf5}:/ELASTIC_BND/data\n'
                       f'\t\t\t\t\t</DataItem>\n'
                       f'\t\t\t\t</DataItem>\n'
                       f'\t\t\t\t</Attribute>\n')

        if rank == 0:
            # set starttime and number of time steps
            time = np.round(-(self.nt_full - 1) / 2 * self.dt, 2)
            if adjoint:
                nt = self.nt_longest_branch + cutting_bonus
            else:
                nt = self.nt_full + cutting_bonus

            # write xdmf file
            with open(filename_xdmf, 'w') as f:
                filename_hdf5_relative = "./" + os.path.basename(filename_hdf5)
                write_xdmf_header(f, filename_hdf5_relative)
                for id_time in range(nt):
                    write_xdmf_timestep(
                        f, filename_hdf5_relative, id_time, time)
                    time = np.round(time + self.dt, 2)
                write_xdmf_footer(f)
        comm.Barrier()
