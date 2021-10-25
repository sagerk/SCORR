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
from typing import Union

import h5py
import numpy as np
from scorr import addons
from scorr.addons import comm, rank, size, group_name
from scorr.noise_source.spectrum import Spectrum
from scorr.wavefield.adjoint_wavefield import AdjointWavefield
from scorr.wavefield.green_function import GreenFunction


class SourceKernel():
    @addons.verbose_mode
    def __init__(self, coordinates=None, connectivity=None, globalElementIds=None,
                 n_elements_global=None, n_elements_local=None, kernel=0.0, verbose: bool = False):
        """
        initialize source kernel

        :param verbose:
        """
        self.coordinates = coordinates
        self.connectivity = connectivity
        self.globalElementIds = globalElementIds
        self.n_elements_global = n_elements_global
        self.n_elements_local = n_elements_local
        self.kernel = kernel

    def __add__(self, other):
        # create class instance that will hold the sum
        source_kernel_sum = SourceKernel()

        # if both are empty, return empty source kernel
        if self.coordinates is None and other.coordinates is None:
            pass
        # if other is emtpy, take information from self
        elif self.coordinates is not None and other.coordinates is None:
            source_kernel_sum.coordinates = self.coordinates
            source_kernel_sum.connectivity = self.connectivity
            source_kernel_sum.globalElementIds = self.globalElementIds
            source_kernel_sum.n_elements_global = self.n_elements_global
            source_kernel_sum.n_elements_local = self.n_elements_local
            source_kernel_sum.kernel = self.kernel
        # if self is empty, take information from other
        elif self.coordinates is None and other.coordinates is not None:
            source_kernel_sum.coordinates = other.coordinates
            source_kernel_sum.connectivity = other.connectivity
            source_kernel_sum.globalElementIds = other.globalElementIds
            source_kernel_sum.n_elements_global = other.n_elements_global
            source_kernel_sum.n_elements_local = other.n_elements_local
            source_kernel_sum.kernel = other.kernel
        # if both are not empty, make sure that both kernels are compatible
        else:
            assert (self.coordinates == other.coordinates).all()
            assert (self.connectivity == other.connectivity).all()
            assert (self.globalElementIds == other.globalElementIds).all()
            assert self.n_elements_global == other.n_elements_global
            assert self.n_elements_local == other.n_elements_local

            source_kernel_sum.coordinates = other.coordinates
            source_kernel_sum.connectivity = other.connectivity
            source_kernel_sum.globalElementIds = other.globalElementIds
            source_kernel_sum.n_elements_global = other.n_elements_global
            source_kernel_sum.n_elements_local = other.n_elements_local

            # add kernel data
            source_kernel_sum.kernel = self.kernel + other.kernel

        # return class instance
        return source_kernel_sum

    def __iadd__(self, other):
        # if both are empty, self does not change
        if self.coordinates is None and other.coordinates is None:
            pass
        # if other is emtpy, self does not change
        elif self.coordinates is not None and other.coordinates is None:
            pass
        # if self is empty, take information from other
        elif self.coordinates is None and other.coordinates is not None:
            self.coordinates = other.coordinates
            self.connectivity = other.connectivity
            self.globalElementIds = other.globalElementIds
            self.n_elements_global = other.n_elements_global
            self.n_elements_local = other.n_elements_local
            self.kernel = other.kernel
        # if both are not empty, make sure that both kernels are compatible
        else:
            assert (self.coordinates == other.coordinates).all()
            assert (self.connectivity == other.connectivity).all()
            assert (self.globalElementIds == other.globalElementIds).all()
            assert self.n_elements_global == other.n_elements_global
            assert self.n_elements_local == other.n_elements_local

            # add kernel data
            self.kernel += other.kernel

        return self

    @classmethod
    @addons.verbose_mode
    def init_with_wavefields(cls, wavefield_fwd: GreenFunction, wavefield_adj: AdjointWavefield, spectrum: Spectrum,
                             id_component_wavefield: int, shortcut: bool = False,
                             verbose: bool = False):
        """
        initialize source kernel with wavefields and directly build kernel

        :param wavefield_fwd:
        :param wavefield_adj:
        :param spectrum:
        :param id_component_wavefield:
        :param shortcut:
        :param verbose:
        :return:
        """
        class_instance = cls()
        class_instance.build_kernel(wavefield_fwd=wavefield_fwd, wavefield_adj=wavefield_adj, spectrum=spectrum,
                                    id_component_wavefield=id_component_wavefield, shortcut=shortcut)
        return class_instance

    @classmethod
    @addons.verbose_mode
    def init_with_kernel_file(cls, filename_kernel_h5: Union[str, Path], verbose: bool = False):
        """
        initialize source kernel with filename and directly read kernel

        :param filename_kernel_h5:
        :param verbose:
        :return:
        """
        class_instance = cls()
        class_instance.read_kernel_file(filename_kernel_h5=filename_kernel_h5)

        return class_instance

    def build_kernel(self, wavefield_fwd: GreenFunction, wavefield_adj: AdjointWavefield, spectrum: Spectrum,
                     id_component_wavefield: int, shortcut: bool = False):
        """
        build source kernel from wavefields

        :param wavefield_fwd:
        :param wavefield_adj:
        :param spectrum:
        :param id_component_wavefield:
        :param shortcut:
        :param verbose:
        :return:
        """
        # specifics of wavefield file
        self.coordinates = wavefield_fwd.coordinates
        self.connectivity = wavefield_fwd.connectivity
        self.globalElementIds = wavefield_fwd.globalElementIds
        self.n_elements_global = wavefield_fwd.n_elements_global
        self.n_elements_local = wavefield_fwd.n_elements_local

        # prepare stf for convolution
        if not shortcut:
            f = np.linspace(-wavefield_fwd.fs / 2,
                            wavefield_fwd.fs / 2, wavefield_fwd.nt_full)
            stf = spectrum.get_stf(f)

        wavefield_fwd_conv = np.zeros(
            (wavefield_fwd.nt_full, self.n_elements_local, wavefield_fwd.n_gll))
        for id_element_local in range(self.n_elements_local):
            for id_gll in range(wavefield_fwd.n_gll):
                wavefield_fwd_padded, _ = wavefield_fwd.get_pad_wavefield_comp_at_gll(id_element=id_element_local,
                                                                                      id_gll=id_gll,
                                                                                      id_component_wavefield=id_component_wavefield)

                if shortcut:
                    wavefield_fwd_conv[:, id_element_local,
                                       id_gll] = wavefield_fwd_padded
                else:
                    assert stf.shape == wavefield_fwd_padded.shape
                    wavefield_fwd_conv[:, id_element_local, id_gll] = np.convolve(stf, wavefield_fwd_padded,
                                                                                  mode="same")

        # integrate over time, for sign see equation (25) in Sager et al. (2018a)
        self.kernel = np.zeros((self.n_elements_local, wavefield_fwd.n_gll))
        for i in range(wavefield_adj.wavefield.shape[0] - 1):
            self.kernel -= wavefield_fwd.sampling_rate * wavefield_fwd_conv[-i - 1, :, :] * \
                wavefield_adj.wavefield[-i - 1, :, id_component_wavefield, :]
        self.kernel -= wavefield_fwd_conv[-1, :, :] * \
            wavefield_adj.wavefield[-1, :, id_component_wavefield, :]

        # BODY WAVE IDEA
        # time_fwd = np.linspace(-wavefield_fwd.endtime, wavefield_fwd.endtime, wavefield_fwd.nt_full)
        # time_adj = np.linspace(-wavefield_fwd.endtime, wavefield_fwd.endtime, wavefield_adj.nt_full) + 7.0

        # src_fwd = np.asarray([0.0, 0.0e3, 6371e3])
        # src_adj = np.asarray([0.0, 30e3, 6371e3])
        # distance_fwd = np.sqrt( np.sum((wavefield_fwd.coordinates[::1,:,:]-src_fwd)**2, 2) )
        # distance_adj = np.sqrt( np.sum((wavefield_adj.coordinates[::1,:,:]-src_adj)**2, 2) )
        # pattern_vel = 4.5e3

        # self.kernel = np.zeros((self.n_elements_local, wavefield_fwd.n_gll))
        # for i in range(wavefield_adj.wavefield.shape[0] - 1):
        #     pattern_fwd = (distance_fwd > pattern_vel * time_fwd[-i - 1]).astype("float32") # * (distance_fwd > 2.0e3).astype("float32")
        #     pattern_adj = (distance_adj > pattern_vel * time_adj[-i - 1]).astype("float32") # * (distance_adj > 2.0e3).astype("float32")

        #     self.kernel -= wavefield_fwd.sampling_rate * wavefield_fwd_conv[-i - 1, :, :] * \
        #                    wavefield_adj.wavefield[-i - 1, :, id_component_wavefield, :] * \
        #                    pattern_fwd * pattern_adj

        # print(time_fwd)
        # print(time_adj)
        # # pattern_fwd = (distance_fwd > pattern_vel * 5.0).astype("float32") * (distance_fwd > 10e3).astype("float32")
        # pattern_adj = (distance_adj > pattern_vel * 5.0).astype("float32") # * (distance_adj > 10e3).astype("float32")
        # self.kernel = pattern_adj

    def read_kernel_file(self, filename_kernel_h5):
        """
        read in kernel file

        :param filename_kernel_h5:
        :param verbose:
        :return:
        """
        with h5py.File(str(filename_kernel_h5), "r") as hdf5:
            self.connectivity = hdf5[group_name + "connectivity"][:]
            self.globalElementIds = hdf5[group_name + "globalElementIds"][:]
            self.n_elements_global = hdf5[group_name + "coordinates"].shape[0]

            rank_elements = addons.rank_element_dictionary(
                self.n_elements_global)
            self.coordinates = hdf5[group_name +
                                    "coordinates"][rank_elements[rank][0]:rank_elements[rank][1], :, :]
            self.n_elements_local = self.coordinates.shape[0]
            self.kernel = hdf5[group_name +
                               "distribution"][rank_elements[rank][0]:rank_elements[rank][1], :]

    # @addons.verbose_mode
    # def write_kernel_to_file(self, filename_kernel_h5, precision, verbose=False):
    #     """
    #     write kernel to file and generate xdmf file
    #
    #     :param filename_kernel_h5:
    #     :param verbose:
    #     :return:
    #     """
    #     assert self.coordinates is not None, "Coordinates are not set!"
    #     assert self.n_elements_global is not None, "'n_elements_global' is not set!"
    #
    #     if rank == 0:
    #         os.makedirs(os.path.dirname(filename_kernel_h5), exist_ok=True)
    #     comm.Barrier()
    #
    #     # set dtype depending on precision
    #     if precision == "single":
    #         dtype = "32"
    #     else:
    #         dtype = "64"
    #
    #     rank_elements = addons.rank_element_dictionary(self.n_elements_global)
    #     with h5py.File(str(filename_kernel_h5), 'w', driver='mpio', comm=comm) as hdf5:
    #         hdf5.create_dataset(group_name + "connectivity", self.connectivity.shape, dtype='float' + dtype,
    #                             data=self.connectivity)
    #         hdf5.create_dataset(group_name + "globalElementIds", self.globalElementIds.shape, dtype='float' + dtype,
    #                             data=self.globalElementIds)
    #
    #         shape_coord = (self.n_elements_global, self.coordinates.shape[1], self.coordinates.shape[2])
    #         dset_coord = hdf5.create_dataset(group_name + "coordinates", shape_coord, dtype='float' + dtype)
    #         dset_coord[rank_elements[rank][0]:rank_elements[rank][1], :, :] = self.coordinates
    #
    #         shape_data = (self.n_elements_global, self.coordinates.shape[1])
    #         dset_data = hdf5.create_dataset(group_name + "distribution", shape_data, dtype='float' + dtype)
    #         dset_data[rank_elements[rank][0]:rank_elements[rank][1], :] = self.kernel
    #
    #     self.write_xdmf_file(os.path.splitext(filename_kernel_h5)[0] + ".xdmf", filename_kernel_h5, verbose)
    #     comm.Barrier()

    @addons.verbose_mode
    def write_kernel_to_file(self, filename_kernel_h5, precision, verbose=False):
        """
        write kernel to file and generate xdmf file

        :param filename_kernel_h5:
        :param verbose:
        :return:
        """
        assert self.coordinates is not None, "Coordinates are not set!"
        assert self.n_elements_global is not None, "'n_elements_global' is not set!"

        if rank == 0:
            os.makedirs(os.path.dirname(filename_kernel_h5), exist_ok=True)
        comm.Barrier()

        # set dtype depending on precision
        if precision == "single":
            dtype = "32"
        else:
            dtype = "64"

        # each rank writes its own file
        filename_hdf5_rank = ".".join(str(filename_kernel_h5).split(".")[
                                      :-1]) + "_" + str(rank) + ".h5"
        with h5py.File(filename_hdf5_rank, "w") as hdf5:
            dset_coord = hdf5.create_dataset(
                group_name + "coordinates", self.coordinates.shape, dtype="float" + dtype)
            dset_coord[:, :, :] = self.coordinates

            dset_data = hdf5.create_dataset(
                group_name + "distribution", self.kernel.shape, dtype='float' + dtype)
            dset_data[:, :] = self.kernel

        # collect all files
        rank_elements = addons.rank_element_dictionary(self.n_elements_global)
        comm.Barrier()
        if rank == 0:
            with h5py.File(str(filename_kernel_h5), "w") as hdf5:
                hdf5.create_dataset(group_name + "connectivity", shape=self.connectivity.shape,
                                    dtype="int" + dtype, data=self.connectivity)
                hdf5.create_dataset(group_name + "globalElementIds", shape=self.globalElementIds.shape,
                                    dtype="int" + dtype, data=self.globalElementIds)

                shape_coord = (
                    self.n_elements_global, self.coordinates.shape[1], self.coordinates.shape[2])
                dset_coord = hdf5.create_dataset(
                    group_name + "coordinates", shape_coord, dtype="float" + dtype)

                shape_data = (self.n_elements_global,
                              self.coordinates.shape[1])
                dset_data = hdf5.create_dataset(
                    group_name + "distribution", shape_data, dtype="float" + dtype)

                # loop over files
                for i_rank in range(size):
                    filename_hdf5_i_rank = ".".join(str(filename_kernel_h5).split(".")[
                                                    :-1]) + "_" + str(i_rank) + ".h5"
                    with h5py.File(filename_hdf5_i_rank, mode="r") as hdf5_i_rank:
                        dset_coord[rank_elements[i_rank][0]:rank_elements[i_rank]
                                   [1], :, :] = hdf5_i_rank[group_name + "coordinates"][:]
                        dset_data[rank_elements[i_rank][0]:rank_elements[i_rank]
                                  [1], :] = hdf5_i_rank[group_name + "distribution"][:]

                    # remove hdf5 file - can be quite large
                    os.remove(filename_hdf5_i_rank)

        self.write_xdmf_file(os.path.splitext(filename_kernel_h5)[
                             0] + ".xdmf", filename_kernel_h5, verbose)
        comm.Barrier()

    @addons.verbose_mode
    def write_xdmf_file(self, filename_kernel_xdmf, filename_kernel_h5, verbose=False):
        """
        write xdmf file for kernel

        :param filename_kernel_xdmf:
        :param filename_kernel_h5:
        :param verbose:
        :return:
        """
        if rank == 0:
            filename_kernel_h5 = Path(filename_kernel_h5)
            with open(filename_kernel_xdmf, "w") as file:
                file.write(f'<Xdmf Version="2.0">\n'
                           f'<Domain>\n'
                           f'   <Topology TopologyType="Quadrilateral" NumberOfElements="{self.connectivity.shape[0]}">\n'
                           f'       <DataItem Format="HDF" DataType="Int" Dimensions="{self.connectivity.shape[0]} 4">\n'
                           f'           {filename_kernel_h5.name}:{group_name}connectivity\n'
                           f'       </DataItem>\n'
                           f'   </Topology>\n'
                           f'    <Geometry GeometryType="XYZ">\n'
                           f'        <DataItem Dimensions="{self.n_elements_global} {self.coordinates.shape[1]} 3" Format="HDF">\n'
                           f'            {filename_kernel_h5.name}:{group_name}coordinates\n'
                           f'        </DataItem>\n'
                           f'    </Geometry>\n'
                           f'    <Grid Name="source_kernel">\n'
                           f'        <Time Value="0" />\n'
                           f'        <Topology Reference="/Xdmf/Domain/Topology[1]" />\n'
                           f'        <Geometry Reference="/Xdmf/Domain/Geometry[1]" />\n'
                           f'        <Attribute AttributeType="Scalar" Center="Node" Name="GRAD_PSD_ZZ">\n'
                           f'            <DataItem Dimensions="{self.n_elements_global} {self.coordinates.shape[1]} " ItemType="HyperSlab" Type="HyperSlab">\n'
                           f'                <DataItem Dimensions="3 2" Format="XML">\n'
                           f'                    0 0\n'
                           f'                    1 1\n'
                           f'                    {self.n_elements_global} {self.coordinates.shape[1]}\n'
                           f'                </DataItem>\n'
                           f'                <DataItem Dimensions="{self.n_elements_global} {self.coordinates.shape[1]}" Format="HDF" Name="Points">\n'
                           f'                    {filename_kernel_h5.name}:{group_name}distribution\n'
                           f'                </DataItem>\n'
                           f'            </DataItem>\n'
                           f'        </Attribute>\n'
                           f'    </Grid>\n'
                           f'</Domain>\n'
                           f'</Xdmf>')
