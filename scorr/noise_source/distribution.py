#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import h5py

from scorr import addons
from scorr.addons import rank, group_name


class Distribution():
    @addons.verbose_mode
    def __init__(self, distribution, coordinates=None, verbose=False):
        """
        initialize distribution

        :param distribution: local, i.e. only specific slice for rank
        :param coordinates: local, i.e. only specific slice for rank
        :param verbose:
        """
        self.distribution = distribution
        self.coordinates = coordinates

    @classmethod
    @addons.verbose_mode
    def init_distribution_with_file(cls, filename, verbose=False):
        """
        initialize distribution with noise source file

        :param filename:
        :param verbose:
        :return:
        """
        try:
            with h5py.File(str(filename), "r") as hdf5:
                n_elements_global = hdf5[group_name + "coordinates"].shape[0]
                rank_elements = addons.rank_element_dictionary(n_elements_global)

                coordinates = hdf5[group_name + "coordinates"][:]
                distribution = hdf5[group_name + "distribution"][rank_elements[rank][0]:rank_elements[rank][1], :]

            return cls(distribution=distribution, coordinates=coordinates)
        except KeyError:
            print("Not a proper noise source file!")
            raise

    def get_value_at_gll(self, id_element, id_gll):
        if self.distribution is None:
            raise ValueError("Distribution cannot be None!")

        if isinstance(self.distribution, float):
            return self.distribution
        else:
            return self.distribution[id_element, id_gll]
