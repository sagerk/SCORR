#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
from scorr import addons
from scorr.noise_source.distribution import Distribution
from scorr.noise_source.spectrum import Spectrum


class NoiseSource():
    __stf_cache = {}

    @addons.verbose_mode
    def __init__(self, data=None, verbose=False):
        """
        initialize noise source

        :param data:
        :param verbose:
        """
        self.list_basis_functions = []
        if data is not None:
            for item in data:
                self.list_basis_functions.append(item)

    @classmethod
    @addons.verbose_mode
    def init_noise_source_with_list(cls, data: list, verbose: bool = False):
        """
        initialize noise source with list

        :param data:
        :param verbose:
        :return:
        """
        return cls(data)

    @classmethod
    @addons.verbose_mode
    def init_noise_source_with_spectrum_and_distribution(cls, spectrum: Spectrum, distribution: Distribution,
                                                         verbose: bool = False):
        """
        initialize noise source with spectrum and distribution

        :param spectrum:
        :param distribution:
        :param verbose:
        :return:
        """
        data = []
        data.append(BasisFunction(spectrum, distribution))
        return cls(data)

    def add_basis_function(self, basisfunction):
        """

        :param basisfunction:
        :return:
        """
        self.list_basis_functions.append(basisfunction)

    def get_value_of_distribution_at_gll(self, id_element, id_gll):
        """
        method to take a short cut, if only one spectrum is considered and the Green function covers the respective
        frequency band

        :param id_element:
        :param id_gll:
        :return:
        """
        assert len(self.list_basis_functions) == 1
        return self.list_basis_functions[0].distribution.get_value_at_gll(id_element=id_element,
                                                                          id_gll=id_gll)

    def _get_cache_key(self, value):
        return (value)

    def compute_stf_at_gll(self, id_element, id_gll, f):
        """
        compute a stf from the psd at a specific point
        caching is for now only implemented for one basis function - first use case

        :param id_element:
        :param id_gll:
        :param f:
        :return:
        """
        if len(self.list_basis_functions) == 1:
            _k = self.list_basis_functions[0].distribution.get_value_at_gll(id_element=id_element,
                                                                            id_gll=id_gll)
            if _k not in self.__stf_cache:
                self.__stf_cache[_k] = self.list_basis_functions[0].spectrum.get_stf(f) * _k

            return self.__stf_cache[_k]

        else:
            stf = 0.0 * f
            for i_bf in range(len(self.list_basis_functions)):
                stf += self.list_basis_functions[i_bf].spectrum.get_stf() * \
                       self.list_basis_functions[i_bf].distribution.get_value_at_gll(id_element=id_element,
                                                                                     id_gll=id_gll)
            return stf

    def optimize_basis_functions(self, spectrum):
        """
        given a spectrum, e.g. lower noise model by Peterson (1993), optimize spectra of basis functions to match
        desired model

        :param spectrum:
        :return:
        """
        pass


class BasisFunction():
    def __init__(self, spectrum: Spectrum, distribution: Distribution):
        self.spectrum = spectrum
        self.distribution = distribution
