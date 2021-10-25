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
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hann

from scorr import addons
from scorr.addons import group_name


class Spectrum():
    __spectrum_cache = {}
    __stf_cache = {}

    @addons.verbose_mode
    def __init__(self, f_peak, bandwidth, strength, verbose=False):
        """
        initialize spectrum

        :param f_peak:
        :param bandwidth:
        :param strength:
        :param verbose:
        """
        self.f_peak = f_peak
        self.bandwidth = bandwidth
        self.strength = strength

    @classmethod
    @addons.verbose_mode
    def init_spectrum_with_dict(cls, config, verbose=False):
        """
        initialize spectrum with dictionary

        :param config:
        :param verbose:
        :return:
        """
        try:
            return cls(f_peak=config["f_peak"], bandwidth=config["bandwidth"], strength=config["strength"])
        except KeyError:
            print("Missing key in dictionary - input file incomplete?")
            raise

    @classmethod
    @addons.verbose_mode
    def init_spectrum_with_file(cls, filename, verbose=False):
        """
        initialize spectrum with noise source file

        :param filename:
        :param verbose:
        :return:
        """
        try:
            with h5py.File(str(filename), "r") as hdf5:
                f_peak, bandwidth, strength = hdf5[group_name + "spectrum"][:]
            return cls(f_peak=f_peak, bandwidth=bandwidth, strength=strength)
        except KeyError:
            print("Not a proper noise source file!")
            raise

    def _get_cache_key(self, f):
        return (self.f_peak, self.bandwidth, self.strength, f.shape, len(f), f.std(), f.mean(), f.ptp())

    def get_spectrum(self, f):
        _k = self._get_cache_key(f)
        if _k not in self.__spectrum_cache:
            self.__spectrum_cache[_k] = self.strength * np.exp(-(abs(f) - self.f_peak) ** 2 / self.bandwidth ** 2)

            # BENCHMARK
            # f_min = 0.005
            # f_max = 0.015
            # ix_f_min = np.argmin(np.abs(f[:] - f_min))
            # ix_f_max = np.argmin(np.abs(f[:] - f_max))
            #
            # spectrum_tmp = np.zeros(f.shape)
            # spectrum_tmp[ix_f_min:ix_f_max + 1] = hann(ix_f_max - ix_f_min + 1)
            # self.__spectrum_cache[_k] = spectrum_tmp

        return self.__spectrum_cache[_k]

    def get_stf(self, f):
        _k = self._get_cache_key(f)
        if _k not in self.__stf_cache:
            psd = self.get_spectrum(f)
            self.__stf_cache[_k] = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(psd) * len(psd))))

        return self.__stf_cache[_k]

    def plot_spectrum(self):
        f = np.linspace(0, self.f_peak + 2 * self.bandwidth, 100)
        spectrum = self.get_spectrum(f)

        plt.figure()
        plt.plot(f, spectrum)
        plt.title("spectrum")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude")
        plt.show()
