#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import numpy as np
import pytest

from scorr.extensions import scorr_extensions
from scorr.noise_source.distribution import Distribution
from scorr.noise_source.noise_source import BasisFunction
from scorr.noise_source.noise_source import NoiseSource
from scorr.noise_source.spectrum import Spectrum
from scorr.test.helpers import wavefield_file_exists, TEST_noise_source, green_function, DIR_TEST_DATA


def test_spectrum():
    """
    TODO: test plotting
    :return:
    """
    spectrum_1 = Spectrum(f_peak=0.08, bandwidth=0.07, strength=1.0)

    config_noise_source = scorr_extensions.load_configuration(TEST_noise_source, type="noise_source")
    spectrum_2 = Spectrum.init_spectrum_with_dict(config_noise_source["spectrum"])

    spectrum_3 = Spectrum.init_spectrum_with_file(filename=DIR_TEST_DATA / "noise_source_homogeneous.h5")

    assert (spectrum_1.f_peak, spectrum_1.bandwidth, spectrum_1.strength) == \
           (spectrum_2.f_peak, spectrum_2.bandwidth, spectrum_2.strength) == \
           (spectrum_3.f_peak, spectrum_3.bandwidth, spectrum_3.strength)

    # Fourier transform stf and check frequency content
    f = np.linspace(-50, 50, 201)
    psd = spectrum_1.get_spectrum(f)
    stf = spectrum_1.get_stf(f)
    psd_ckeck = np.real(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(stf)) / len(stf)))
    assert sum(psd) == pytest.approx(sum(psd_ckeck), rel=1e-6, abs=1e-10)

    # init with a file that is not a noise source file
    with pytest.raises(KeyError):
        Spectrum.init_spectrum_with_file(filename=wavefield_file_exists)


def test_distribution():
    # normal initialization
    dist_1 = Distribution(distribution=1.0)
    assert dist_1.get_value_at_gll(id_element=0, id_gll=0) == 1.0

    # init with noise source file with homogeneous distribution
    dist_2 = Distribution.init_distribution_with_file(filename=DIR_TEST_DATA / "noise_source_homogeneous.h5")
    assert dist_2.get_value_at_gll(id_element=0, id_gll=0) == 1.0

    # init with a file that is not a noise source file
    with pytest.raises(KeyError):
        Distribution.init_distribution_with_file(filename=wavefield_file_exists)


def test_basis_function():
    """
    TODO: test plotting
    :return:
    """
    spectrum = Spectrum(f_peak=0.08, bandwidth=0.07, strength=1.0)
    distribution = Distribution(distribution=1.0)
    BasisFunction(spectrum, distribution)


def test_noise_source():
    spectrum_1 = Spectrum(f_peak=0.08, bandwidth=0.07, strength=1.0)
    spectrum_2 = Spectrum(f_peak=0.08, bandwidth=0.07, strength=1.0)
    spectrum_3 = Spectrum(f_peak=0.08, bandwidth=0.07, strength=1.0)
    distribution = Distribution(distribution=1.0)

    bf_list = []
    bf_list.append(BasisFunction(spectrum_1, distribution))
    bf_list.append(BasisFunction(spectrum_2, distribution))
    bf_list.append(BasisFunction(spectrum_3, distribution))

    noise_source_1 = NoiseSource()
    noise_source_1.add_basis_function(bf_list[0])
    assert len(noise_source_1.list_basis_functions) == 1
    assert noise_source_1.get_value_of_distribution_at_gll(id_element=0, id_gll=0) == 1

    noise_source_2 = NoiseSource(bf_list)
    assert len(noise_source_2.list_basis_functions) == 3
    with pytest.raises(AssertionError):
        noise_source_2.get_value_of_distribution_at_gll(id_element=0, id_gll=0)

    noise_source_3 = NoiseSource([bf_list[0]])
    assert len(noise_source_3.list_basis_functions) == 1
    assert noise_source_1.get_value_of_distribution_at_gll(id_element=0, id_gll=0) == 1

    noise_source_4 = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum_1, distribution)
    assert len(noise_source_4.list_basis_functions) == 1
    assert noise_source_1.get_value_of_distribution_at_gll(id_element=0, id_gll=0) == 1

    noise_source_4.add_basis_function(BasisFunction(spectrum_3, distribution))
    assert len(noise_source_4.list_basis_functions) == 2
    with pytest.raises(AssertionError):
        noise_source_2.get_value_of_distribution_at_gll(id_element=0, id_gll=0)

    noise_source_5 = NoiseSource.init_noise_source_with_list(bf_list)
    assert len(noise_source_5.list_basis_functions) == 3
    with pytest.raises(AssertionError):
        noise_source_2.get_value_of_distribution_at_gll(id_element=0, id_gll=0)


def test_compute_stf_at_point():
    """
    TODO: test plotting and time
    :return:
    """
    spectrum = Spectrum(f_peak=0.08, bandwidth=0.07, strength=1.0)
    distribution = Distribution(distribution=1.0)
    noise_source = NoiseSource.init_noise_source_with_spectrum_and_distribution(spectrum, distribution)

    # set up a meaningful time and frequency vector
    nt_full = 2 * green_function.nt_longest_branch - 1
    max_abs_time = (green_function.nt_longest_branch - 1) * green_function.dt

    time = np.linspace(-max_abs_time, max_abs_time, nt_full)
    f = np.linspace(-green_function.fs / 2, green_function.fs / 2, nt_full)

    # direct way: compute stf at coordinate point
    stf = noise_source.compute_stf_at_gll(id_element=0, id_gll=0, f=f)
    assert sum(stf) == pytest.approx(5.4444534e1, rel=1e-4)
