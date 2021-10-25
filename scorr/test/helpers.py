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

from mpi4py import MPI

from scorr.noise_source.spectrum import Spectrum
from scorr.wavefield.green_function import GreenFunction

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# most generic way to get the test directory.
DIR_TEST = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
DIR_TEST_TMP = DIR_TEST / "tmp"
DIR_TEST_DATA = DIR_TEST / "data"

TEST_config = DIR_TEST / ".." / "config" / "scorr_template.json"
TEST_noise_source = DIR_TEST / ".." / "config" / "noise_source_template.json"
TEST_site = DIR_TEST / ".." / "config" / "site_template.json"

wavefield_file_exists = DIR_TEST_DATA / "wavefield_BND_green.h5"

try:
    green_function = GreenFunction(str(wavefield_file_exists), starttime=0.0, endtime=1.0)
except:
    print("problem reading green function")
    raise

try:
    spectrum = Spectrum(f_peak=0.08, bandwidth=0.07, strength=1)
except:
    print("problem with spectrum")
    raise
