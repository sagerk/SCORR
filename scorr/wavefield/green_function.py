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
from scorr.wavefield.wavefield import Wavefield


class GreenFunction(Wavefield):
    @addons.verbose_mode
    def __init__(self, filename, starttime, endtime, sampling_rate=1, rotate=False, verbose=False):
        """
        initialize green function

        :param filename:
        :param starttime:
        :param endtime:
        :param verbose:
        """
        assert starttime <= 0.0, "starttime has to be <= 0"
        Wavefield.__init__(self, filename=filename,
                           starttime=starttime,
                           endtime=endtime,
                           sampling_rate=sampling_rate,
                           rotate=rotate,
                           verbose=verbose)

    def __str__(self):
        return f"dt of Green function wavefield: {self.dt}"

    @classmethod
    @addons.verbose_mode
    def init_green_function_with_dict(cls, filename, config):
        """
        initialize green function with dictionary

        :param config:
        :param verbose:
        :return:
        """
        try:
            return cls(filename=filename,
                       starttime=config["simulation"]["green_starttime"],
                       endtime=config["simulation"]["corr_max_lag"],
                       sampling_rate=config["simulation"]["sampling_rate_boundary"],
                       rotate=config["simulation"]["spherical"],
                       verbose=config["verbose"])
        except KeyError:
            print("Missing key in dictionary - input file incomplete?")
            raise
