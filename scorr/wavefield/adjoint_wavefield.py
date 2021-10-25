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


class AdjointWavefield(Wavefield):
    @addons.verbose_mode
    def __init__(self, filename, starttime, endtime, sampling_rate=1, rotate=False, verbose=False):
        """
        initialize adjoint wavefield

        :param filename:
        :param starttime:
        :param endtime:
        :param verbose:
        """

        # adjoint wavefields in SCORR always cover acausal and causal branch
        # needed four source kernel and source for second part of structure kernel
        # UPDATE: NOT ANY MORE! we do not need the full causal branch
        # assert abs(starttime) == abs(endtime), "abs(starttime) == abs(endtime)"
        assert starttime <= 0.0, "starttime has to be <= 0"

        # initialize wavefield
        Wavefield.__init__(self, filename=filename, starttime=starttime, endtime=endtime, sampling_rate=sampling_rate,
                           rotate=rotate, verbose=verbose)

    def __str__(self):
        return f"dt of adjoint wavefield: {self.dt}"

    @classmethod
    @addons.verbose_mode
    def init_adjoint_wavefield_function_with_dict(cls, filename, config):
        """
        initialize adjoint wavefield with dictionary

        :param config:
        :param verbose:
        :return:
        """
        try:
            if config["simulation"]["corr_max_lag_causal"]:
                endtime = config["simulation"]["corr_max_lag_causal"]
            else:
                endtime = config["simulation"]["corr_max_lag"]

            return cls(filename=filename,
                       starttime=-config["simulation"]["corr_max_lag"],
                       endtime=endtime,
                       sampling_rate=config["simulation"]["sampling_rate_boundary"],
                       rotate=config["simulation"]["spherical"],
                       verbose=config["verbose"])
        except KeyError:
            print("Missing key in dictionary - input file incomplete?")
            raise
