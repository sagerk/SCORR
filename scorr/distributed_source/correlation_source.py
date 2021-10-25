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
from scorr.distributed_source.distributed_source import DistributedSource


class CorrelationSource(DistributedSource):
    @addons.verbose_mode
    def __init__(self, noise_source, green_function, verbose=False):
        """
        initialize correlation source

        :param noise_source:
        :param green_function:
        :param verbose:
        """
        DistributedSource.__init__(
            self, noise_source=noise_source, wavefield=green_function, verbose=verbose)

        # test if correlation source makes sense with given noise source and green function
        self.test_frequency_band()

    def test_frequency_band(self):
        """
        TODO: check if combination of noise source and green function is possible
        IDEA 1: check spectrum of green function and noise source (probably too expensive)
        IDEA 2: via CFL criterion and element size (but we only know the element size at the surface...)

        SOLUTION: follow IDEA 1, but only for one time series of green function
        :return:
        """
        pass
