#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.spatial import cKDTree
import numpy as np


class SphericalNearestNeighbour(object):
    """
    Spherical nearest neighbour queries using scipy's fast
    kd-tree implementation.

    credit: LION
    """

    def __init__(self, data):
        cart_data = self.spherical2cartesian(data)
        self.data = data
        self.kd_tree = cKDTree(data=cart_data, leafsize=10)

    def query(self, points, k=1):
        points = self.spherical2cartesian(points)
        d, i = self.kd_tree.query(points, k=k)
        return d, i

    @staticmethod
    def spherical2cartesian(data):
        """
        Converts an array of shape (x, 2) containing latitude/longitude
        pairs into an array of shape (x, 3) containing x/y/z assuming a
        radius of one for points on the surface of a sphere.
        """
        # Convert data from lat/lng to x/y/z, assume radius of 1
        colat = 90 - data["lat"]
        cart_data = np.empty((data.shape[0], 3))

        cart_data[:, 0] = np.sin(np.deg2rad(colat)) * np.cos(np.deg2rad(data["lon"]))
        cart_data[:, 1] = np.sin(np.deg2rad(colat)) * np.sin(np.deg2rad(data["lon"]))
        cart_data[:, 2] = np.cos(np.deg2rad(colat))

        return cart_data