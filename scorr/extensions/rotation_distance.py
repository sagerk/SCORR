#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import math
from collections import namedtuple

import numpy as np
from geographiclib import geodesic


def from_cartesian_to_latlon_r(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    lat = np.rad2deg(np.arctan2(z, hxy))
    lon = np.rad2deg(np.arctan2(y, x))
    return lat, lon, r


def from_cartesian_to_latlon(x, y, z):
    return from_cartesian_to_latlon_r(x, y, z)[:-1]


def get_transformation_matrix_3d(latitude, longitude, inverse=False):
    """
    transformation matrix to convert x, y, z to ZNE
    multiplying the matrix with a [x, y, z].T will yield a vector with [Z, N, E].T
    """
    theta = np.deg2rad(90 - latitude)
    phi = np.deg2rad(longitude)

    M = np.array([
        # r unit vector 
        [np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
         np.cos(theta)],
        # theta
        [-np.cos(theta) * np.cos(phi),
         -np.cos(theta) * np.sin(phi),
         np.sin(theta)],
        # phi
        [-np.sin(phi), np.cos(phi), 0]
    ])

    if inverse:
        return np.linalg.inv(M)
    else:
        return M


def rotate_vector(M, vector):
    assert vector.shape[1] == 3

    vector_rotated = np.zeros_like(vector)
    for i in range(vector.shape[0]):
        vector_rotated[i, :] = np.dot(M, vector[i, :])

    return vector_rotated


def great_circle_distance(point1, point2):
    """
    Convenience function to calculate the great circle distance between two
    points on a spherical Earth.

    This method uses the Vincenty formula in the special case of a spherical
    Earth. For more accurate values use the geodesic distance calculations of
    geopy (https://github.com/geopy/geopy).

    :return: Distance in degrees as a floating point number.
    """

    # Convert to radians.
    lat1 = math.radians(point1[0])
    lat2 = math.radians(point2[0])
    long1 = math.radians(point1[1])
    long2 = math.radians(point2[1])
    long_diff = long2 - long1

    gd = math.degrees(math.atan2(math.sqrt((math.cos(lat2) * math.sin(long_diff)) ** 2 +
                                           (math.cos(lat1) * math.sin(lat2) - math.sin(lat1) *
                                            math.cos(lat2) * math.cos(long_diff)) ** 2),
                                 math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) *
                                 math.cos(long_diff)))

    return gd


def compute_distance_in_m(loc_src, loc_rec, spherical):
    if spherical:
        rec_lat, rec_lon = from_cartesian_to_latlon(x=loc_rec[0], y=loc_rec[1], z=loc_rec[2])
        src_lat, src_lon = from_cartesian_to_latlon(x=loc_src[0], y=loc_src[1], z=loc_src[2])

        distance_in_m = great_circle_distance([src_lat, src_lon], [rec_lat, rec_lon]) / 360.0 * (
                2 * np.pi * 6371 * 1e3)
    else:
        distance_in_m = np.linalg.norm(loc_src - loc_rec)

    return distance_in_m


Point = namedtuple("Point", ["lat", "lon"])


def great_circle_points(point_1, point_2, max_extension=None,
                        max_npts=100):
    """
    Generator yielding a number points along a greatcircle from point_1 to
    point_2. Max extension is the normalization factor. If the distance between
    point_1 and point_2 is exactly max_extension, then 3000 points will be
    returned, otherwise a fraction will be returned.

    If max_extension is not given, the generator will yield exactly max_npts
    points.
    """
    point = geodesic.Geodesic.WGS84.Inverse(
        lat1=point_1[0], lon1=point_1[1], lat2=point_2[0], lon2=point_2[1])
    line = geodesic.Geodesic.WGS84.Line(
        point_1[0], point_1[1], point["azi1"])

    if max_extension:
        npts = int((point["a12"] / float(max_extension)) * max_npts)
    else:
        npts = max_npts - 1

    if npts == 0:
        npts = 1
    for i in range(npts + 1):
        line_point = line.Position(i * point["s12"] / float(npts))
        yield Point(line_point["lat2"], line_point["lon2"])
