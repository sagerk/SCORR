#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import io
import json
import os
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from decorator import decorate
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

group_name = "/DIST_BND/"
# group_name = "/ELASTIC/"

def git_hash_and_diff(filename):
    if rank == 0:
        with open(str(filename), mode="w") as fh:
            subprocess.call(['echo', 'git hash:'], stdout=fh)
            subprocess.call(['git', 'rev-parse', 'HEAD'], stdout=fh)
            subprocess.call(['echo', ''], stdout=fh)
            subprocess.call(['git', 'diff'], stdout=fh)
    comm.Barrier()


def to_json(python_object):
    if isinstance(python_object, Path):
        return {"__class__": "Path", "__value__": str(python_object)}
    raise TypeError(repr(python_object) + ' is not JSON serializable')


def from_json(json_object):
    if '__class__' in json_object:
        if json_object["__class__"] == "Path":
            return Path(json_object["__value__"])
    return json_object


def write_json_file(filename, configuration):
    comm.Barrier()
    if rank == 0:
        with io.open(str(filename), 'w') as fh:
            json.dump(OrderedDict(configuration), fh, default=to_json,
                      sort_keys=True, indent=4, separators=(",", ": "))
    comm.Barrier()


def load_json_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError

    with io.open(str(filename), "r") as fh:
        return json.load(fh, object_hook=from_json)


def _verbose_mode(f, *args, **kw):
    if len(args) >= 1 and args[-1] == True:
        if rank == 0:
            print("{}".format(trim_docstring(f.__doc__)))

    return f(*args, **kw)


def verbose_mode(f):
    return decorate(f, _verbose_mode)


def trim_docstring(docstring):
    """
    from https://www.python.org/dev/peps/pep-0257/
    modifications:
    - changed maxint to maxsize for python 3
    - only return first line

    :param docstring:
    :return:
    """
    if not docstring:
        return ''

    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()

    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))

    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
            break

    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Return a single string:
    return '\n'.join(trimmed)


def make_rank_element_dictionary(size):
    def wrapper(n_elements):
        ideal_size = int(n_elements / size)
        remainder = np.mod(n_elements, size)

        # more elements than ranks
        if ideal_size >= 1:
            rank_element = {}
            for key in range(size):
                if key < remainder:
                    # The first 'remainder' ranks get 'count + 1' tasks each
                    start = key * (ideal_size + 1)
                    stop = start + ideal_size + 1
                else:
                    # The remaining 'size - remainder' ranks get 'count' task each
                    start = key * ideal_size + remainder
                    stop = start + ideal_size

                rank_element[key] = (start, stop)

            # make sure that the last element is always included
            rank_element[size - 1] = (rank_element[size - 1][0], n_elements)

        # rare case: less elements than ranks; last size-n_elements_global will idle
        else:
            rank_element = {}
            for i in range(size):
                if i < n_elements:
                    rank_element[i] = (i, i + 1)
                else:
                    rank_element[i] = (n_elements, n_elements)

        return rank_element

    return wrapper


rank_element_dictionary = make_rank_element_dictionary(size)
