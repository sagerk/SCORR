#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
from setuptools import setup, find_packages

setup_config = dict(
    name="scorr",
    version="0.0.1",
    description="",
    author="Korbinian Sager",
    author_email="korbinian_sager@brown.edu",
    url="https://gitlab.com/sagerk/SCORR",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "prettytable",
        "geographiclib"
    ],
    entry_points="""
        [console_scripts]
        scorr=scorr.main:main
    """
)

if __name__ == "__main__":
    setup(**setup_config)
