#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
from pathlib import Path

from scorr.extensions import job_tracker, scorr_extensions

# DIR_PROJECT = Path.home() / "scorr_inversion_data_50_final"
# DIR_PROJECT = Path.home() / "scorr_inversion_synthetic"
# DIR_PROJECT = Path("/scratch/snx3000/sagerk/scorr_test_grenoble_cartesian")
DIR_PROJECT = Path("/home/ksager/scorr_test_grenoble")
# DIR_PROJECT = Path("/scratch/snx3000/sagerk/scorr_patrick")

id = "1000"
id_new = "1000"

# id = "10007"
# id_new = "100007"

# ref_station_list = scorr_extensions.load_reference_stations(DIR_PROJECT / "reference_stations.json").keys()
ref_station_list = ["syn_0"]

for item in ref_station_list:
    # if item == "syn_BK_CMB":
    #     continue

    # job_tracker.copy_reference_station(item + "_" + id, item + "_" + id_new)
    # job_tracker.reset_jobs_of_reference_station(item + "_" + id_new, job_id_adjoint_1=True, job_id_source_kernel=True)
    # job_tracker.reset_jobs_of_reference_station(item + "_" + id_new, job_id_adjoint_1=True, job_id_dist_adjstf=True,
    #                                             job_id_adjoint_2=True, job_id_source_kernel=True)
    # job_tracker.reset_jobs_of_reference_station(item + "_" + id_new, job_id_corr_source=True)#, job_id_correlation=True)
    # job_tracker.reset_jobs_of_reference_station(item + "_" + id_new, job_id_dist_adjstf=True)
    # job_tracker.reset_jobs_of_reference_station(item + "_" + id_new, job_id_corr_source=True, job_id_correlation=True)
    job_tracker.remove_reference_station(item + "_" + id)
    # job_tracker.remove_reference_station(item)
