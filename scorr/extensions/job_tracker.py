#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import arrow


class retry():
    """
    Decorator that will keep retrying the operation after a timeout.
    """

    def __init__(self, retries: int):
        self.retries = retries

    def __call__(self, f):
        def wrapped_f(*args, **kwargs):
            for _ in range(self.retries):
                try:
                    retval = f(*args, **kwargs)
                except sqlite3.OperationalError:
                    continue
                else:
                    return retval
                raise

        return wrapped_f


# Register adapter and converter for arrow datetime objects and enforce that
# they are UTC.
def arrow_adapter(dt):
    return dt.to("UTC").isoformat()


def arrow_converter(value):
    return arrow.get(value.decode())


sqlite3.register_adapter(arrow.Arrow, arrow_adapter)

sqlite3.register_converter("utc_datetime", arrow_converter)

RefInfo = collections.namedtuple(
    "RefInfo",
    ["ref_station",
     "job_id_green", "job_id_corr_source", "job_id_correlation",
     "job_id_adjoint_1", "job_id_source_kernel", "job_id_dist_adjstf", "job_id_adjoint_2",
     "inversion_type", "last_updated"])


@contextmanager
def sqlite_cursor():
    filename = Path.home() / Path(".scorr-ref_station-tracker.sqlite")
    conn = sqlite3.connect(str(filename),
                           detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ref_stations (
            ref_station text NOT NULL UNIQUE,
            job_id_green text,
            job_id_corr_source text,
            job_id_correlation text,
            job_id_adjoint_1 text,
            job_id_source_kernel text,
            job_id_dist_adjstf text,
            job_id_adjoint_2 text,
            inversion_type text,
            last_updated utc_datetime NOT NULL
        )
    """)
    yield c
    conn.commit()
    conn.close()


def add_reference_station(ref_station, job_id_green=None, job_id_corr_source=None, job_id_correlation=None,
                          job_id_adjoint_1=None, job_id_source_kernel=None, job_id_dist_adjstf=None,
                          job_id_adjoint_2=None, inversion_type=None):
    now = arrow.utcnow()
    with sqlite_cursor() as c:
        c.execute("""
            INSERT INTO ref_stations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (f"{ref_station}", job_id_green, job_id_corr_source, job_id_correlation,
              job_id_adjoint_1, job_id_source_kernel, job_id_dist_adjstf, job_id_adjoint_2,
              inversion_type, now))

    return RefInfo(ref_station=ref_station, job_id_green=job_id_green, job_id_corr_source=job_id_corr_source,
                   job_id_correlation=job_id_correlation, job_id_adjoint_1=job_id_adjoint_1,
                   job_id_source_kernel=job_id_source_kernel, job_id_dist_adjstf=job_id_dist_adjstf,
                   job_id_adjoint_2=job_id_adjoint_2, inversion_type=inversion_type, last_updated=now)


def get_all_reference_stations():
    with sqlite_cursor() as c:
        c.execute("SELECT * FROM ref_stations ORDER BY ref_station ASC")
        return [RefInfo(*_i) for _i in c.fetchall()]


def get_jobs_of_reference_station(ref_station):
    """
    return None if no job_id exists for job
    """
    with sqlite_cursor() as c:
        c.execute("""
            SELECT * FROM ref_stations WHERE ref_station = ?
        """, (ref_station,))
        return [RefInfo(*_i) for _i in c.fetchall()][0]


def copy_reference_station(ref_station_src, ref_station_dst, project_dir=None):
    """
    copy reference station

    :param ref_station_src:
    :param ref_station_dst:
    :return:
    """
    ref_info_src = get_jobs_of_reference_station(ref_station=ref_station_src)
    ref_info_dst = add_reference_station(ref_station=ref_station_dst,
                                         job_id_green=ref_info_src.job_id_green,
                                         job_id_corr_source=ref_info_src.job_id_corr_source,
                                         job_id_correlation=ref_info_src.job_id_correlation,
                                         job_id_adjoint_1=ref_info_src.job_id_adjoint_1,
                                         job_id_source_kernel=ref_info_src.job_id_source_kernel,
                                         job_id_dist_adjstf=ref_info_src.job_id_dist_adjstf,
                                         job_id_adjoint_2=ref_info_src.job_id_adjoint_2,
                                         inversion_type=ref_info_src.inversion_type)

    # # copy correlations, adjoint source time functions and kernels if they exist
    # if project_dir:
    #     project_dir = Path(project_dir)
    #     if ref_info_dst.job_id_correlation:
    #         try:
    #             shutil.copytree(src=project_dir / "correlations" / "synthetics" / ref_station_src,
    #                             dst=project_dir / "correlations" / "synthetics" / ref_station_dst)
    #         except FileNotFoundError:
    #             print("There are no synthetics to copy.")
    #
    #         try:
    #             shutil.copytree(src=project_dir / "correlations" / "observations" / ref_station_src,
    #                             dst=project_dir / "correlations" / "observations" / ref_station_dst)
    #         except FileNotFoundError:
    #             print("There are no observations to copy.")
    #
    #         try:
    #             shutil.copytree(src=project_dir / "correlations" / ref_station_src,
    #                             dst=project_dir / "correlations" / ref_station_dst)
    #         except FileNotFoundError:
    #             print("There are no other correlations to copy.")
    #
    #         try:
    #             shutil.copytree(src=project_dir / "adjoint_stf" / ref_station_src,
    #                             dst=project_dir / "adjoint_stf" / ref_station_dst)
    #         except FileNotFoundError:
    #             print("There are no adjoint source time functions to copy.")
    #
    #     if ref_info_dst.job_id_adjoint_1 or ref_info_dst.job_id_source_kernel or ref_info_dst.job_id_adjoint_2:
    #         try:
    #             shutil.copytree(src=project_dir / "kernels" / ref_station_src,
    #                             dst=project_dir / "kernels" / ref_station_dst)
    #         except FileNotFoundError:
    #             print("There are no kernels to copy.")

    return ref_info_dst


def rename_reference_station(ref_station_src, ref_station_dst, project_dir=None):
    """
    rename reference station

    :param ref_station_src:
    :param ref_station_dst:
    :return:
    """
    # change entries in database
    ref_info_dst = copy_reference_station(ref_station_src=ref_station_src, ref_station_dst=ref_station_dst)
    remove_reference_station(ref_station=ref_station_src)

    # copy correlations, adjoint source time functions and kernels if they exist
    if project_dir:
        project_dir = Path(project_dir)
        if ref_info_dst.job_id_correlation:
            try:
                shutil.move(src=project_dir / "correlations" / "synthetics" / ref_station_src,
                            dst=project_dir / "correlations" / "synthetics" / ref_station_dst)
            except FileNotFoundError:
                print("There are no synthetics to move.")

            try:
                shutil.move(src=project_dir / "correlations" / "observations" / ref_station_src,
                            dst=project_dir / "correlations" / "observations" / ref_station_dst)
            except FileNotFoundError:
                print("There are no observations to move.")

            try:
                shutil.move(src=project_dir / "correlations" / ref_station_src,
                            dst=project_dir / "correlations" / ref_station_dst)
            except FileNotFoundError:
                print("There are no other correlations to move.")

            try:
                shutil.move(src=project_dir / "adjoint_stf" / ref_station_src,
                            dst=project_dir / "adjoint_stf" / ref_station_dst)
            except FileNotFoundError:
                print("There are no adjoint source time functions to move.")

        if ref_info_dst.job_id_adjoint_1 or ref_info_dst.job_id_source_kernel or ref_info_dst.job_id_adjoint_2:
            try:
                shutil.move(src=project_dir / "kernels" / ref_station_src,
                            dst=project_dir / "kernels" / ref_station_dst)
            except FileNotFoundError:
                print("There are no kernels to move.")

    # return new ref_info
    return ref_info_dst


def reset_reference_station(ref_station, inversion_type=None):
    """
    could be implemented in a nicer way
    """
    remove_reference_station(ref_station=ref_station)
    return add_reference_station(ref_station=ref_station, inversion_type=inversion_type)


def remove_reference_station(ref_station):
    """
    Remove a job from the database.
    """
    with sqlite_cursor() as c:
        c.execute("""
            DELETE FROM ref_stations
            WHERE ref_station = ?
        """, (f"{ref_station}",))


def reset_jobs_of_reference_station(ref_station, job_id_green=False, job_id_corr_source=False, job_id_correlation=False,
                                    job_id_adjoint_1=False, job_id_source_kernel=False, job_id_dist_adjstf=False,
                                    job_id_adjoint_2=False):
    ref_info = get_jobs_of_reference_station(ref_station=ref_station)

    command = "UPDATE ref_stations SET "
    arguments = ()

    if job_id_green:
        command += f"job_id_green = ?, "
        arguments += (None,)
        ref_info = ref_info._replace(job_id_green=None)
    if job_id_corr_source:
        command += f"job_id_corr_source = ?, "
        arguments += (None,)
        ref_info = ref_info._replace(job_id_corr_source=None)
    if job_id_correlation:
        command += f"job_id_correlation = ?, "
        arguments += (None,)
        ref_info = ref_info._replace(job_id_correlation=None)
    if job_id_adjoint_1:
        command += f"job_id_adjoint_1 = ?, "
        arguments += (None,)
        ref_info = ref_info._replace(job_id_adjoint_1=None)
    if job_id_source_kernel:
        command += f"job_id_source_kernel = ?, "
        arguments += (None,)
        ref_info = ref_info._replace(job_id_source_kernel=None)

        if ref_info.inversion_type == "structure" or ref_info.inversion_type == "joint":
            if not job_id_green or not job_id_dist_adjstf or not job_id_adjoint_2:
                # raise ValueError("You also have to reset job_id_green, job_id_dist_adjstf and job_id_adjoint_2!")
                print("You might also have to reset job_id_green, job_id_dist_adjstf and job_id_adjoint_2!")

    if job_id_dist_adjstf:
        command += f"job_id_dist_adjstf = ?, "
        arguments += (None,)
        ref_info = ref_info._replace(job_id_dist_adjstf=None)
    if job_id_adjoint_2:
        command += f"job_id_adjoint_2 = ?, "
        arguments += (None,)
        ref_info = ref_info._replace(job_id_adjoint_2=None)

    command += f"last_updated = ? WHERE ref_station = ?"
    now = arrow.utcnow()
    arguments += (now, f"{ref_station}")
    ref_info = ref_info._replace(last_updated=now)

    with sqlite_cursor() as c:
        c.execute(command, arguments)

    return ref_info


@retry(5)
def update_reference_station(ref_station, job_id_green=None, job_id_corr_source=None, job_id_correlation=None,
                             job_id_adjoint_1=None, job_id_source_kernel=None, job_id_dist_adjstf=None,
                             job_id_adjoint_2=None, inversion_type=None):
    ref_info = get_jobs_of_reference_station(ref_station=ref_station)

    command = "UPDATE ref_stations SET "
    arguments = ()

    if job_id_green:
        command += f"job_id_green = ?, "
        arguments += (job_id_green,)
        ref_info = ref_info._replace(job_id_green=job_id_green)
    if job_id_corr_source:
        command += f"job_id_corr_source = ?, "
        arguments += (job_id_corr_source,)
        ref_info = ref_info._replace(job_id_corr_source=job_id_corr_source)
    if job_id_correlation:
        command += f"job_id_correlation = ?, "
        arguments += (job_id_correlation,)
        ref_info = ref_info._replace(job_id_correlation=job_id_correlation)
    if job_id_adjoint_1:
        command += f"job_id_adjoint_1 = ?, "
        arguments += (job_id_adjoint_1,)
        ref_info = ref_info._replace(job_id_adjoint_1=job_id_adjoint_1)
    if job_id_source_kernel:
        command += f"job_id_source_kernel = ?, "
        arguments += (job_id_source_kernel,)
        ref_info = ref_info._replace(job_id_source_kernel=job_id_source_kernel)
    if job_id_dist_adjstf:
        command += f"job_id_dist_adjstf = ?, "
        arguments += (job_id_dist_adjstf,)
        ref_info = ref_info._replace(job_id_dist_adjstf=job_id_dist_adjstf)
    if job_id_adjoint_2:
        command += f"job_id_adjoint_2 = ?, "
        arguments += (job_id_adjoint_2,)
        ref_info = ref_info._replace(job_id_adjoint_2=job_id_adjoint_2)
    if inversion_type is not None or inversion_type == "":
        command += f"inversion_type = ?, "
        arguments += (inversion_type,)
        ref_info = ref_info._replace(inversion_type=inversion_type)

    command += f"last_updated = ? WHERE ref_station = ?"
    now = arrow.utcnow()
    arguments += (now, f"{ref_station}")
    ref_info = ref_info._replace(last_updated=now)

    with sqlite_cursor() as c:
        c.execute(command, arguments)

    return ref_info
