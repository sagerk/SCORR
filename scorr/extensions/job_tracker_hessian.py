#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import arrow


# Register adapter and converter for arrow datetime objects and enforce that
# they are UTC.
def arrow_adapter(dt):
    return dt.to("UTC").isoformat()


def arrow_converter(value):
    return arrow.get(value.decode())


sqlite3.register_adapter(arrow.Arrow, arrow_adapter)

sqlite3.register_converter("utc_datetime", arrow_converter)

RefInfo_hessian = collections.namedtuple(
    "RefInfo_hessian",
    ["ref_station",
     "job_id_p", "job_id_source_z", "job_id_z", "job_id_source_q", "job_id_q", "job_id_w",
     "job_id_source_kernel_pPG", "job_id_source_kernel_uPdG", "inversion_type", "last_updated"])


@contextmanager
def sqlite_cursor():
    filename = Path.home() / Path(".scorr-ref_station-tracker-hessian.sqlite")
    conn = sqlite3.connect(str(filename),
                           detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ref_stations (
            ref_station text NOT NULL UNIQUE,
            job_id_p text,
            job_id_source_z text,
            job_id_z text,
            job_id_source_q text,
            job_id_q text,
            job_id_w text,
            job_id_source_kernel_pPG text,
            job_id_source_kernel_uPdG text,
            inversion_type text,
            last_updated utc_datetime NOT NULL
        )
    """)
    yield c
    conn.commit()
    conn.close()


def add_reference_station(ref_station, job_id_p=None, job_id_source_z=None, job_id_z=None, job_id_source_q=None,
                          job_id_q=None, job_id_w=None, job_id_source_kernel_pPG=None, job_id_source_kernel_uPdG=None,
                          inversion_type=None):
    now = arrow.utcnow()
    with sqlite_cursor() as c:
        c.execute("""
            INSERT INTO ref_stations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )""", (
            f"{ref_station}", job_id_p, job_id_source_z, job_id_z, job_id_source_q, job_id_q, job_id_w,
            job_id_source_kernel_pPG, job_id_source_kernel_uPdG, inversion_type, now))

    return RefInfo_hessian(ref_station=ref_station, job_id_p=job_id_p, job_id_source_z=job_id_source_z,
                           job_id_z=job_id_z, job_id_source_q=job_id_source_q, job_id_q=job_id_q, job_id_w=job_id_w,
                           job_id_source_kernel_pPG=job_id_source_kernel_pPG,
                           job_id_source_kernel_uPdG=job_id_source_kernel_uPdG,
                           inversion_type=inversion_type, last_updated=now)


def get_all_reference_stations():
    with sqlite_cursor() as c:
        c.execute("SELECT * FROM ref_stations ORDER BY ref_station ASC")
        return [RefInfo_hessian(*_i) for _i in c.fetchall()]


def get_jobs_of_reference_station(ref_station):
    """
    return None if no job_id exists for job
    """
    with sqlite_cursor() as c:
        c.execute("""
            SELECT * FROM ref_stations WHERE ref_station = ?
        """, (ref_station,))
        return [RefInfo_hessian(*_i) for _i in c.fetchall()][0]


def copy_reference_station(ref_station_src, ref_station_dst):
    """
    copy reference station

    :param ref_station_src:
    :param ref_station_dst:
    :return:
    """
    ref_info_src = get_jobs_of_reference_station(ref_station=ref_station_src)
    ref_info_dst = add_reference_station(ref_station=ref_station_dst,
                                         job_id_p=ref_info_src.job_id_p,
                                         job_id_source_z=ref_info_src.job_id_source_z,
                                         job_id_z=ref_info_src.job_id_z,
                                         job_id_source_q=ref_info_src.job_id_source_q,
                                         job_id_q=ref_info_src.job_id_q,
                                         job_id_w=ref_info_src.job_id_w,
                                         job_id_source_kernel_pPG=ref_info_src.job_id_source_kernel_pPG,
                                         job_id_source_kernel_uPdG=ref_info_src.job_id_source_kernel_uPdG,
                                         inversion_type=ref_info_src.inversion_type)
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


def reset_jobs_of_reference_station(ref_station, job_id_p=False, job_id_source_z=False, job_id_z=False,
                                    job_id_source_q=False, job_id_q=False, job_id_w=False,
                                    job_id_source_kernel_pPG=False, job_id_source_kernel_uPdG=False):
    ref_info_hessian = get_jobs_of_reference_station(ref_station=ref_station)

    command = "UPDATE ref_stations SET "
    arguments = ()

    if job_id_p:
        command += f"job_id_p = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_p=None)
    if job_id_source_z:
        command += f"job_id_source_z = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_z=None)
    if job_id_z:
        command += f"job_id_z = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_z=None)
    if job_id_source_q:
        command += f"job_id_source_q = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_q=None)
    if job_id_q:
        command += f"job_id_q = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_q=None)
    if job_id_w:
        command += f"job_id_w = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_w=None)
    if job_id_source_kernel_pPG:
        command += f"job_id_source_kernel_pPG = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_kernel_pPG=None)
    if job_id_source_kernel_uPdG:
        command += f"job_id_source_kernel_uPdG = ?, "
        arguments += (None,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_kernel_uPdG=None)

    command += f"last_updated = ? WHERE ref_station = ?"
    now = arrow.utcnow()
    arguments += (now, f"{ref_station}")
    ref_info_hessian = ref_info_hessian._replace(last_updated=now)

    with sqlite_cursor() as c:
        c.execute(command, arguments)

    return ref_info_hessian


def update_reference_station(ref_station, job_id_p=None, job_id_source_z=None, job_id_z=None, job_id_source_q=None,
                             job_id_q=None, job_id_w=None, job_id_source_kernel_pPG=None,
                             job_id_source_kernel_uPdG=None, inversion_type=None):
    ref_info_hessian = get_jobs_of_reference_station(ref_station=ref_station)

    command = "UPDATE ref_stations SET "
    arguments = ()

    if job_id_p:
        command += f"job_id_p = ?, "
        arguments += (job_id_p,)
        ref_info_hessian = ref_info_hessian._replace(job_id_p=job_id_p)
    if job_id_source_z:
        command += f"job_id_source_z = ?, "
        arguments += (job_id_source_z,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_z=job_id_source_z)
    if job_id_z:
        command += f"job_id_z = ?, "
        arguments += (job_id_z,)
        ref_info_hessian = ref_info_hessian._replace(job_id_z=job_id_z)
    if job_id_source_q:
        command += f"job_id_source_q = ?, "
        arguments += (job_id_source_q,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_q=job_id_source_q)
    if job_id_q:
        command += f"job_id_q = ?, "
        arguments += (job_id_q,)
        ref_info_hessian = ref_info_hessian._replace(job_id_q=job_id_q)
    if job_id_w:
        command += f"job_id_w = ?, "
        arguments += (job_id_w,)
        ref_info_hessian = ref_info_hessian._replace(job_id_w=job_id_w)
    if job_id_source_kernel_pPG:
        command += f"job_id_source_kernel_pPG = ?, "
        arguments += (job_id_source_kernel_pPG,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_kernel_pPG=job_id_source_kernel_pPG)
    if job_id_source_kernel_uPdG:
        command += f"job_id_source_kernel_uPdG = ?, "
        arguments += (job_id_source_kernel_uPdG,)
        ref_info_hessian = ref_info_hessian._replace(job_id_source_kernel_uPdG=job_id_source_kernel_uPdG)
    if inversion_type is not None or inversion_type == "":
        command += f"inversion_type = ?, "
        arguments += (inversion_type,)
        ref_info_hessian = ref_info_hessian._replace(inversion_type=inversion_type)

    command += f"last_updated = ? WHERE ref_station = ?"
    now = arrow.utcnow()
    arguments += (now, f"{ref_station}")
    ref_info_hessian = ref_info_hessian._replace(last_updated=now)

    with sqlite_cursor() as c:
        c.execute(command, arguments)

    return ref_info_hessian
