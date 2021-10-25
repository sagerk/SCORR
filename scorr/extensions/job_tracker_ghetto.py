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

RefInfo_ghetto = collections.namedtuple(
    "RefInfo_ghetto",
    ["ref_station",
     "job_id_dGm",
     "job_id_dudagger",
     "inversion_type", "last_updated"])


@contextmanager
def sqlite_cursor():
    filename = Path.home() / Path(".scorr-ref_station-tracker-ghetto.sqlite")
    conn = sqlite3.connect(str(filename),
                           detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ref_stations (
            ref_station text NOT NULL UNIQUE,
            job_id_dGm text,
            job_id_dudagger text,
            inversion_type text,
            last_updated utc_datetime NOT NULL
        )
    """)
    yield c
    conn.commit()
    conn.close()


def add_reference_station(ref_station, job_id_dGm=None, job_id_dudagger=None, inversion_type=None):
    now = arrow.utcnow()
    with sqlite_cursor() as c:
        c.execute("""
            INSERT INTO ref_stations VALUES (
                ?, ?, ?, ?, ?
            )""", (
        f"{ref_station}", job_id_dGm, job_id_dudagger, inversion_type, now))

    return RefInfo_ghetto(ref_station=ref_station, job_id_dGm=job_id_dGm, job_id_dudagger=job_id_dudagger,
                          inversion_type=inversion_type, last_updated=now)


def get_all_reference_stations():
    with sqlite_cursor() as c:
        c.execute("SELECT * FROM ref_stations ORDER BY ref_station ASC")
        return [RefInfo_ghetto(*_i) for _i in c.fetchall()]


def get_jobs_of_reference_station(ref_station):
    """
    return None if no job_id exists for job
    """
    with sqlite_cursor() as c:
        c.execute("""
            SELECT * FROM ref_stations WHERE ref_station = ?
        """, (ref_station,))
        return [RefInfo_ghetto(*_i) for _i in c.fetchall()][0]


## TODO: add copy function


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


def reset_jobs_of_reference_station(ref_station, job_id_dGm=False, job_id_dudagger=False):
    ref_info_ghetto = get_jobs_of_reference_station(ref_station=ref_station)

    command = "UPDATE ref_stations SET "
    arguments = ()

    if job_id_dGm:
        command += f"job_id_dGm = ?, "
        arguments += (None,)
        ref_info_ghetto = ref_info_ghetto._replace(job_id_dGm=None)
    if job_id_dudagger:
        command += f"job_id_dudagger = ?, "
        arguments += (None,)
        ref_info_ghetto = ref_info_ghetto._replace(job_id_dudagger=None)

    command += f"last_updated = ? WHERE ref_station = ?"
    now = arrow.utcnow()
    arguments += (now, f"{ref_station}")
    ref_info_ghetto = ref_info_ghetto._replace(last_updated=now)

    with sqlite_cursor() as c:
        c.execute(command, arguments)

    return ref_info_ghetto


def update_reference_station(ref_station, job_id_dGm=None, job_id_dudagger=None,
                             inversion_type=None):
    ref_info_ghetto = get_jobs_of_reference_station(ref_station=ref_station)

    command = "UPDATE ref_stations SET "
    arguments = ()

    if job_id_dGm:
        command += f"job_id_dGm = ?, "
        arguments += (job_id_dGm,)
        ref_info_ghetto = ref_info_ghetto._replace(job_id_dGm=job_id_dGm)
    if job_id_dudagger:
        command += f"job_id_dudagger = ?, "
        arguments += (job_id_dudagger,)
        ref_info_ghetto = ref_info_ghetto._replace(job_id_dudagger=job_id_dudagger)
    if inversion_type is not None or inversion_type == "":
        command += f"inversion_type = ?, "
        arguments += (inversion_type,)
        ref_info_ghetto = ref_info_ghetto._replace(inversion_type=inversion_type)

    command += f"last_updated = ? WHERE ref_station = ?"
    now = arrow.utcnow()
    arguments += (now, f"{ref_station}")
    ref_info_ghetto = ref_info_ghetto._replace(last_updated=now)

    with sqlite_cursor() as c:
        c.execute(command, arguments)

    return ref_info_ghetto
