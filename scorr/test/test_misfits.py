#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
import numpy as np
import pytest
from matplotlib import pyplot as plt

from scorr.extensions import misfit_helpers, misfits


def check_generic_1st(misfit_function):
    """
    test is now done by visual inspection

    :return:
    """
    # user specific choice
    scale = 1.0e10
    nt = 1501
    dt = 0.01

    # generate synthetics and observations
    u_obs = 2 * (np.random.rand(nt) - 0.5)
    u_ini = 2 * (np.random.rand(nt) - 0.5)
    du = 2 * (np.random.rand(nt) - 0.5)

    # design window
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
    t_min = 3.0
    t_max = 9.0
    win = misfit_helpers.evaluate_window(time, t_min, t_max, type="hann")

    # assert right format for test
    assert u_obs.shape == u_ini.shape == du.shape == win.shape

    # compute misfit and adjoint source time function
    j, adstf = misfit_function(u=u_ini, u_0=u_obs, win=win, dt=dt, scale=scale)

    # compute change of misfit with du via adjoint source time function
    djdu = np.sum(adstf * du)  # / scale

    # finite difference approximation of change of misfit by adding du to u_ini
    dcheck = []
    steps = np.arange(-12, 0, 0.1)
    for step in steps:
        duh = u_ini + du * np.power(10.0, step)
        jh = misfit_function(u=duh, u_0=u_obs, win=win, dt=dt, scale=scale)[0]
        djduh = (jh - j) / np.power(10.0, step)  # / scale
        dcheck.append(abs(djdu - djduh) / abs(djdu))

    # plot approximation
    plt.figure()
    plt.semilogy(steps, dcheck)
    plt.title(
        f"check adjoint source time function for {misfit_function.__name__}")
    plt.show()


def check_generic_2nd(misfit_function):
    """
    test is now done by visual inspection

    :return:
    """
    # user specific choice
    scale = 1.0e10
    nt = 1501
    dt = 0.01

    # generate synthetics and observations
    u_obs = 2 * (np.random.rand(nt) - 0.5)
    u_ini = 2 * (np.random.rand(nt) - 0.5)
    du1 = 2 * (np.random.rand(nt) - 0.5)
    du2 = 2 * (np.random.rand(nt) - 0.5)

    # design window
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
    t_min = 3.0
    t_max = 9.0
    win = misfit_helpers.evaluate_window(time, t_min, t_max, type="hann")

    # assert right format for test
    assert u_obs.shape == u_ini.shape == win.shape == du1.shape == du2.shape

    # compute adjoint source time function for 2nd order
    _, adstf_2nd = misfit_function(u=u_ini, u_0=u_obs, win=win, dt=dt, scale=scale,
                                   order=2, du=du1)

    # compute change of misfit with du via adjoint source time function
    djdu = np.sum(adstf_2nd * du2)  # / scale

    # compute adjoint source time function for first order as reference
    _, adstf_ref = misfit_function(
        u=u_ini, u_0=u_obs, win=win, dt=dt, scale=scale, order=1)

    # finite difference approximation of change of misfit by change of first order adjoint source time function
    dcheck = []
    steps = np.arange(-12, 0, 0.1)
    for step in steps:
        duh = u_ini + du1 * np.power(10.0, step)
        _, adstf_h = misfit_function(
            u=duh, u_0=u_obs, win=win, dt=dt, scale=scale)
        djduh = np.sum((adstf_h - adstf_ref) /
                       np.power(10.0, step) * du2)  # / scale
        dcheck.append(abs(djdu - djduh) / abs(djdu))

    # plot approximation
    plt.figure()
    plt.semilogy(steps, dcheck)
    plt.title(
        f"check adjoint source time function 2nd order for {misfit_function.__name__}")
    plt.show()


def check_log_amplitude_ratio():
    """
    test is now done by visual inspection

    :return:
    """
    # user specific choice
    scale = 1.0e10
    nt = 1501
    dt = 0.01

    # generate synthetics and observations
    u_obs = 2 * (np.random.rand(nt) - 0.5)
    u_ini = 2 * (np.random.rand(nt) - 0.5)
    du = 2 * (np.random.rand(nt) - 0.5)

    # design window
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
    t_min = 3.0
    t_max = 9.0
    win_caus = misfit_helpers.evaluate_window(time, t_min, t_max, type="hann")
    win_acaus = misfit_helpers.evaluate_window(
        time, -t_max, -t_min, type="hann")

    # assert right format for test
    assert u_obs.shape == u_ini.shape == du.shape == win_caus.shape == win_acaus.shape

    # compute misfit and adjoint source time function
    j, adstf = misfits.log_amplitude_ratio(u=u_ini, u_0=u_obs, win_caus=win_caus, win_Acaus=win_acaus, dt=dt,
                                           scale=scale)

    # compute change of misfit with du via adjoint source time function
    djdu = np.sum(adstf * du)  # / scale

    # finite difference approximation of change of misfit by adding du to u_ini
    dcheck = []
    steps = np.arange(-12, 0, 0.1)
    for step in steps:
        duh = u_ini + du * np.power(10.0, step)
        jh = misfits.log_amplitude_ratio(u=duh, u_0=u_obs, win_caus=win_caus, win_Acaus=win_acaus, dt=dt,
                                         scale=scale)[0]
        djduh = (jh - j) / np.power(10.0, step)  # / scale
        dcheck.append(abs(djdu - djduh) / abs(djdu))

    # plot approximation
    plt.figure()
    plt.semilogy(steps, dcheck)
    plt.title("check adjoint source time function for ASYMMETRY measurement")
    plt.show()


def check_log_amplitude_ratio_2nd():
    """
    test is now done by visual inspection

    :return:
    """
    # user specific choice
    scale = 1.0e10
    nt = 1501
    dt = 0.01

    # generate synthetics and observations
    u_obs = 2 * (np.random.rand(nt) - 0.5)
    u_ini = 2 * (np.random.rand(nt) - 0.5)
    du1 = 2 * (np.random.rand(nt) - 0.5)
    du2 = 2 * (np.random.rand(nt) - 0.5)

    # design window
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
    t_min = 3.0
    t_max = 9.0
    win_caus = misfit_helpers.evaluate_window(time, t_min, t_max, type="hann")
    win_acaus = misfit_helpers.evaluate_window(
        time, -t_max, -t_min, type="hann")

    # assert right format for test
    assert u_obs.shape == u_ini.shape == win_caus.shape == win_acaus.shape == du1.shape == du2.shape

    # compute adjoint source time function for 2nd order
    _, adstf_2nd = misfits.log_amplitude_ratio(u=u_ini, u_0=u_obs, win_caus=win_caus, win_Acaus=win_acaus, dt=dt,
                                               scale=scale, du=du1, order=2)

    # compute change of misfit with du via adjoint source time function
    djdu = np.sum(adstf_2nd * du2)  # / scale

    # compute adjoint source time function for first order as reference
    _, adstf_ref = misfits.log_amplitude_ratio(u=u_ini, u_0=u_obs, win_caus=win_caus, win_Acaus=win_acaus, dt=dt,
                                               scale=scale, order=1)

    # finite difference approximation of change of misfit by adding du to u_ini
    dcheck = []
    steps = np.arange(-12, -2, 0.1)
    for step in steps:
        duh = u_ini + du1 * np.power(10.0, step)
        _, adstf_h = misfits.log_amplitude_ratio(u=duh, u_0=u_obs, win_caus=win_caus, win_Acaus=win_acaus, dt=dt,
                                                 scale=scale, order=1)
        djduh = np.sum((adstf_h - adstf_ref) /
                       np.power(10.0, step) * du2)  # / scale
        dcheck.append(abs(djdu - djduh) / abs(djdu))

    # plot approximation
    plt.figure()
    plt.semilogy(steps, dcheck)
    plt.title("check adjoint source time function for ASYMMETRY measurement")
    plt.show()


def test_cc_time_shift():
    """
    test time shifts

    :return:
    """
    # user specific choice
    nt = 3001
    dt = 0.01
    lag_1 = 10
    lag_2 = 20

    # design window
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
    t_min = 3.0
    t_max = 9.0
    win = misfit_helpers.evaluate_window(time, t_min, t_max, type="cos")

    # generate synthetics and observations
    u_obs = 2 * (np.random.rand(nt) - 0.5)
    u_ini = np.zeros_like(u_obs)
    u_ini[:-lag_1] = u_obs[lag_1:]

    # compute misfit and adjoint source time function
    j, *_ = misfits.traveltime_shift_book(
        u=u_ini, u_0=u_obs, win=win, dt=dt, scale=1)

    # test time shift
    assert j == pytest.approx(0.5 * (lag_1 * dt) ** 2, rel=1e-4)

    # generate new synthetics - opposite direction
    u_ini = np.zeros_like(u_obs)
    u_ini[lag_2:] = u_obs[:-lag_2]

    # compute misfit and adjoint source time function
    j, *_ = misfits.traveltime_shift_book(
        u=u_ini, u_0=u_obs, win=win, dt=dt, scale=1)

    # test time shift
    assert j == pytest.approx(0.5 * (-lag_2 * dt) ** 2, rel=1e-4)


# def check_cc_time_shift_2nd():
#     """
#     small helper function - not a test for the final implementation
#
#     :return:
#     """
#     # user specific choice
#     scale = 1.0e10
#     nt = 1501
#     dt = 0.01
#
#     lag_1 = 10
#
#     # design window
#     time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
#     t_min = 3.0
#     t_max = 9.0
#     win = misfit_helpers.evaluate_window(time, t_min, t_max, type="cos")
#
#     # generate synthetics and observations
#     u_obs = 2 * (np.random.rand(nt) - 0.5)
#     u_ini = np.zeros_like(u_obs)
#     u_ini[:-lag_1] = u_obs[lag_1:]
#
#     # generate perturbations
#     du1 = 2 * (np.random.rand(nt) - 0.5)
#     du2 = 2 * (np.random.rand(nt) - 0.5)
#
#     # design window
#     time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
#     t_min = 3.0
#     t_max = 9.0
#     win_caus = misfit_helpers.evaluate_window(time, t_min, t_max, type="hann")
#     win_acaus = misfit_helpers.evaluate_window(time, -t_max, -t_min, type="hann")
#
#     # assert right format for test
#     assert u_obs.shape == u_ini.shape == win_caus.shape == win_acaus.shape == du1.shape == du2.shape
#
#     # compute adjoint source time function for 2nd order
#     _, adstf_2nd, _ = misfits.traveltime_shift_book(u=u_ini, u_0=u_obs, win=win, dt=dt, scale=scale, order=2, du=du1)
#
#     # compute change of misfit with du via adjoint source time function
#     djdu = np.sum(adstf_2nd * du2)  # / scale
#
#     # compute adjoint source time function for first order as reference
#     _, adstf_ref, _ = misfits.traveltime_shift_book(u=u_ini, u_0=u_obs, win=win, dt=dt, scale=scale, order=1)
#
#     # finite difference approximation of change of misfit by adding du to u_ini
#     dcheck = []
#     steps = np.arange(-12, -2, 0.1)
#     for step in steps:
#         duh = u_ini + du1 * np.power(10.0, step)
#         _, adstf_h, _ = misfits.traveltime_shift_book(u=duh, u_0=u_obs, win=win, dt=dt, scale=scale, order=1)
#         djduh = np.sum((adstf_h - adstf_ref) / np.power(10.0, step) * du2)  # / scale
#         dcheck.append(abs(djdu - djduh) / abs(djdu))
#
#     # plot approximation
#     plt.figure()
#     plt.semilogy(steps, dcheck)
#     plt.title("check adjoint source time function for TIME SHIFT measurement")
#     plt.show()


if __name__ == "__main__":
    check_generic_1st(misfits.waveform_differences)
    check_generic_2nd(misfits.waveform_differences)
    check_generic_1st(misfits.energy_difference)
    check_generic_2nd(misfits.energy_difference)
    check_generic_1st(misfits.envelope_difference)
    check_generic_2nd(misfits.envelope_difference)
    check_generic_1st(misfits.envelope_squared_difference)
    check_generic_2nd(misfits.envelope_squared_difference)
    check_log_amplitude_ratio()
    check_log_amplitude_ratio_2nd()

    # check_cc_time_shift_2nd()
