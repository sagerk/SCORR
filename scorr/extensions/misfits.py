#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCORR - Salvus Correlation

:copyright:
    Korbinian Sager (korbinian_sager@brown.edu), 2021
:license:
    MIT License
"""
from tempfile import TemporaryFile

import numpy as np
from scipy.signal import hilbert

from scorr.extensions import misfit_helpers


def make_measurement(config_measurement, u, u_0, du, starttime, dt, distance_in_m, order, fh=TemporaryFile(mode="w")):
    """
    returns 'None, None, message' if no measurement can be made
    TODO for cc_time_shifts: only perform measurement if necessary

    :param config_measurement:
    :param u:
    :param u_0:
    :param du:
    :param starttime:
    :param dt:
    :param distance_in_m:
    :param order:
    :return:
    """
    nt = np.size(u)
    time = np.linspace(starttime, starttime + (nt - 1) * dt, nt)

    ############################
    ###      PICK WINDOW     ###
    ############################
    t_min, t_max, message = misfit_helpers.pick_window(config_measurement=config_measurement,
                                                       time=time, u=u, u_0=u_0, distance_in_m=distance_in_m)
    # no measurement if t_min and t_max are not valid
    if t_min is None or t_max is None:
        print("No measurement. " + message)
        return None, None, message

    ############################
    ###    QUALITY CONTROL   ###
    ############################
    # if config_measurement["type"] == "log_amplitude_ratio":
    #     only_snr = True
    # else:
    #     # currently max_cc is only calculated/needed for time shift measurements
    #     only_snr = True

    # snr_caus, snr_Acaus, max_cc_caus, max_cc_Acaus, message = \
    #     misfit_helpers.quality_control(config_measurement=config_measurement, u=u, u_0=u_0,
    #                                    starttime=starttime, dt=dt, distance_in_m=distance_in_m, fh=fh,
    #                                    only_snr=only_snr)

    ############################
    ###    MISFIT & ADSTF    ###
    ############################
    if config_measurement["type"] == "waveform_differences":
        # get windows
        win = misfit_helpers.evaluate_window(
            time=time, t_min=t_min, t_max=t_max, type="cos")
        # win += evaluate_window(time=time, t_min=-t_max, t_max=-t_min, type="cos")

        # make measurement and compute corresponding adjoint source time function
        misfit, adstf = waveform_differences(u=u, u_0=u_0, win=win, dt=dt, scale=config_measurement["scale"],
                                             du=du, order=order)

    elif "difference" in config_measurement["type"]:
        # TODO: quality control

        # get windows
        win_caus = misfit_helpers.evaluate_window(
            time=time, t_min=t_min, t_max=t_max, type="hann")
        win_Acaus = misfit_helpers.evaluate_window(
            time=time, t_min=-t_max, t_max=-t_min, type="hann")

        # perform measurements
        if config_measurement["type"] == "energy_difference":
            misfit_caus, adstf_caus = energy_difference(u=u, u_0=u_0, win=win_caus, dt=dt,
                                                        scale=config_measurement["scale"], du=du, order=order)
            misfit_Acaus, adstf_Acaus = energy_difference(u=u, u_0=u_0, win=win_Acaus, dt=dt,
                                                          scale=config_measurement["scale"], du=du, order=order)

        elif config_measurement["type"] == "envelope_difference":
            misfit_caus, adstf_caus = envelope_difference(u=u, u_0=u_0, win=win_caus, dt=dt,
                                                          scale=config_measurement["scale"], du=du, order=order)
            misfit_Acaus, adstf_Acaus = envelope_difference(u=u, u_0=u_0, win=win_Acaus, dt=dt,
                                                            scale=config_measurement["scale"], du=du, order=order)

        elif config_measurement["type"] == "envelope_squared_difference":
            misfit_caus, adstf_caus = envelope_squared_difference(u=u, u_0=u_0, win=win_caus, dt=dt,
                                                                  scale=config_measurement["scale"], du=du,
                                                                  order=order)
            misfit_Acaus, adstf_Acaus = envelope_squared_difference(u=u, u_0=u_0, win=win_Acaus, dt=dt,
                                                                    scale=config_measurement["scale"], du=du,
                                                                    order=order)

        else:
            raise NotImplementedError(
                f"{config_measurement['type']} is not implemented!")

        # assemble misfit and adstf
        if order == 1:
            misfit = misfit_caus + misfit_Acaus
        else:
            misfit = None
        adstf = adstf_caus + adstf_Acaus

    elif config_measurement["type"] == "log_amplitude_ratio":
        if snr_caus < config_measurement["snr"] or snr_Acaus < config_measurement["snr"]:
            print("No measurement. " + message)
            return None, None, message

        # get windows
        win_caus = misfit_helpers.evaluate_window(
            time=time, t_min=t_min, t_max=t_max, type="hann")
        win_Acaus = misfit_helpers.evaluate_window(
            time=time, t_min=-t_max, t_max=-t_min, type="hann")

        # make measurement and compute corresponding adjoint source time function
        misfit, adstf = log_amplitude_ratio(u=u, u_0=u_0, win_caus=win_caus, win_Acaus=win_Acaus, dt=dt,
                                            scale=config_measurement["scale"], du=du, order=order, fh=fh)

    elif config_measurement["type"] == "cc_time_shift":
        # get windows
        win_caus = misfit_helpers.evaluate_window(
            time=time, t_min=t_min, t_max=t_max, type="tukey")
        # win_Acaus = misfit_helpers.evaluate_window(
        #     time=time, t_min=-t_max, t_max=-t_min, type="cos")

        # make measurement and compute corresponding adjoint source time function
        misfit_caus, adstf_caus, time_shift_caus, max_cc_caus = \
            traveltime_shift_book(u=u, u_0=u_0, win=win_caus, dt=dt, scale=config_measurement["scale"],
                                  du=du, order=order, fh=fh)
        # misfit_Acaus, adstf_Acaus, time_shift_Acaus, max_cc_Acaus = \
        #     traveltime_shift_book(u=u, u_0=u_0, win=win_Acaus, dt=dt, scale=config_measurement["scale"],
        #                           du=du, order=order, fh=fh)

        # check range and quality - could check cc-requirement before calculating time shifts
        # if abs(time_shift_caus) >= config_measurement["min_period_in_s"] / 2.0:
        #     message += f"Time shift of {time_shift_caus} is larger than half a period!"
        #     adstf_caus = None
        # if max_cc_caus < config_measurement["correlation_coeff"]:
        #     adstf_caus = None

        # if abs(time_shift_Acaus) >= config_measurement["min_period_in_s"] / 2.0:
        #     message += f"Time shift of {time_shift_Acaus} is larger than half a period!"
        #     adstf_Acaus = None
        # if max_cc_Acaus < config_measurement["correlation_coeff"]:
        #     adstf_Acaus = None

        # combine causal and acausal measurements
        # if adstf_caus is None and adstf_Acaus is None:
        #     print("No measurement. " + message)
        #     return None, None, message
        if adstf_caus is None:
            message = "Only measured on acausal: " + message
            print(message)
            misfit = misfit_Acaus
            adstf = adstf_Acaus
        # elif adstf_Acaus is None:
        #     message = "Only measured on causal: " + message
        #     print(message)
        #     misfit = misfit_caus
        #     adstf = adstf_caus
        else:
            adstf = adstf_caus # + adstf_Acaus
            if order == 1:
                misfit = misfit_caus # + misfit_Acaus
            else:
                misfit = None

    elif config_measurement["type"] == "cc_time_asymmetry":
        # get windows
        win_caus = misfit_helpers.evaluate_window(
            time=time, t_min=t_min, t_max=t_max, type="cos")
        win_Acaus = misfit_helpers.evaluate_window(
            time=time, t_min=-t_max, t_max=-t_min, type="cos")

        # make measurement and compute corresponding adjoint source time function
        misfit, adstf, time_shift_caus, time_shift_Acaus = \
            traveltime_asymmetry_book(u=u, u_0=u_0, win_caus=win_caus, win_Acaus=win_Acaus, dt=dt,
                                      scale=config_measurement["scale"], du=du, order=order, fh=fh)

        # check range and quality
        if abs(time_shift_caus) >= config_measurement["min_period_in_s"] / 2.0:
            message += f"Time shift of {time_shift_caus} is larger than half a period!"
            adstf = None
        if max_cc_caus < config_measurement["correlation_coeff"]:
            adstf = None

        if abs(time_shift_Acaus) >= config_measurement["min_period_in_s"] / 2.0:
            message += f"Time shift of {time_shift_Acaus} is larger than half a period!"
            adstf = None
        if max_cc_Acaus < config_measurement["correlation_coeff"]:
            adstf = None

        # check if measurement was possible
        if adstf is None:
            print("No measurement. " + message)
            return None, None, message

    else:
        raise NotImplementedError(f"Measurement type {config_measurement['type']} is not implemented!\n"
                                  f"Available options: [waveform_differences, log_amplitude_ratio, "
                                  f"cc_time_shift, cc_time_asymmetry]")

    # return negative adstf, see equation (24) in Sager et al. (2018a)
    # flip adstf due to terminal condition of adjoint run
    return misfit, -np.flipud(adstf), message


def waveform_differences(u, u_0, win, dt, scale, du=None, order=1):
    assert u.shape == u_0.shape == win.shape

    if order == 1:
        misfit = 0.5 * scale * np.sum(win ** 2 * (u - u_0) ** 2) * dt
        adstf = win ** 2 * (u - u_0) * dt
    elif order == 2:
        assert du.shape == u.shape
        misfit = None
        adstf = win ** 2 * du * dt
    else:
        raise ValueError("Order has to be either 1 or 2!")

    return misfit, scale * adstf


def energy_difference(u, u_0, win, dt, scale, du=None, order=1):
    assert u.shape == u_0.shape == win.shape

    # compute energy
    energy = np.sum((win * u) ** 2) * dt
    energy_0 = np.sum((win * u_0) ** 2) * dt

    # compute measurement
    measurement = energy - energy_0

    if order == 1:
        misfit = 0.5 * scale * measurement ** 2
        adstf = measurement * (2 * win ** 2 * u * dt)
    elif order == 2:
        misfit = None
        adstf = measurement * (2 * win ** 2 * du * dt) + \
            np.sum(2 * win ** 2 * u * du * dt) * (2 * win ** 2 * u * dt)

    else:
        raise ValueError("Order has to be either 1 or 2!")

    return misfit, scale * adstf


def envelope_difference(u, u_0, win, dt, scale, du=None, order=1):
    assert u.shape == u_0.shape == win.shape

    envelope_0 = misfit_helpers.compute_envelope(u_0)
    envelope = misfit_helpers.compute_envelope(u)
    d_env_1 = u
    d_env_2 = np.imag(hilbert(u))

    # compute measurement
    measurement = envelope - envelope_0

    if order == 1:
        misfit = 0.5 * scale * np.sum(win ** 2 * measurement ** 2) * dt
        adstf = win ** 2 * measurement / envelope * d_env_1 * dt - \
            np.imag(hilbert(win ** 2 * measurement / envelope * d_env_2)) * dt
    elif order == 2:
        misfit = None

        d_c1_1 = du
        d_c1_2 = np.imag(hilbert(du))
        a = win ** 2 * measurement / envelope * dt
        b = win ** 2 * envelope_0 / \
            (envelope ** 3) * (d_env_1 * d_c1_1 + d_env_2 * d_c1_2) * dt

        adstf = a * d_c1_1 - \
            np.imag(hilbert(a * d_c1_2)) + b * d_env_1 - \
            np.imag(hilbert(b * d_env_2))
    else:
        raise ValueError("Order has to be either 1 or 2!")

    return misfit, scale * adstf


def envelope_squared_difference(u, u_0, win, dt, scale, du=None, order=1):
    assert u.shape == u_0.shape == win.shape

    envelope_squared_0 = misfit_helpers.compute_envelope_squared(u_0)
    envelope_squared = misfit_helpers.compute_envelope_squared(u)
    d_env_1 = u
    d_env_2 = np.imag(hilbert(u))

    measurement = envelope_squared - envelope_squared_0

    if order == 1:
        misfit = 0.5 * scale * np.sum(win ** 2 * measurement ** 2) * dt
        adstf = win ** 2 * measurement * 2 * d_env_1 * dt - \
            np.imag(hilbert(win ** 2 * measurement * 2 * d_env_2)) * dt
    elif order == 2:
        misfit = None

        d_d_env_2 = np.imag(hilbert(du))

        term1_1 = 4 * win ** 2 * u ** 2 * du
        term1_2 = -4 * np.imag(hilbert(win ** 2 * d_env_1 * d_env_2 * du))
        term1_3 = 2 * win ** 2 * measurement * du

        term2_1 = 4 * np.imag(hilbert(win ** 2 * d_d_env_2 * d_env_2 ** 2))
        term2_2 = -4 * win ** 2 * d_d_env_2 * d_env_2 * d_env_1
        term2_3 = 2 * np.imag(hilbert(win ** 2 * measurement * d_d_env_2))

        term1 = term1_1 + term1_2 + term1_3
        term2 = term2_1 + term2_2 + term2_3

        # assemble full adjoint source time function
        adstf = (term1 - term2) * dt
    else:
        raise ValueError("Order has to be either 1 or 2!")

    return misfit, scale * adstf


def log_amplitude_ratio(u, u_0, win_caus, win_Acaus, dt, scale, du=None, order=1, fh=TemporaryFile(mode="w")):
    assert u.shape == u_0.shape == win_caus.shape == win_Acaus.shape

    # compute energies of synthetic correlation function
    e_caus = np.sum(np.power(win_caus * np.squeeze(u), 2)) * dt
    e_Acaus = np.sum(np.power(win_Acaus * np.squeeze(u), 2)) * \
        dt + np.finfo(float).eps

    # compute energies of observed correlation function
    e_0_caus = np.sum(np.power(win_caus * np.squeeze(u_0), 2)) * dt
    e_0_Acaus = np.sum(np.power(win_Acaus * np.squeeze(u_0), 2)
                       ) * dt + np.finfo(float).eps

    # compute asymmetries
    A = np.log(e_caus / e_Acaus)
    A_0 = np.log(e_0_caus / e_0_Acaus)

    fh.write(str(A) + " ")
    fh.write(str(A_0) + " ")

    if order == 1:
        # compute misfit
        misfit = 0.5 * scale * (A - A_0) ** 2

        # compute adjoint source time function
        de_caus = 2.0 * np.power(win_caus, 2) * np.squeeze(u) * dt
        de_Acaus = 2.0 * np.power(win_Acaus, 2) * np.squeeze(u) * dt

        adstf = (A - A_0) * (de_caus / e_caus - de_Acaus / e_Acaus)

    elif order == 2:
        misfit = None

        de_caus = np.sum(2 * win_caus ** 2 * u * du) * dt
        de_Acaus = np.sum(2 * win_Acaus ** 2 * u * du) * dt

        adstf = (A - A_0) * (2 / e_caus * win_caus ** 2 - 2 / e_Acaus * win_Acaus ** 2) * du * dt + \
                (A - A_0) * (-2 / e_caus ** 2 * win_caus ** 2 * u * de_caus +
                             2 / e_Acaus ** 2 * win_Acaus ** 2 * u * de_Acaus) * dt + \
                (2 / e_caus * win_caus ** 2 * u - 2 / e_Acaus * win_Acaus ** 2 * u) * \
                (de_caus / e_caus - de_Acaus / e_Acaus) * dt

    else:
        raise ValueError("Order has to be either 1 or 2!")

    # return misfit and adjoint source time function
    return misfit, scale * adstf


def traveltime_shift_book(u, u_0, win, dt, scale, du=None, order=1, fh=TemporaryFile(mode="w")):
    assert u.shape == u_0.shape == win.shape

    # compute velocity and accelaration seismograms
    v = np.zeros_like(u)
    v[:-1] = np.diff(u) / dt
    a = np.zeros_like(u)
    a[:-1] = np.diff(v) / dt

    # estimate time shift
    # time_shift = misfit_helpers.compute_time_shift(u, u_0, win, dt)
    if np.sum(u_0) == 0.0:
        time_shift = 1.0
        max_cc = 1.0
    else:
        time_shift, max_cc = misfit_helpers.compute_time_shift_and_max_cc(
            u, u_0, win, dt)
    fh.write(str(time_shift) + " ")

    # compute misfit and adjoint source time functions
    if order == 1:
        # compute misfit
        misfit = 0.5 * scale * time_shift ** 2

        # compute adjoint source time function
        N = -np.sum(win * u * win * a) * dt
        adstf = scale * time_shift * (win * v) / N

    elif order == 2:
        # misfit
        misfit = None

        # compute velocity and acceleration version of du
        dv = np.zeros_like(du)
        dv[:-1] = np.diff(du) / dt
        da = np.zeros_like(du)
        da[:-1] = np.diff(dv) / dt

        # compute adjoint source time function
        N = -np.sum(win * u * win * a) * dt
        dN = -np.sum(win * du * win * a) * dt - np.sum(win * u * win * da) * dt

        adstf_1 = -scale * time_shift * (win * v) / (N ** 2) * dN
        adstf_2 = scale * time_shift * (win * dv) / N
        adstf_3 = scale * dv * ((win * v) / N) ** 2
        adstf = adstf_1 + adstf_2 + adstf_3

    else:
        raise ValueError("Order has to be either 1 or 2!")

    # return misfit and adjoint source time function
    return misfit, adstf, time_shift, max_cc


def traveltime_shift_pure(u, u_0, win, dt, scale, du=None, order=1, fh=TemporaryFile(mode="w")):
    assert u.shape == u_0.shape == win.shape

    # compute velocity seismograms and apply windows
    v_0 = np.zeros_like(u_0)
    v_0[:-1] = np.diff(u_0) / dt
    v_0 = win * v_0

    # compute acceleration seismograms and apply windows
    a_0 = np.zeros_like(u_0)
    a_0[:-1] = np.diff(v_0) / dt
    a_0 = win * a_0

    # estimate time shift
    # time_shift = misfit_helpers.compute_time_shift(u, u_0, win, dt)
    time_shift, max_cc = misfit_helpers.compute_time_shift_and_max_cc(
        u, u_0, win, dt)
    fh.write(str(time_shift) + " ")

    # prepare shifted versions
    v_0_shift = np.zeros_like(u_0)
    a_0_shift = np.zeros_like(u_0)
    lag = int(abs(time_shift) / dt)

    if (time_shift < 0):
        v_0_shift[lag:] = v_0[:-lag]
        a_0_shift[lag:] = a_0[:-lag]
    elif (time_shift > 0):
        v_0_shift[:-lag] = v_0[lag:]
        a_0_shift[:-lag] = a_0[lag:]
    else:
        v_0_shift = v_0
        a_0_shift = a_0

    # compute misfit and adjoint source time functions
    if order == 1:
        # compute misfit
        misfit = 0.5 * scale * time_shift ** 2

        # compute adjoint source time function
        N = -np.sum(win * u * a_0_shift) * dt
        adstf = scale * time_shift * (v_0_shift / N)

    elif order == 2:
        # misfit
        misfit = None

        # compute adjoint source time function
        N = -np.sum(win * u * a_0_shift) * dt
        dN = -np.sum(win * du * a_0_shift) * dt

        adstf_1 = -scale * time_shift * v_0_shift / (N ** 2) * dN
        adstf_2 = scale * du * (v_0_shift / N) ** 2
        adstf = adstf_1 + adstf_2

    else:
        raise ValueError("Order has to be either 1 or 2!")

    # nt = np.size(u)
    # # time = np.linspace(-21600, 12000, nt)
    # time = np.linspace(-25, 25, nt)
    # plt.figure(figsize=(12, 8))
    # plt.plot(time, win * u, "b")
    # plt.plot(time, win * u_0, "k")
    # plt.plot(time, adstf, 'r')
    # # plt.plot(time, win, "r")
    # # plt.xlim(-12000, 12000)
    # plt.xlim(-25, 25)
    # plt.show()

    # return misfit and adjoint source time function
    return misfit, adstf, time_shift, max_cc


def traveltime_asymmetry_book(u, u_0, win_caus, win_Acaus, dt, scale, du=None, order=1, fh=TemporaryFile(mode="w")):
    assert u.shape == u_0.shape == win_caus.shape == win_Acaus.shape

    # compute velocity and acceleration seismograms
    v = np.zeros_like(u)
    v[:-1] = np.diff(u) / dt
    a = np.zeros_like(u)
    a[:-1] = np.diff(v) / dt

    # estimate time shifts
    time_shift_caus = misfit_helpers.compute_time_shift(u, u_0, win_caus, dt)
    fh.write(str(time_shift_caus) + " ")

    time_shift_Acaus = misfit_helpers.compute_time_shift(u, u_0, win_Acaus, dt)
    fh.write(str(time_shift_Acaus) + " ")

    # compute time difference
    t_diff = time_shift_caus + time_shift_Acaus

    # set up parts for adjoint source time function
    N_caus = -np.sum(win_caus * u * win_caus * a) * dt
    N_Acaus = -np.sum(win_Acaus * u * win_Acaus * a) * dt
    adstf_caus = (win_caus * v) / N_caus
    adstf_Acaus = (win_Acaus * v) / N_Acaus

    # compute misfit and adjoint source time functions
    if order == 1:
        # misfit
        misfit = 0.5 * scale * t_diff ** 2

        # assemble complete adjoint source time function
        adstf = scale * t_diff * (adstf_caus + adstf_Acaus)

    elif order == 2:
        # misfit
        misfit = None

        # compute velocity and acceleration version of du
        dv = np.zeros_like(du)
        dv[:-1] = np.diff(du) / dt
        da = np.zeros_like(du)
        da[:-1] = np.diff(dv) / dt

        # compute adjoint source time function
        dN_caus = -np.sum(win_caus * du * win_caus * a) * \
            dt - np.sum(win_caus * u * win_caus * da) * dt
        dN_Acaus = -np.sum(win_Acaus * du * win_Acaus * a) * \
            dt - np.sum(win_Acaus * u * win_Acaus * da) * dt

        adstf_1_new = scale * time_shift_caus * win_caus * dv / N_caus \
            - scale * time_shift_caus * win_caus * v / N_caus ** 2 * dN_caus \
            + scale * ((win_caus * v)/N_caus) ** 2 * dv
        adstf_2_new = scale * time_shift_caus * win_Acaus * dv / N_Acaus \
            - scale * time_shift_caus * win_Acaus * v / N_Acaus ** 2 * dN_Acaus \
            + scale * (win_caus * v)/N_caus * (win_Acaus * v)/N_Acaus * dv
        adstf_3_new = scale * time_shift_Acaus * win_caus * dv / N_caus \
            - scale * time_shift_Acaus * win_caus * v / N_caus ** 2 * dN_caus \
            + scale * (win_caus * v)/N_caus * (win_Acaus * v)/N_Acaus * dv
        adstf_4_new = scale * time_shift_Acaus * win_Acaus * dv / N_Acaus \
            - scale * time_shift_Acaus * win_Acaus * v / N_Acaus ** 2 * dN_Acaus \
            + scale * ((win_Acaus * v)/N_Acaus) ** 2 * dv

        # assemble complete adjoint source time function
        adstf = adstf_1_new + adstf_2_new + adstf_3_new + adstf_4_new

    else:
        raise ValueError("Order has to be either 1 or 2!")

    # nt = np.size(u)
    # # time = np.linspace(-21600, 12000, nt)
    # time = np.linspace(-25, 25, nt)
    # plt.figure(figsize=(12, 8))
    # plt.plot(time, win * u, "b")
    # plt.plot(time, win * u_0, "k")
    # plt.plot(time, adstf, 'r')
    # # plt.plot(time, win, "r")
    # # plt.xlim(-12000, 12000)
    # plt.xlim(-25, 25)
    # plt.show()

    # return misfit and adjoint source time function
    return misfit, adstf, time_shift_caus, time_shift_Acaus
