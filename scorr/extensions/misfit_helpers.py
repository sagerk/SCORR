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
from matplotlib import pyplot as plt
from scipy.signal import hilbert, tukey


def compute_time_shift(u, u_0, win, dt):
    # estimate time shifts
    cc = np.correlate(win * u_0, win * u, mode="full")
    time_shift = (cc.argmax() - len(u_0) + 1) * dt
    print("Time shift: ", time_shift)

    return time_shift


def compute_max_cc(u, u_0, win):
    # estimate time shifts
    cc = np.correlate(win * u_0, win * u, mode="full")

    # compute correlation coefficient
    max_cc_value = cc.max() / np.sqrt(((win * u) ** 2).sum() * ((win * u_0) ** 2).sum())
    print("Max cc value: ", max_cc_value)

    return max_cc_value


def compute_time_shift_and_max_cc(u, u_0, win, dt):
    # compute correlation function
    cc = np.correlate(win * u_0, win * u, mode="full")

    # estimate time shift
    time_shift = (cc.argmax() - len(u_0) + 1) * dt
    print("Time shift: ", time_shift)

    # compute correlation coefficient
    max_cc_value = cc.max() / np.sqrt(((win * u) ** 2).sum() * ((win * u_0) ** 2).sum())
    print("Max cc value: ", max_cc_value)

    return time_shift, max_cc_value


def compute_envelope(u):
    return np.sqrt(u ** 2 + np.imag(hilbert(u)) ** 2)


def compute_envelope_squared(u):
    return compute_envelope(u) ** 2


def evaluate_window(time, t_min, t_max, type="cos"):
    win = np.ones_like(time)

    if type == "cos":
        win *= (time > t_min) * (time < t_max)
        width = abs(t_max - t_min) * 0.2
        win += (0.5 + 0.5 * np.cos(np.pi * (t_max - time) / (width))) * \
            (time >= t_max) * (time < t_max + width)
        win += (0.5 + 0.5 * np.cos(np.pi * (t_min - time) / (width))) * \
            (time > t_min - width) * (time <= t_min)
    elif type == "hann":
        win *= (time >= t_min) * (time <= t_max)
        indices = np.where(win == 1)
        index_start = indices[0][0]
        index_last = indices[0][-1]
        win[index_start:index_last + 1] = np.hanning(len(indices[0]))
    elif type == "tukey":
        win *= (time >= t_min) * (time <= t_max)
        indices = np.where(win == 1)
        index_start = indices[0][0]
        index_last = indices[0][-1]
        win[index_start:index_last + 1] = tukey(len(indices[0]))
    elif type == "box":
        win *= (time >= t_min) * (time <= t_max)
    else:
        raise NotImplementedError(f"Window type {type} is not implemented!\n"
                                  f"Available options: [box, cos, hann]")

    return win


def pick_window(config_measurement, time, u, u_0, distance_in_m):
    t_min = time[0]
    t_max = time[-1]

    if config_measurement["type"] == "waveform_differences" or config_measurement["type"] == "cc_time_shift":
        if config_measurement["pick_window"]:
            if config_measurement["pick_manual"]:
                fig = plt.figure()
                plt.plot(time, u)
                plt.plot(time, u_0)
                points = [(-1.0, -1.0), (1.0, 1.0)]
                while points[0][0] * points[1][0] < 0.0:
                    points = plt.ginput(2, show_clicks=True)
                plt.close(fig)

                if points[0][1] < -0.5 and points[1][1] < -0.5:
                    message = "Manual stop during picking!"
                    return None, None, message

                t_min = min(points[0][0], points[1][0])
                t_max = max(points[0][0], points[1][0])
            else:
                # t_center = distance_in_m / \
                #     config_measurement["surface_wave_velocity_in_mps"]
                # t_min = t_center - \
                #     config_measurement["window_halfwidth_in_sec"]
                # t_max = t_center + \
                #     config_measurement["window_halfwidth_in_sec"]

                # t_min = distance_in_m / 5.0e3 + 0.0 - 0.2
                # t_max = distance_in_m / 5.0e3 + 0.0 + 0.2
                # t_min = distance_in_m / 6.0e3 - 0.4 - 0.5
                # t_max = distance_in_m / 6.0e3 - 0.4 + 0.5
                if distance_in_m < 20e3:
                    print("3")
                    t_min = 1.8
                    t_max = 4.2
                    print(t_min, t_max)
                else:
                    print("4")
                    t_min = 2.7
                    t_max = 5.2
                    print(t_min, t_max)

    else:
        t_center = distance_in_m / \
            config_measurement["surface_wave_velocity_in_mps"]
        t_min = t_center - config_measurement["window_halfwidth_in_sec"]
        t_max = t_center + config_measurement["window_halfwidth_in_sec"]

        # get rid of overlapping windows
        if t_min < 0:
            message = f"Windows are overlapping!"
            return None, None, message

        # check if t_max is in appropriate range
        if t_max > time[-1]:
            t_max = time[-1]
            if t_min >= t_max:
                message = f"t_min larger than t_max!"
                return None, None, message

        assert t_max > 0, f"Error in window selection for asymmetry measurement: t_max has to be > 0."

    return t_min, t_max, ""


def estimate_SNR(config_measurement, u_0, starttime, dt, distance_in_m, branch):
    nt = np.size(u_0)
    time = np.linspace(starttime, starttime + (nt - 1) * dt, nt)

    if branch == "causal":
        sign = 1.0
    elif branch == "acausal":
        sign = -1.0
    else:
        raise ValueError("Options are [causal, acausal]!")

    # get times for signal window
    t_center_signal = sign * distance_in_m / \
        config_measurement["surface_wave_velocity_in_mps"]
    t_min_signal = t_center_signal - \
        config_measurement["window_halfwidth_in_sec"]
    t_max_signal = t_center_signal + \
        config_measurement["window_halfwidth_in_sec"]

    # check if t_max_signal is in appropriate range
    if t_max_signal > time[-1]:
        t_max_signal = time[-1]
        if t_min_signal >= t_max_signal:
            raise ValueError(
                "Cannot estimate SNR! t_min_signal larger than t_max_signal!")

    if t_min_signal < time[0]:
        t_min_signal = time[0]
        if t_min_signal >= t_max_signal:
            raise ValueError(
                "Cannot estimate SNR! t_min_signal larger than t_max_signal!")

    # estimate signal energy
    win_signal = evaluate_window(
        time=time, t_min=t_min_signal, t_max=t_max_signal, type="cos")
    normalizer = (t_max_signal - t_min_signal) / dt + 1
    signal_energy = np.sum((win_signal * u_0) ** 2) / normalizer

    # get times for noise window
    t_center_noise = t_center_signal + sign * 4 * \
        config_measurement["window_halfwidth_in_sec"]
    t_min_noise = t_center_noise - \
        config_measurement["window_halfwidth_in_sec"]
    t_max_noise = t_center_noise + \
        config_measurement["window_halfwidth_in_sec"]

    # check if t_max_noise is in appropriate range
    if t_max_noise > time[-1]:
        t_max_noise = time[-1]
        if t_min_noise >= t_max_noise:
            # raise ValueError(
            #     "Cannot estimate SNR! t_min_signal larger than t_max_signal!")
            return None

    if t_min_noise < time[0]:
        t_min_noise = time[0]
        if t_min_noise >= t_max_signal:
            raise ValueError(
                "Cannot estimate SNR! t_min_signal larger than t_max_signal!")

    # estimate noise energy
    win_noise = evaluate_window(
        time=time, t_min=t_min_noise, t_max=t_max_noise, type="cos")
    normalizer = (t_max_noise - t_min_noise) / dt + 1
    noise_energy = np.sum((win_noise * u_0) ** 2) / normalizer

    # nt = np.size(u)
    # time = np.linspace(-21600, 12000, nt)
    # plt.figure(figsize=(12, 8))
    # plt.plot(time, u_0, "k")
    # plt.plot(time, win_signal, "r")
    # plt.plot(time, win_noise, "b")
    # plt.xlim(-12000, 12000)
    # plt.show()

    return signal_energy / noise_energy + np.finfo(np.float32).eps


def estimate_CC_signal(config_measurement, u, u_0, starttime, dt, distance_in_m, branch):
    nt = np.size(u_0)
    time = np.linspace(starttime, starttime + (nt - 1) * dt, nt)

    if branch == "causal":
        sign = 1.0
    elif branch == "acausal":
        sign = -1.0
    else:
        raise ValueError("Options are [causal, acausal]!")

    # get times for signal window
    t_center_signal = sign * distance_in_m / \
        config_measurement["surface_wave_velocity_in_mps"]
    t_min_signal = t_center_signal - \
        config_measurement["window_halfwidth_in_sec"]
    t_max_signal = t_center_signal + \
        config_measurement["window_halfwidth_in_sec"]

    # check if t_max_signal is in appropriate range
    if t_max_signal > time[-1]:
        t_max_signal = time[-1]
        if t_min_signal >= t_max_signal:
            raise ValueError(
                "Cannot estimate SNR! t_min_signal larger than t_max_signal!")

    if t_min_signal < time[0]:
        t_min_signal = time[0]
        if t_min_signal >= t_max_signal:
            # raise ValueError(
            #     "Cannot estimate SNR! t_min_signal larger than t_max_signal!")
            return None

    # get windows
    win_signal = evaluate_window(
        time=time, t_min=t_min_signal, t_max=t_max_signal, type="cos")

    # compute cc
    return compute_max_cc(u, u_0, win_signal)


def quality_control(config_measurement, u, u_0, starttime, dt, distance_in_m, fh=TemporaryFile(mode="w"), only_snr=False):
    ########################
    ###   estimate SNR   ###
    ########################
    snr_caus = estimate_SNR(config_measurement=config_measurement, u_0=u_0, starttime=starttime, dt=dt,
                            distance_in_m=distance_in_m, branch="causal")
    snr_Acaus = estimate_SNR(config_measurement=config_measurement, u_0=u_0, starttime=starttime, dt=dt,
                             distance_in_m=distance_in_m, branch="acausal")
    # due to shorter simulations, an estimate for the causal branch might not be possible
    if snr_caus is None:
        snr_caus = snr_Acaus

    ########################
    ###   estimate CC    ###
    ########################
    if not only_snr:
        max_cc_caus = estimate_CC_signal(config_measurement=config_measurement, u=u, u_0=u_0, starttime=starttime, dt=dt,
                                         distance_in_m=distance_in_m, branch="causal")
        max_cc_Acaus = estimate_CC_signal(config_measurement=config_measurement, u=u, u_0=u_0, starttime=starttime, dt=dt,
                                          distance_in_m=distance_in_m, branch="acausal")
    else:
        max_cc_caus = 1.0
        max_cc_Acaus = 1.0

    ########################
    ###      summary     ###
    ########################
    # write to log file
    fh.write(str(snr_caus) + " ")
    fh.write(str(snr_Acaus) + " ")
    fh.write(str(max_cc_caus) + " ")
    fh.write(str(max_cc_Acaus) + " ")

    message = f"SNR_c {snr_caus}, CC_c {max_cc_caus}; SNR_a {snr_Acaus}, CC_a {max_cc_Acaus}"
    return snr_caus, snr_Acaus, max_cc_caus, max_cc_Acaus, message
