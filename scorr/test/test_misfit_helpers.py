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
from scipy.signal import chirp
from matplotlib import pyplot as plt

from scorr.extensions import misfit_helpers


def check_envelope():
    nt = 1501
    dt = 0.01
    time = np.linspace(0, (nt - 1) * dt, nt)

    # construct signal
    signal = chirp(time, 1 / (100 * dt), time[-1], 1 / (20 * dt))
    signal *= (1.0 + 0.5 * np.sin(2.0 * np.pi * 0.5 * time))

    # compute envelope
    envelope = misfit_helpers.compute_envelope(signal)
    envelope_squared = misfit_helpers.compute_envelope_squared(signal)

    # plot signal and envelope
    plt.figure(figsize=(12, 8))
    plt.plot(time, signal, "r", label="signal")
    plt.plot(time, envelope, "b", label="envelope")
    plt.plot(time, envelope_squared, "g", label="envelope squared")
    plt.legend()
    plt.xlim(0, time[-1])
    plt.show()


def test_pick_window():
    config_measurement = {}
    config_measurement["type"] = "waveform_differences"
    config_measurement["pick_window"] = False
    config_measurement["window_halfwidth_in_sec"] = 30.0
    config_measurement["pick_manual"] = False
    config_measurement["surface_wave_velocity_in_mps"] = 4000

    # generate random seismograms
    nt = 1501
    dt = 0.1
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)
    u_obs = 2 * (np.random.rand(nt) - 0.5)
    u_ini = 2 * (np.random.rand(nt) - 0.5)

    # no window is picked
    t_min, t_max, _ = misfit_helpers.pick_window(config_measurement=config_measurement,
                                                 time=time, u=u_ini, u_0=u_obs, distance_in_m=4000.0)
    assert t_min == -(nt - 1) / 2 * dt
    assert t_max == (nt - 1) / 2 * dt

    # asymmetry measurement
    config_measurement["type"] = "log_amplitude_ratio"
    t_min, t_max, _ = misfit_helpers.pick_window(config_measurement=config_measurement,
                                                 time=time, u=u_ini, u_0=u_obs, distance_in_m=160000.0)
    assert t_min == 10.0
    assert t_max == 70.0

    # asymmetry measurement - t_max would be larger than time[-1]
    t_min, t_max, _ = misfit_helpers.pick_window(config_measurement=config_measurement,
                                                 time=time, u=u_ini, u_0=u_obs, distance_in_m=200000.0)
    assert t_min == 20.0
    assert t_max == 75.0

    # asymmetry measurement - t_min would be smaller than 0.0
    t_min, t_max, _ = misfit_helpers.pick_window(config_measurement=config_measurement,
                                                 time=time, u=u_ini, u_0=u_obs, distance_in_m=4000.0)
    assert t_min is None
    assert t_max is None


def test_compute_window():
    # evaluate_window(time, t_min, t_max, type="cos"):
    config_measurement = {}
    config_measurement["type"] = "waveform_differences"
    config_measurement["pick_window"] = False
    config_measurement["window_halfwidth_in_sec"] = 30.0
    config_measurement["pick_manual"] = False
    config_measurement["surface_wave_velocity_in_mps"] = 4000

    # generate random seismograms
    nt = 1501
    dt = 0.1
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)

    # box window
    win = misfit_helpers.evaluate_window(
        time=time, t_min=time[0], t_max=time[-1], type="box")
    assert np.sum(win) == nt
    win = misfit_helpers.evaluate_window(
        time=time, t_min=-1, t_max=1, type="box")
    assert np.sum(win) == 21

    # cos window
    win = misfit_helpers.evaluate_window(
        time=time, t_min=time[0], t_max=time[-1], type="cos")
    assert np.sum(win) == nt

    # hanning window
    win = misfit_helpers.evaluate_window(
        time=time, t_min=time[200], t_max=time[-200], type="hann")
    assert np.round(win[751], 5) == 1.0
    assert win[200] == 0.0
    assert win[-200] == 0.0

    # tukey window
    win = misfit_helpers.evaluate_window(
        time=time, t_min=time[200], t_max=time[-200], type="tukey")
    assert np.round(win[751], 5) == 1.0
    assert win[200] == 0.0
    assert win[-200] == 0.0


def check_compute_window():
    """
    test is now done by visual inspection

    :return:
    """
    config_measurement = {}
    config_measurement["type"] = "waveform_differences"
    config_measurement["pick_window"] = False
    config_measurement["window_halfwidth_in_sec"] = 30.0
    config_measurement["pick_manual"] = False
    config_measurement["surface_wave_velocity_in_mps"] = 4000

    # generate random seismograms
    nt = 1501
    dt = 0.1
    time = np.linspace(-(nt - 1) / 2 * dt, (nt - 1) / 2 * dt, nt)

    # cos window
    win = misfit_helpers.evaluate_window(
        time=time, t_min=time[400], t_max=time[-400], type="cos")
    plt.plot(time, win)
    plt.title("cos window")
    plt.show()

    # hanning window
    win = misfit_helpers.evaluate_window(
        time=time, t_min=time[200], t_max=time[-200], type="hann")
    plt.plot(time, win)
    plt.title("hanning window")
    plt.show()

    # tukey window
    win = misfit_helpers.evaluate_window(
        time=time, t_min=time[200], t_max=time[-200], type="tukey")
    plt.plot(time, win)
    plt.title("tukey window")
    plt.show()


if __name__ == "__main__":
    check_envelope()
    check_compute_window()
