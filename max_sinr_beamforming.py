import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def compute_variances(SIR, SINR, source_loc, interference_loc, mic_loc, sigma_s=1):
    """
    This function will compute the powers (i.e. variance) of an interference
    source and the microphone self noise power given values for the
    SIR (signal-to-interference ratio) and SINR (signal-to-interference-and-noise ratio)
    and optionaly a target source power.
    """

    if SINR > SIR:
        raise ValueError("SINR must be less or equal to SIR.")

    d_s = np.linalg.norm(source_loc - mic_loc)
    d_i = np.linalg.norm(interference_loc - mic_loc)

    sigma_i = sigma_s * d_i / d_s * 10 ** (-SIR / 20)
    sigma_n = sigma_s / d_s * np.sqrt(10 ** (-SINR / 10) - 10 ** (-SIR / 10))

    return sigma_i, sigma_n


def compute_gain(w, X, ref, clip_up=None, clip_down=None, sqm=False):
    """

    Parameters
    ----------
    w: array_like (n_bins, n_channels)
    X: array_like (n_frames, n_bins, n_channels)
        The STFT data
    ref: array_like (n_frames, n_bins)
        The reference signal
    n_lambda: int, optional
        The number of lagrange multiplier value to try in the approximation (default: 20)
    clip_up: float, optional
        Limits the maximum value of the gain (default no limit)
    clip_down: float, optional
        Limits the minimum value of the gain (default no limit)
    """

    y = np.sum(X * np.conj(w), axis=2)

    num = np.sum(np.conj(ref) * y, axis=0)
    denom = np.sum(np.abs(y) ** 2, axis=0)

    c = np.ones(num.shape, dtype=complex)
    I = denom > 0.0
    c[I] = num[I] / denom[I]

    if clip_up is not None:
        I = np.logical_and(np.abs(c) > clip_up, np.abs(c) > 0)
        c[I] *= clip_up / np.abs(c[I])

    if clip_down is not None:
        I = np.logical_and(np.abs(c) < clip_down, np.abs(c) > 0)
        c[I] *= clip_down / np.abs(c[I])

    return c
