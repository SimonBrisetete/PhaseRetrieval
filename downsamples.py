import numpy as np
import pandas as pd


def movmean(D, N):
    """

    :param D: array
    :type D:
    :param N: window size
    :type N:
    :return:
    :rtype:
    """
    ret = np.cumsum(D, dtype=float)
    ret[N:] = ret[N:] - ret[:-N]
    return ret[N - 1:] / N


def movmeanC(D, N, mode='same'):
    """

    :param D:
    :type D:
    :param N:
    :type N:
    :param mode: 'same' -  same shape as the original data (default)
                 'valid' - values at the edges that did not see the entire kernel are discarded.
                 The output array is smaller in shape than the input array.
    :type mode:
    :return:
    :rtype:
    """
    kernel_size = N
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(D, kernel, mode=mode)
    return data_convolved


def DownSampleS(D, DwnSampleS):
    """
    DOWNSAMPLE:
    This functions samples the data down by specified factors for the space dimension by applying a moving mean.
    :param D:
    :type D:
    :param DwnSampleS:
    :type DwnSampleS:
    :return:
    :rtype:
    """
    D = pd.DataFrame(D).rolling(DwnSampleS, axis=0, min_periods=1).mean().values
    D = pd.DataFrame(D).rolling(DwnSampleS, axis=1, min_periods=1).mean().values
    D = D[::DwnSampleS, ::DwnSampleS]
    Nx, Ny = D.shape

    return D, Nx, Ny