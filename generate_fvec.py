import numpy as np


def f_vec(Fs, nbpoints, positive_f_only=False):
    """
    F_VEC creates the frequency vector with the sampling frequency Fs and nbpoints samples.
    :param Fs: Sampling frequency
    :type Fs:
    :param nbpoints: number of points
    :type nbpoints:
    :param positive_f_only: positive_f_only whether to calculate only the positive frequencies
    :type positive_f_only:
    :return f: sampled frequency vector
    :rtype:
    :return df: frequency step
    :rtype:
    :return idx_0: index of the frequency f=0. If positive_f_only == true,
                   idx_0 == 1 else idx_0 points to the middle of the vector
    :rtype:
    """
    df = Fs / nbpoints

    if nbpoints % 2 == 0:      # even number of points
        if positive_f_only:
            f = np.arange(0, Fs / 2, df)
            idx_0 = 1
        else:
            f = np.arange(-Fs / 2, Fs / 2 - df, df)
            idx_0 = nbpoints // 2 + 1
            f[idx_0] = 0
    else:    # uneven number of points
        if positive_f_only:
            f = np.arange(0, Fs / 2 - df / 2, df)
            idx_0 = 1
        else:
            f = np.arange(-Fs / 2 + df / 2, Fs / 2 - df / 2, df)
            idx_0 = int(np.ceil(nbpoints / 2))
            f[idx_0] = 0
    return f, df, idx_0