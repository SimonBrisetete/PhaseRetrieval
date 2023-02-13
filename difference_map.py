import numpy as np
from numpy.fft import fftn, ifftn


# PROJECTION OPERATORS
def P_S(x, S_in):
    """
    Projection of x onto support S_in
    :param x:
    :type x:
    :param S_in:
    :type S_in:
    :return:
    :rtype:
    """
    x_new = x*S_in['supp']
    return x_new


def P_M(x, M_in):
    """
    Projection of x onto Magnitude.
    """
    X = fftn(x)
    X_new = X/np.abs(X)*M_in['M_data']
    # X_new = M_in['M_data'] * np.exp(1j * np.angle(X))
    x_new = ifftn(X_new)
    return x_new


# THE DIFFERENCE MAP
# "Phase Retrieval by Iterative Projections" (2003) J Opt Soc Ann A, VOl 20, pp40-55
# x_{n+1} = x_n + beta*(P_S*R_M*x_n - P_M*R_S*x_n)
# R_M*x = (1+gamma_M)*P_M*x - gamma_M*x
# R_S*x = (1+gamma_S)*P_S*x - gamma_S*x


def R_M(x, gamma_M, M_in):
    return (1+gamma_M)*P_M(x, M_in) - gamma_M*x


def R_S(x, gamma_S, S_in):
    return (1+gamma_S)*P_S(x, S_in) - gamma_S*x


def DifferenceMap(x, beta, gamma_S, gamma_M, M_in, S_in):
    x_PMRS = P_M(R_S(x, gamma_S, S_in), M_in)
    x_PSRM = P_S(R_M(x, gamma_M, M_in), S_in)   # relaxed projection of x on the support constraint

    x_new = x + beta*(x_PMRS - x_PSRM)
    return x_new, x_PSRM


def convolution_filter(x, kernel):
    return ifftn(fftn(x)*kernel)


def initialize_error_factor():
    # CM parameters
    beta = 0.99
    gamma_M = -1/beta   # Random projections and the optimization of an algorithm for phase
                        # retrieval, J. Phys. A-Math Gen, Vol 36, pp 2995-3007
    gamma_S = 1/beta
    return beta, gamma_M, gamma_S