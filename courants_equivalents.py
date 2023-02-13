import numpy as np
import scipy
from scipy.optimize import minimize


def initial_guess(Ns, mode='zeros', G=None, b=None, x0=None, LStol=None, LSmaxit=None):
    # mode = 'zeros' (default) - array of zero values;
    # mode = 'random' (default) - array of random values;
    # mode = 'approximate' (default) - try to find an approximate solution;
    # Ns - number of equivalent sources
    # G - propagator matrix with size (2*Nobs, 2*Ns)
    # b - measurement vector with size (2*Nobs, 1)
    # x0 - initial guess if any (2*Ns, 1)
    # LStol - tolerance for minimization process
    # LSmaxit - maximum number of iterations before exiting minimization process

    if mode == "zeros":
        guess = np.zeros((2*Ns,), dtype=complex)
    elif mode == "random":
        guess = np.random.random((2*Ns,), dtype=complex)
    elif mode == "approximate":
        if G is None or b is None:
            return
        if x0 is None:
            guess0 = np.random.random((2*Ns,), dtype=complex)
        else:
            guess0 = x0
        # calcul des J2D par G2D.J2D=EmXY
        # Try to find approximate solution from x0=JeqIni
        # 1 - Compute a residual vector r0 = b - A@x0.
        # 2 - Use LSQR to solve the system A@dx = r0.
        # 3 - Add the correction dx to obtain a final solution x = x0 + dx.
        print('Try to find an initial condition: \t')
        r0 = b[:, 0] - G.dot(guess0)
        # If x0 is “good”, norm(r0) will be smaller than norm(b).
        if np.linalg.norm(r0) < np.linalg.norm(b):
            print("r0 is a 'good' initial solution.")
            btol0 = LStol * np.linalg.norm(b) / np.linalg.norm(r0)
        else:
            print("r0 is not a 'good' initial solution.")
            btol0 = LStol
        # Suppose LSQR takes k1 iterations to solve A@x = b and k2 iterations to solve A@dx = r0.
        # If the same stopping tolerances atol and btol are used for each system,
        # k1 and k2 will be similar, but the final solution x0 + dx should be more accurate.
        # The only way to reduce the total work is to use a larger stopping tolerance for the second system.
        # If some value btol is suitable for A@x = b, the larger value btol*norm(b)/norm(r0) should be
        # suitable for A@dx = r0.

        dx0 = scipy.sparse.linalg.lsqr(A=G, b=r0, atol=btol0, btol=btol0, iter_lim=LSmaxit)
        guess = np.add(guess0, dx0[0])
    return guess


def apply_polar_rules(Nobs, Em, polar='V'):
    # Weighted Field in function of the polarization
    # horizontal - Ex = Em
    # vertical - Ey = Em
    # circular - Ex = Ey = Em/2 ???
    # elliptical - Ex = alpha*Em; Ey = (1-alpha)*Em with 0 <= alpha <= 1 ???
    # Nobs - number of observations
    # Em - observations array (Nobs, 1)

    EmXY = np.zeros((2 * Nobs, 1), dtype=complex)
    if polar == 'H':
        alpha = 1
    elif polar == 'V':
        alpha = 0
    elif polar == 'Circular':
        alpha = 0.5
    elif polar == 'Elliptical':
        alpha = 0.2  # à définir
    EmXY[::2] = alpha * Em  # Ex definition
    EmXY[1::2] = (1 - alpha) * Em  # Ey definition
    return EmXY


def real_to_complex(z):      # real vector of length 2n -> complex of length n
    return z[:len(z)//2] + 1j * z[len(z)//2:]


def complex_to_real(z):      # complex vector of length n -> real of length 2n
    return np.concatenate((np.real(z), np.imag(z)))


def model(A, x):
    """
    Return square amplitude  |<A,x>|^2 (order=2)
    :param A:
    :type A:
    :param x:
    :type x:
    :return:
    :rtype:
    """
    #return np.abs(A.dot(x.T))
    return A.dot(x.T)[:, np.newaxis]


def residuals(A, b, x):
    """
    Calculate residuals r = b - A@x = |y_m|^2 - |<A,x>|^2
    :param params:
    :type params:
    :param x:
    :type x:
    :return:
    :rtype:
    """
    # 1 - Compute a residual vector r = b - A@x.
    # 2 - Use LSQR to solve the system A@x = r.

    diff = b - model(A, x)
    return diff.real ** 2 + diff.imag ** 2


def residuals1(x, A, b):
    diff = b - model(A, x)
    return diff.real ** 2 + diff.imag ** 2


def cost_cpl(x, A, b):
    #return (np.dot(A, x.view(np.complex128)) - b).view(np.double)
    if len(b.shape) == 2:
        b = b[:, 0]
    return (A@x - b.view(np.complex128)).view(np.double)


def cost_cpl1(x, A, b):
    #return (np.dot(A, x.view(np.complex128)) - b).view(np.double)
    if len(b.shape) == 2:
        b = b[:, 0]
    return (A@x - b).view(np.double)


def phase_retrieval(A, b, p_guess, *resfunc):
    """
    Solve |y_m|^2 = |<A,x>|^2
    :param resfunc:
    :type resfunc:
    :return:
    :rtype:
    """
    # The minimization methods of SciPy work with real arguments only.
    # But minimization on the complex space C^n amounts to minimization on R^2n,
    # the algebra of complex numbers never enters the consideration.
    # Thus, adding two wrappers for conversion from Cn to R2n and back, you can optimize over complex numbers.

    params, cov = scipy.optimize.leastsq(resfunc, p_guess, args=(A, b))
    # params = [A, b]
    # sol = minimize(lambda z: residuals(params=params, x=real_to_complex(z)), x0=complex_to_real(p_guess))
    # print(real_to_complex(sol.x))
    #
    # # params, cov, infodict, mesg, ier = scipy.optimize.leastsq(resfunc, p_guess, args=(A, b), full_output=True)
    # ssr = abs(((model(A, *params) - b) ** 2).sum())

    fmt = '{:15} | {:37} | {:17} | {:6}'
    header = fmt.format('name', 'params', 'sum(residuals**2)', 'ncalls')
    print('{}\n{}'.format(header, '-' * len(header)))
    fmt = '{:15} | {:37} | {:17.2f} | {:6}'
    print(fmt.format(resfunc.__name__, params, ssr, infodict['nfev']))
