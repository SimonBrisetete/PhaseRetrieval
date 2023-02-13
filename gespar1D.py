import numpy as np
from scipy.fft import fft, ifft


def generate_fft_problem_function_oversampling(n, k):
    offset = 3
    maxVal = 1
    n = 2*n
    F = fft(np.eye(n))    # F is the DFT matrix
    n = n//2
    locs = np.random.permutation(n)
    x = np.zeros((n, 1))
    x[locs[0:k]] = (np.random.rand(k, 1) * maxVal + offset)*(-1)**(np.floor(np.random.rand(k, 1)*2)+1)
    c = np.abs(fft(x, 2*n))**2
    x_real = x
    return F, x_real, c


def gradF(w, c, x):
    # Returns the gradient of f
    z = fft(x)
    out = len(x)*4*ifft(w*z*(np.abs(z)**2 - c))
    return out


def objectiveFun(w, c, x):
    # Returns the squared 2 norm of the consistency error
    out = sum(w*((np.abs(fft(x))**2 - c)**2))
    return out


def bestMatch(x1, x2):
    # bestMatch finds the permutation of x1 that matches x2 the best in
    # terms of the following ambiguities: circular shift, sign, flipping
    [n, m] = np.shape(x1)
    minErr = np.inf
    for kk in range(1, n):
        for signInd in range(1, 2):
            for flip in range(0, 1):
                if flip:
                    x1shift = np.flipud(np.circshift(x1, kk)*(-1)^signInd)
                else:
                    x1shift = np.circshift(x1, kk)*(-1)^signInd
                err = np.linalg.norm(x2-x1shift)
                if err < minErr:
                    xBest = x1shift
                    minErr = err
    return xBest


def DGN(w, S, c, n, x0, iterations, F):
    # Damped Gauss Newton
    x = np.zeros(2*n, 1)    # 2*n equations
    x[S] = x0
    s = 0.5
    for i in range(1, iterations):
        s = np.min([2*s, 1])
        y = fft(x)
        b = np.sqrt(w)*(np.abs(y)**2 + c)
        B = (np.dot(np.real(y)*np.sqrt(w), np.real(F[:, S])) +
             np.dot(np.imag(y)*np.sqrt(w), np.imag(F[:, S])))
        xold = x
        fold = objectiveFun(w, c, xold)
        x = np.zeros(2*n, 1)
        x[S] = 2*np.linalg.lstsq(B, b)
        if np.linalg.matrix_rank(B) < len(S):
            pass

        xnew = x
        while objectiveFun(w, c, xold + s*(xnew - xold)) > fold:     # && (s > 1e-5):
            s = 0.5*s

        x = xold + s*(xnew - xold)
        if np.norm(x-xold) < 1e-4:
            return x
    return x


def GESPAR_1DF(c, n, k, iterations, verbose, F, ac, noisy, replacementsSoFar,
               totalReplacements, thresh, randomWeights):
    """
    Performs the 2 - opt method.
    # Parameters:
    # -------------------------------------
    # c = Fourier magnitude measurements.
    # n = length of input signal
    # iterations = number of DGN inner iterations
    # verbose = write stuff(1) or not (0)
    # F = DFT matrix
    # ac = autocorrelation sequence
    # noisy = use ac info for support(0) or not (1)
    # replacementsSoFar = How many index replacements were done so far in previous initial points
    # totalReplacements = Max.number of allowed replacements
    # thresh = stopping criteria for objective function
    # randomWeights = use random weights for different measurements(1)
    # or all measurements have equal weights (0)
    # -----------------------------------------
    :param c:
    :type c:
    :param n:
    :type n:
    :param k:
    :type k:
    :param iterations:
    :type iterations:
    :param verbose:
    :type verbose:
    :param F:
    :type F:
    :param ac:
    :type ac:
    :param noisy:
    :type noisy:
    :param replacementsSoFar:
    :type replacementsSoFar:
    :param totalReplacements:
    :type totalReplacements:
    :param thresh:
    :type thresh:
    :param randomWeights:
    :type randomWeights:
    :return:
    :rtype:
    """
    # Initialize support
    if noisy:
        noACSupp = 1    # Do not use Autocorrelation support info
    else:
        noACSupp = 0    # Use Autocorrelation support info only in the noiseless case

    ac[abs(ac) < 1e-8] = 0
    acOffSupp = np.where(ac == 0)
    maxAC = np.max(np.where(ac))
    acOffSuppMax = maxAC + 1 - np.where(ac == 0)
    acOffSuppMax[acOffSuppMax < 1] = 0
    acOffSuppMax[acOffSuppMax > n] = 0
    acOffSuppMax = np.nonzero(acOffSuppMax)
    acOffSupp = np.unique([acOffSupp, acOffSuppMax])
    if noACSupp:
        acOffSupp = []

    p = np.random.permutation(n - 1) + 1
    p = np.setdiff1d(p, acOffSupp)
    p = p[np.random.permutation(len(p))]
    replacements = 0    # counts how many index replacements are done
    if randomWeights:
        w = 1 + (np.random.rand((2 * n, 1)) < 0.5)   # Use random weights
    else:
        w = np.ones((2 * n, 1))

    p[p == 1] = 0
    p[p == maxAC] = 0
    p = np.nonzero(p).T
    if len(p) + 2 < k:
        k = max(2, len(p) + 2)
    supp = [1, maxAC, p[1:k - 2]]
    if noACSupp:
        supp = [1, p[1:k - 1]]

    x_k = DGN(w, supp, c, n, np.random.randn(k, 1), iterations, F)  # Damped-Gauss-Newton Initial guess
    replacements = replacements + 1
    fMin = objectiveFun(w, c, x_k)
    it = 0
    while 1:
        it = it + 1
        # Main iteration
        [junk, idx] = np.sort(abs(x_k[supp]))
        supp = supp[idx]    # Sorting supp from min(abs(x_k)) to max
        fGrad = gradF(w, c, x_k)
        offSupp = np.setdiff1d[1:n, supp]
        offSupp = np.setdiff1d[offSupp, acOffSupp]
        junk, idx = np.sort[- np.abs(fGrad[offSupp])]
        offSupp = offSupp[idx]
        pSupp = np.arange(1, len(supp))
        pOffSupp = np.arange(1, len(offSupp))
        improved = 0
        for iInd in range(1, np.min(1, len(supp))):
            i = supp[pSupp[iInd]]   # Index to remove
            if noACSupp:
                if i == 1:
                    continue    # Never remove 1st element
            else:
                if i == 1 or i == maxAC:
                    continue    # Never remove 1st and last element

            for jInd in range(1, (min(1, len(offSupp))) ):
                j = offSupp[pOffSupp[jInd]] # Index to insert
                # Check replacement
                suppTemp = supp
                suppTemp[suppTemp == i] = j
                # Solve GN with given support
                xTemp = DGN(w, suppTemp, c, n, x_k(suppTemp), iterations, F)
                fTemp = objectiveFun(w, c, xTemp)
                replacements = replacements + 1
                if fTemp < fMin:
                    if verbose:
                        print('replacement: %d  Replacing %d with %d   f= %3.3f\n' %(replacements - 1, i, j, fTemp))

                    x_k = xTemp
                    x_n = x_k
                    supp = suppTemp
                    improved = 1
                    fMin = fTemp
                    if fTemp < thresh:
                        if verbose:
                            print('******************************************Success!,'
                                  f' iteration={replacements:%d}\n')
                        return
                    break
                else:
                    x_n = xTemp
                if replacementsSoFar + replacements + 1 > totalReplacements:
                    return
            if improved:
                break

        if not improved:
            if verbose:
                print('no possible improvement - trying new initial guess\n')
            x_n = x_k
            return x_n, fMin, replacements

    x_n = x_k
    return x_n, fMin, replacements