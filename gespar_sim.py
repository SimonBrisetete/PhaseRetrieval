import numpy as np
from numpy.fft import ifft
import matplotlib.pyplot as plt
import time
from gespar1D import generate_fft_problem_function_oversampling, GESPAR_1DF, bestMatch


# This scripts runs a 1D Fourier GESPAR simulation:
# It generates random vectors and tries to recover them from their (possibly noisy) 1D Fourier
# magnitude measurements, with x2 oversampling.

# Important parameters:
# n= number of measurements.  The length of the signal will be n/2
# kVec = range of simulated sparsity levels
# maxIt = Simulation iterations, for each k
# totIterations = Number of replacements allowed per signal
# snr = noise added to measurements.  1000 and above is treated as noiseless.
# Below 1000 no autocorrelation-derived support information will be used

def add_gaussian_noise(x, target_snr_db):
    """
    Adding noise using target SNR.
    :param x: signal
    :type x:
    :param target_snr_db: target SNR in dBW
    :type target_snr_db:
    :return:
    :rtype:
    """
    # Calculate signal power and convert to dB
    x_watts = x ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to SNR_dB = P_(signal, dB) - P_(noise, dB) then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate a sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    # Noise up the original signal
    x_noisy = x + np.expand_dims(noise, axis=1)
    return x_noisy


# Seeding random generator
n = 0
np.random.seed(n)
# print(np.random.uniform(0, 1, size=n))
# s = RandStream('mcg16807','Seed',0)
# RandStream.setDefaultStream(s)

kVec = np.arange(1, 8, 1)     # Range of sparsity values in simulations
n = 128   # Number of measurements.  The length of the signal will be n/2
kInd = 0
iterations = 100    # GN inner iterations
draw = 1
verbose = 0
maxIt = 10  # Simulation iterations, for each k
errMat = np.zeros((len(kVec), maxIt))
itMat = errMat
trialsMat = errMat
timeMat = errMat
totIterations = 6400    # Number of replacements allowed per signal
failedCases = []
failInd = 0
recData = np.zeros((len(kVec), maxIt, n))
trueData = np.zeros((len(kVec), maxIt, n))
snr = 1001      # snr>1000 is treated as noiseless
# Run iterations
for k in kVec:
    kInd = kInd+1
    for it in range(1, maxIt):
        tic = time.time()
        success = 0
        # Generating random data + measurements
        F, x_real, c = generate_fft_problem_function_oversampling(n//2, k)
        # Calculating autocorrelation - Adding noise and symmetrizing c due to noise to get
        # a real autocorrelation function
        cn = add_gaussian_noise(c, snr)
        cn = 0.5*(cn + np.array([cn[0], cn[-1:-1:2].T]))
        ac = ifft[cn]
        ac = ac[0: n//2]

        locs = np.where(x_real)
        supp = locs.T - np.min(locs).T + 1
        x_real = [[x_real], [np.zeros((n//2, 1))]]
        trueData[kInd, it, :] = x_real
        itSoFar = 0
        trial = 0
        fValueMin = np.inf
        while itSoFar < totIterations:
            if verbose:
                print('total replacements so far = %d \n' % itSoFar)

            # using GESPAR to recover x from cn (noisy measurements)
            noisy = 1
            fThresh = 1e-3
            randomWeights = 1
            [x_n, fValue, its] = GESPAR_1DF(cn, n/2, k, iterations, verbose, F, ac, noisy, itSoFar,
                                            totIterations, fThresh, randomWeights)
            trial = trial + 1
            itSoFar = itSoFar + its
            if fValue < 2*np.norm(c-cn)**2 or fValue < 1e-4:   # Breaking condition for a successful recovery
                toc = time.time() - tic
                t = np.round(toc*100)/100
                success = 1
                fValueMin = fValue
                x_n_best = x_n
                print(f'{it:%d}. succeeded! k = {k:%d}   total evaluations {itSoFar:%d} in {trial:%d}'
                      f' initial points took {t:%2.2f} secs\n')
                break

            if fValue < fValueMin:
                fValueMin = fValue
                x_n_best = x_n

        if not success:
            toc = time.time() - tic
            t = np.round(toc*100)/100
            print(f'{it:%d}. k = {k:%d}  {itSoFar:%d} Evaluations that took {t:%2.2f} secs'
                  f' were not enough\n')
            failInd = failInd + 1
            failedCases.append([failInd, x_real])

        x_nB = bestMatch(x_n_best, x_real)
        errMat[kInd, it] = np.norm(x_nB - x_real) / np.norm(x_real)
        itMat[kInd, it] = itSoFar
        recData[kInd, it, :] = x_nB
        trialsMat[kInd, it] = trial
        t = time.time() - tic
        timeMat[kInd, it] = t

        if draw:
            plt.figure(13)
            plt.plot(np.arange(1, n), x_nB)
            plt.plot(np.arange(1, n), x_real, '*')

    l2erVec = np.mean(errMat, 2)
    erMat = errMat
    erMat[erMat < 1e-3] = 0
    erMat[erMat > 1e-3] = 1
    erVec = np.sum(erMat, 2)
    timeVec = np.mean(timeMat, 2)
    # Saving Data
    print(f'\n SAVING... \n', fValueMin)
    fileName = 'GESPAR_results_snr_' + str(snr)
    #save(fileName,'errMat','erVec','kVec','snr','n','totIterations','recData','trueData',
    #      'l2erVec','maxIt','timeMat','timeVec','itMat','trialsMat');


# Plotting
if draw:
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(kVec,(1-erVec/maxIt),'-*')
    plt.title(f'Recovery probability, N={n}')
    plt.xlabel('k')
    plt.ylim([0, 1.1])
    plt.ylabel('Recovery probability')

    itVec = np.mean(itMat,2)
    plt.subplot(3,1,2)
    plt.plot(kVec,itVec,'-*')
    plt.title('mean # of iterations vs k')
    plt.xlabel('k')
    plt.ylabel('Iterations')
    trialsVec = np.mean(trialsMat, 2)
    plt.subplot(3, 1, 3)
    plt.plot(kVec, trialsVec, '-*')
    plt.title('mean # of trials vs k')
    plt.xlabel('k')
    plt.ylabel('Trials')

    plt.figure()
    plt.plot(kVec,l2erVec)
    plt.xlabel('k')
    plt.ylabel('l2 error')
    plt.title('mean l2 rec. error')
