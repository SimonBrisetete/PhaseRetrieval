"""
The first main problem we see is that we don’t have an equally sampled time space.
We can solve this problem by considering a continuous time space and adding the values related to the
closest day available.

The second problem is that Fourier analysis works well only for stationary data and we can clearly see that this
 time series is increasing during the years.
We are fixing this using a Polynomial Regression to find the best-fit polynomial function that fits the data.
Then we will remove this line and obtain the stationary time-series:
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import pywt
#from scipy import signal
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq, irfft
from scipy import stats


def filter_signal(th):
    f_s = fft_filter(th)
    return np.real(np.fft.ifft(f_s))


def fft_filter(perc):
    fft_signal = np.fft.fft(signal)
    fft_abs = np.abs(fft_signal)
    th=perc*(2*fft_abs[0:int(len(signal)/2.)]/len(new_Xph)).max()
    fft_tof=fft_signal.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/len(new_Xph)
    fft_tof[fft_tof_abs<=th]=0
    return fft_tof


def fft_filter_amp(th):
    fft = np.fft.fft(signal)
    fft_tof=fft.copy()
    fft_tof_abs=np.abs(fft_tof)
    fft_tof_abs=2*fft_tof_abs/len(new_Xph)
    fft_tof_abs[fft_tof_abs<=th]=0
    return fft_tof_abs[0:int(len(fft_tof_abs)/2.)]


# plt.rcParams['figure.figsize'] = [16, 10]
# plt.rcParams.update({'font.size': 18})
# #Create a simple signal with two frequencies
# data_step = 0.001
# t = np.arange(start=0, stop=1, step=data_step)
# f_clean = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
# f_noise = f_clean + 2.5*np.random.randn(len(t))
#
# plt.figure()
# plt.plot(t, f_noise, color='c', linewidth=1.5, label='Noisy')
# plt.plot(t, f_clean, color='k', linewidth=2, label='Clean')
# plt.legend()
#
# n = len(t)
# yf = rfft(f_noise)
# xf = rfftfreq(n, data_step)
# plt.figure()
# plt.plot(xf, np.abs(yf))
#
# # Remove the noise frequencies
# yf_abs = np.abs(yf)
# indices = yf_abs > 300   # filter out those value under 300
# yf_clean = indices * yf  # noise frequency will be set to 0
# plt.figure()
# plt.plot(xf, np.abs(yf_clean))
#
# # Inverse back to Time-Domain data
# new_f_clean = irfft(yf_clean)
# plt.figure()
# plt.plot(t, new_f_clean)
# plt.ylim(-6, 8)

# Plot Filtered data

# th_opt = th_list[np.array(corr_values).argmin()]
# opt_signal = filter_signal(th_opt)
# plt.plot(x[1000:1100],signal[1000:1100],color='navy',label='Original Signal')
# plt.plot(x[1000:1100],opt_signal[1000:1100],color='firebrick',label='Optimal signal (Th=%.3f)'%(th_opt))
# plt.plot(x[1000:1100],(signal-opt_signal)[1000:1100],color='darkorange',label='Difference')
# plt.xlabel('Time')
# plt.ylabel('Signal')
# plt.legend()

# data = pd.read_csv('images/GOOGL.csv', sep=';')
# data.Date = pd.to_datetime(data.Date)
# data.head()
#
# # Data pre-processing
# start_date = data.Date.loc[0]
# end_date = data.Date.loc[len(data) - 1]
# start_year = start_date.year
# start_month = start_date.month
# start_day = start_date.day
# end_year = end_date.year
# end_month = end_date.month
# end_day = end_date.day
# number_of_days = abs((end_date - start_date).days)
# start_date = datetime.date(start_date.year, start_date.month, start_date.day)
# date_list = []
# for day in range(number_of_days):
#     a_date = (start_date + datetime.timedelta(days=day)).isoformat()
#     date_list.append(a_date)
# date_list = pd.to_datetime(date_list, utc=True)
# new_data = pd.DataFrame({'Date': date_list})
# x = new_data.Date
# old_x = pd.to_datetime(data.Date, utc=True)
# y = []
# for i in range(len(x)):
#     x_i = x.loc[i]
#     diff_list = []
#     for j in range(len(data)):
#         diff_list.append(abs((x_i - old_x.loc[j]).days))
#     diff_list = np.array(diff_list)
#     y.append(data.Close[diff_list.argmin()])
#
# #y = np.asarray(y)
# data_t = {}
# data_t["Date"] = data.Date
# data_t["Close"] = data.Close
# data_t["x"] = x
# df_y = pd.DataFrame({'y': y})
# data_t["y"] = df_y
# df = pd.DataFrame.from_dict(data_t, orient="index")
# df.to_csv("images/googl_data.csv")

# Load data from transformed csv
data = pd.read_csv('images/googl_data.csv', sep=';')


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title('Original Data', color='red', fontsize=20)
plt.scatter(data.Date, data.Close, s=2)
plt.xlabel('Date')
plt.ylabel('Close')
plt.subplot(1, 2, 2)
plt.title('Smoothed Data', color='navy', fontsize=20)
plt.scatter(data.x, data.y, s=2, color='navy')
plt.xlabel('Date')
plt.ylabel('Close')

plt.show()
