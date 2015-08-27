import matplotlib
import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.fftpack as fftpack
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker
from itertools import cycle
import glob

try:
  plt.style.use('ian')
except AttributeError:
  print "old matplotlib version?"
except ValueError:
  print "Cannot find the requested stylesheet"

#c = ['#0A5F02', '#1c567a', '#814292', '#d7383b', '#fdae61', '#c0f8b8']
#c = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'] 
#c = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

rho = 3300
alpha_0 = 4.e-5
k = 4.1
cp = 1250.
kappa = k/rho/cp
eta_0 = 1.e21
D = 6371.e3
g = 9.81

seconds_to_Gyr = 3600.*24.*365.25*1.e9
omega_E = 7.29e-5
omega = 4.e-5

frac = 50

def get_spin_spectrum(text_file):

  #load the data
  data = np.loadtxt(text_file,skiprows=50, usecols=(1,17,18,19,20))
  #get rid of duplicate times (why might these exist?)
  stats = []
  for i in range(len(data)-1):
    if data[i,0] != data[i+1, 0]:
      stats.append(data[i,:])
  stats = np.array(stats) 
        
  eig1tck = interp.splrep(stats[:,0],stats[:,3], k=3 )
  eig2tck = interp.splrep(stats[:,0],stats[:,4], k=3)
  axistck = interp.splrep(stats[:,0],stats[:,1], k=3)
  mismatchtck = interp.splrep(stats[:,0],stats[:,2], k=3)

  times = np.linspace(stats[0,0], stats[-1,0], len(stats[:,0]))
  lambda1 = interp.splev(times, eig1tck, der=0)
  lambda2 = interp.splev(times, eig2tck, der=0)
  axis = interp.splev(times, axistck, der=0)
  mismatch = interp.splev(times, mismatchtck, der=0)

  # nondimensionalize
  diff = (lambda1-lambda2)/( np.mean(lambda1 + lambda2)/2.0 )
  times = times/1.e9 #* kappa / D / D
 
  frequency = times[-1]/len(times)
  autocorr = lambda x: np.correlate(x,x, mode='same')
  psd = lambda x : signal.welch(x,  fs=frequency,scaling='spectrum', return_onesided=True, nfft=100, nperseg=10)

  acorrdiff = autocorr(diff)
  p = psd(diff)
  peak = p[0][np.argmax(p[1])]
  print text_file, peak

  return times, diff, acorrdiff, peak



output_files = [f for f in glob.glob("statistics_*")]
sort_key = lambda s: float(s.split('_')[1])
output_files.sort(key=sort_key)
print output_files

amplitudes = []
timescales = []
Rayleighs = []

#plt.subplots_adjust(wspace=0.0, hspace=0.0)

colors = cycle(plt.rcParams['axes.color_cycle'])
ax = plt.subplot(1,2,1)

for i,f in enumerate(output_files):

  delta_T = float(f.split('_')[1])
  alpha = alpha_0
  eta = eta_0 
  Ra = rho * g * alpha * delta_T * D * D * D / eta / kappa


  times, diff, acorrdiff, peak = get_spin_spectrum(f)
  tmax = times[-1]
  tmin = times[0]
  times = times-tmin
  
  Rayleighs.append(Ra)
  amplitudes.append(diff.mean())
  timescales.append(peak)

  #plot a subset of the timeseries (plotting all of them got too busy)
  if f in ['statistics_50', 'statistics_200', 'statistics_500']:
    ax.plot(times, diff, next(colors),\
            label=r'$\mathrm{Ra} = %.1f \times 10^{%i}$' % ( Ra/np.power(10., np.floor(np.log10(Ra))) , np.floor(np.log10(Ra))))
  else:
    next(colors)



ax.set_xlim(0,4)
ax.set_ylim(0, 8.0e-5)
ax.set_xlabel("Time (Gyr)")
ax.set_ylabel(r'$(\lambda_2-\lambda_1)/I_0$')
ax.legend(loc="upper right")
print np.polyfit(np.log(Rayleighs[3:]), np.log(amplitudes[3:]),1)

ax = plt.subplot(122)
ax.grid(False)
ax.set_xscale('log')
ax.set_yscale('log')
ax.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())
ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext())

Ra=np.logspace(6.6, 8.3)
Moment = 2.5*np.power(Ra, -2./3.)

ax.plot(Ra, Moment, '--', c='0.5', linewidth=4)

clist = plt.rcParams['axes.color_cycle']
ax.scatter(Rayleighs, np.array(amplitudes), c = clist, s=100)

ax.set_ylim(3.0e-6, 3e-4)
ax.set_xlim(1.e6, 4.0e8)
ax.set_xlabel("Rayleigh number")
ax.set_ylabel("Average relative moment")

fig = plt.gcf()
fig.set_size_inches(12.0,6.0)
plt.savefig("eigengap.pdf")
#plt.show()
