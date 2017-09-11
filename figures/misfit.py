import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

import scipy.signal as signal
import scipy.interpolate as interp

try:
  plt.style.use('./tpw_rate.mplstyle')
except AttributeError:
  print("old matplotlib version?")
except ValueError:
  print("Cannot find the requested stylesheet")

data = np.loadtxt('statistics_500', skiprows=50, usecols=(1,17,18,19,20))

rho = 3300
alpha_0 = 4.e-5
delta_T_0 = 200.0
k = 4.1
cp = 1250.
kappa = k/rho/cp
eta_0 = 1.e21
D = 6371.e3
g = 9.81

time = data[:,0]/1.e9
t_start = 9.
time = time - t_start
tmin = 0.
tmax = 2.

mismatch = data[:,2]
#Interpolate using splines for a bit nicer-looking curves
theta = interp.UnivariateSpline( time, mismatch, s=8.e4)

eig1 = data[:,3]
eig2 = data[:,4]
eigmean = np.mean(eig1 + eig2)/2.
eigdiff = (eig1-eig2)/eigmean

spin = data[:,1]
spin_unwrap = np.unwrap(spin*np.pi*2./180.)*180./np.pi/2.
spin_rate = interp.UnivariateSpline(time, spin_unwrap).derivative()(time)

pos = np.where(np.abs(np.diff(spin)) >= 10.0)[0]
time[pos] = np.nan
spin[pos] = np.nan

fig = plt.figure()
ax = fig.add_subplot(311)

ax.plot(time, eigdiff, label=r'$\Lambda_{21} = (\lambda_2-\lambda_1)/I_0$')
ax.set_xlim(tmin, tmax)
ax.set_ylim(0., 2.e-5)
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, prune=None))
ax.set_ylabel(r'Eigengap')

ax = fig.add_subplot(312)

ax.plot(np.array([]), np.array([]))
ax.plot(time, spin, label=r'Spin axis')
ax.plot(np.array([]), np.array([]))
ax.plot(time, theta(time), label=r'$\theta$')
ax.set_xlim(tmin, tmax)
ax.set_ylim(0., 180.)
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3, prune=None))
ax.set_ylabel(r"Angle ($^\circ$)")

ax = fig.add_subplot(313)

ax.plot(np.array([]), np.array([]))
ax.plot(np.array([]), np.array([]))
ax.plot(time, np.abs(spin_rate)/1000., label=r'Spin axis')
ax.set_xlim(tmin, tmax)
ax.set_ylim(0., 6.)
ax.set_xlabel(r"Time (Gyr)")
ax.set_ylabel(r'TPW rate ($^\circ$/Myr)')


plt.savefig("misfit.pdf", bbox_inches='tight')
#plt.show()
