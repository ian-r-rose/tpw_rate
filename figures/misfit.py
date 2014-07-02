
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

c = ['#0A5F02', '#1c567a', '#814292', '#d7383b', '#fdae61', '#c0f8b8']
data = np.loadtxt('statistics_500', skiprows=50, usecols=(1,15,16,17,18))

rho = 3300
alpha_0 = 4.e-5
delta_T_0 = 200.0
k = 4.1
cp = 1250.
kappa = k/rho/cp
eta_0 = 1.e21
D = 6371.e3
g = 9.81


time = data[:,0]*kappa/D/D
mismatch = data[:,2]
eig1 = data[:,3]
eig2 = data[:,4]
eigmean = np.mean(eig1+eig2)/2.0
eig1 = eig1/eigmean - 1.
eig2 = eig2/eigmean - 1.

spin = data[:,1]

pos = np.where(np.abs(np.diff(spin)) >= 10.0)[0]
time[pos] = np.nan
spin[pos] = np.nan

fig = plt.figure()
fig.subplots_adjust(wspace=0.0, hspace=0.0)

ax = fig.add_subplot(211)

ax.plot(time, eig1, color=c[3], label=r'$\lambda_1$')
ax.plot(time, eig2, color=c[1], label=r'$\lambda_2$')
plt.xlim(0.0, 0.5e-10)
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
plt.ylabel(r'Relative moment')

ax = fig.add_subplot(212)

ax.plot(time, spin, color=c[0], label=r'Spin axis')
ax.plot(time, mismatch, color=c[2], label=r'$\theta$')
plt.xlim(0.0, 0.5e-10)
ax.legend(loc='upper right')
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6, prune='both'))
plt.xlabel(r"Time")
plt.ylabel(r"Angle ($^\circ$)")




plt.show()
plt.savefig("misfit.pdf")
