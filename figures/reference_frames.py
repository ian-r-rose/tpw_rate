import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import rc

rc('text', usetex=True)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def Rx(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])

# define origin
o = np.array([0,0,0])

# define ox0y0z0 axes
x0 = np.array([1,0,0])
y0 = np.array([0,1,0])
z0 = np.array([0,0,1])
L3=3.0
L2=2.0
L1=1.0
E = np.array([ [L1, 0, 0], [0, L2, 0], [0, 0, L3] ])


# spin axis specification
theta = 30. * np.pi/180.
phi = 0. * np.pi/180.
omega = np.array([ np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
#relative motion of the two frames
alpha = 50. * np.pi/180.
beta = 70. * np.pi/180.
psi = np.array([ np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta), np.cos(alpha)])
# tpw vector
omega_dot = np.dot(E, omega.T) - np.dot(omega, np.dot(E, omega.T))*omega.T
omega_dot_prime = omega_dot + np.cross(psi, omega)
omega_dot = 0.3*omega_dot/norm(omega_dot)
omega_dot_prime = 0.3*omega_dot_prime/norm(omega_dot_prime)

# produce figure
alpha_val = 0.4
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
basis_arrow_prop_dict = dict(mutation_scale=20, lw=2, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0, alpha=alpha_val)
arrow_prop_dict = dict(mutation_scale=20, lw=2, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0, alpha=1.0)

# plot ox0y0z0 axes
e1 = Arrow3D([o[0], x0[0]], [o[1], x0[1]], [o[2], x0[2]], **basis_arrow_prop_dict)
ax.add_artist(e1)
e2 = Arrow3D([o[0], y0[0]], [o[1], y0[1]], [o[2], y0[2]], **basis_arrow_prop_dict)
ax.add_artist(e2)
e3 = Arrow3D([o[0], z0[0]], [o[1], z0[1]], [o[2], z0[2]], **basis_arrow_prop_dict)
ax.add_artist(e3)

# add dotted arcs to axes
arc_angles = np.linspace(-5., 95., 200)*np.pi/180.
arc = np.array([np.sin(arc_angles)*0., np.sin(arc_angles), np.cos(arc_angles)])
ax.plot( arc[0,:], arc[1,:], arc[2,:], 'k--', alpha=alpha_val)
arc = np.array([np.cos(arc_angles), np.sin(arc_angles), np.cos(arc_angles)*0.])
ax.plot( arc[0,:], arc[1,:], arc[2,:], 'k--', alpha=alpha_val)

# plot spin axis
spin_arrow = Arrow3D([o[0], omega[0]], [o[1], omega[1]], [o[2], omega[2]], **arrow_prop_dict)
ax.add_artist(spin_arrow)
# add arc for spin axis
arc_angles = np.linspace(0., theta, 100)
arc = 0.5*np.array([np.sin(arc_angles)*np.cos(phi), np.sin(arc_angles)*np.sin(phi), np.cos(arc_angles)])
ax.plot( arc[0,:], arc[1,:], arc[2,:], 'k')
# add helper lines for omega
span = np.linspace(0, 1., 100)
guideline = np.array([np.cos(phi)*np.sin(theta)*np.ones_like(span), np.sin(phi)*np.sin(theta)*np.ones_like(span), np.cos(theta)*span])
ax.plot( guideline[0,:], guideline[1,:], guideline[2,:], 'k--', alpha=alpha_val)

# plot relative rotation
psi_arrow = Arrow3D([o[0], psi[0]], [o[1], psi[1]], [o[2], psi[2]], **arrow_prop_dict)
ax.add_artist(psi_arrow)
# add helper lines for psi
span = np.linspace(0, 1., 100)
guideline = np.array([np.cos(beta)*np.sin(alpha)*np.ones_like(span), np.sin(beta)*np.sin(alpha)*np.ones_like(span), np.cos(alpha)*span])
ax.plot( guideline[0,:], guideline[1,:], guideline[2,:], 'k--', alpha=alpha_val)
guideline = np.array([np.cos(beta)*np.sin(alpha)*(span), np.sin(beta)*np.sin(alpha)*(span), np.zeros_like(span)])
ax.plot( guideline[0,:], guideline[1,:], guideline[2,:], 'k--', alpha=alpha_val)
guideline = np.array([omega * s + psi * (1.-s) for s in span] )
for i,p in enumerate(guideline):
  guideline[i] = 0.45*p/norm(p)
ax.plot( guideline[:,0], guideline[:,1], guideline[:,2], 'k')

#plot omega_dot
omega_dot_arrow = Arrow3D([omega[0], omega[0]+omega_dot[0]], \
                          [omega[1], omega[1]+omega_dot[1]], \
                          [omega[2], omega[2]+omega_dot[2]], \
                          **arrow_prop_dict)
ax.add_artist(omega_dot_arrow)
omega_dot_prime_arrow = Arrow3D([omega[0], omega[0]+omega_dot_prime[0]], \
                                [omega[1], omega[1]+omega_dot_prime[1]], \
                                [omega[2], omega[2]+omega_dot_prime[2]], \
                                **arrow_prop_dict)
ax.add_artist(omega_dot_prime_arrow)


text_options = {'horizontalalignment': 'center',
                'verticalalignment': 'center',
                'fontsize': 16}

# add labels for axes
ax.text(1.1*x0[0],1.1*x0[1],1.1*x0[2],r'$e_1$', **text_options)
ax.text(1.1*y0[0],1.1*y0[1],1.1*y0[2],r'$e_2$', **text_options)
ax.text(1.1*z0[0],1.1*z0[1],1.1*z0[2],r'$e_3$', **text_options)
ax.text(1.1*omega[0],1.1*omega[1],1.1*omega[2],r'$\omega$', **text_options)
ax.text(1.1*psi[0],1.1*psi[1],1.1*psi[2],r'$\psi$', **text_options)
ax.text(omega[0] + omega_dot[0]*1.1, omega[1] + omega_dot[1]*1.1, omega[2]+omega_dot[2]*1.1, r'$\dot{\omega}$', **text_options)
ax.text(omega[0] + omega_dot_prime[0]*1.1, omega[1] + omega_dot_prime[1]*1.1, omega[2]+omega_dot_prime[2]*1.1, r'$\dot{\omega}^\prime$', **text_options)

# add text for angles
p = 0.45*(omega + z0)/2.0
ax.text( p[0], p[1], p[2], r'$\theta$', **text_options)
p = 0.3* (psi + omega) / 2.0
ax.text( p[0], p[1], p[2], r'$\gamma$', **text_options)

# show figure
ax.view_init(elev=36, azim=18)
ax.set_axis_off()
plt.show()
