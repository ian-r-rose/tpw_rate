import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import rc

rc('text', usetex=True)

d2r = np.pi/180.
r2d = 180./np.pi

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def construct_euler_rotation_matrix(alpha, beta, gamma):
    """
    Make a 3x3 matrix which represents a rigid body rotation,
    with alpha being the first rotation about the z axis,
    beta being the second rotation about the y axis, and
    gamma being the third rotation about the z axis.
 
    All angles are assumed to be in radians
    """
    rot_alpha = np.array( [ [np.cos(alpha), -np.sin(alpha), 0.],
                            [np.sin(alpha), np.cos(alpha), 0.],
                            [0., 0., 1.] ] )
    rot_beta = np.array( [ [np.cos(beta), 0., np.sin(beta)],
                           [0., 1., 0.],
                           [-np.sin(beta), 0., np.cos(beta)] ] )
    rot_gamma = np.array( [ [np.cos(gamma), -np.sin(gamma), 0.],
                            [np.sin(gamma), np.cos(gamma), 0.],
                            [0., 0., 1.] ] )
    rot = np.dot( rot_gamma, np.dot( rot_beta, rot_alpha ) )
    return rot

def spherical_to_cartesian( longitude, latitude, norm ):
    assert(np.all(longitude >= 0.) and np.all(longitude <= 360.))
    assert(np.all(latitude >= -90.) and np.all(latitude <= 90.))
    assert(np.all(norm >= 0.))
    colatitude = 90.-latitude
    return np.array([ norm * np.sin(colatitude*d2r)*np.cos(longitude*d2r),
                      norm * np.sin(colatitude*d2r)*np.sin(longitude*d2r),
                      norm * np.cos(colatitude*d2r) ] )

def cartesian_to_spherical( vecs ):
    v = np.reshape(vecs, (3,-1))
    norm = np.sqrt(v[0,:]*v[0,:] + v[1,:]*v[1,:] + v[2,:]*v[2,:])
    latitude = 90. - np.arccos(v[2,:]/norm)*r2d
    longitude = np.arctan2(v[1,:], v[0,:] )*r2d
    return longitude, latitude, norm

def plot_rotation_arrow( axes, arrow, ccw=True, frac = 0.75, *args, **kwargs):
    origin = np.array([ arrow._verts3d[0][0], arrow._verts3d[1][0], arrow._verts3d[2][0]])
    tip = np.array([ arrow._verts3d[0][1], arrow._verts3d[1][1], arrow._verts3d[2][1]])
    vec = tip-origin
    lon,lat,norm = cartesian_to_spherical(vec)

    # Make points around the arrow
    colat = 4.
    azimuths = np.linspace(0., 330., 331.)
    lats = np.ones_like(azimuths)*(90. - colat)
    norms = np.ones_like(azimuths)*norm[0]*frac
    points = spherical_to_cartesian(azimuths, lats, norms)
    rotation_matrix = construct_euler_rotation_matrix( 0.,  (90.-lat[0])*d2r, lon[0]*d2r)
    rotated_points = np.dot( rotation_matrix, points )

    # plot the points
    dline = 10
    line = axes.plot(rotated_points[0,:-dline], rotated_points[1,:-dline], rotated_points[2,:-dline], lw=1, color='k')
    # add the arrowhead
    if ccw is True:
        head = Arrow3D( [rotated_points[0,-1-dline], rotated_points[0,-1]],
                        [rotated_points[1,-1-dline], rotated_points[1,-1]],
                        [rotated_points[2,-1-dline], rotated_points[2,-1]], 
                        arrowstyle='-|>', mutation_scale=10, color='k')
    else:
        head = Arrow3D( [rotated_points[0,dline], rotated_points[0,0]],
                        [rotated_points[1,dline], rotated_points[1,0]],
                        [rotated_points[2,dline], rotated_points[2,0]], 
                        arrowstyle='-|>', mutation_scale=10, color='k')
    axes.add_artist(head)
    return line,head


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
beta = 110. * np.pi/180.
psi = np.array([ np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta), np.cos(alpha)])
# tpw vector
omega_dot = np.dot(E, omega.T) - np.dot(omega, np.dot(E, omega.T))*omega.T
omega_dot_prime = omega_dot + np.cross(psi, omega)
#omega_dot = 0.3*omega_dot/norm(omega_dot)
#omega_dot_prime = 0.3*omega_dot_prime/norm(omega_dot_prime)
omega_dot *= 0.35
omega_dot_prime *= 0.35

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
plot_rotation_arrow(ax, spin_arrow)
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
plot_rotation_arrow(ax, psi_arrow, ccw=False)
# add helper lines for psi
span = np.linspace(0, 1., 100)
guideline = np.array([np.cos(beta)*np.sin(alpha)*np.ones_like(span), np.sin(beta)*np.sin(alpha)*np.ones_like(span), np.cos(alpha)*span])
ax.plot( guideline[0,:], guideline[1,:], guideline[2,:], 'k--', alpha=alpha_val)
guideline = np.array([np.cos(beta)*np.sin(alpha)*(span), np.sin(beta)*np.sin(alpha)*(span), np.zeros_like(span)])
ax.plot( guideline[0,:], guideline[1,:], guideline[2,:], 'k--', alpha=alpha_val)

tmpvec = np.array([ np.cos(beta), np.sin(beta), 0.])
guideline = np.array([tmpvec * s + psi * (1.-s) for s in span] )
for i,p in enumerate(guideline):
  guideline[i] = 0.45*p/norm(p)
ax.plot( guideline[:,0], guideline[:,1], guideline[:,2], 'k')
tmpvec2 = np.array([ 1., 0., 0.])
guideline = np.array([tmpvec * s + tmpvec2 * (1.-s) for s in span] )
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
ax.text(1.1*x0[0],1.1*x0[1],1.1*x0[2],r'$\mathbf{e}_1$', **text_options)
ax.text(1.1*y0[0],1.1*y0[1],1.1*y0[2],r'$\mathbf{e}_2$', **text_options)
ax.text(1.1*z0[0],1.1*z0[1],1.1*z0[2],r'$\mathbf{e}_3$', **text_options)
ax.text(1.1*omega[0],1.1*omega[1],1.1*omega[2],r'$\omega$', **text_options)
ax.text(1.1*psi[0],1.1*psi[1],1.1*psi[2],r'$\Psi$', **text_options)
ax.text(omega[0] + omega_dot[0]*1.15, omega[1] + omega_dot[1]*1.15, omega[2]+omega_dot[2]*1.15, r'$\dot{\omega}$', **text_options)
ax.text(omega[0] + omega_dot_prime[0]*1.2, omega[1] + omega_dot_prime[1]*1.2, omega[2]+omega_dot_prime[2]*1.2, r'$\dot{\omega}^\prime$', **text_options)

# add text for angles
p = 0.45*(omega + z0)/2.0
ax.text( p[0], p[1], p[2], r'$\theta$', **text_options)
p = 0.4* (tmpvec + psi) / 2.0
ax.text( p[0], p[1], p[2], r'$\alpha$', **text_options)
p = 0.6* (tmpvec + tmpvec2) / 2.0
ax.text( p[0], p[1], p[2], r'$\beta$', **text_options)

# show figure
ax.view_init(elev=25, azim=49)
ax.set_axis_off()
#plt.show()
plt.savefig("reference_frames.pdf")
