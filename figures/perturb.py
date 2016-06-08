from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rc('text', usetex=True)

d2r = np.pi/180.

fig = plt.figure(figsize=(8,4), frameon=False)


ax = fig.add_subplot(111)
ax.axis('off')
ax.axes.get_yaxis().set_major_formatter(plt.NullFormatter())
ax.axes.get_xaxis().set_major_formatter(plt.NullFormatter())
ax.tick_params( top='off', bottom='off', left='off', right='off')
ax.set_xlim(-2,2)
ax.set_ylim(-1,1)

ellipse_args = { 'edgecolor': 'k', 'facecolor':'w', 'lw':2 } 
arrow_args = {'arrowstyle':'-|>', 'lw':2, 'mutation_scale':20, 'facecolor':'k'}
text_args = {'horizontalalignment': 'center', 'verticalalignment': 'center', 'fontsize': 20}

##FLATTENED ELLIPSE

#ellipse
flat_ellipse = patches.Ellipse( xy = (-1,0), width=1.6, height=1.2, angle=20., **ellipse_args)
ax.add_patch(flat_ellipse)
#perturbation
perturbation = patches.Ellipse( xy = (-1.3, 0.4), width=0.15, height=0.15, facecolor='darkgray', edgecolor='w')
ax.add_patch(perturbation)

#spin axis
omega = patches.FancyArrowPatch( (-1,0), (-0., 0.1), **arrow_args)
ax.add_patch(omega)
ax.text( 0.04, 0.1, r'$\Omega$', **text_args)

#principal axes
lambda_1 = patches.FancyArrowPatch( (-1,0), (-1. + 0.8*np.cos(30.*d2r), 0.8*np.sin(30.*d2r)), **arrow_args)
lambda_2 = patches.FancyArrowPatch( (-1,0), (-1. + 0.6*np.cos(-60.*d2r), 0.6*np.sin(-60.*d2r)), **arrow_args)
ax.add_patch(lambda_1)
ax.add_patch(lambda_2)
ax.text( 1.5, -0.25, r'$\lambda_1$', **text_args)
ax.text( 0.97, 0.5, r'$\lambda_2$', **text_args)

#arc
angles = np.linspace( 30.*d2r, np.arctan2( 0.1, 0.9), 100)
ax.plot( -1. + 0.5*np.cos(angles), 0. + 0.5*np.sin(angles), lw=1, color='k' )
ax.text( -0.6,  0.13, r'$\xi$', **text_args)

#smooth out center
ax.add_patch( patches.Ellipse(xy=(-1,0), width=0.02, height=0.02, facecolor='k', edgecolor='k') )

#### ROUND ELLIPSE

#ellipse
round_ellipse = patches.Ellipse( xy = (1,0), width=1.4, height=1.4, angle=20., **ellipse_args)
ax.add_patch(round_ellipse)
#perturbation
perturbation = patches.Ellipse( xy = (0.7, 0.4), width=0.15, height=0.15, facecolor='darkgray', edgecolor='w')
ax.add_patch(perturbation)

#spin axis
omega = patches.FancyArrowPatch( (1,0), (1.9, 0.1), **arrow_args)
ax.add_patch(omega)
ax.text( 1.94, 0.1, r'$\Omega$', **text_args)

#principal axes
lambda_1 = patches.FancyArrowPatch( (1,0), (1. + 0.7*np.cos(80.*d2r), 0.7*np.sin(80.*d2r)), **arrow_args)
lambda_2 = patches.FancyArrowPatch( (1,0), (1. + 0.7*np.cos(-10.*d2r), 0.7*np.sin(-10.*d2r)), **arrow_args)
ax.add_patch(lambda_1)
ax.add_patch(lambda_2)
ax.text( -0.86, -0.45, r'$\lambda_1$', **text_args)
ax.text( -0.52, 0.4, r'$\lambda_2$', **text_args)

#arc
angles = np.linspace( 80.*d2r, np.arctan2( 0.1, 0.9), 100)
ax.plot( 1. + 0.3*np.cos(angles), 0. + 0.3*np.sin(angles), lw=1, color='k' )
ax.text( 1.15,  0.13, r'$\xi$', **text_args)

#smooth out center
ax.add_patch( patches.Ellipse(xy=(1,0), width=0.02, height=0.02, facecolor='k', edgecolor='k') )

#plt.show()
plt.savefig('perturb.pdf', bbox_inches='tight')
