import simulation
import numpy as np

import matplotlib
import time
import sys

matplotlib.use('MACOSX') # For mac

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import timeit
import argparse

parser = argparse.ArgumentParser(description='Gravity Simulator')
parser.add_argument('--plot', action='store_true', help='plot graphs')

args = parser.parse_args()

# ---------------------------------
# Initalise parameters
G = 1.0    # Gravitational Constant
#G = 6.6743 * 10**(-11) # m^3/(kg*s^2)
N = 200             # number of particles
dt = 0.01           # timestep
t = 0               # current time of the simulation
t_max = 10.0        # how many seconds simulation runs for
soft_param = 0.1    # softening parameter

# ---------------------------------
# Run simulation
start = timeit.default_timer()

pos_t = simulation.simulate(G, N, dt, t, t_max, soft_param)

end = timeit.default_timer()

print('Executed in %.2fs' % (end - start))

# ---------------------------------
# Draw graph
if not args.plot:
    sys.exit(0)

def draw(i, fps, scatter, pos_t):
    t = i / fps
    title.set_text('Gravity Simulator\nN=%i dt=%.2f t=%.2fs t_max=%.2fs' % (N, dt, t, t_max))

    scatter._offsets3d = (pos_t[:,0,i], pos_t[:,1,i], pos_t[:,2,i])

fig = plt.figure('Gravity Simulator', dpi=100)
ax = fig.add_subplot(projection='3d')
title = ax.set_title('Gravity Simulator\n N=%i dt=%.2f t=0.00 t_max=%.2f' % (N, dt, t_max))

scatter = ax.scatter(pos_t[:,0,0], pos_t[:,1,0], pos_t[:,2,0], s=100, marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(np.min(pos_t[:,0,:]),np.max(pos_t[:,0,:]))
ax.set_ylim(np.min(pos_t[:,1,:]),np.max(pos_t[:,1,:]))
ax.set_zlim(np.min(pos_t[:,2,:]),np.max(pos_t[:,2,:]))

fps = pos_t.shape[2] / t_max

ani = matplotlib.animation.FuncAnimation(fig, draw, frames=pos_t.shape[2], fargs=(fps, scatter, pos_t), interval=round(1000 / fps), repeat=False)

plt.show()
