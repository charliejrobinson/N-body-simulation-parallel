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


G = 1.0    # Gravitational Constant
#G = 6.6743 * 10**(-11) # m^3/(kg*s^2)

def calc_acc(pos, mass, soft_param):
    '''
    Parameters
    ----------
    pos : (N,3) Matrix of position vectors
    mass : (N,1) Matrix of masses
    soft_param : Softening parameter

    Returns
    -------
    acc : Matrix of accelerations
    '''

    # particle positions r=(x,y,z)
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # calculate particle seperations
    dx = np.transpose(x) - x
    dy = np.transpose(y) - y
    dz = np.transpose(z) - z

    # matrix of inverse seperations cubed (1/r^3)
    inv_sep = (dx**2 + dy**2 + dz**2 + soft_param**2)**(-1.5)

    # calculate acceleration components
    ax = np.matmul(G * (dx * inv_sep), mass)
    ay = np.matmul(G * (dy * inv_sep), mass)
    az = np.matmul(G * (dz * inv_sep), mass)

    # create acceleration matrix
    acc = np.hstack((ax,ay,az))

    return acc


def simulate(N, dt, t, t_max, soft_param):
    '''
    Calculate values for simulation
    '''

    # generate initial conditions
    np.random.seed(17)

    pos  = np.random.randn(N,3) # normally distributed positions
    vel  = np.zeros((N,3)) #np.random.randn(N,3) # normally distributed velocities
    mass = 1.0 * np.ones((N,1))  # particle mass is 1.0

    # convert to Center-of-Mass frame (??)
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial conditions
    acc = calc_acc(pos, mass, soft_param)

    # setup initial heavy one
    pos[0] = [0,0,0]
    mass[0] = 1.0 * 10**(30)

    # calculate timesteps for plotting
    steps = int(np.ceil(t_max/dt))

    # data store for plotting, define t=0
    pos_t = np.zeros((N,3,steps+1))
    pos_t[:,:,0] = pos

	# Iteration loop by leapfrog integration
    for i in range(steps):
		# first kick
        vel += acc * dt/2.0

		# drift
        pos += vel * dt

		# recalculate accelerations
        acc = calc_acc(pos, mass, soft_param)

		# second kick
        vel += acc * dt/2.0

		# new time
        t += dt

		# get energy of system
        pos_t[:,:,i+1] = np.copy(pos)

    return pos_t

# ---------------------------------
# Initalise parameters
N = 200             # number of particles
dt = 0.01           # timestep
t = 0               # current time of the simulation
t_max = 10.0        # how many seconds simulation runs for
soft_param = 0.1    # softening parameter

# ---------------------------------
# Run simulation
start = timeit.default_timer()

pos_t = simulate(N, dt, t, t_max, soft_param)

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
