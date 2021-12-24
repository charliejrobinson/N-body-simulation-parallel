import simulation_python
import simulation_cython
import numpy as np

import matplotlib
import time
import sys

matplotlib.use('MACOSX')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import timeit
import argparse

parser = argparse.ArgumentParser(description='Gravity Simulator')
parser.add_argument('simulation', choices=['python', 'cython'], help='Which simulation to use')
parser.add_argument('--animate', action='store_true', help='plot animated graphs')
parser.add_argument('--plot_start', action='store_true', help='plots start graph')
parser.add_argument('--plot_end', action='store_true', help='plots end graph')
parser.add_argument('--seed', type=int, default=17, help='random seed to use')
parser.add_argument('--N', type=int, default=3, help='number of particles')
parser.add_argument('--dt', type=float, default=0.01, help='timestep')
parser.add_argument('--t_max', type=float, default=10.0, help='how many seconds simulation runs for')

args = parser.parse_args()

# ---------------------------------
# Initalise parameters
np.random.seed(args.seed)

N = args.N
dt = args.dt
t_max = args.t_max

G = 1.0    # Gravitational Constant - G = 6.6743 * 10**(-11) # m^3/(kg*s^2)
t = 0   # current time of the simulation
soft_param = 0.1    # softening parameter

# ---------------------------------
# Initalise conditions

# ---------------------------------
# Run simulation
start = timeit.default_timer()

pos_t = None
if args.simulation == 'python':
    pos_t = simulation_python.simulate(G, N, dt, t, t_max, soft_param)
elif args.simulation == 'cython':
    pos_t = simulation_cython.simulate(G, N, dt, t, t_max, soft_param)

end = timeit.default_timer()

print('Executed %s simulation in %.2fs' % (args.simulation, end - start))

# ---------------------------------
# Draw graph
def plot_at_index(name, i, fps, pos_t):
    t = i / fps

    fig = plt.figure(name, dpi=100)
    ax = fig.add_subplot(projection='3d')
    title = ax.set_title('Gravity Simulator\n N=%i dt=%.2f t=%.2f t_max=%.2f' % (N, dt, t, t_max))

    scatter = ax.scatter(pos_t[:,0,i], pos_t[:,1,i], pos_t[:,2,i], s=1, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(np.min(pos_t[:,0,:]),np.max(pos_t[:,0,:]))
    ax.set_ylim(np.min(pos_t[:,1,:]),np.max(pos_t[:,1,:]))
    ax.set_zlim(np.min(pos_t[:,2,:]),np.max(pos_t[:,2,:]))

    return fig, scatter, title

def draw(i, title, fps, scatter, pos_t):
    t = i / fps
    title.set_text('Gravity Simulator\nN=%i dt=%.2f t=%.2fs t_max=%.2fs' % (N, dt, t, t_max))

    scatter._offsets3d = (pos_t[:,0,i], pos_t[:,1,i], pos_t[:,2,i])

fps = (pos_t.shape[2]-1) / t_max

if args.plot_start:
    plot_at_index('Gravity Simulator - Start', 0, fps, pos_t)

if args.plot_end:
    plot_at_index('Gravity Simulator - End', pos_t.shape[2]-1, fps, pos_t)

if args.animate:
    fig, scatter, title = plot_at_index('Gravity Simulator - Animated', 0, fps, pos_t)
    ani = matplotlib.animation.FuncAnimation(fig, draw, frames=pos_t.shape[2], fargs=(title, fps, scatter, pos_t), interval=round(1000 / fps), repeat=False)

plt.show()
