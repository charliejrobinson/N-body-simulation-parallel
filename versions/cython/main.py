import os
os.environ ["MKL_NUM_THREADS"] = "1"
os.environ ["NUMEXPR_NUM_THREADS"] = "1"
os.environ ["OMP_NUM_THREADS"] = "1"

import simulation_python
import simulation_cython

import numpy as np
import pickle
import matplotlib
import time
import sys
import datetime

matplotlib.use('MACOSX')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import timeit
import argparse

parser = argparse.ArgumentParser(description='Gravity Simulator')
parser.add_argument('--simulation', type=str, nargs='*', default=[], choices=['python', 'cython'], help='Which simulation to use')
parser.add_argument('--animate', action='store_true', help='plot animated graphs')
parser.add_argument('--plot_start', action='store_true', help='plots start graph')
parser.add_argument('--plot_end', action='store_true', help='plots end graph')
parser.add_argument('--stats', action='store_true', help='plots stats')
parser.add_argument('--average_over', type=int, default=3, help='Number of runs to average over')
parser.add_argument('--seed', type=int, default=17, help='random seed to use')
parser.add_argument('--N', type=int, nargs='+', default=[3, 50, 100, 200, 400], help='number of particles')
parser.add_argument('--dt', type=float, default=0.01, help='timestep')
parser.add_argument('--t_max', type=float, default=10.0, help='how many seconds simulation runs for')

args = parser.parse_args()

def run_simulation(simulation, G, N, dt, t, t_max, soft_param):
    start = timeit.default_timer()

    pos_t = None
    if simulation == 'python':
        pos_t = simulation_python.simulate(G, N, dt, t, t_max, soft_param)
    elif simulation == 'cython':
        pos_t = simulation_cython.simulate(G, N, dt, t, t_max, soft_param)

    end = timeit.default_timer()

    return end-start, pos_t

def plot_at_index(name, i, fps, pos_t, N, dt, t, t_max):
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

def draw(i, title, fps, scatter, pos_t, N, dt, t, t_max):
    t = i / fps
    title.set_text('Gravity Simulator\nN=%i dt=%.2f t=%.2fs t_max=%.2fs' % (N, dt, t, t_max))

    scatter._offsets3d = (pos_t[:,0,i], pos_t[:,1,i], pos_t[:,2,i])

# ---------------------------------
# Initalise parameters
np.random.seed(args.seed)

dt = args.dt
t_max = args.t_max

G = 1.0    # Gravitational Constant - G = 6.6743 * 10**(-11) # m^3/(kg*s^2)
t = 0   # current time of the simulation
soft_param = 0.1    # softening parameter

# ---------------------------------
# Load stats
stats = pickle.load(open('stats.p', 'rb'))

# ---------------------------------
# Run simulation
for simulation in args.simulation:
    if simulation not in stats:
        stats[simulation] = []

    stat = {
        'date': datetime.datetime.now(),
        'dt': dt, 't': t, 't_max': t_max, 'soft_param': soft_param,
        'runs': {'N': [], 'duration': []},
    }

    for N in args.N:
        durations = []
        pos_t = None
        for i in range(args.average_over):
            duration, _pos_t = run_simulation(simulation, G, N, dt, t, t_max, soft_param)
            durations.append(duration)
            pos_t = _pos_t

        duration = np.average(durations)

        print('Executed %s simulation %i times with N=%i in average %.2fs' % (simulation,args.average_over,  N, duration))
        stat['runs']['N'].append(N)
        stat['runs']['duration'].append(duration)

        # Draw graph
        fps = (pos_t.shape[2]-1) / t_max

        if args.plot_start:
            plot_at_index('Gravity Simulator - Start', 0, fps, pos_t, N, dt, t, t_max)

        if args.plot_end:
            plot_at_index('Gravity Simulator - End', pos_t.shape[2]-1, fps, pos_t, N, dt, t, t_max)

        if args.animate:
            fig, scatter, title = plot_at_index('Gravity Simulator - Animated', 0, fps, pos_t, N, dt, t, t_max)
            ani = matplotlib.animation.FuncAnimation(fig, draw, frames=pos_t.shape[2], fargs=(title, fps, scatter, pos_t, N, dt, t, t_max), interval=round(1000 / fps), repeat=False)

        plt.show()

    stats[simulation].append(stat)

# ---------------------------------
# Write stats
pickle.dump(stats, open('stats.p', 'wb'))

# ---------------------------------

if args.stats:
    for simulation_name, simulation_stats in stats.items():
        last_run = simulation_stats[-1]
        plt.plot(last_run['runs']['N'], last_run['runs']['duration'], marker='o', label=simulation_name)

    plt.xlabel('N')
    plt.ylabel('Duratation (s)')
    plt.title('Stats')
    plt.legend()
    plt.show()
