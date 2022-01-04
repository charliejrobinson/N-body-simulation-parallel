import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import simulation_python
import simulation_python_original
import simulation_python_sqrt
import simulation_chris_final
import simulation_python_without_numpy
import simulation_cython_without_numpy
import simulation_cython_openmp
#import simulation_cython

import openmp_api_wraper

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

# TODO remove
np.set_printoptions(precision=8)

simulations = ['python', 'cython', 'python_original', 'python_sqrt', 'chris', 'python_without_numpy', 'cython_without_numpy', 'cython_openmp']

parser = argparse.ArgumentParser(description='Gravity Simulator')

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--seed', type=int, default=17, help='random seed to use')
parent_parser.add_argument('--threads', type=int, default=1, help='number of threads to use for parallel code')
parent_parser.add_argument('--dt', type=float, default=0.01, help='timestep')
parent_parser.add_argument('--t_max', type=float, default=10.0, help='how many seconds simulation runs for')
parent_parser.add_argument('--simulation', type=str, nargs='*', default=[], choices=simulations, help='Which simulation to use')

subparsers = parser.add_subparsers(dest='command')

parser_run = subparsers.add_parser('run', help='Run simulations', parents=[parent_parser])
parser_run.add_argument('--N', type=int, nargs='+', default=[50], help='number of particles')
parser_run.add_argument('--animate', action='store_true', help='plot animated graphs')
parser_run.add_argument('--plot_start', action='store_true', help='plots start graph')
parser_run.add_argument('--plot_end', action='store_true', help='plots end graph')

parser_profile = subparsers.add_parser('profile', help='Record stats for simulations', parents=[parent_parser])
parser_profile.add_argument('--N', type=int, nargs='+', default=[3, 50, 100, 200, 400, 600, 800], help='number of particles')
parser_profile.add_argument('--average_over', type=int, default=3, help='Number of runs to average over')

parser_validate = subparsers.add_parser('validate', help='Validate a simulation is correct', parents=[parent_parser])
parser_validate.add_argument('--N', type=int, nargs='+', default=[12], help='number of particles')
parser_validate.add_argument('--validation_simulation', type=str, default='python_original', choices=simulations, help='Which simulation to use for validation')

parser_load = subparsers.add_parser('load', help='Load a positions file')
parser_load.add_argument('path', type=str, help='File path')

parser_stats = subparsers.add_parser('stats', help='Plot statistics')


args = parser.parse_args()

def run_simulation(threads, simulation, pos, mass, vel, G, N, dt, t_max, soft_param):
    # start = timeit.default_timer()
    start = openmp_api_wraper.get_wtime()

    pos_t = None
    if simulation == 'python':
        pos_t = simulation_python.simulate(pos, mass, vel, G, N, dt, t_max, soft_param)
    elif simulation == 'python_original':
        pos_t = simulation_python_original.simulate(pos, mass, vel, G, N, dt, t_max, soft_param)
    elif simulation == 'python_sqrt':
        pos_t = simulation_python_sqrt.simulate(pos, mass, vel, G, N, dt, t_max, soft_param)
    elif simulation == 'chris':
        pos_t = simulation_chris_final.simulate(pos, mass, vel, G, N, dt, t_max, soft_param)
    elif simulation == 'python_without_numpy':
        pos_t = simulation_python_without_numpy.simulate(pos, mass, vel, G, N, dt, t_max, soft_param)
    elif simulation == 'cython_without_numpy':
        pos_t = simulation_cython_without_numpy.simulate(pos, mass, vel, G, N, dt, t_max, soft_param)
    elif simulation == 'cython_openmp':
        pos_t = simulation_cython_openmp.simulate(threads, pos, mass, vel, G, N, dt, t_max, soft_param)
    elif simulation == 'cython':
        pass # pos_t = simulation_cython.simulate(pos, mass, vel, G, N, dt, t_max, soft_param)

    # end = timeit.default_timer()
    end = openmp_api_wraper.get_wtime()

    return end-start, pos_t

def initialise_environment(seed, N):
    np.random.seed(seed)

    # ---------------------------------
    # Initalise enviroment
    pos  = np.random.randn(N,3).astype(np.double) # normally distributed positions
    vel  = np.random.randn(N,3).astype(np.double) # normally distributed velocities
    mass = np.ones((N,1)).astype(np.double) # particle mass is 1.0

    vel -= np.mean(mass * vel, 0) / np.mean(mass) # convert to Center-of-Mass frame (??)

    # setup initial heavy one
    # pos[0] = [0,0,0]
    # mass[0] = 1.0 * 10**(30)

    return pos, vel, mass

def plot_at_index(name, i, fps, pos_t, N, dt, t_max, simulation):
    t = i / fps

    fig = plt.figure(name, dpi=100)
    ax = fig.add_subplot(projection='3d')
    title = ax.set_title('Gravity Simulator - %s\n N=%i dt=%.2f t=%.2f t_max=%.2f' % (simulation, N, dt, t, t_max))

    scatter = ax.scatter(pos_t[:,0,i], pos_t[:,1,i], pos_t[:,2,i], s=1, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(np.min(pos_t[:,0,:]),np.max(pos_t[:,0,:]))
    ax.set_ylim(np.min(pos_t[:,1,:]),np.max(pos_t[:,1,:]))
    ax.set_zlim(np.min(pos_t[:,2,:]),np.max(pos_t[:,2,:]))

    return fig, scatter, title

def draw(i, title, fps, scatter, pos_t, N, dt, t_max, simulation):
    t = i / fps
    title.set_text('Gravity Simulator - %s \nN=%i dt=%.2f t=%.2fs t_max=%.2fs' % (simulation, N, dt, t, t_max))

    scatter._offsets3d = (pos_t[:,0,i], pos_t[:,1,i], pos_t[:,2,i])

# ---------------------------------
# Load stats
stats = pickle.load(open('stats.p', 'rb'))

# ---------------------------------
# Run simulation
if args.command in ['run', 'profile', 'validate']:
    # ---------------------------------
    # Initalise parameters
    dt = args.dt
    t_max = args.t_max

    G = 1 # 6.6743 * 10**(-11) # m^3/(kg*s^2)    # Gravitational Constant - G =
    soft_param = 1e-5    # softening parameter, what should this value be?
    threads = args.threads

    # pos_offset = 20000 # TODO remove

    valid_pos_t = None
    if args.command == 'validate':
        pos, vel, mass = initialise_environment(args.seed, args.N[0])
        _, valid_pos_t = run_simulation(threads, args.validation_simulation, pos, mass, vel, G, args.N[0], dt, t_max, soft_param)

    for simulation in args.simulation:
        if simulation not in stats:
            stats[simulation] = []

        stat = {
            'date': datetime.datetime.now(),
            'dt': dt, 't_max': t_max, 'soft_param': soft_param, 'threads': threads,
            'runs': {'N': [], 'duration': []},
        }

        for N in args.N:
            durations = []
            pos_t = None

            runs = 1
            if 'average_over' in args:
                runs = args.average_over

            for i in range(runs):
                pos, vel, mass = initialise_environment(args.seed, N)

                duration, pos_t = run_simulation(threads, simulation, pos, mass, vel, G, N, dt, t_max, soft_param)
                durations.append(duration)

                if args.command == 'validate':
                    if np.allclose(valid_pos_t, pos_t):
                        print('VALID against %s for %i' % (args.validation_simulation, args.N[0]))
                    else:
                        # for i in range(len(valid_pos_t)):
                        #     for j in range(len(valid_pos_t[i])):
                        #         for k in range(len(valid_pos_t[i,j])):
                        #             if not np.allclose(valid_pos_t[i,j,k], pos_t[i,j,k]):
                        #                 print(i,j,k, valid_pos_t.dtype, valid_pos_t[i,j,k])
                        #                 print(i,j,k, pos_t.dtype, pos_t[i,j,k])
                        #                 break
                        # print(valid_pos_t[-1])
                        # print(pos_t[-1])
                        print('*** INVALID against %s for %i' % (args.validation_simulation, args.N[0]))

            duration = np.average(durations)

            print('Executed %s simulation %i times with N=%i in average %.2fs' % (simulation,runs,  N, duration))
            stat['runs']['N'].append(N)
            stat['runs']['duration'].append(duration)

            # Draw graph
            fps = (pos_t.shape[2]-1) / t_max

            if 'plot_start' in args and args.plot_start:
                plot_at_index('Gravity Simulator - %s - Start' % simulation, 0, fps, pos_t, N, dt, t_max, simulation)

            if 'plot_end' in args and args.plot_end:
                plot_at_index('Gravity Simulator - %s - End' % simulation, pos_t.shape[2]-1, fps, pos_t, N, dt, t_max, simulation)

            if 'animate' in args and args.animate:
                fig, scatter, title = plot_at_index('Gravity Simulator - %s - Animated' % simulation, 0, fps, pos_t, N, dt, t_max, simulation)
                ani = matplotlib.animation.FuncAnimation(fig, draw, frames=pos_t.shape[2], fargs=(title, fps, scatter, pos_t, N, dt, t_max, simulation), interval=round(1000 / fps), repeat=False)

            plt.show()

        stats[simulation].append(stat)

# ---------------------------------
# Write stats
if args.command == 'profile':
    print('Saving to stats.p')
    pickle.dump(stats, open('stats.p', 'wb'))

# ---------------------------------

if args.command in ['load']:
    data = pickle.load(open(args.path, 'rb'))
    pos_t = data['pos_t']

    fps = (pos_t.shape[2]-1) / data['t_max']

    fig, scatter, title = plot_at_index('Gravity Simulator - %s - Animated' % data['simulation'], 0, fps, pos_t, data['N'], data['dt'], data['t_max'], data['simulation'])
    ani = matplotlib.animation.FuncAnimation(fig, draw, frames=pos_t.shape[2], fargs=(title, fps, scatter, pos_t, data['N'], data['dt'], data['t_max'], data['simulation']), interval=round(1000 / fps), repeat=False)

    plt.show()

if args.command in ['stats', 'profile']:
    for simulation_name, simulation_stats in stats.items():
        last_run = simulation_stats[-1]
        plt.plot(last_run['runs']['N'], last_run['runs']['duration'], marker='o', label=simulation_name)

    plt.xlabel('N')
    plt.ylabel('Duratation (s)')
    plt.title('Stats')
    plt.legend()
    plt.show()
