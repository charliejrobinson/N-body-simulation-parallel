import os
import platform
import pathlib

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI

import simulation_python_gpu
import simulation_cython_without_numpy
import simulation_cython_openmp_static
import simulation_cython_openmp_dynamic
import simulation_cython_openmp_guided
import simulation_cython_mpi
import simulation_cython_mpi_ring

import horizons_data
import datetime

import openmp_api_wraper

import numpy as np
import pickle
import matplotlib
import time
import sys
import datetime

if platform.system() == 'Darwin':
    matplotlib.use('MACOSX')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import timeit
import argparse

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
is_master = rank == 0

simulations = ['python_gpu', 'cython_without_numpy', 'cython_openmp', 'cython_mpi', 'cython_mpi_ring']

parser = argparse.ArgumentParser(description='Gravity Simulator')

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument('--experiment', type=str, default='none', help='experiment')
parent_parser.add_argument('--seed', type=int, default=17, help='random seed to use')
parent_parser.add_argument('--threads', type=int, default=1, help='number of threads to use for parallel code')
parent_parser.add_argument('--schedule', type=str, default='static', help='schedule to use for parallel code')
parent_parser.add_argument('--chunks', type=int, default=1, help='number of chunks to use for parallel code')
parent_parser.add_argument('--delta_t', type=float, default=0.01, help='timestep')
parent_parser.add_argument('--t_max', type=float, default=10.0, help='how many seconds simulation runs for')
parent_parser.add_argument('--simulation', type=str, nargs='*', default=['cython_without_numpy'], choices=simulations, help='Which simulation to use')

subparsers = parser.add_subparsers(dest='command')

parser_run = subparsers.add_parser('run', help='Run simulations', parents=[parent_parser])
parser_run.add_argument('--bodies', type=str, nargs='*', default=[], help='List of NAIF IDs to simulate from NASA Horizons or "solar_system" for all planets and moons')
parser_run.add_argument('--date', type=datetime.date.fromisoformat, default=datetime.date.today(), help='Date to use for Horizons data')
parser_run.add_argument('--N', type=int, nargs='+', default=[50], help='number of particles')
parser_run.add_argument('--animate', action='store_true', help='plot animated graphs')
parser_run.add_argument('--plot_start', action='store_true', help='plots start graph')
parser_run.add_argument('--plot_end', action='store_true', help='plots end graph')
parser_run.add_argument('--save', action='store_true', help='saves the output to a file')

parser_profile = subparsers.add_parser('profile', help='Record stats for simulations', parents=[parent_parser])
parser_profile.add_argument('--N', type=int, nargs='+', default=[3, 50, 100, 200, 400, 600, 800], help='number of particles')
parser_profile.add_argument('--average_over', type=int, default=3, help='Number of runs to average over')

parser_validate = subparsers.add_parser('validate', help='Validate a simulation is correct', parents=[parent_parser])
parser_validate.add_argument('--N', type=int, nargs='+', default=[12], help='number of particles')
parser_validate.add_argument('--validation_simulation', type=str, default='python_original', choices=simulations, help='Which simulation to use for validation')
parser_validate.add_argument('--print', action='store_true', help='Print results to compare')

parser_load = subparsers.add_parser('load', help='Load a positions file')
parser_load.add_argument('path', type=str, nargs='+', help='File path')
parser_load.add_argument('--compare', action='store_true', help='Checks results are equal to one another')
parser_load.add_argument('--animate', action='store_true', help='Animate the results')

parser_stats = subparsers.add_parser('stats', help='Plot statistics')

args = parser.parse_args()

'''Passes the arguments to the simulation function'''
def run_simulation(schedule, chunk_size, threads, simulation, positions, masses, velocities, G, N, delta_t, t_max, softening_parameter):
    start = openmp_api_wraper.get_wtime()

    positions_over_time = None
    if simulation == 'python_gpu':
        positions_over_time = simulation_python_gpu.simulate(positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)
    elif simulation == 'cython_without_numpy':
        positions_over_time = simulation_cython_without_numpy.simulate(positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)
    elif simulation == 'cython_openmp':
        if schedule == 'static':
            positions_over_time = simulation_cython_openmp_static.simulate(schedule, chunk_size, threads, positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)
        elif schedule == 'dynamic':
            positions_over_time = simulation_cython_openmp_dynamic.simulate(schedule, chunk_size, threads, positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)
        elif schedule == 'guided':
            positions_over_time = simulation_cython_openmp_guided.simulate(schedule, chunk_size, threads, positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)
    elif simulation == 'cython_mpi':
        positions_over_time = simulation_cython_mpi.simulate(positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)
    elif simulation == 'cython_mpi_ring':
        positions_over_time = simulation_cython_mpi_ring.simulate(positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)

    end = openmp_api_wraper.get_wtime()

    return end-start, positions_over_time

'''Initalises randomly normally distributed point like particles with unit masses, arbitrary units'''
def initialise_environment(seed, N):
    np.random.seed(seed)

    positions  = np.random.randn(N,3).astype(np.double) # normally distributed positions
    velocities  = np.random.randn(N,3).astype(np.double) # normally distributed velocities
    masses = np.ones((N,1)).astype(np.double) # particle masses is 1.0

    velocities -= np.mean(masses * velocities, 0) / np.mean(masses) # convert to Center-of-masses frame

    G = 1 # Gravitational Constant

    return positions, velocities, masses, G

'''Initialises the environment based on NASA Horizons data'''
def initialise_environment_bodies(bodies, date):
    if 'solar_system' in bodies:
        bodies = None

    masses, positions, velocities = horizons_data.get_bodies(bodies, date=str(date))

    G = 6.6743 * 10**(-11) # m^3/(kg*s^2)

    return positions, velocities, masses, G

'''Saves timing data for later plotting'''
def save(simulation, N, positions_over_time, t_max, delta_t):
    data = {
        'simulation': simulation,
        'N': N,
        'positions_over_time': positions_over_time,
        't_max': t_max,
        'delta_t': delta_t
    }

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = 'data_%s_%s.p' % (simulation, date)

    pickle.dump(data, open(path, 'wb'))

    print('Saved to %s' % path)

'''Plots the results of the simulation'''
def plot_at_index(name, i, fps, positions_over_time, N, delta_t, t_max, simulation):
    t = i / fps

    fig = plt.figure(name, dpi=100)
    ax = fig.add_subplot(projection='3d')
    title = ax.set_title('Gravity Simulator - %s\n N=%i delta_t=%.2f t=%.2f t_max=%.2f' % (simulation, N, delta_t, t, t_max))

    scatter = ax.scatter(positions_over_time[:,0,i], positions_over_time[:,1,i], positions_over_time[:,2,i], s=1, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(np.min(positions_over_time[:,0,:]),np.max(positions_over_time[:,0,:]))
    ax.set_ylim(np.min(positions_over_time[:,1,:]),np.max(positions_over_time[:,1,:]))
    ax.set_zlim(np.min(positions_over_time[:,2,:]),np.max(positions_over_time[:,2,:]))

    return fig, scatter, title

'''Animates the results of the simulation'''
def draw(i, title, fps, scatter, positions_over_time, N, delta_t, t_max, simulation):
    t = i / fps
    title.set_text('Gravity Simulator - %s \nN=%i delta_t=%.2f t=%.2fs t_max=%.2fs' % (simulation, N, delta_t, t, t_max))

    scatter._offsets3d = (positions_over_time[:,0,i], positions_over_time[:,1,i], positions_over_time[:,2,i])

# ---------------------------------
# Load stats
stats = {}
stats_path = 'stats.p'
if os.path.exists(stats_path):
    stats = pickle.load(open(stats_path, 'rb'))

# ---------------------------------
# Run simulation
if args.command in ['run', 'profile', 'validate']:
    delta_t = args.delta_t
    t_max = args.t_max

    G = 1 # Gravitational Constant, this is overridden by the initialisation function
    softening_parameter = 1e-5    # Softening parameter
    threads = args.threads
    schedule = args.schedule
    chunk_size = args.chunks

    experiment = args.experiment # Experiment name used for saving data

    # Validates simulation by comparing the results of the simulation to known result
    valid_pos_t = None
    if args.command == 'validate' and is_master:
        positions, velocities, masses, G = initialise_environment(args.seed, args.N[0])
        _, valid_pos_t = run_simulation(schedule, chunk_size, threads, args.validation_simulation, positions, masses, velocities, G, args.N[0], delta_t, t_max, softening_parameter)

    for simulation in args.simulation:
        if experiment not in stats:
            stats[experiment] = {}
        if simulation not in stats[experiment]:
            stats[experiment][simulation] = []

        for N in args.N:
            stat = {
                'date': datetime.datetime.now(),
                'delta_t': delta_t, 't_max': t_max, 'softening_parameter': softening_parameter,
                'schedule': schedule,
                'chunk_size': chunk_size,
                'N': N,
            }

            durations = []
            positions_over_time = None

            runs = 1
            if 'average_over' in args:
                runs = args.average_over

            for i in range(runs):

                # Initialise environment
                positions, velocities, masses, G = initialise_environment(args.seed, N)
                if args.bodies and len(args.bodies):
                    print('** --bodies set, overriding --N')
                    positions, velocities, masses, G = initialise_environment_bodies(args.bodies, args.date)
                    N = masses.shape[0]

                # Run simulation
                duration, positions_over_time = run_simulation(schedule, chunk_size, threads, simulation, positions, masses, velocities, G, N, delta_t, t_max, softening_parameter)
                durations.append(duration)

                if args.command == 'validate' and is_master:
                    if np.allclose(valid_pos_t, positions_over_time):
                        print('VALID against %s for %i' % (args.validation_simulation, args.N[0]))
                    else:
                        print('*** INVALID against %s for %i' % (args.validation_simulation, args.N[0]))

                        if args.print:
                            print(valid_pos_t[:,:,-1])
                            print(positions_over_time[:,:,-1])

            duration = np.mean(durations)
            duration_low = np.min(durations)
            duration_high = np.max(durations)

            # Print summary
            if is_master:
                extra = ''
                if runs > 1:
                    extra += '[error %.4fs - %.4fs]' % (duration_low, duration_high)

                if 'mpi' in simulation:
                    extra += '[%i processes]' % size
                    stat['procs'] = size
                elif 'openmp' in simulation:
                    extra += '[%i threads] [%s schedule] [%i chunk size]' % (threads, schedule, chunk_size)
                    stat['procs'] = threads

                print('Executed %s simulation %i times with N=%i in average %.4fs %s' % (simulation,runs,  N, duration, extra))
            else:
                continue

            stat['duration'] = duration
            stat['duration_low'] = duration_low
            stat['duration_high'] = duration_high

            stats[experiment][simulation].append(stat)

            if 'save' in args and args.save and is_master:
                save(simulation, N, positions_over_time, t_max, delta_t)

            # Draw graph
            fps = (positions_over_time.shape[2]-1) / t_max

            if 'plot_start' in args and args.plot_start and is_master:
                plot_at_index('Gravity Simulator - %s - Start' % simulation, 0, fps, positions_over_time, N, delta_t, t_max, simulation)

            if 'plot_end' in args and args.plot_end and is_master:
                plot_at_index('Gravity Simulator - %s - End' % simulation, positions_over_time.shape[2]-1, fps, positions_over_time, N, delta_t, t_max, simulation)

            if 'animate' in args and args.animate and is_master:
                fig, scatter, title = plot_at_index('Gravity Simulator - %s - Animated' % simulation, 0, fps, positions_over_time, N, delta_t, t_max, simulation)
                ani = matplotlib.animation.FuncAnimation(fig, draw, frames=positions_over_time.shape[2], fargs=(title, fps, scatter, positions_over_time, N, delta_t, t_max, simulation), interval=round(1000 / fps), repeat=False)

            plt.show()

# ---------------------------------
# Write stats
if args.command == 'profile' and is_master:
    print('Saving to stats.p')
    pickle.dump(stats, open('stats.p', 'wb'))

# ---------------------------------
# Load results for plotting
if args.command in ['load']:
    compare = None
    for path in args.path:
        data = pickle.load(open(path, 'rb'))
        positions_over_time = data['positions_over_time']

        if args.compare and compare is not None:
            if np.allclose(positions_over_time, compare):
                print('EQUAL')
            else:
                print('NOT EQUAL')

        compare = positions_over_time

        if args.animate:
            fps = (positions_over_time.shape[2]-1) / data['t_max']

            fig, scatter, title = plot_at_index('Gravity Simulator - %s - Animated' % data['simulation'], 0, fps, positions_over_time, data['N'], data['delta_t'], data['t_max'], data['simulation'])
            ani = matplotlib.animation.FuncAnimation(fig, draw, frames=positions_over_time.shape[2], fargs=(title, fps, scatter, positions_over_time, data['N'], data['delta_t'], data['t_max'], data['simulation']), interval=round(1000 / fps), repeat=False)

            plt.show()

# Plot stats
if args.command in ['stats'] and is_master:
    for simulation_name, simulation_stats in stats.items():
        last_run = simulation_stats[-1]
        plt.plot(last_run['runs']['N'], last_run['runs']['duration'], marker='o', label=simulation_name)

    plt.xlabel('N')
    plt.ylabel('Duratation (s)')
    plt.title('Stats')
    plt.legend()
    plt.show()
