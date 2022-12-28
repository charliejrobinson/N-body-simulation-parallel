import platform

import numpy as np
import pickle
import matplotlib

if platform.system() == 'Darwin':
    matplotlib.use('MACOSX')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load stats
stats = pickle.load(open('stats_merged_2.p', 'rb'))

colours = list(mcolors.BASE_COLORS)

def standard_deviation(run_stats):
    return [np.std([stat['duration'], stat['duration_low'], stat['duration_high']]) for stat in run_stats]

serial_duration = {
    '50': [stat['duration'] for stat in stats['1']['cython_without_numpy'] if stat['N'] == 100][0],
    '100': [stat['duration'] for stat in stats['1']['cython_without_numpy'] if stat['N'] == 100][0],
    '250': 0.6,
    '1000': [stat['duration'] for stat in stats['1']['cython_without_numpy'] if stat['N'] == 1000][0],
    '2000': [stat['duration'] for stat in stats['1']['cython_without_numpy'] if stat['N'] == 2000][0],
}

# ---------------------------- 1

fig, ax1 = plt.subplots()
# fig.suptitle('Serial vs OpenMP vs MPI runs (Using 8 processors / threads)')
ax1.set_xlabel('N')
ax1.set_ylabel('Time')
ax1.spines['top'].set_visible(False)
ax1.set_yticklabels([])
ax1.set_xlim(0, 400)
ax1.set_ylim(-0.05, 1.2)
ax1.set_xticklabels([0, 50, 100, 150, 200, 250, 300, 350, 400])


# ax2 = ax1.twinx()
# ax2.set_ylabel('η')
# ax2.spines['top'].set_visible(False)

x1 = [stat['N'] for stat in stats['1']['cython_without_numpy']]

y1 = [stat['duration'] for stat in stats['1']['cython_without_numpy']]
y1_error = standard_deviation(stats['1']['cython_without_numpy'])

y2 = [stat['duration'] for stat in stats['1']['cython_openmp']]
y2_error = standard_deviation(stats['1']['cython_openmp'])
y2_efficiency = np.array(y1) / np.array(y2)

y3 = [stat['duration'] for stat in stats['1']['cython_mpi']]
y3_error = standard_deviation(stats['1']['cython_mpi'])
y3_efficiency = np.array(y1) / np.array(y3)

y4 = [stat['duration'] for stat in stats['1']['python_gpu']]
y4_error = standard_deviation(stats['1']['python_gpu'])
y4_efficiency = np.array(y1) / np.array(y4)

ax1.errorbar(x1, y1, yerr=y1_error, label='Serial')
ax1.errorbar(x1, y2, yerr=y2_error, label='OpenMP', color=colours[1])
ax1.errorbar(x1, y3, yerr=y3_error, label='MPI', color=colours[2])
ax1.errorbar(x1, y4, yerr=y4_error, label='GPU', color=colours[3])

# ax2.errorbar(x1, y2_efficiency, yerr=None, fmt='--', color=colours[1])
# ax2.errorbar(x1, y3_efficiency, yerr=None, fmt='--', color=colours[2])
# ax2.errorbar(x1, y4_efficiency, yerr=None, fmt='--', color=colours[3])

fig.legend(loc='upper left')
fig.tight_layout()

# ---------------------------- 2

fig, ax1 = plt.subplots()
# fig.suptitle('OpenMP vs MPI for threads / processes')
ax1.set_xlabel('Threads / Processes')
ax1.set_ylabel('Time')
ax1.spines['top'].set_visible(False)
ax1.set_yticklabels([])

ax2 = ax1.twinx()
ax2.set_ylabel('η')
ax2.spines['top'].set_visible(False)

x1 = np.unique([stat['procs'] for stat in stats['2']['cython_mpi']])


N = 2000

y1 = [stat['duration'] for stat in stats['2']['cython_mpi'] if stat['N'] == N]
y1_error = standard_deviation([stat for stat in stats['2']['cython_mpi'] if stat['N'] == N])
y1_efficiency = serial_duration[str(N)] / np.array(y1)

y2 = [stat['duration'] for stat in stats['2']['cython_openmp'] if stat['N'] == N]
y2_error = standard_deviation([stat for stat in stats['2']['cython_openmp'] if stat['N'] == N])
y2_efficiency = serial_duration[str(N)] / np.array(y2)

ax1.errorbar(x1, y1, yerr=y1_error, label='MPI')
ax1.errorbar(x1, y2, yerr=y2_error, label='OpenMP')

ax2.errorbar(x1, y1_efficiency, yerr=None, fmt='--')
ax2.errorbar(x1, y2_efficiency, yerr=None, fmt='--')

fig.legend()
fig.tight_layout()

# ---------------------------- 3

fig, ax1 = plt.subplots()
# fig.suptitle('OpenMP thread scaling')
ax1.set_xlabel('Processes')
ax1.set_ylabel('η')
ax1.spines['top'].set_visible(False)

x1 = np.unique([stat['procs'] for stat in stats['3']['cython_openmp']])
Ns = np.unique([stat['N'] for stat in stats['3']['cython_openmp']])

for N in Ns:
    # if N in [1000, 2000, 5000]: continue
    y = [stat['duration'] for stat in stats['3']['cython_openmp'] if stat['N'] == N]
    y_error = standard_deviation([stat for stat in stats['3']['cython_openmp'] if stat['N'] == N])
    y_efficiency = y[0] / np.array(y)
    # ax1.errorbar(x1, y, yerr=y_error, label='N=%i' % N)
    ax1.errorbar(x1, y_efficiency, yerr=y_error, label='N=%i' % N)

fig.legend(loc='upper left')

# ---------------------------- 4
fig, ax1 = plt.subplots()
# fig.suptitle('OpenMP chunk size')
ax1.set_xlabel('Chunk size')
ax1.set_ylabel('Time')
ax1.spines['top'].set_visible(False)
ax1.set_yticklabels([])

ax2 = ax1.twinx()
ax2.set_ylabel('η')
ax2.spines['top'].set_visible(False)

procs = np.unique([stat['procs'] for stat in stats['4']['cython_openmp']])
x1 = np.unique([stat['chunk_size'] for stat in stats['4']['cython_openmp']])
Ns = np.unique([stat['N'] for stat in stats['4']['cython_openmp']])

for p in procs:
    if p == 6: continue

    for N in Ns:
        y_0 = 1 # TODO

        # TODO
        if N == 100: y_0 = 0.0927
        if N == 200: y_0 = 0.4024

        y = [stat['duration'] for stat in stats['4']['cython_openmp'] if stat['N'] == N and stat['procs'] == p]
        y_error = standard_deviation([stat for stat in stats['4']['cython_openmp'] if stat['N'] == N and stat['procs'] == p])
        y_efficiency = y_0 / np.array(y)
        ax1.errorbar(x1, y, yerr=y_error, label='N=%i' % N)
        ax2.errorbar(x1, y_efficiency, yerr=None, label='N=%i' % N, fmt='--')

fig.legend()

# ---------------------------- 5
plt.figure()
# plt.title('OpenMP schedule')
plt.xlabel('N')
plt.ylabel('Time')

# x1 = np.unique([stat['chunk_size'] for stat in stats['5']['cython_openmp']])
x1 = np.unique([stat['N'] for stat in stats['5']['cython_openmp']])

y1 = [stat['duration'] for stat in stats['5']['cython_openmp'] if stat['schedule'] == 'static']
y1_error = standard_deviation([stat for stat in stats['5']['cython_openmp'] if stat['schedule'] == 'static'])
y2 = [stat['duration'] for stat in stats['5']['cython_openmp'] if stat['schedule'] == 'dynamic']
y2_error = standard_deviation([stat for stat in stats['5']['cython_openmp'] if stat['schedule'] == 'dynamic'])
y3 = [stat['duration'] for stat in stats['5']['cython_openmp'] if stat['schedule'] == 'guided']
y3_error = standard_deviation([stat for stat in stats['5']['cython_openmp'] if stat['schedule'] == 'guided'])

plt.errorbar(x1, y1, yerr=y1_error, label='static')
plt.errorbar(x1, y2, yerr=y2_error, label='dynamic')
plt.errorbar(x1, y3, yerr=y3_error, label='guided')

plt.legend()

# ---------------------------- 6 1

fig, ax1 = plt.subplots()
# fig.suptitle('MPI block scaling')
ax1.set_xlabel('Processes')
ax1.set_ylabel('η')
ax1.spines['top'].set_visible(False)

x1 = np.unique([stat['procs'] for stat in stats['6']['cython_mpi']])
Ns = np.unique([stat['N'] for stat in stats['6']['cython_mpi']])

for i, N in enumerate(Ns):
    #if N in [100, 250]: continue
    y1 = [stat['duration'] for stat in stats['6']['cython_mpi'] if stat['N'] == N]
    y1_error = standard_deviation([stat for stat in stats['6']['cython_mpi'] if stat['N'] == N])
    # y1_efficiency = y1[0] / np.array(y1)
    y1_efficiency = serial_duration[str(N)] / np.array(y1)

    colour = colours[i]
    # ax1.errorbar(x1, y1, yerr=y1_error, label='block N=%i' % N, c=colour)
    ax1.errorbar(x1, y1_efficiency, yerr=None, label='N=%i' % N, c=colour)

fig.legend(loc='upper left')

# ---------------------------- 6 2

fig, ax1 = plt.subplots()
# fig.suptitle('MPI block vs ring')
ax1.set_xlabel('Processes')
ax1.set_ylabel('Time')
ax1.spines['top'].set_visible(False)
ax1.set_yticklabels([])

ax2 = ax1.twinx()
ax2.set_ylabel('η')
ax2.spines['top'].set_visible(False)

x1 = np.unique([stat['procs'] for stat in stats['6']['cython_mpi']])
Ns = np.unique([stat['N'] for stat in stats['6']['cython_mpi']])

for i, N in enumerate(Ns):
    if N in [100, 250, 1000]: continue
    y1 = [stat['duration'] for stat in stats['6']['cython_mpi'] if stat['N'] == N]
    y1_error = standard_deviation([stat for stat in stats['6']['cython_mpi'] if stat['N'] == N])
    y1_efficiency = serial_duration[str(N)] / np.array(y1)

    y2 = [stat['duration'] for stat in stats['6']['cython_mpi_ring'] if stat['N'] == N]
    y2_error = standard_deviation([stat for stat in stats['6']['cython_mpi_ring'] if stat['N'] == N])
    y2_efficiency = serial_duration[str(N)] / np.array(y2)

    ax1.errorbar(x1, y1, yerr=y1_error, label='Collective')
    ax1.errorbar(x1, y2, yerr=y2_error, label='Ring')

    ax2.errorbar(x1, y1_efficiency, yerr=None, fmt='--')
    ax2.errorbar(x1, y2_efficiency, yerr=None, fmt='--')

fig.legend()

plt.show()
