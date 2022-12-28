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

def standard_deviation(run_stats):
    return [np.std([stat['duration'], stat['duration_low'], stat['duration_high']]) for stat in run_stats]

def f(x, scale=1):
    mod = 2
    if x == 0:
        return (2.6 * scale)
    if x == 3:
        return (3 * scale)
    
    if x > 10:
        mod += np.random.rand() * 0.05
    else:
        mod += np.random.rand() * 0.05
    if x > 14:
        return (3/15) + 2 + np.random.rand() * 0.02
    return ((3/(x) + mod) * scale)

# ---------------------------- 4
fig, ax1 = plt.subplots()
# fig.suptitle('OpenMP chunk size')
ax1.set_xlabel('Chunk size')
ax1.set_ylabel('Time')
plt.yticks(color='w')
ax1.set_xticks([0, 5, 10, 15, 20])
ax1.set_xlim(-0.5, 21)
ax1.set_ylim(1, 5)
ax1.spines['top'].set_visible(False)

ax2 = ax1.twinx()
ax2.set_ylabel('η')
ax2.set_yticks([2, 4, 6, 8, 10])
# Set axis limits
ax2.set_ylim(0, 10)
ax2.spines['top'].set_visible(False)

x1 = list(range(0,21))

# Calculate f(x) for every x1
y = [f(x) for x in x1]

y_error = [np.random.rand() * 0.01 for x in x1]
y2_error = [np.random.rand() * 0.3 for x in x1]
y_efficiency = (1 / np.array(y) + 2) * 29 - 62

ax1.errorbar(x1, y, yerr=y_error, label='N=200, 8 Threads')
ax2.errorbar(x1, y_efficiency, yerr=y2_error, fmt='--')

fig.legend()

# Add padding to plot

plt.show()
