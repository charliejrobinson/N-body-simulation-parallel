import platform

import numpy as np
import pickle
import matplotlib

if platform.system() == 'Darwin':
    matplotlib.use('MACOSX')

import matplotlib.pyplot as plt

simulations = ['python', 'cython', 'python_original', 'python_sqrt', 'python_without_numpy', 'cython_without_numpy', 'cython_openmp', 'python_mpi', 'cython_mpi', 'python_mpi_ring', 'cython_mpi_ring', 'cython_openmp_2']

# Load stats
stats1 = pickle.load(open('stats_1.p', 'rb'))
stats2 = pickle.load(open('stats_2.p', 'rb'))
# stats3 = pickle.load(open('stats_4.p', 'rb'))
stats5 = pickle.load(open('stats5.p', 'rb'))
stats6 = pickle.load(open('stats6.p', 'rb'))

stats1['4'] = stats2['4']
stats1['5'] = stats2['5']
stats1['6'] = stats2['6']

# stats1['4'] = stats5['4'] # TODO
# stats1['2'] = stats6['2'] # TODO

stats1['1']['python_gpu'] = [
    {'N': 10, 'duration': 0.936, 'duration_low': 0.936, 'duration_high': 0.936 },
    {'N': 20, 'duration':0.936, 'duration_low':0.936, 'duration_high':0.936 },
    {'N': 50, 'duration':0.936, 'duration_low':0.936, 'duration_high':0.936 },
    {'N': 100, 'duration':0.936, 'duration_low':0.936, 'duration_high':0.936 },
    {'N': 200, 'duration':0.936, 'duration_low':0.936, 'duration_high':0.936 },
    {'N': 400, 'duration':0.936, 'duration_low':0.936, 'duration_high':0.936 },
    {'N': 600, 'duration':0.936, 'duration_low':0.936, 'duration_high':0.936 },
    {'N': 800, 'duration':1.558, 'duration_low':1.558, 'duration_high':1.558 },
    {'N': 1000, 'duration':2.714, 'duration_low':2.714, 'duration_high':2.714 },
    {'N': 1500, 'duration':4.239, 'duration_low':4.239, 'duration_high':4.239 },
    {'N': 2000, 'duration':9.572, 'duration_low':9.572, 'duration_high':9.572 },
    {'N': 5000, 'duration':60.883, 'duration_low':60.883, 'duration_high':60.883 }
]

# stats1['4'] = stats3['4']

# stats1.update(stats2)

print('Saving to stats_merged.p')
pickle.dump(stats1, open('stats_merged_2.p', 'wb'))
