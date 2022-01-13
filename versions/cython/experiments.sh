#!/usr/bin/env bash



cd /app

# Compile
python3 setup.py build_ext --inplace

echo "** Experiments **"

# mpirun -n 1 python3 main.py run --simulation python_mpi --N 60
mpirun -n 2 python3 main.py run --simulation python_mpi --N 120
mpirun -n 4 python3 main.py run --simulation python_mpi --N 120
# mpirun -n 6 python3 main.py run --simulation python_mpi --N 60
# mpirun -n 2 --oversubscribe python3 main.py run --simulation python_mpi --N 12
# mpirun -n 4 --oversubscribe python3 main.py run --simulation python_mpi --N 12
# mpirun -n 6 --oversubscribe python3 main.py run --simulation python_mpi --N 12

# Run
# python3 main.py run --simulation cython_openmp --N 800 --t_max 10.0 --threads 1
# python3 main.py run --simulation cython_openmp --N 800 --t_max 10.0 --threads 2
# python3 main.py run --simulation cython_openmp --N 800 --t_max 10.0 --threads 4
# python3 main.py run --simulation cython_openmp --N 800 --t_max 10.0 --threads 8
# python3 main.py run --simulation cython_openmp --N 800 --t_max 10.0 --threads 16

echo "** Experiments Done **"
