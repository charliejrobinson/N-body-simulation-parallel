#!/usr/bin/env bash



cd /app

# Compile
python3 setup.py build_ext --inplace

echo "*** Experiments ***"

echo "***** Serial vs OpenMP vs MPI runs (Using 8 processors / threads) (Using static scheduling)"
python3 main.py profile --simulation cython_without_numpy --N 10 20 50 100 200 400 600 800 1000 1500 2000 5000 --experiment 1
python3 main.py profile --simulation cython_openmp --N 10 20 50 100 200 400 600 800 1000 1500 2000 5000 --threads 8 --schedule guided --experiment 1
mpirun -n 8 -bind-to hwthread python3 main.py profile --simulation cython_mpi --N 10 20 50 100 200 400 600 800 1000 1500 2000 5000 --experiment 1

echo "***** OpenMP vs MPI for threads / processes"
for i in {0..24..2}; do
   mpirun -n $i -bind-to hwthread python3 main.py profile --simulation cython_mpi --N 2000 --experiment 2;
   python3 main.py profile --simulation cython_openmp --N 2000 --threads $i --schedule guided --experiment 2;
done

echo "***** OpenMP thread scaling"
for i in {0..24..2}; do
   python3 main.py profile --simulation cython_openmp --N 100 250 1000 2000 --threads $i --experiment 3;
done

echo "***** OpenMP chunk size"
for i in {1..20..1}; do
   python3 main.py profile --simulation cython_openmp --N 200 --threads 8 --chunks $i --schedule dynamic --experiment 4;
   python3 main.py profile --simulation cython_openmp --N 200 --threads 6 --chunks $i --schedule dynamic --experiment 4;
   python3 main.py profile --simulation cython_openmp --N 100 --threads 8 --chunks $i --schedule dynamic --experiment 4;
done

echo "***** OpenMP schedule"
python3 main.py profile --simulation cython_openmp --N 100 250 1000 2000 5000 --threads 8 --schedule static --experiment 5;
python3 main.py profile --simulation cython_openmp --N 100 250 1000 2000 5000 --threads 8 --schedule dynamic --experiment 5;
python3 main.py profile --simulation cython_openmp --N 100 250 1000 2000 5000 --threads 8 --schedule guided --experiment 5;

echo "***** MPI block vs ring"
for i in {0..24..2}; do
   mpirun -n $i -bind-to hwthread python3 main.py profile --simulation cython_mpi --N 100 250 1000 2000 --experiment 6;
   mpirun -n $i -bind-to hwthread python3 main.py profile --simulation cython_mpi_ring --N 100 250 1000 2000 --experiment 6;
done

echo "*** Experiments Done ***"
