#======================= # run_picalc_pyx_omp . py #=======================
import os

threads = 2

os.environ ["MKL_NUM_THREADS"] = str(threads)
os.environ ["NUMEXPR_NUM_THREADS"] = str(threads)
os.environ ["OMP_NUM_THREADS"] = str(threads)

from hellocombi import printinfo
from mpi4py import MPI

print('hello')

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

printinfo(rank, size, name)
