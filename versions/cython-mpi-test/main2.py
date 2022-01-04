import sys
import numpy as np

from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()

def main():
    N = 6
    pos = np.ones((N, 2), dtype='d')

    comm = MPI.COMM_SELF.Spawn(sys.executable, args=['gather_test.py'], maxprocs=1)

    params = {
        'N': N
    }

    comm.bcast(params, root=MPI.ROOT)
    comm.Disconnect()

    comm.Disconnect()


    print(pos)

main()
