from mpi4py import MPI
import numpy as np

def do_work(array, rank):
    array = array + rank + 1
    return array

def main():
    # comm = MPI.COMM_WORLD
    comm = MPI.Comm.Get_parent()
    size = comm.Get_size()
    rank = comm.Get_rank()

    comm = MPI.Comm.Get_parent()

    params = None
    comm.bcast(params, root=0)

    comm.Disconnect()

    print(rank, params)

    return

    pos = np.zeros((N, 2), dtype='d')

    # comm.Bcast(pos, root=0)

    count = int(np.ceil(N / size))

    for i in range(count):
        working_pos = np.array_split(pos, size)[rank]

        working_pos = do_work(working_pos, rank)

        recvbuf = np.empty([size, count, 2], dtype='d')

        comm.Allgather(working_pos, recvbuf)

        pos = recvbuf.reshape((N, 2))

    # if rank == 0:
    #     print(pos)

    return pos

main

# main(5)
# print('rank: %i' % rank)
