#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
cdef int size = comm.Get_size()
cdef int rank = comm.Get_rank()
cdef int is_master = rank == 0

cdef calc_acc(int index_from, int index_to, double[:,:] acc, float G, double[:,:] pos, double[:,:] mass, float soft_param):
    cdef Py_ssize_t i, j
    cdef double x1, y1, z1, x2, yz, z2
    cdef int N = np.array(pos).shape[0]
    cdef double inv_sep
    cdef double dx, dy, dz

    for i in range(index_from, index_to):
        acc[i,0] = 0
        acc[i,1] = 0
        acc[i,2] = 0

        for j in range(N):
            # calculate particle seperations
            dx = pos[j,0] - pos[i,0]
            dy = pos[j,1] - pos[i,1]
            dz = pos[j,2] - pos[i,2]

            # matrix of inverse seperations cubed (1/r^3)
            inv_sep = (dx**2 + dy**2 + dz**2 + soft_param**2)**(-1.5)

            # calculate acceleration components
            acc[i,0] += G * (dx * inv_sep) * mass[j][0]
            acc[i,1] += G * (dy * inv_sep) * mass[j][0]
            acc[i,2] += G * (dz * inv_sep) * mass[j][0]

cpdef simulate(double[:,:] pos, double[:,:] mass, double[:,:] vel, float G, int N, float dt, float t_max, float soft_param):
    params = None
    pos_t = None

    cdef double[:,:] acc

    if is_master:
        acc = np.zeros((N, 3))
        calc_acc(0, N, acc, G, pos, mass, soft_param)

        params = {'pos': np.array(pos), 'vel': np.array(vel), 'mass': np.array(mass), 'acc': np.array(acc)}

        steps = int(np.ceil(t_max / dt))
        pos_t = np.zeros((N,3,steps+1))
        pos_t[:,:,0] = pos

    # Broadcast / Recive enviroment
    params = comm.bcast(params, root=0)
    # TODO scatter with cyclic_mpi_t vel and pos, mass is global

    # Unpack environment
    pos = params['pos']
    vel = params['vel']
    mass = params['mass']
    acc = params['acc']

    cdef float t = 0

    steps = int(np.ceil(t_max/dt))
    cdef int count = int(np.ceil(N / size))

    cdef int width = np.floor(N / size)
    cdef int width_padded = width + (N % size)
    cdef int index_from = width * rank
    cdef int index_to   = width * (rank + 1)
    cdef int index_to_padded = index_to + (N % size)

    cdef Py_ssize_t x, i, j

    cdef double[:,:,:] _recvbuf

    for x in range(steps):
        for i in range(index_from, index_to_padded):
            # first kick
            vel[i,0] += acc[i,0] * (dt/2.0)
            vel[i,1] += acc[i,1] * (dt/2.0)
            vel[i,2] += acc[i,2] * (dt/2.0)

            # drift
            pos[i,0] += vel[i,0] * dt
            pos[i,1] += vel[i,1] * dt
            pos[i,2] += vel[i,2] * dt

        recvbuf = np.empty([size, width_padded, 3])
        comm.Allgather(pos[index_from:index_to_padded], recvbuf)
        _recvbuf = recvbuf

        for j in range(size):
          pos[index_from:index_to_padded] = _recvbuf[j]

        calc_acc(index_from, index_to_padded, acc, G, pos, mass, soft_param)

        for i in range(index_from, index_to_padded):
            # second kick
            vel[i,0] += acc[i,0] * (dt/2.0)
            vel[i,1] += acc[i,1] * (dt/2.0)
            vel[i,2] += acc[i,2] * (dt/2.0)

        t += dt

        if is_master:
            pos_t[:,:,x+1] = pos

    return np.array(pos_t)
