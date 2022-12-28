#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
cdef int size = comm.Get_size()
cdef int rank = comm.Get_rank()
cdef int is_master = rank == 0

cdef calc_acceleration(int index_from, int index_to, double[:,:] accelerations, float G, double[:,:] positions, double[:,:] masses, float softening_parameter):
    cdef Py_ssize_t i, j
    cdef double x1, y1, z1, x2, yz, z2
    cdef int N = np.array(positions).shape[0]
    cdef double inverse_separations
    cdef double dx, dy, dz

    for i in range(index_from, index_to):
        accelerations[i,0] = 0
        accelerations[i,1] = 0
        accelerations[i,2] = 0

        for j in range(N):
            # calculate distance between particle pairs for x,y,z
            dx = positions[j,0] - positions[i,0]
            dy = positions[j,1] - positions[i,1]
            dz = positions[j,2] - positions[i,2]

            # matrix of inverse seperations cubed (1/r^3)
            inverse_separations = (dx**2 + dy**2 + dz**2 + softening_parameter**2)**(-1.5)

            # calculate accelerations for x,y,z components
            accelerations[i,0] += G * (dx * inverse_separations) * masses[j][0]
            accelerations[i,1] += G * (dy * inverse_separations) * masses[j][0]
            accelerations[i,2] += G * (dz * inverse_separations) * masses[j][0]

cpdef simulate(double[:,:] positions, double[:,:] masses, double[:,:] velocities, float G, int N, float delta_t, float t_max, float softening_parameter):
    params = None
    positions_by_timestep = None

    cdef double[:,:] accelerations

    if is_master:
        accelerations = np.zeros((N, 3))
        calc_acceleration(0, N, accelerations, G, positions, masses, softening_parameter)

        params = {'positions': np.array(positions), 'velocities': np.array(velocities), 'masses': np.array(masses), 'accelerations': np.array(accelerations)}

        steps = int(np.ceil(t_max / delta_t))
        positions_by_timestep = np.zeros((N,3,steps+1))
        positions_by_timestep[:,:,0] = positions

    # Broadcast / Recive enviroment
    params = comm.bcast(params, root=0)

    # Unpack environment
    positions = params['positions']
    velocities = params['velocities']
    masses = params['masses']
    accelerations = params['accelerations']

    cdef float t = 0

    steps = int(np.ceil(t_max/delta_t))
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
            velocities[i,0] += accelerations[i,0] * (delta_t/2.0)
            velocities[i,1] += accelerations[i,1] * (delta_t/2.0)
            velocities[i,2] += accelerations[i,2] * (delta_t/2.0)

            # drift
            positions[i,0] += velocities[i,0] * delta_t
            positions[i,1] += velocities[i,1] * delta_t
            positions[i,2] += velocities[i,2] * delta_t

        recvbuf = np.empty([size, width_padded, 3])
        comm.Allgather(positions[index_from:index_to_padded], recvbuf)
        _recvbuf = recvbuf

        for j in range(size):
          positions[index_from:index_to_padded] = _recvbuf[j]

        calc_acceleration(index_from, index_to_padded, accelerations, G, positions, masses, softening_parameter)

        for i in range(index_from, index_to_padded):
            # second kick
            velocities[i,0] += accelerations[i,0] * (delta_t/2.0)
            velocities[i,1] += accelerations[i,1] * (delta_t/2.0)
            velocities[i,2] += accelerations[i,2] * (delta_t/2.0)

        t += delta_t

        if is_master:
            positions_by_timestep[:,:,x+1] = positions

    return np.array(positions_by_timestep)
