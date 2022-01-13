#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
cdef int size = comm.Get_size()
cdef int rank = comm.Get_rank()
cdef int is_master = rank == 0

cdef void calc_acc(double[:,:] loc_acc, float G, double[:,:] loc_pos, double[:,:] mass, float soft_param, double[:,:] tmp_data, int n, int loc_n):
  cdef int src = (rank + 1) % size
  cdef int dest = (rank - 1 + size) % size

  # tmp_data = np.zeros((2*loc_n, 3)).astype(np.double)
  tmp_data[:,:] = 0.0
  # loc_acc  = np.zeros((loc_n, 3)).astype(np.double)
  loc_acc[:,:] = 0.0

  # Compute accelerations from my particles interactions with themselves
  calc_accel_2(mass, tmp_data, loc_acc, loc_pos, loc_n, rank, loc_n, rank, n, size, G, soft_param)

  cdef Py_ssize_t i, loc_part
  cdef int other_proc

  # Now compute forces resulting from my particles' interactions with other processes' particles
  for i in range(1, size):
    other_proc = (rank + i) % size
    comm.Sendrecv_replace(tmp_data, dest, source=src)
    calc_accel_2(mass, tmp_data, loc_acc, loc_pos, loc_n, rank, loc_n, other_proc, n, size, G, soft_param)

  comm.Sendrecv_replace(tmp_data, dest, source=src)
  for loc_part in range(loc_n):
      loc_acc[loc_part][0] += tmp_data[loc_n+loc_part][0]
      loc_acc[loc_part][1] += tmp_data[loc_n+loc_part][1]
      loc_acc[loc_part][2] += tmp_data[loc_n+loc_part][2]

# Given a global index glb1 assigned to process rk1, find the next higher global index assigned to process rk2
cdef int First_index(int gbl1, int rank1, int rank2, int size):
    if rank1 < rank2:
        return gbl1 + (rank2 - rank1)
    else:
        return gbl1 + (rank2 - rank1) + size

# Convert a global particle index to a global permuted index
cdef int Global_to_local(int gbl_part, int rank, int size):
    return (gbl_part - rank) / size

# Compute the forces on particles owned by process rk1 due to interaction with particles owned by procss rk2.  Exploit the symmetry (force on particle i due to particle k) = -(force on particle k due to particle i)
# rk1 = process owning particles in pos1
# rk2 = process owning contributed particles
# p = number of processes in communicator containing processes rk1 and rk2
# loc_n1 = number of my particles in pos1
# loc_n2 = number of my particles in pos2
cdef void calc_accel_2(double[:,:] mass, double[:,:] tmp_data, double[:,:] loc_acc, double[:,:] loc_pos, int loc_n1, int rank1, int loc_n2, int rank2, int n, int size, float G, float soft_param):
    cdef int gbl_part1 = rank1

    cdef Py_ssize_t i, j
    cdef int gbl_part2
    cdef double inv_sep
    cdef double dx, dy, dz

    for i in range(loc_n1):
        gbl_part2 = First_index(gbl_part1, rank1, rank2, size)
        for j in range(Global_to_local(gbl_part2, rank2, size), loc_n2):
            # print(i, j, gbl_part1, gbl_part2)

            # calculate particle seperations
            dx = loc_pos[j,0] - loc_pos[i,0]
            dy = loc_pos[j,1] - loc_pos[i,1]
            dz = loc_pos[j,2] - loc_pos[i,2]

            # matrix of inverse seperations cubed (1/r^3)
            inv_sep = (dx**2 + dy**2 + dz**2 + soft_param**2)**(-1.5)

            # calculate acceleration components
            loc_acc[i,0] += G * (dx * inv_sep) * mass[gbl_part2][0]
            loc_acc[i,1] += G * (dy * inv_sep) * mass[gbl_part2][0]
            loc_acc[i,2] += G * (dz * inv_sep) * mass[gbl_part2][0]
            tmp_data[loc_n2+j,0] -= G * (dx * inv_sep) * mass[gbl_part2][0]
            tmp_data[loc_n2+j,1] -= G * (dy * inv_sep) * mass[gbl_part2][0]
            tmp_data[loc_n2+j,2] -= G * (dz * inv_sep) * mass[gbl_part2][0]

            gbl_part2 += size

        gbl_part1 += size

cpdef simulate(double[:,:] pos, double[:,:] mass, double[:,:] vel, float G, int N, float dt, float t_max, float soft_param):
    params = None
    pos_t = None

    if is_master:
        assert N % size == 0, 'N and number of processes must devide equally'

        params = {'pos': np.array(pos), 'vel': np.array(vel), 'mass': np.array(mass)}

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

    cdef float t = 0

    steps = int(np.ceil(t_max / dt))
    cdef int count = int(np.ceil(N / size))

    cdef int loc_n = int(N / size) # Number of particales locally
    cdef int index_from = int(loc_n * rank)
    cdef int index_to   = int(loc_n * (rank + 1))

    cdef double[:,:] loc_pos = pos[index_from:index_to]
    cdef double[:,:] loc_vel = vel[index_from:index_to]
    cdef double[:,:] loc_acc = np.zeros((loc_n, 3)).astype(np.double)
    cdef double[:,:] tmp_data = np.zeros((2*loc_n, 3)).astype(np.double)

    calc_acc(loc_acc, G, loc_pos, mass, soft_param, tmp_data, N, loc_n)

    cdef Py_ssize_t x, i
    # cdef double[:,:, :] recvbuf

    for x in range(steps):
        for i in range(loc_n):
            # first kick
            loc_vel[i,0] += loc_acc[i,0] * (dt/2.0)
            loc_vel[i,1] += loc_acc[i,1] * (dt/2.0)
            loc_vel[i,2] += loc_acc[i,2] * (dt/2.0)

            # drift
            loc_pos[i,0] += loc_vel[i,0] * dt
            loc_pos[i,1] += loc_vel[i,1] * dt
            loc_pos[i,2] += loc_vel[i,2] * dt

        calc_acc(loc_acc, G, loc_pos, mass, soft_param, tmp_data, N, loc_n)
        # print(loc_pos)

        for i in range(loc_n):
            # second kick
            loc_vel[i,0] += loc_acc[i,0] * (dt/2.0)
            loc_vel[i,1] += loc_acc[i,1] * (dt/2.0)
            loc_vel[i,2] += loc_acc[i,2] * (dt/2.0)

        t += dt

        # TODO slow, if don't need to output sim each time can speed up
        recvbuf = np.empty([size, loc_n, 3])
        comm.Gather(loc_pos, recvbuf)
        recvbuf = recvbuf.reshape((N, 3))
        pos = recvbuf

        if is_master:
            pos_t[:,:,x+1] = pos

    return np.array(pos_t)
