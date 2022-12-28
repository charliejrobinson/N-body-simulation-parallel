#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
import numpy as np

from cython.parallel cimport prange
cimport openmp
# from libc.math cimport sqrt

cdef void calc_acc(str schedule, int chunk_size, int threads, double[:,:] acc, float G, double[:,:] pos, double[:,:] mass, float soft_param):
    '''
    Parameters
    ----------
    pos : (N,3) Matrix of position vectors
    mass : (N,1) Matrix of masses
    soft_param : Softening parameter

    Returns
    -------
    acc : Matrix of accelerations
    '''
    cdef Py_ssize_t i, j
    cdef double x1, y1, z1, x2, yz, z2
    cdef int N = pos.shape[0]
    cdef double inv_sep
    cdef double temp
    cdef double dx, dy, dz

    for i in prange(N, nogil=True, num_threads=threads, schedule='static', chunksize=chunk_size):
        # Zero the array
        # TODO will this change the vectorisation
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
            temp = G * inv_sep * mass[j][0]

            # calculate acceleration components
            acc[i,0] += temp * dx
            acc[i,1] += temp * dy
            acc[i,2] += temp * dz

cdef void leapfrog(str schedule, int chunk_size, int threads, double[:,:] acc, double[:,:] vel, double[:,:] pos, double[:,:] mass, float soft_param, float G, double dt):
  cdef int N = pos.shape[0]
  cdef Py_ssize_t i

  for i in prange(N, nogil=True, num_threads=threads, schedule='static', chunksize=chunk_size):
      # first kick
      vel[i,0] += acc[i,0] * (dt/2.0)
      vel[i,1] += acc[i,1] * (dt/2.0)
      vel[i,2] += acc[i,2] * (dt/2.0)

      # drift
      pos[i,0] += vel[i,0] * dt
      pos[i,1] += vel[i,1] * dt
      pos[i,2] += vel[i,2] * dt

  # recalculate accelerations
  calc_acc(schedule, chunk_size, threads, acc, G, pos, mass, soft_param)

  for i in prange(N, nogil=True, num_threads=threads, schedule='static', chunksize=chunk_size):
      # second kick
      vel[i,0] += acc[i,0] * (dt/2.0)
      vel[i,1] += acc[i,1] * (dt/2.0)
      vel[i,2] += acc[i,2] * (dt/2.0)

cpdef simulate(str schedule, int chunk_size, int threads, double[:,:] pos, double[:,:] mass, double[:,:] vel, float G, int N, double dt, float t_max, float soft_param):
    '''
    Calculate values for simulation
    '''
    # TODO use triangular matrix to avoid calcualting same particales twice

    # data store for plotting, define t=0
    cdef int steps = int(np.ceil(t_max/dt))
    cdef double[:,:,:] pos_t = np.zeros((N,3,steps+1))
    pos_t[:,:,0] = pos

    # calculate initial conditions
    # cdef np.ndarray[np.double_t, ndim=2] acc = np.zeros(pos.shape).astype(np.double)
    cdef double[:,:] acc = np.zeros(np.array(pos).shape).astype(np.double)
    calc_acc(schedule, chunk_size, threads, acc, G, pos, mass, soft_param)

    # Iteration loop by leapfrog integration
    cdef float t = 0
    cdef Py_ssize_t i
    for i in range(steps):
        leapfrog(schedule, chunk_size, threads, acc, vel, pos, mass, soft_param, G, dt)

	      # new time
        t += dt

	      # get energy of system
        pos_t[:,:,i+1] = pos # TODO this can be lots of memory, maybe don't inlcude?

    return np.array(pos_t)
