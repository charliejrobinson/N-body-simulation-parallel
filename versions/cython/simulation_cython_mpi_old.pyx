#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
import numpy as np
# cimport numpy as np
from libc.math cimport sqrt, pow

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ctypedef np.double_t np.double

cdef void calc_acc(double[:,:] acc, float G, double[:,:] pos, double[:,:] mass, float soft_param):
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
    cdef int N = np.array(pos).shape[0]
    cdef double inv_sep
    cdef double dx, dy, dz

    # TODO fast array
    for i in range(N):
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

            # calculate acceleration components
            acc[i,0] += G * (dx * inv_sep) * mass[j][0]
            acc[i,1] += G * (dy * inv_sep) * mass[j][0]
            acc[i,2] += G * (dz * inv_sep) * mass[j][0]

cdef void leapfrog(double[:,:] acc, double[:,:] vel, double[:,:] pos, double[:,:] mass, float soft_param, float G, double dt):
  cdef int N = pos.shape[0]
  cdef Py_ssize_t i

  for i in range(N):
      # first kick
      vel[i,0] += acc[i,0] * (dt/2.0)
      vel[i,1] += acc[i,1] * (dt/2.0)
      vel[i,2] += acc[i,2] * (dt/2.0)

      # drift
      pos[i,0] += vel[i,0] * dt
      pos[i,1] += vel[i,1] * dt
      pos[i,2] += vel[i,2] * dt

  # recalculate accelerations
  comm.Bcast(pos, root=0)
  calc_acc(acc, G, pos, mass, soft_param)

  for i in range(N):
      # second kick
      vel[i,0] += acc[i,0] * (dt/2.0)
      vel[i,1] += acc[i,1] * (dt/2.0)
      vel[i,2] += acc[i,2] * (dt/2.0)

cpdef simulate(double[:,:] pos, double[:,:] mass, double[:,:] vel, float G, int N, double dt, float t_max, float soft_param):
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
    calc_acc(acc, G, pos, mass, soft_param)

    comm.Bcast(acc, root=0)
    comm.Bcast(pos, root=0)

    # Iteration loop by leapfrog integration
    cdef float t = 0
    cdef Py_ssize_t i
    for i in range(steps):
        leapfrog(acc, vel, pos, mass, soft_param, G, dt)

	      # new time
        t += dt

	      # get energy of system
        pos_t[:,:,i+1] = pos # TODO this can be lots of memory, maybe don't inlcude?

    return np.array(pos_t)
