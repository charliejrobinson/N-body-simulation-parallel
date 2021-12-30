import numpy as np
from libc.math cimport sqrt

cdef double[:,:] calc_acc(float G, double[:,:] pos, double[:,:] mass, float soft_param):
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
    cdef double[:,:] acc = np.zeros(np.array(pos).shape).astype(np.double)

    cdef Py_ssize_t i, j
    cdef double x1, y1, z1, x2, yz, z2
    cdef int N = pos.shape[0]

    for i in range(N):
        for j in range(N):
            # calculate particle seperations
            dx = pos[j,0] - pos[i,0]
            dy = pos[j,1] - pos[i,1]
            dz = pos[j,2] - pos[i,2]

            # matrix of inverse seperations cubed (1/r^3)
            inv_sep = dx**2 + dy**2 + dz**2 + soft_param
            inv_sep = (1.0 / sqrt(inv_sep)) ** 3

            # calculate acceleration components
            acc[i,0] = G * (dx * inv_sep) * mass[j][0]
            acc[i,1] = G * (dy * inv_sep) * mass[j][0]
            acc[i,2] = G * (dz * inv_sep) * mass[j][0]

    return np.array(acc)

cdef double[:,:] leapfrog(double[:,:] acc, double[:,:] vel, double[:,:] pos, double[:,:] mass, float soft_param, float G, double dt):
  cdef int N = pos.shape[0]
  cdef Py_ssize_t i

  for i in range(N):
      # first kick
      vel[i,0] += acc[i,0] * dt/2.0
      vel[i,1] += acc[i,1] * dt/2.0
      vel[i,2] += acc[i,2] * dt/2.0

      # drift
      pos[i,0] += vel[i,0] * dt
      pos[i,1] += vel[i,1] * dt
      pos[i,2] += vel[i,2] * dt

  # recalculate accelerations
  acc = calc_acc(G, pos, mass, soft_param)

  for i in range(N):
      # second kick
      vel[i,0] += acc[i,0] * dt/2.0
      vel[i,1] += acc[i,1] * dt/2.0
      vel[i,2] += acc[i,2] * dt/2.0

  return pos

def simulate(double[:,:] pos, double[:,:] mass, double[:,:] vel, float G, int N, double dt, float t_max, float soft_param):
    '''
    Calculate values for simulation
    '''
    # TODO use triangular matrix to avoid calcualting same particales twice

    # data store for plotting, define t=0
    cdef int steps = int(np.ceil(t_max/dt))
    cdef double[:,:,:] pos_t = np.zeros((N,3,steps+1))
    pos_t[:,:,0] = pos

    # calculate initial conditions
    acc = calc_acc(G, pos, mass, soft_param)

    # Iteration loop by leapfrog integration
    cdef float t = 0
    cdef Py_ssize_t i
    for i in range(steps):
        pos = leapfrog(acc, vel, pos, mass, soft_param, G, dt)

	      # new time
        t += dt

	      # get energy of system
        pos_t[:,:,i+1] = pos # TODO this can be lots of memory, maybe don't inlcude?

    return np.array(pos_t)
