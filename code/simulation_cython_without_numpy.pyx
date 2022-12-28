#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython
import numpy as np

from libc.math cimport sqrt, pow

cdef void calc_acceleration(double[:,:] accelerations, float G, double[:,:] positions, double[:,:] masses, float softening_parameter):
    cdef Py_ssize_t i, j
    cdef double x1, y1, z1, x2, yz, z2
    cdef int N = positions.shape[0]
    cdef double inverse_separations
    cdef double dx, dy, dz

    for i in range(N):
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

cdef void leapfrog(double[:,:] accelerations, double[:,:] velocities, double[:,:] positions, double[:,:] masses, float softening_parameter, float G, double delta_t):
  cdef int N = positions.shape[0]
  cdef Py_ssize_t i

  for i in range(N):
      # first kick
      velocities[i,0] += accelerations[i,0] * (delta_t/2.0)
      velocities[i,1] += accelerations[i,1] * (delta_t/2.0)
      velocities[i,2] += accelerations[i,2] * (delta_t/2.0)

      # drift
      positions[i,0] += velocities[i,0] * delta_t
      positions[i,1] += velocities[i,1] * delta_t
      positions[i,2] += velocities[i,2] * delta_t

  # recalculate accelerations
  calc_acceleration(accelerations, G, positions, masses, softening_parameter)

  for i in range(N):
      # second kick
      velocities[i,0] += accelerations[i,0] * (delta_t/2.0)
      velocities[i,1] += accelerations[i,1] * (delta_t/2.0)
      velocities[i,2] += accelerations[i,2] * (delta_t/2.0)

cpdef simulate(double[:,:] positions, double[:,:] masses, double[:,:] velocities, float G, int N, double delta_t, float t_max, float softening_parameter):
    cdef int steps = int(np.ceil(t_max/delta_t))
    cdef double[:,:,:] positions_by_timestep = np.zeros((N,3,steps+1))
    positions_by_timestep[:,:,0] = positions

    # calculate initial conditions
    cdef double[:,:] accelerations = np.zeros(np.array(positions).shape).astype(np.double)
    calc_acceleration(accelerations, G, positions, masses, softening_parameter)

    # Iteration loop by leapfrog integration
    cdef float t = 0
    cdef Py_ssize_t i
    for i in range(steps):
        leapfrog(accelerations, velocities, positions, masses, softening_parameter, G, delta_t)
        
        t += delta_t

        positions_by_timestep[:,:,i+1] = positions 

    return np.array(positions_by_timestep)
