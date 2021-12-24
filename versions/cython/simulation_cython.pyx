import numpy as np

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

    # particle positions r=(x,y,z)
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # calculate particle seperations
    # TODO calculating twice as matrix is symmetric
    dx = np.transpose(x) - x
    dy = np.transpose(y) - y
    dz = np.transpose(z) - z

    # matrix of inverse seperations cubed (1/r^3)
    inv_sep = (dx**2 + dy**2 + dz**2 + soft_param**2)**(-1.5)

    # calculate acceleration components
    ax = np.matmul(G * (dx * inv_sep), mass)
    ay = np.matmul(G * (dy * inv_sep), mass)
    az = np.matmul(G * (dz * inv_sep), mass)

    # create acceleration matrix
    acc = np.hstack((ax,ay,az))

    return acc


def simulate(float G, int N, float dt, float t, float t_max, float soft_param):
  # Initalise particals
  cdef double[:,:] pos  = np.random.randn(N,3).astype(np.double) # normally distributed positions
  cdef double[:,:] vel  = np.zeros((N,3)).astype(np.double) #np.random.randn(N,3) # normally distributed velocities
  cdef double[:,:] mass = 1.0 * np.ones((N,1)).astype(np.double)  # particle mass is 1.0

  # Initial heavy partical
  cdef double[:] heavy_partical = np.array([0,0,0]).astype(np.double)
  pos[0] = heavy_partical
  mass[0] = 1.0 * 10**(30)

  # convert to Center-of-Mass frame (??)
  vel -= np.mean(np.multiply(mass, vel), 0) / np.mean(mass)

  # TODO use triangular matrix to avoid calcualting same particales twice

  # data store for plotting, define t=0
  cdef int steps = int(np.ceil(t_max/dt)) # calculate timesteps for plotting
  cdef double[:,:,:] pos_t = np.zeros((N,3,steps+1)).astype(np.double)
  pos_t[:,:,0] = pos

  # Calculate inital accelerations
  acc = calc_acc(G, pos, mass, soft_param)

  # Iteration loop by leapfrog integration
  # TODO explain in report
  cdef Py_ssize_t i
  for i in range(steps):
      # first kick
      vel += np.multiply(acc, dt/2.0)

      # drift
      pos += np.multiply(vel, dt)

      # recalculate accelerations
      acc = calc_acc(G, pos, mass, soft_param)

      # second kick
      vel += np.multiply(acc, dt/2.0)

      # new time
      t += dt

      # get energy of system
      pos_t[:,:,i+1] = pos

  return np.array(pos_t)
