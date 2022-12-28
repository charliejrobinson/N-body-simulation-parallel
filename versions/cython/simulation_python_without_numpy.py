import numpy as np
import math

# Test

def calc_acc(acc, G, pos, mass, soft_param):
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
    # Zero acc
    acc = np.zeros(pos.shape)

    N = pos.shape[0]

    for i in range(N):
        for j in range(N):
            # calculate particle seperations
            dx = pos[j,0] - pos[i,0]
            dy = pos[j,1] - pos[i,1]
            dz = pos[j,2] - pos[i,2]

            # matrix of inverse seperations cubed (1/r^3)
            inv_sep = (dx**2 + dy**2 + dz**2 + soft_param**2)**(-1.5)

            # calculate acceleration components
            acc[i,0] += G * (dx * inv_sep) * mass[j]
            acc[i,1] += G * (dy * inv_sep) * mass[j]
            acc[i,2] += G * (dz * inv_sep) * mass[j]

    return acc

def leapfrog(acc, vel, pos, mass, soft_param, G, dt):
    N = pos.shape[0]
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
    acc = calc_acc(acc, G, pos, mass, soft_param)

    for i in range(N):
        # second kick
        vel[i,0] += acc[i,0] * (dt/2.0)
        vel[i,1] += acc[i,1] * (dt/2.0)
        vel[i,2] += acc[i,2] * (dt/2.0)

    return acc

def simulate(pos, mass, vel, G, N, dt, t_max, soft_param):
    '''
    Calculate values for simulation
    '''
    # TODO use triangular matrix to avoid calcualting same particales twice

    # data store for plotting, define t=0
    steps = int(np.ceil(t_max/dt))
    pos_t = np.zeros((N,3,steps+1))
    pos_t[:,:,0] = pos

    # calculate initial conditions
    acc = np.zeros(pos.shape)
    acc = calc_acc(acc, G, pos, mass, soft_param)

    # Iteration loop by leapfrog integration
    t = 0
    for i in range(steps):
        acc = leapfrog(acc, vel, pos, mass, soft_param, G, dt)

	    # new time
        t += dt

	    # get energy of system
        pos_t[:,:,i+1] = np.copy(pos) # TODO this can be lots of memory, maybe don't inlcude?


    return pos_t
