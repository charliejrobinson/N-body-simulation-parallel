import numpy as np
import math

def calc_acc(G, pos, mass, soft_param):
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
    acc = np.zeros(pos.shape)

    for i, (x1, y1, z1) in enumerate(pos):
        for (x2, y2, z2) in pos:
            # calculate particle seperations
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            # matrix of inverse seperations cubed (1/r^3)
            inv_sep = dx**2 + dy**2 + dz**2 + soft_param
            inv_sep = (1.0 / math.sqrt(inv_sep)) ** 3

            # calculate acceleration components
            acc[i][0] = G * (dx * inv_sep) * mass[i][0]
            acc[i][1] = G * (dy * inv_sep) * mass[i][0]
            acc[i][2] = G * (dz * inv_sep) * mass[i][0]

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
    acc = calc_acc(G, pos, mass, soft_param)

    # Iteration loop by leapfrog integration
    t = 0
    for i in range(steps):
	    # first kick
        vel += acc * dt/2.0

	    # drift
        pos += vel * dt

	    # recalculate accelerations
        acc = calc_acc(G, pos, mass, soft_param)

        # second kick
        vel += acc * dt/2.0

	    # new time
        t += dt

	    # get energy of system
        pos_t[:,:,i+1] = np.copy(pos) # TODO this can be lots of memory, maybe don't inlcude?

    return pos_t
