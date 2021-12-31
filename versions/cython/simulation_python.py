import numpy as np
from scipy.spatial.distance import pdist, squareform

# TODO remove
np.set_printoptions(precision=8)

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
    # particle positions r=(x,y,z)
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # calculate particle seperations
    dx = pdist(x)
    dy = pdist(y)
    dz = pdist(z)

    # matrix of inverse seperations cubed (1/r^3)
    inv_sep = (dx**2 + dy**2 + dz**2 + soft_param**2)**(-1.5)

    dx2 = (x.T - x) * squareform(inv_sep)
    dy2 = (y.T - y) * squareform(inv_sep)
    dz2 = (z.T - z) * squareform(inv_sep)

    # print(dx2)
    #
    # print(squareform(dx * inv_sep))

    # calculate acceleration components
    ax = np.matmul(dx2, G*mass)
    ay = np.matmul(dy2, G*mass)
    az = np.matmul(dz2, G*mass)

    # create acceleration matrix
    acc = np.hstack((ax,ay,az))

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
