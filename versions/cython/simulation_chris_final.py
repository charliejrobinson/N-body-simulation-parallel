import numpy as np
from scipy.spatial.distance import pdist, squareform
import timeit

# TODO remove
np.set_printoptions(precision=3)

def calc_acc(G, r, mass, soft_param):
    '''
    Parameters
    ----------
    r : (N,3) Matrix of position vectors
    mass : (N,1) Matrix of masses
    soft_param : Softening parameter

    Returns
    -------
    acc : Matrix of accelerations
    '''

    # print(r)
    # (N,N,3)
    # dr_ijk = r_ik - r_jk
    N = r.shape[0]
    dr = (r.reshape(1, N, 3) - r.reshape(N, 1, 3))
    #dr = r[:, None, :] - r[None, :, :]
    # print(dr.shape)
    # print(dr)

    # (N,N)
    dr_squared = np.sum(np.square(dr), axis=2) + soft_param

    # Make |r21| / r21^2
    # Shape: (N,N,3)
    # r_ijk = dr_ijk * (dr_squared_ij)^3/2, don't sum over i and j
    inv_r_squared = dr * np.expand_dims(dr_squared**(-3/2), axis=2)
    #inv_r_squared = dr / dr_squared
    # print(inv_r_squared.shape)
    # print(inv_r_squared)

    # (N,3)
    # acc_ij = inv_r_squared_ikj * G * mass_k
    acc = np.einsum('ikj,k', inv_r_squared, G*mass.reshape(N))
    # print(acc.shape)
    # print(acc)

    # calculate acceleration components
    #ax = np.matmul( inv_r_squared[:,:,0], G*mass)
    #ay = np.matmul( inv_r_squared[:,:,1], G*mass)
    #az = np.matmul( inv_r_squared[:,:,2], G*mass)

    # create acceleration matrix
    #acc = np.hstack((ax,ay,az))

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
