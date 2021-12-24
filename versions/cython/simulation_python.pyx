import numpy as np

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

def simulate(G, N, dt, t, t_max, soft_param):
    '''
    Calculate values for simulation
    '''

    # generate initial conditions
    np.random.seed(17)

    pos  = np.random.randn(N,3) # normally distributed positions
    vel  = np.zeros((N,3)) #np.random.randn(N,3) # normally distributed velocities
    mass = 1.0 * np.ones((N,1))  # particle mass is 1.0

    # convert to Center-of-Mass frame (??)
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial conditions
    acc = calc_acc(G, pos, mass, soft_param)

    # setup initial heavy one
    pos[0] = [0,0,0]
    mass[0] = 1.0 * 10**(30)

    # calculate timesteps for plotting
    steps = int(np.ceil(t_max/dt))

    # TODO use triangular matrix to avoid calcualting same particales twice

    # data store for plotting, define t=0
    pos_t = np.zeros((N,3,steps+1))
    pos_t[:,:,0] = pos

    # Iteration loop by leapfrog integration
    # TODO explain in report
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
        pos_t[:,:,i+1] = np.copy(pos)

    return pos_t
