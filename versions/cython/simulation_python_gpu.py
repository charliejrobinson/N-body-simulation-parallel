import numpy as np

try:
    import cupy as cp
except ImportError:
    print('**** CUPY not installed')

def calc_acc(G, pos, mass, soft_param):
    # particle positions r=(x,y,z)
    x = pos[:,0:1]
    y = pos[:,1:2]
    z = pos[:,2:3]

    # calculate particle seperations
    dx = x.T - x
    dy = x.T - y
    dz = x.T - z

    # matrix of inverse seperations cubed (1/r^3)
    inv_sep = (dx**2 + dy**2 + dz**2 + soft_param**2)**(-1.5)

    # calculate acceleration components
    ax = (G * (dx * inv_sep)) @ mass
    ay = (G * (dy * inv_sep)) @ mass
    az = (G * (dz * inv_sep)) @ mass

    return (ax, ay, az)

def simulate(pos, mass, vel, G, N, dt, t_max, soft_param):
    # data store for plotting, define t=0
    steps = int(np.ceil(t_max/dt))
    pos_t = np.zeros((N,3,steps+1))
    pos_t[:,:,0] = pos

    pos_t = cp.asarray(pos_t)
    pos = cp.asarray(pos)
    mass = cp.asarray(mass)
    vel = cp.asarray(vel)

    # calculate initial conditions
    acc = cp.hstack(calc_acc(G, pos, mass, soft_param))

    # Iteration loop by leapfrog integration
    t = 0
    for i in range(steps):
	      # first kick
        vel += acc * dt/2.0

	      # drift
        pos += vel * dt

	      # recalculate accelerations
        acc = cp.hstack(calc_acc(G, pos, mass, soft_param))

        # second kick
        vel += acc * dt/2.0

	      # new time
        t += dt

	      # get energy of system
        pos_t[:,:,i+1] = pos.copy()

    pos_t = cp.asnumpy(pos_t)

    return pos_t
