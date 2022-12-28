import numpy as np

try:
    import cupy as cp
except ImportError:
    print('**** CUPY not installed')

def calculate_acceleration(G, positions, masses, softening_parameter):
    # particle positions r=(x,y,z)
    x = positions[:,0:1]
    y = positions[:,1:2]
    z = positions[:,2:3]

    # calculate distance between pairs of particles for x,y,z
    dx = x.T - x
    dy = x.T - y
    dz = x.T - z

    # matrix of inverse seperations cubed (1/r^3)
    inverse_separations = (dx**2 + dy**2 + dz**2 + softening_parameter**2)**(-1.5)

    # calc accelerations for x,y,z
    ax = (G * (dx * inverse_separations)) @ masses
    ay = (G * (dy * inverse_separations)) @ masses
    az = (G * (dz * inverse_separations)) @ masses

    return (ax, ay, az)

def simulate(positions, masses, velocities, G, N, delta_t, t_max, softening_parameter):
    # Store data for plotting
    steps = int(np.ceil(t_max/delta_t))
    positions_over_time = np.zeros((N,3,steps+1))
    positions_over_time[:,:,0] = positions

    positions_over_time = cp.asarray(positions_over_time)
    positions = cp.asarray(positions)
    masses = cp.asarray(masses)
    velocities = cp.asarray(velocities)

    # calculate initial conditions
    accelerations = cp.hstack(calculate_acceleration(G, positions, masses, softening_parameter))

    # Iteration with leapfrog
    t = 0
    for i in range(steps):
	    # first kick
        velocities += accelerations * delta_t/2.0

	    # drift
        positions += velocities * delta_t

	    # recalculate accelerations
        accelerations = cp.hstack(calculate_acceleration(G, positions, masses, softening_parameter))

        # second kick
        velocities += accelerations * delta_t/2.0

        t += delta_t

        positions_over_time[:,:,i+1] = positions.copy()

    positions_over_time = cp.asnumpy(positions_over_time)

    return positions_over_time
