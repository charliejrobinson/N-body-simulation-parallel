from mpi4py import MPI
import numpy as np

def calc_acc(index_from, index_to, acc, G, pos, mass, soft_param):
    # Zero acc
    acc = np.zeros(pos.shape)

    for i in range(index_from, index_to):
        for j in range(pos.shape[0]):
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

def simulate(pos, mass, vel, G, N, dt, t_max, soft_param):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    is_master = rank == 0

    params = None
    pos_t = None

    if is_master:
        assert N % size == 0, 'N and number of processes must devide equally'

        acc = np.zeros(pos.shape)
        acc = calc_acc(0, N, acc, G, pos, mass, soft_param)

        params = {'pos': pos, 'vel': vel, 'mass': mass, 'acc': acc}

        steps = int(np.ceil(t_max / dt))
        pos_t = np.zeros((N,3,steps+1))
        pos_t[:,:,0] = pos

    # Broadcast / Recive enviroment
    params = comm.bcast(params, root=0)

    # Unpack environment
    pos = params['pos']
    vel = params['vel']
    mass = params['mass']
    acc = params['acc']

    t = 0

    steps = int(np.ceil(t_max / dt))
    count = int(np.ceil(N / size))

    width = int(N / size)
    index_from = int(width * rank)
    index_to   = int(width * (rank + 1))

    for x in range(steps):
        for i in range(index_from, index_to):
            # first kick
            vel[i,0] += acc[i,0] * (dt/2.0)
            vel[i,1] += acc[i,1] * (dt/2.0)
            vel[i,2] += acc[i,2] * (dt/2.0)

            # drift
            pos[i,0] += vel[i,0] * dt
            pos[i,1] += vel[i,1] * dt
            pos[i,2] += vel[i,2] * dt

        recvbuf = np.empty([size, width, 3])
        comm.Allgather(pos[index_from:index_to], recvbuf)
        pos = recvbuf.reshape((N, 3))

        acc = calc_acc(index_from, index_to, acc, G, pos, mass, soft_param)

        for i in range(index_from, index_to):
            # second kick
            vel[i,0] += acc[i,0] * (dt/2.0)
            vel[i,1] += acc[i,1] * (dt/2.0)
            vel[i,2] += acc[i,2] * (dt/2.0)

        t += dt

        if is_master:
            pos_t[:,:,x+1] = np.copy(pos)

    return pos_t
