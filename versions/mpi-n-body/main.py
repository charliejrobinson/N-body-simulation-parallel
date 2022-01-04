from mpi4py import MPI
import numpy as np
import pickle

def do_work(array, rank):
    array = array + rank + 1
    return array

def initialise_environment(seed, N):
    np.random.seed(seed)

    # ---------------------------------
    # Initalise enviroment
    pos  = np.random.randn(N,3).astype(np.double) # normally distributed positions
    vel  = np.random.randn(N,3).astype(np.double) # normally distributed velocities
    mass = np.ones((N,1)).astype(np.double) # particle mass is 1.0

    vel -= np.mean(mass * vel, 0) / np.mean(mass) # convert to Center-of-Mass frame (??)

    # setup initial heavy one
    # pos[0] = [0,0,0]
    # mass[0] = 1.0 * 10**(30)

    return pos, vel, mass

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

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    is_master = rank == 0

    params = None
    pos_t = None

    if is_master:
        seed = 17
        N = 100
        dt = 0.01
        t_max = 0.04
        G = 1
        soft_param = 1e-5

        assert N % size == 0, 'N and number of processes must devide equally'

        pos, vel, mass = initialise_environment(seed, N)
        acc = np.zeros(pos.shape)
        acc = calc_acc(0, N, acc, G, pos, mass, soft_param)

        params = {'N': N, 'dt': dt, 't_max': t_max, 'G': G, 'soft_param': soft_param, 'pos': pos, 'vel': vel, 'mass': mass, 'acc': acc}

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

    dt = params['dt']
    t = 0

    steps = int(np.ceil(params['t_max'] / dt))
    count = int(np.ceil(params['N'] / size))

    width = int(params['N'] / size)
    index_from = int(width * rank)
    index_to   = int(width * (rank + 1))

    # print('%i: %i-%i' % (rank, index_from, index_to))

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

        acc = calc_acc(index_from, index_to, acc, params['G'], pos, mass, params['soft_param'])

        for i in range(index_from, index_to):
            # second kick
            vel[i,0] += acc[i,0] * (dt/2.0)
            vel[i,1] += acc[i,1] * (dt/2.0)
            vel[i,2] += acc[i,2] * (dt/2.0)

        t += dt

        recvbuf = np.empty([size, width, 3])
        comm.Allgather(pos[index_from:index_to], recvbuf)
        pos = recvbuf.reshape((params['N'], 3))

        if is_master:
            pos_t[:,:,x+1] = np.copy(pos)

    if is_master:
        print('Saving...')
        data = {
            'simulation': 'MPI_python',
            'N': params['N'],
            'pos_t': pos_t,
            't_max': params['t_max'],
            'dt': params['dt']
        }

        pickle.dump(data, open('data.p', 'wb'))

main()

# main(5)
# print('rank: %i' % rank)
