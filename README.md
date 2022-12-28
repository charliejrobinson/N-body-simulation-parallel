# High performance computing project
Investigations into parallel computing for an N-body gravity simulation for a 4th year Physics high performance computing unit [University of Bristol] (result: 89%). Pdf available.

![Front of research paper](https://github.com/charliejrobinson/N-body-simulation-parallel/blob/main/HPC.PNG)


## Compiling
`python3 setup.py build_ext --inplace`

## Running
`python3 main.py`

### MPI running
`mpirun -n 2 --oversubscribe python3 main.py run --simulation python_mpi --N 12 --plot_end`

### OpenMP running
`python3 main.py run --simulation cython_openmp --threads 4 --N 12 --plot_end`

## cython profile
`cython -a *.pyx`
then open *.html
