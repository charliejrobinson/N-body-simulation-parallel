# Gravity Simulator
N-body simulation with multiple parallel approaches

## Installing
`pip install -r requirements.txt`

## Running on Google Cloud Compute Engine
We use a docker virtualised enviroment (https://www.docker.com) which we upload to Google Cloud Container Repository and then deploy to Compute Engine. The Docker script can be viewed in the `Dockerfile`, it installs dependences and then runs `experiments.sh`.

Before starting you'll need:
- Docker installed locally
- A Google Cloud project
- Install `gcloud` command line utility (https://cloud.google.com/sdk/gcloud)
- Login to Google Cloud project locally

```
$ docker build -t hpc .
$ gcloud config set project <PROJECT_ID>
$ gcloud builds submit --tag gcr.io/<CONTAINER_PATH>
$ gcloud compute instances create-with-container instance-1 --project=<PROJECT_ID> --zone=us-central1-a --machine-type=c2-standard-30 --container-image=gcr.io/<CONTAINER_PATH> --container-tty --container-stdin --container-restart-policy=never
```

NOTE: Make you may be charged a few £s so make sure to stop the containers

Once the instances have been created, SSH into them to monitor the results of the experiments.

## Running on BlueCrystal
As BlueCrystal was down for maintaince during the writing of this project we ran our experiments on Google Cloud. However, this project should be able to run on by compiling and executing `experiments.sh` on BlueCrystal.

## Running locally
We use a single python file which you pass arguments of which simulation varient you want to run.

### Compiling
`python3 setup.py build_ext --inplace`

### Running
`python3 main.py --help`

Data:
- Randomly distributed point like particles `python3 main.py run --simulation cython_without_numpy --N 12 --animate`
- Solar system bodies using NASA Horizons data
  - Planets and large moons in Solar system using current date: `python3 main.py run --simulation cython_without_numpy --bodies solar_system --date 2022-02-06 --animate`
  - Particular bodies (lookup via NAIF ID and date) `python3 main.py run --simulation cython_without_numpy --bodies 10 199 299 399 499 599 699 799 899 999 --date 2022-02-06 --animate`

Simulations:
- Serial `python3 main.py run --simulation cython_without_numpy --N 12 --animate`
- MPI `mpirun -n 2 python3 main.py run --simulation cython_mpi --N 12 --animate`
- OpenMP `python3 main.py run --simulation cython_openmp --threads 4 --N 12 --animate`
- GPU `python3 main.py run --simulation python_gpu --N 12 --animate` - REQUIES CUPY installed

Running experiments suite (NOTE takes a long time):
`$ ./experiments.sh`

## NOTES
`horizons_data.py` is modified from https://github.com/hannorein/rebound/blob/main/rebound/horizons.py
