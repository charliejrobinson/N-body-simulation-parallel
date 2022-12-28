from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import platform

os.environ["CC"] = "/usr/bin/gcc"
if platform.system() == 'Darwin':
    os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"

names = ['simulation_cython_without_numpy', 'simulation_cython_openmp_static', 'simulation_cython_openmp_dynamic', 'simulation_cython_openmp_guided', 'openmp_api_wraper', 'simulation_cython_mpi', 'simulation_cython_mpi_ring']

extensions = []
for name in names:
    extensions.append(Extension(name, sources=["%s.pyx" % name], extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-ftree-vectorize', '-funroll-loops'], extra_link_args=['-fopenmp'], include_dirs=[numpy.get_include()]))

setup(ext_modules=cythonize(extensions))
