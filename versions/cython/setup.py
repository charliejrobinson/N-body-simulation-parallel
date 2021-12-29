from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os

os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"

# define an extension that will be cythonized and compiled
ext = Extension('simulation_cython', sources=["simulation_cython.pyx"], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize(ext))

# all .pyx files in a folder
# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(name = 'simulation', ext_modules = cythonize(["*.pyx"]))
