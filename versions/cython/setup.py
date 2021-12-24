from distutils.core import setup
from Cython.Build import cythonize

setup(name = 'simulation', ext_modules = cythonize(["*.pyx"]))
