from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"

ext_modules = [ Extension (
    "picalc_pyx_omp",
    ["picalc_pyx_omp.pyx"],
    extra_compile_args =['-fopenmp'], extra_link_args =['-fopenmp']
)]

setup(name="picalc_pyx_omp", ext_modules=cythonize(ext_modules))
