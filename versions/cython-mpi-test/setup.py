from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"

ext_modules = [ Extension (
    "hellocombi",
    ["hellocombi.pyx"],
    extra_compile_args =['-fopenmp'], extra_link_args =['-fopenmp']
)]

setup(name="hellocombi", ext_modules=cythonize(ext_modules))
