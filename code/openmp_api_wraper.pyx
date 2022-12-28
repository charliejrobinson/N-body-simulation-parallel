#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

cimport cython

from cython.parallel cimport parallel
cimport openmp

cpdef get_wtime():
    return openmp.omp_get_wtime()
