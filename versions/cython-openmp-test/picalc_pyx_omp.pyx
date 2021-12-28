#==================== # picalc_pyx_omp . pyx #==================== import time
from libc .math cimport sqrt
from cython.parallel cimport prange
cimport openmp

def main( int nmax, int threads ):
  cdef:
    double pibyfour = 0.0
    double dx = 1.0 / nmax
    double initial , final
    int i

  print("Threads set to {:2d}".format(threads))
  initial = openmp.omp_get_wtime()

  for i in prange(nmax, nogil=True , num_threads=threads):
    pibyfour += sqrt(1-(i*dx)**2)

  final = openmp.omp_get_wtime()
  print("Elapsed time: {:8.6f} s".format(final-initial))
  pi = 4.0 * pibyfour * dx
  print("Pi = {:18.16f}".format(pi))
  return 0
