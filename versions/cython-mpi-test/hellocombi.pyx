#================
# hellocombi . pyx
#================
from cython.parallel cimport parallel, threadid
cimport openmp
def printinfo(rank, size, name):
  cdef int num_threads, thread
  with nogil, parallel():
    num_threads = openmp.omp_get_num_threads()
    thread = threadid ()
    with gil:
      print("This is thread {} of {} threads, on rank {} out of {} tasks, running on {}".format(thread, num_threads, rank, size, name))

  return 0
