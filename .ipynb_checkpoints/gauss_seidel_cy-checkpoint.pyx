import numpy as np
cimport numpy as np


def gauss_seidel_cython(np.ndarray[np.float32_t, ndim=2] f, int iterations):
    cdef int i, j, it
    cdef int N = f.shape[0]
    cdef np.ndarray[np.float32_t, ndim=2] newf = np.copy(f)

    for it in range(iterations):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                newf[i, j] = 0.25 * (newf[i+1, j] + newf[i-1, j] + 
                                     newf[i, j+1] + newf[i, j-1])
        np.copyto(f, newf)  # Update the grid
    
    return f