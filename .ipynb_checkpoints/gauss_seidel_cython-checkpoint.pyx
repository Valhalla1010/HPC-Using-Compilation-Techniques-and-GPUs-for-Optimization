def gauss_seidel_cython(double [:, :]f, int iterations):
    newf = f.copy()
    cdef int i, j , k
    cdef int rows
    cdef int cols

    rows = f.shape[0]
    cols = f.shape[1]

    for k in range(iterations):
        for i in range (1, rows - 1):
            for j in range(1, cols - 1):
                newf[i, j] = 0.25 * (newf[i+1, j] + newf[i-1, j] 
                + newf[i, j+1] + newf[i, j-1])



