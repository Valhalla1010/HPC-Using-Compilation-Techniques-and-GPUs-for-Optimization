import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from gauss_seidel_cy import gauss_seidel_cython



def gauss_seidel(f):

    newf = f.copy()
    for i in range(1, newf.shape[0] - 1):
        for j in range(1, newf.shape[0] - 1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i, j-1]+
                                newf[i+1,j] + newf[i-1,j])
            
    return newf



def gauss_seidel_list(f):
    newf = [row[:] for row in f]
    for i in range(1, len(newf) - 1):
        for j in range(1, len(newf[i])- 1):
            newf[i][j] = 0.25 * (newf[i+1][j] + newf[i-1][j] + 
                                 newf[i][j+1] + newf[i][j-1])
    return newf



def initialize(grid_sizes):
    times_numpy = []
    times_list = []
    times_cython = []
    
    for N in grid_sizes:
       f_list = [[np.random.rand() for _ in range(N)] for _ in range(N)]
       for i in range(N):
           f_list[i][0] = f_list[i][-1] = 0
       for j in range(N):
           f_list[0][j] = f_list[-1][j] = 0


       f_numpy = np.random.rand(N,N)
       f_numpy[0, :] = f_numpy[-1, :] = f_numpy[:, 0] = f_numpy[:, -1] = 0

       f_cython = np.random.rand(N, N).astype(np.float32)
        
       start_time = timer()
       for _ in range(1000):
           f_list = gauss_seidel_list(f_list)
       end_time = timer()
       times_list.append(end_time - start_time)

       start_time = timer()
       for _ in range(1000):
           f_numpy = gauss_seidel(f_numpy)
       end_time = timer()
       times_numpy.append(end_time - start_time)


       start_time = timer()
       f_cython = gauss_seidel_cython(f_cython, 1000)
       times_cython.append(timer() - start_time)

       print(f"Grid size: {N}x{N} | List time: {times_list[-1]:.4f}s | Numpy time: {times_numpy[-1]:.4f}s | Cython: {times_cython[-1]:.4f}s")

    return times_numpy, times_list, times_cython


 
if __name__ == "__main__":

    grid_sizes = [10, 50, 100, 200, 300]
    times_numpy, times_list, times_cython = initialize(grid_sizes) 
    


    #plt.plot(grid_sizes, times_list, marker='o', linestyle='-', label='List')
    #plt.plot(grid_sizes, times_numpy, marker='s', linestyle='-', label='NumPy')
    #plt.xlabel('Grid size')
    #plt.ylabel('Time')
    #plt.title('Gauss-Seidel')
    #plt.legend()
    #plt.grid()
    #plt.show()

     




