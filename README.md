# Using Compilation Techniques and GPUs for Optimization 

# Exercise 1 - Gauss-Seidel for Poisson Solver

In this exercise, we develop and optimize the Gauss-Seidel solver for solving the **2D** Poisson Equation:

$$
\frac{{∂f}^2}{{∂x}^2} + \frac{{∂f}^2}{{∂y}^2} = S
$$

We solve Poisson's equation by it to convergence by a forward time-centered space differencing (FTCS) on a grid using the Gauss-Seidel Method:

    - We discretize the function f on a square box of size 1.
    - We use a uniform grid with N grid points in the x-direction and N grid points in the y-direction. 
    - We impose the values at the boundary equal to zero.
    - We have no source S
    - We initialize the simulation with random numbers

The Gauss-Seidel iteration for the Poisson Equation in 2D is:

$$
{f}_(i,j) = \frac{1}{4}[{f}^n_(i+1,j) + {f}^n_(i-1,j) + {f}^n_(i,j+1) + {f}^n_(i, j-1)]
$$

In Python code, a Gauss-Seidel iteration can be written as follows:

    
    def gauss_seidel(f):
    newf = f.copy()
    
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                   newf[i+1,j] + newf[i-1,j])
    return newf

where the grid values at the boundaries are fixed to zero, and no source is included.

Then, for running the 1,000 iterations:


    for i in range(1000):    
    x = gauss_seidel(x)


# **Task 1.1**

Develop the Gauss-Seidel solver with Python List, array, or NumPy. Plot the performance varying the grid size.

# **Task 1.2**

Profile the code to identify the part of the code to optimize. You can use the tool of your choice.

# **Task 1.3**

Use the Cython Annotation tool to identify the parts to use Cython

# **Task 1.4**

Use Cython to optimize the part you identified as the most computationally expensive. Compare the performance with the results obtained in Task 2.1.

# **Task 1.5**

Use PyTorch to port your code to Nvidia GPUs. You must express the two nested loop operations as NumPy roll operations in 2D as we did for the diffusion code. Or use the Google Colab.

**For this task, we use a simpler numerical scheme called Jacobi, where  **f[i-1,j]** and **f[i,j-1]** are calculated at time n instead on n+1. In this way, we remove the calculation dependencies on the new time step, which can be easily vectorized.**

# **Task 1.6**

Use CuPy to port your code to Nvidia GPUs

# **Task 1.7**

Measure the performance (execution time) with GPU (PyTorch and CuPy) and make a plot of the execution time varying the size of the grid. Compare and comment on the performance difference with and without GPU.

# **Task 1.8**

Save the newgrid matrix as an hdf5 file using **h5py**

