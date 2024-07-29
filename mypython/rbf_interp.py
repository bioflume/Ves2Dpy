import numpy as np
import time

def rbfinterp(x, options):
    """
    Interpolates using RBF.

    Parameters:
    x : ndarray
        Points at which to interpolate.
    options : dict
        Dictionary containing RBF properties and coefficients.
    
    Returns:
    f : ndarray
        Interpolated values at points x.
    """
    start_time = time.time()
    phi = options['rbfphi']
    rbfconst = options['RBFConstant']
    nodes = options['x']
    rbfcoeff = options['rbfcoeff']

    dim, n = nodes.shape
    dim_points, n_points = x.shape

    if dim != dim_points:
        raise ValueError('x should have the same number of rows as an array used to create RBF interpolation')

    f = np.zeros(n_points)

    for i in range(n_points):
        r = np.linalg.norm(x[:, i].reshape(-1, 1) - nodes, axis=0)
        s = rbfcoeff[n] + np.sum(rbfcoeff[:n] * phi(r, rbfconst))
        
        for k in range(dim):
            s += rbfcoeff[k + n + 1] * x[k, i]  # linear part
        
        f[i] = s

    if options['Stats'] == 'on':
        print(f'Interpolation at {len(f)} points was computed in {time.time() - start_time:.6e} sec')
    
    return f

# # Radial Base Functions
# def rbfphi_linear(r, const):
#     return r

# def rbfphi_cubic(r, const):
#     return r ** 3

# def rbfphi_gaussian(r, const):
#     return np.exp(-0.5 * r ** 2 / const ** 2)

# def rbfphi_multiquadrics(r, const):
#     return np.sqrt(1 + r ** 2 / const ** 2)

# def rbfphi_thinplate(r, const):
#     return r ** 2 * np.log(r + 1)
