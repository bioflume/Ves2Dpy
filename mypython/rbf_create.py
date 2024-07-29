import numpy as np
import time

def rbfcreate(x=None, y=None, **kwargs):
    """
    Creates an RBF interpolation.
    
    Parameters:
    x : ndarray
        dim by n matrix of coordinates for the nodes
    y : ndarray
        1 by n vector of values at nodes
    kwargs : dict
        Additional name-value pairs to set RBF properties
    
    Returns:
    options : dict
        Dictionary containing RBF properties and coefficients
    """
    start_time = time.time()
    
    # Print out possible values of properties if no arguments are provided
    if x is None and y is None:
        print('               x: [ dim by n matrix of coordinates for the nodes ]')
        print('               y: [   1 by n vector of values at nodes ]')
        print('     RBFFunction: [ gaussian  | thinplate | cubic | multiquadrics | {linear} ]')
        print('     RBFConstant: [ positive scalar     ]')
        print('       RBFSmooth: [ positive scalar {0} ]')
        print('           Stats: [ on | {off} ]')
        print('\n')
        return
    
    Names = ['RBFFunction', 'RBFConstant', 'RBFSmooth', 'Stats']
    names = [name.lower() for name in Names]
    
    options = {name: None for name in Names}
    
    #**************************************************************************
    # Check input arrays 
    #**************************************************************************
    nXDim, nXCount = x.shape
    nYDim, nYCount = y.shape

    if nXCount != nYCount:
        raise ValueError('x and y should have the same number of rows')

    if nYDim != 1:
        raise ValueError('y should be n by 1 vector')

    options['x'] = x
    options['y'] = y

    #**************************************************************************
    # Default values 
    #**************************************************************************
    options['RBFFunction'] = 'linear'
    options['RBFConstant'] = (np.prod(np.max(x, axis=1) - np.min(x, axis=1)) / nXCount) ** (1 / nXDim)  # approx. average distance between the nodes 
    options['RBFSmooth'] = 0
    options['Stats'] = 'off'

    #**************************************************************************
    # Argument parsing code: similar to ODESET.m
    #**************************************************************************
    # if (len(kwargs) % 2) != 0:
    #     raise ValueError('Arguments must occur in name-value pairs.')
    
    for arg, val in kwargs.items():
        if arg.lower() not in names:
            raise ValueError(f"Unrecognized property name '{arg}'")
        options[Names[names.index(arg.lower())]] = val
    
    #**************************************************************************
    # Creating RBF Interpolation
    #**************************************************************************
    rbf_functions = {
        'linear': rbfphi_linear,
        'cubic': rbfphi_cubic,
        'multiquadrics': rbfphi_multiquadrics,
        'thinplate': rbfphi_thinplate,
        'gaussian': rbfphi_gaussian
    }

    rbf_function_name = options['RBFFunction'].lower()
    options['rbfphi'] = rbf_functions.get(rbf_function_name, rbfphi_linear)
    phi = options['rbfphi']

    A = rbfAssemble(x, phi, options['RBFConstant'], options['RBFSmooth'])
    b = np.concatenate([y.reshape(-1), np.zeros(nXDim + 1)])
    rbfcoeff = np.linalg.solve(A, b)
    options['rbfcoeff'] = rbfcoeff

    if options['Stats'] == 'on':
        print(f"{len(y)} point RBF interpolation was created in {time.time() - start_time:.6e} sec")
        print('\n')
    
    return options


def rbfAssemble(x, phi, const, smooth):
    dim, n = x.shape
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            r = np.linalg.norm(x[:, i] - x[:, j])
            temp = phi(r, const)
            A[i, j] = temp
            A[j, i] = temp
        A[i, i] -= smooth
    
    P = np.hstack((np.ones((n, 1)), x.T))
    A = np.block([
        [A, P],
        [P.T, np.zeros((dim + 1, dim + 1))]
    ])
    return A

#**************************************************************************
# Radial Base Functions
#************************************************************************** 
def rbfphi_linear(r, const):
    return r

def rbfphi_cubic(r, const):
    return r ** 3

def rbfphi_gaussian(r, const):
    return np.exp(-0.5 * r ** 2 / const ** 2)

def rbfphi_multiquadrics(r, const):
    return np.sqrt(1 + r ** 2 / const ** 2)

def rbfphi_thinplate(r, const):
    return r ** 2 * np.log(r + 1)
