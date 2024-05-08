import numpy as np
from curve import Curve
from MLARM import MLARM_py
import time

# Load curve_py
oc = Curve()

# File name
fileName = './output/test.bin'  # To save simulation data

def set_bg_flow(bgFlow, speed):
    if bgFlow == 'relax':
        return lambda X: np.zeros_like(X)  # Relaxation
    elif bgFlow == 'shear':
        return lambda X: speed * np.vstack((X[N:], np.zeros_like(X[:N])))  # Shear
    elif bgFlow == 'tayGreen':
        return lambda X: speed * np.vstack((np.sin(X[:N]) * np.cos(X[N:]), -np.cos(X[:N]) * np.sin(X[N:])))  # Taylor-Green
    elif bgFlow == 'parabolic':
        return lambda X: np.vstack((speed * (1 - (X[N:] / 1.3) ** 2), np.zeros_like(X[:N])))  # Parabolic
    elif bgFlow == 'rotation':
        # need check
        return lambda X: speed * np.vstack((-np.sin(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N:] ** 2 + X[N:] ** 2),
                                    np.cos(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N] ** 2 + X[N:] ** 2))) 
    else:
        return lambda X: np.zeros_like(X)

# Flow specification
bgFlow = 'parabolic'
speed = 12000
vinf = set_bg_flow(bgFlow, speed)

# Time stepping
dt = 1e-5  # Time step size
# Th = 0.15  # Time horizon
Th = 2*dt

# Vesicle discretization
N = 128  # Number of points to discretize vesicle

# Vesicle initialization
nv = 1  # Number of vesicles
ra = 0.65  # Reduced area -- networks trained with vesicles of ra = 0.65
X0 = oc.ellipse(N, ra)  # Initial vesicle as ellipsoid
# X0.shape: (256,1)

# Arc-length is supposed to be 1 so divide by the initial length
area0, len0 = oc.geomProp(X0)[1:]
X0 = X0 / len0

center = np.array([0, 0.065])  # Center in [x, y]
IA = np.pi / 2  # Inclination angle
X = np.zeros_like(X0)
X[:N] = np.cos(IA) * X0[:N] - np.sin(IA) * X0[N:] + center[0]
X[N:] = np.sin(IA) * X0[:N] + np.cos(IA) * X0[N:] + center[1]

# Build MLARM class to take time steps using networks
# Load the normalization (mean, std) values for the networks
# adv_net_input_norm = np.load('./shannets/ves_fft_in_param.npy')
# adv_net_output_norm = np.load('./shannets/ves_fft_out_param.npy')
# relax_net_input_norm = np.load('./shannets/ves_relax_dt1E5.npy')
# relax_net_output_norm = np.load('./shannets/ves_relax_dt1E5.npy')
adv_net_input_norm = np.load("../ves_adv_trained/ves_fft_in_para.npy")
adv_net_output_norm = np.load("../ves_adv_trained/ves_fft_out_para.npy")
relax_net_input_norm = [-2.321531638801999e-12, 0.0626436322927475,
                         -3.3723935102814018e-12, 0.13317303359508514]
relax_net_output_norm = [-3.548781546403035e-10, 0.06260271370410919,
                          -6.015386522228994e-10, 0.13323774933815002]

mlarm = MLARM_py(dt, vinf, oc, adv_net_input_norm, adv_net_output_norm,
                 relax_net_input_norm, relax_net_output_norm)
area0, len0 = oc.geomProp(X)[1:]
mlarm.area0 = area0
mlarm.len0 = len0

# Save the initial data
with open(fileName, 'wb') as fid:
    np.array([N, nv]).tofile(fid)
    X.flatten().astype('float64').tofile(fid)

# Evolve in time
currtime = 0
it = 0
while currtime < Th:
    # Take a time step
    tStart = time.time()
    X = mlarm.time_step(X)
    tEnd = time.time()

    # Find error in area and length
    area, length = oc.geomProp(X)[1:]
    errArea = np.max(np.abs(area - mlarm.area0) / mlarm.area0)
    errLen = np.max(np.abs(length - mlarm.len0) / mlarm.len0)

    # Update counter and time
    it += 1
    currtime += dt

    # Print time step info
    print('********************************************')
    print(f'{it}th time step, time: {currtime}')
    print(f'Solving with networks takes {tEnd - tStart} sec.')
    print(f'Error in area and length: {max(errArea, errLen)}')
    print('********************************************\n')

    # Save data
    output = np.concatenate(([currtime], X.flatten())).astype('float64')
    with open(fileName, 'ab') as fid:
        output.tofile(fid)
