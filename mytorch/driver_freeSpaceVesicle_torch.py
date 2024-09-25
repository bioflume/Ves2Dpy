
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from curve import Curve
from wrapper_MLARM_torch import MLARM_py
import time
from scipy.io import loadmat

device = torch.device("cpu")
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
        return lambda X: np.vstack((speed * (1 - (X[N:] / width) ** 2), np.zeros_like(X[:N])))  # Parabolic
    elif bgFlow == 'rotation':
        # need check
        return lambda X: speed * np.vstack((-np.sin(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N:] ** 2 + X[N:] ** 2),
                                    np.cos(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N] ** 2 + X[N:] ** 2))) 
    else:
        return lambda X: np.zeros_like(X)

# Flow specification
bgFlow = 'parabolic'
speed = 750
width = 0.32275
vinf = set_bg_flow(bgFlow, speed)

# Time stepping
dt = 1e-5  # Time step size
# Th = 0.15  # Time horizon
Th = 100*dt

# Vesicle discretization
N = 128  # Number of points to discretize vesicle

# Vesicle initialization
nv = 1  # Number of vesicles
ra = 0.65  # Reduced area -- networks trained with vesicles of ra = 0.65
X0 = oc.ellipse(N, torch.tensor([ra]))  # Initial vesicle as ellipsoid
# X0.shape: (256,1)

nv = X0.shape[1]

# Arc-length is supposed to be 1 so divide by the initial length
area0, len0 = oc.geomProp(X0)[1:]
X0 = X0 / len0
center = torch.tensor([0, 0.065]).double() # Center in [x, y]
IA = torch.tensor(torch.pi / 2)  # Inclination angle
X = torch.zeros_like(X0)
X[:N] = torch.cos(IA) * X0[:N] - torch.sin(IA) * X0[N:] + center[0]
X[N:] = torch.sin(IA) * X0[:N] + torch.cos(IA) * X0[N:] + center[1]


# Build MLARM class to take time steps using networks
# Load the normalization (mean, std) values for the networks

adv_net_input_norm = np.load("../trained/ves_adv_trained/ves_fft_in_para.npy")
adv_net_output_norm = np.load("../trained/ves_adv_trained/ves_fft_out_para.npy")
# Relax Net for dt = 1E-5 (DIFF_June8)
relax_net_input_norm = np.array([7.684710645605719e-09, 0.06278636306524277,
                         7.071167829053593e-08, 0.13339479267597198])
relax_net_output_norm = np.array([4.258172037197028e-09, 0.001633652369491756,
                                  7.698989001880818e-09, 0.0014213572721928358])
# nearNetInputNorm = np.load("../trained/in_param_allmode.npy")
# nearNetOutputNorm = np.load("../trained/out_param_allmode.npy")
# tenSelfNetInputNorm = np.array([1.375491734401102e-11, 0.060100164264440536,
#                         2.017387923380909e-10, 0.13698625564575195])
# tenSelfNetOutputNorm = np.array([327.23284912109375, 375.04327392578125])
# tenAdvNetInputNorm = np.load("../trained/ves_advten_models/ves_advten_in_param.npy")
# tenAdvNetOutputNorm = np.load("../trained/ves_advten_models/ves_advten_out_param.npy")

mlarm = MLARM_py(dt, vinf, oc, 
                torch.from_numpy(adv_net_input_norm), torch.from_numpy(adv_net_output_norm),
                torch.from_numpy(relax_net_input_norm), torch.from_numpy(relax_net_output_norm),
                device=device)

area0, len0 = oc.geomProp(X)[1:]
mlarm.area0 = area0
mlarm.len0 = len0
# print(f"area0 {area0} len0 {len0}")
for w in range(5):
    X, _, _ = oc.redistributeArcLength(X)

# area0, len0 = oc.geomProp(X)[1:]
# mlarm.area0 = area0
# mlarm.len0 = len0
# print(f"area0 {area0} len0 {len0}")

# Save the initial data
# with open(fileName, 'wb') as fid:
#     np.array([N, nv]).tofile(fid)
#     X.cpu().numpy().flatten().astype('float64').tofile(fid)

# Evolve in time
currtime = 0
it = 0

while currtime < Th:
    # Take a time step
    tStart = time.time()
    # X = np.random.rand(24,3)
    # Ten = np.random.rand(12,3)
    # print("testing X:")
    # print(X)
    # print("testing ten:")
    # print(Ten)
    
    X = mlarm.time_step(X)
    tEnd = time.time()

    # Find error in area and length
    area, length = oc.geomProp(X)[1:]
    errArea = torch.max(torch.abs(area - mlarm.area0) / mlarm.area0)
    errLen = torch.max(torch.abs(length - mlarm.len0) / mlarm.len0)

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
    # output = np.concatenate(([currtime], X.flatten())).astype('float64')
    # with open(fileName, 'ab') as fid:
    #     output.tofile(fid)

# np.save("X_final_multi.npy", X)
# np.save("X_final1.npy", X)