import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from curve_batch import Curve
# from wrapper_MLARM import MLARM_manyfree_py
# from wrapper_MLARM_nearSubtract import MLARM_manyfree_py
from wrapper_MLARM_batch import MLARM_manyfree_py
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cpu")
# Load curve_py
oc = Curve()

# File name
# fileName = './output/48inTG.bin'  # To save simulation data
fileName = './output/linshi.bin'  # To save simulation data

# def set_bg_flow(bgFlow, speed):
#     if bgFlow == 'relax':
#         return lambda X: np.zeros_like(X)  # Relaxation
#     elif bgFlow == 'shear':
#         return lambda X: speed * np.vstack((X[N:], np.zeros_like(X[:N])))  # Shear
#     elif bgFlow == 'tayGreen':
#         return lambda X: speed * np.vstack((np.sin(X[:N]) * np.cos(X[N:]), -np.cos(X[:N]) * np.sin(X[N:])))  # Taylor-Green
#     elif bgFlow == 'parabolic':
#         return lambda X: np.vstack((speed * (1 - (X[N:] / 0.375) ** 2), np.zeros_like(X[:N])))  # Parabolic
#     elif bgFlow == 'rotation':
#         # need check
#         return lambda X: speed * np.vstack((-np.sin(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N:] ** 2 + X[N:] ** 2),
#                                     np.cos(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N] ** 2 + X[N:] ** 2))) 
#     else:
#         return lambda X: np.zeros_like(X)

def set_bg_flow(bgFlow, speed):
    def get_flow(X):
        N = X.shape[0] // 2  # Assuming the input X is split into two halves
        if bgFlow == 'relax':
            return torch.zeros_like(X)  # Relaxation
        elif bgFlow == 'shear':
            return speed * torch.vstack((X[N:], torch.zeros_like(X[:N])))  # Shear
        elif bgFlow == 'tayGreen':
            return speed * torch.vstack((torch.sin(X[:N]) * torch.cos(X[N:]), -torch.cos(X[:N]) * torch.sin(X[N:])))  # Taylor-Green
        elif bgFlow == 'parabolic':
            return torch.vstack((speed * (1 - (X[N:] / 0.375) ** 2), torch.zeros_like(X[:N])))  # Parabolic
        elif bgFlow == 'rotation':
            r = torch.sqrt(X[:N] ** 2 + X[N:] ** 2)
            theta = torch.atan2(X[N:], X[:N])
            return speed * torch.vstack((-torch.sin(theta) / r, torch.cos(theta) / r))  # Rotation
        elif bgFlow == 'vortex':
            chanWidth = 2.5
            return speed * torch.cat([
                torch.sin(X[:X.shape[0]//2] / chanWidth * torch.pi) * torch.cos(X[X.shape[0]//2:] / chanWidth * torch.pi),
                -torch.cos(X[:X.shape[0]//2] / chanWidth * torch.pi) * torch.sin(X[X.shape[0]//2:] / chanWidth * torch.pi)], dim=0)
        else:
            return torch.zeros_like(X)
    
    return get_flow

# Flow specification
bgFlow = 'shear'
speed = 2000
vinf = set_bg_flow(bgFlow, speed)

# bgFlow = 'vortex'
# speed = 200
# vinf = set_bg_flow(bgFlow, speed)


# Time stepping
dt = 1e-5  # Time step size
Th = 300*dt # Time horizon

# Vesicle discretization
N = 128  # Number of points to discretize vesicle


init_data = loadmat("../shearIC.mat") ### INIT SHAPES FROM THE DATA SET
# init_data = loadmat("../48vesiclesInTG_N128.mat") ### INIT SHAPES FROM THE DATA SET
Xics = init_data.get('Xic')
X0 = torch.from_numpy(Xics).to(device)
nv = X0.shape[1]
area0, len0 = oc.geomProp(X0)[1:]
print(f"len0 is {len0}")
X = X0.clone().to(device)

print(f"We have {nv} vesicles")
Ten = torch.from_numpy(np.zeros((128,nv))).to(device)


# Build MLARM class to take time steps using networks
# Load the normalization (mean, std) values for the networks
# ADV Net retrained in Oct 2024
adv_net_input_norm = np.load("../trained/2024Oct_adv_fft_tot_in_para.npy")
adv_net_output_norm = np.load("../trained/2024Oct_adv_fft_tot_out_para.npy")
# Relax Net for dt = 1E-5 (DIFF_June8)
relax_net_input_norm = np.array([-8.430413700466488e-09, 0.06278684735298157,
                                6.290720477863943e-08, 0.13339413702487946])
relax_net_output_norm = np.array([-2.884585348361668e-10, 0.00020574081281665713,
                                -5.137390512999218e-10, 0.0001763451291481033])
# nearNetInputNorm = np.load("../trained/in_param_allmode.npy")
# nearNetOutputNorm = np.load("../trained/out_param_allmode.npy")
nearNetInputNorm = np.load("../trained/in_param_disth_allmode.npy")
nearNetOutputNorm = np.load("../trained/out_param_disth_allmode.npy")
# tenSelfNetInputNorm = np.array([2.980232033378272e-11, 0.06010082736611366, 
#                         -1.0086939616904544e-10, 0.13698545098304749])
# tenSelfNetOutputNorm = np.array([327.26141357421875, 375.0673828125 ])

# self ten network updated by using a 156k dataset
tenSelfNetInputNorm = np.array([0.00017108717293012887, 0.06278623640537262, 
                        0.002038202714174986,0.13337858021259308])
tenSelfNetOutputNorm = np.array([337.7627868652344, 466.6429138183594])

# tenAdvNetInputNorm = np.load("../trained/ves_advten_models/ves_advten_in_param.npy")
# tenAdvNetOutputNorm = np.load("../trained/ves_advten_models/ves_advten_out_param.npy")
tenAdvNetInputNorm = np.load("../trained/2024Oct_advten_in_para_allmodes.npy")
tenAdvNetOutputNorm = np.load("../trained/2024Oct_advten_out_para_allmodes.npy")

mlarm = MLARM_manyfree_py(dt, vinf, oc, 
                torch.from_numpy(adv_net_input_norm), torch.from_numpy(adv_net_output_norm),
                torch.from_numpy(relax_net_input_norm), torch.from_numpy(relax_net_output_norm),
                torch.from_numpy(nearNetInputNorm),torch.from_numpy(nearNetOutputNorm), 
                torch.from_numpy(tenSelfNetInputNorm), torch.from_numpy(tenSelfNetOutputNorm),
                torch.from_numpy(tenAdvNetInputNorm), torch.from_numpy(tenAdvNetOutputNorm), device=device)

area0, len0 = oc.geomProp(X)[1:]
mlarm.area0 = area0
mlarm.len0 = len0
for _ in range(5):
    X, flag = oc.redistributeArcLength(X)
    if flag:
        break

# Save the initial data
with open(fileName, 'wb') as fid:
    np.array([N, nv]).flatten().astype('float64').tofile(fid)
    X.cpu().numpy().T.flatten().astype('float64').tofile(fid)

# Evolve in time
currtime = 0
# it = 0

# while currtime < Th:
for it in tqdm(range(int(Th//dt))): 
    # Take a time step
    tStart = time.time()
    
    X, Ten = mlarm.time_step_many(X, Ten)
    # np.save(f"shape_t{currtime}.npy", X)
    tEnd = time.time()

    # Find error in area and length
    area, length = oc.geomProp(X)[1:]
    errArea = torch.max(torch.abs(area - mlarm.area0) / mlarm.area0)
    errLen = torch.max(torch.abs(length - mlarm.len0) / mlarm.len0)

    # Update counter and time
    # it += 1
    currtime += dt

    # Print time step info
    print('********************************************')
    print(f'{it+1}th time step, time: {currtime}')
    print(f'Solving with networks takes {tEnd - tStart} sec.')
    print(f'Error in area and length: {max(errArea, errLen)}')
    print('********************************************\n')

    # Save data
    output = np.concatenate(([currtime], X.cpu().numpy().T.flatten())).astype('float64')
    with open(fileName, 'ab') as fid:
        output.tofile(fid)
