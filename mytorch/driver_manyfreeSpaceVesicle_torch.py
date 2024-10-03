
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from curve import Curve
from wrapper_MLARM_torch import MLARM_py, MLARM_manyfree_py
# from MLARM_rbfcheck import MLARM_py, MLARM_manyfree_py
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        return lambda X: np.vstack((speed * (1 - (X[N:] / 0.375) ** 2), np.zeros_like(X[:N])))  # Parabolic
    elif bgFlow == 'rotation':
        # need check
        return lambda X: speed * np.vstack((-np.sin(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N:] ** 2 + X[N:] ** 2),
                                    np.cos(np.arctan2(X[N:], X[:N])) / np.sqrt(X[:N] ** 2 + X[N:] ** 2))) 
    else:
        return lambda X: np.zeros_like(X)

# Flow specification
bgFlow = 'shear'
speed = 2000
vinf = set_bg_flow(bgFlow, speed)

# Time stepping
dt = 1e-5  # Time step size
# Th = 0.15  # Time horizon
Th = 30*dt

# Vesicle discretization
N = 128  # Number of points to discretize vesicle

# Vesicle initialization
# nv = 3  # Number of vesicles
# ra = torch.tensor([0.65])  # Reduced area -- networks trained with vesicles of ra = 0.65
# X0 = oc.ellipse(N, ra)  # Initial vesicle as ellipsoid
# X0 = torch.repeat_interleave(X0, 3, dim=1)
# X0.shape: (256,1)

init_data = loadmat("../shearIC.mat") ### INIT SHAPES FROM THE DATA SET
Xics = init_data.get('Xic')
X0 = torch.from_numpy(Xics)
nv = X0.shape[1]
area0, len0 = oc.geomProp(X0)[1:]
print(f"len0 is {len0}")
X = X0.clone().to(device)

# init_data = loadmat("../initShapes.mat") ### INIT SHAPES FROM THE DATA SET
# Xics = init_data.get('Xics')
# X0 = torch.from_numpy(Xics[:,0:2])
# # X0[:128, 1] = X0[:128, 1] + 0.62
# # X0[128:, 1] = X0[128:, 1] + 0.6
# X0[:128, 1] = X0[:128, 1] + 0.23
# X0[128:, 1] = X0[128:, 1] + 0.2
# # X0[:128, 2] = X0[:128, 2] - 0.2
# # X0[128:, 2] = X0[128:, 2] + 0.2
# # X0[:128, 3] = X0[:128, 3] + 0.2
# # X0[128:, 3] = X0[128:, 3] + 0.6
# nv = X0.shape[1]
print(f"We have {nv} vesicles")
Ten = torch.from_numpy(np.zeros((128,nv))).to(device)


# Arc-length is supposed to be 1 so divide by the initial length
# area0, len0 = oc.geomProp(X0)[1:]
# X0 = X0 / len0
# X = X0.clone()
# center = torch.tensor([0, 0.065]).double() # Center in [x, y]
# IA = torch.tensor(torch.pi / 2)  # Inclination angle
# X = torch.zeros_like(X0)
# X[:N] = torch.cos(IA) * X0[:N] - torch.sin(IA) * X0[N:] + center[0]
# X[N:] = torch.sin(IA) * X0[:N] + torch.cos(IA) * X0[N:] + center[1]
# X[:128, 0] += 0.1
# X[128:, 0] += 0.2
# X[:128, 1] += 0.28
# X[128:, 1] += 0.4
# X[:128, 2] += 0.5
# X[128:, 2] -= 0.4


# plt.figure()
# plt.scatter(X[:128,0], X[128:,0], label="0")
# plt.scatter(X[:128,1], X[128:,1], label="1")
# plt.scatter(X[:128,2], X[128:,2], label="2")
# plt.axis("scaled")
# plt.legend()
# plt.show()


# Build MLARM class to take time steps using networks
# Load the normalization (mean, std) values for the networks

adv_net_input_norm = np.load("../trained/ves_adv_trained/ves_fft_in_para.npy")
adv_net_output_norm = np.load("../trained/ves_adv_trained/ves_fft_out_para.npy")
# Relax Net for dt = 1E-5 (DIFF_June8)
relax_net_input_norm = np.array([7.684710645605719e-09, 0.06278636306524277,
                         7.071167829053593e-08, 0.13339479267597198])
relax_net_output_norm = np.array([4.258172037197028e-09, 0.001633652369491756,
                                  7.698989001880818e-09, 0.0014213572721928358])
nearNetInputNorm = np.load("../trained/in_param_allmode.npy")
nearNetOutputNorm = np.load("../trained/out_param_allmode.npy")
tenSelfNetInputNorm = np.array([2.980232033378272e-11, 0.06010082736611366, 
 -1.0086939616904544e-10, 0.13698545098304749])
tenSelfNetOutputNorm = np.array([327.26141357421875, 375.0673828125 ])
tenAdvNetInputNorm = np.load("../trained/ves_advten_models/ves_advten_in_param.npy")
tenAdvNetOutputNorm = np.load("../trained/ves_advten_models/ves_advten_out_param.npy")

mlarm = MLARM_manyfree_py(dt, vinf, oc, 
                torch.from_numpy(adv_net_input_norm), torch.from_numpy(adv_net_output_norm),
                torch.from_numpy(relax_net_input_norm), torch.from_numpy(relax_net_output_norm),
                torch.from_numpy(nearNetInputNorm),torch.from_numpy(nearNetOutputNorm), 
                torch.from_numpy(tenSelfNetInputNorm), torch.from_numpy(tenSelfNetOutputNorm),
                torch.from_numpy(tenAdvNetInputNorm), torch.from_numpy(tenAdvNetOutputNorm), device=device)

area0, len0 = oc.geomProp(X)[1:]
mlarm.area0 = area0
mlarm.len0 = len0
for w in range(5):
    X, _, _ = oc.redistributeArcLength(X)

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
    # X = np.random.rand(24,3)
    # Ten = np.random.rand(12,3)
    # print("testing X:")
    # print(X)
    # print("testing ten:")
    # print(Ten)
    
    X, newTen = mlarm.time_step(X, Ten)
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
    print(f'{it}th time step, time: {currtime}')
    print(f'Solving with networks takes {tEnd - tStart} sec.')
    print(f'Error in area and length: {max(errArea, errLen)}')
    print('********************************************\n')

    # Save data
    output = np.concatenate(([currtime], X.cpu().numpy().T.flatten())).astype('float64')
    with open(fileName, 'ab') as fid:
        output.tofile(fid)
