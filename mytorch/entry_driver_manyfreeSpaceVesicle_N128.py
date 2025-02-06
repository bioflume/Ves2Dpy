# %%
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
from curve_batch import Curve
# from wrapper_MLARM import MLARM_manyfree_py
# from wrapper_MLARM_nearSubtract import MLARM_manyfree_py
# from wrapper_MLARM_batch import MLARM_manyfree_py
from wrapper_MLARM_batch_opt_N128 import MLARM_manyfree_py
# from wrapper_MLARM_batch_profile import MLARM_manyfree_py
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
from filter import interpft

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cur_dtype = torch.float32
# Load curve_py
oc = Curve()

# File name
# fileName = './output/48inTG.bin'  # To save simulation data
# fileName = './output/linshi.bin'  # To save simulation data
fileName = './output/ls.bin'  # To save simulation data

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
            chanWidth = 7.25
            return speed * torch.cat([
                torch.sin(X[:X.shape[0]//2] / chanWidth * torch.pi) * torch.cos(X[X.shape[0]//2:] / chanWidth * torch.pi),
                -torch.cos(X[:X.shape[0]//2] / chanWidth * torch.pi) * torch.sin(X[X.shape[0]//2:] / chanWidth * torch.pi)], dim=0)
        else:
            return torch.zeros_like(X)
    
    return get_flow

# Flow specification
# bgFlow = 'shear'
# speed = 2000
# vinf = set_bg_flow(bgFlow, speed)

bgFlow = 'vortex'
speed = 500
vinf = set_bg_flow(bgFlow, speed)


# Time stepping
dt = 1e-5  # Time step size
Th = 50*dt # Time horizon

# Vesicle discretization
N = 128  # Number of points to discretize vesicle


# init_data = loadmat("../shearIC.mat") ### INIT SHAPES FROM THE DATA SET
# init_data = loadmat("../48vesiclesInTG_N128.mat") ### INIT SHAPES FROM THE DATA SET
Xics = loadmat("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/ManyVesICsTaylorGreen/nv504IC.mat").get('X')
# Xics = init_data.get('Xic')
# Xics = np.load("TG_new_start.npy")
# Xics = loadmat("../3VesNearCheck.mat").get("X")
X0 = torch.from_numpy(Xics).to(device)

if X0.shape[0] != 2*N:
    X0 = torch.concat((interpft(X0[:32], N), interpft(X0[32:], N)), dim=0)

X0 = X0.float()
nv = X0.shape[1]
area0, len0 = oc.geomProp(X0)[1:]
print(f"area0 is {area0}")
print(f"len0 is {len0}")
X = X0.clone().to(device)

# X0 = [0.8543, 0.8469, 0.8567, 0.8546, 0.8172, 0.8049, 0.8166, 0.8141, 0.7809,
#         0.7729, 0.7910, 0.8095, 0.8225, 0.8390, 0.8599, 0.8702, 0.9030, 0.9206,
#         0.9166, 0.8989, 0.9052, 0.8698, 0.8860, 0.9075, 0.9192, 0.9445, 0.9529,
#         0.9491, 0.9371, 0.9187, 0.9004, 0.8795, 0.8603, 0.8380, 0.8195, 0.7968,
#         0.7786, 0.7511, 0.7309, 0.7015, 0.6890, 0.6566, 0.6481, 0.6533, 0.6860,
#         0.6994, 0.7257, 0.7448, 0.7733, 0.7937, 0.8192, 0.8387, 0.8647, 0.8864,
#         0.9145, 0.9354, 0.9572, 0.9722, 0.9877, 0.9692, 0.9481, 0.9318, 0.9054,
#         0.8844, 0.8561, 0.8370, 0.8110, 0.7934, 0.7656, 0.7480, 0.7203, 0.7037,
#         0.6737, 0.6574, 0.6286, 0.6207, 0.5978, 0.5937, 0.6122, 0.6358, 0.6609,
#         0.6856, 0.7154, 0.7416, 0.7730, 0.7971, 0.8266, 0.8490, 0.8794, 0.9016,
#         0.9317, 0.9520, 0.9827, 1.0039, 1.0369, 1.0578, 1.0909, 1.1086, 1.1427,
#         1.1542, 1.1711, 1.1397, 1.1271, 1.0979, 1.0905, 1.1123, 1.0855, 1.0658,
#         1.0702, 1.0495, 1.0273, 1.0292, 1.0036, 0.9854, 0.9773, 0.9505, 0.9328,
#         0.9467, 0.9452, 0.9180, 0.9255, 0.9323, 0.9045, 0.9156, 0.9096, 0.8773,
#         0.8890, 0.8865, 1.8717, 1.8622, 1.8282, 1.8138, 1.8109, 1.8020, 1.7676,
#         1.7490, 1.7393, 1.7158, 1.6932, 1.6794, 1.6513, 1.6399, 1.6613, 1.6859,
#         1.6832, 1.6716, 1.6767, 1.7217, 1.7243, 1.7557, 1.7527, 1.7327, 1.7339,
#         1.7160, 1.7219, 1.7554, 1.7779, 1.8016, 1.8254, 1.8452, 1.8669, 1.8867,
#         1.9070, 1.9266, 1.9464, 1.9676, 1.9866, 2.0077, 2.0203, 2.0434, 2.0528,
#         2.0495, 2.0190, 2.0073, 1.9847, 1.9696, 1.9473, 1.9305, 1.9112, 1.8952,
#         1.8761, 1.8578, 1.8401, 1.8271, 1.8131, 1.8020, 1.8030, 1.8205, 1.8415,
#         1.8561, 1.8777, 1.8914, 1.9127, 1.9273, 1.9501, 1.9643, 1.9863, 2.0003,
#         2.0236, 2.0378, 2.0616, 2.0745, 2.0995, 2.1099, 2.1388, 2.1408, 2.1516,
#         2.1587, 2.1503, 2.1470, 2.1356, 2.1280, 2.1151, 2.1069, 2.0938, 2.0850,
#         2.0701, 2.0601, 2.0467, 2.0395, 2.0262, 2.0178, 2.0025, 1.9934, 1.9775,
#         1.9715, 1.9568, 1.9531, 1.9412, 1.9536, 1.9633, 1.9755, 1.9857, 2.0129,
#         2.0298, 2.0404, 2.0709, 2.0755, 2.0889, 2.1121, 2.1270, 2.1153, 2.1005,
#         2.1106, 2.1062, 2.0765, 2.0495, 2.0451, 2.0199, 2.0017, 1.9807, 1.9515,
#         1.9306, 1.9147, 1.8919, 1.8765]

# %matplotlib inline
# plt.figure()
# plt.plot(X0[:128], X0[128:])
# plt.axis('scaled')
# plt.show()

# %%
print(f"We have {nv} vesicles")
Ten = torch.from_numpy(np.zeros((128,nv))).float().to(device)


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

tenAdvNetInputNorm = np.load("../trained/2024Oct_advten_in_para_allmodes.npy")
tenAdvNetOutputNorm = np.load("../trained/2024Oct_advten_out_para_allmodes.npy")

mlarm = MLARM_manyfree_py(dt, vinf, oc,  False, 1e2, 
                torch.from_numpy(adv_net_input_norm).to(cur_dtype), torch.from_numpy(adv_net_output_norm).to(cur_dtype),
                torch.from_numpy(relax_net_input_norm).to(cur_dtype), torch.from_numpy(relax_net_output_norm).to(cur_dtype),
                torch.from_numpy(nearNetInputNorm).to(cur_dtype), torch.from_numpy(nearNetOutputNorm).to(cur_dtype), 
                torch.from_numpy(tenSelfNetInputNorm).to(cur_dtype), torch.from_numpy(tenSelfNetOutputNorm).to(cur_dtype),
                torch.from_numpy(tenAdvNetInputNorm).to(cur_dtype), torch.from_numpy(tenAdvNetOutputNorm).to(cur_dtype), 
                device=device,
                )

area0, len0 = oc.geomProp(X)[1:]
# mlarm.area0 = area0
mlarm.area0 = torch.ones((nv), device=X.device, dtype=torch.float32) * 0.0524
# mlarm.len0 = len0
mlarm.len0 = torch.ones((nv), device=X.device, dtype=torch.float32)

for _ in range(10):
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
    
    # X, Ten = mlarm.time_step_many(X, Ten)
    X, Ten = mlarm.time_step_many_timing(X, Ten)
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
