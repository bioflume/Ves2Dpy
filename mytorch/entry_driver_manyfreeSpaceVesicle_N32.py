# %%
import numpy as np
import torch
torch.set_default_dtype(torch.float32)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from curve_batch import Curve
# from wrapper_MLARM import MLARM_manyfree_py
# from wrapper_MLARM_nearSubtract import MLARM_manyfree_py
# from wrapper_MLARM_batch import MLARM_manyfree_py
# from wrapper_MLARM_batch_profiling import MLARM_manyfree_py
from wrapper_MLARM_batch_opt_N32 import MLARM_manyfree_py

import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
from filter import interpft
# from utils.load_bin import load_single_ves_file_py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cur_dtype = torch.float32
# Load curve_py
oc = Curve()

# nv = 504
# File name

# fileName = './output/48inTG_steporder.bin'  # To save simulation data
# fileName = './output/linshi.bin'  # To save simulation data
# fileName = './output/up.bin'  # To save simulation data
fileName = './output/ls.bin'  # To save simulation data
# fileName = './output/lsls.bin'  # To save simulation data
# fileName = './output/ls_repulse1e4.bin'  # To save simulation data


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
            # chanWidth = loadmat(f"../ManyVesICsTaylorGreen/nv{nv}IC.mat").get('chanWidth')[0][0]
            # print(f"chanWidth is {chanWidth}")
            chanWidth = 2.5
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
speed = 200
vinf = set_bg_flow(bgFlow, speed)

# vel_field = vinf(torch.meshgrid(torch.linspace()))
def generate_indices(start, step, length, range_size):
    indices = []
    for i in range(length):
        start_range = start + i * step
        indices.extend(range(start_range, start_range + range_size))
    return indices

# Example: Generate indices for 0-5, 12-17, 24-29, ...
start = 0       # Starting index
step = 12       # Step size
length = 9     # Number of groups
range_size = 6  # Range size

indices = generate_indices(start, step, length, range_size)
# print(indices)

# Time stepping
num_steps = 6000
dt = 1e-5  # Time step size
Th = num_steps * dt # Time horizon

# Vesicle discretization
N = 32  # Number of points to discretize vesicle

# /work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/

# Xics = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/shear_N32.npy") ### INIT SHAPES FROM THE DATA SET
Xics = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/48vesTG_N32.npy") ### INIT SHAPES FROM THE DATA SET
# Xics = loadmat("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/ManyVesICsTaylorGreen/nv504IC.mat").get('X')
# Xics = loadmat("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/ManyVesICsTaylorGreen/nv1020IC.mat").get('X')
# Xics = loadmat(f"../ManyVesICsTaylorGreen/nv504IC.mat").get('X')
# Xics = loadmat("/work/09452/alberto47/stampede3/ves_MATLAB/Ves2Dpy/matlab_N32/4VesNearCheck_N32.mat").get('X')
# Xics = np.load("TG_new_start.npy")
# vesx, vesy, _, _, _, _, _ = load_single_ves_file_py("output/lsls.bin")
# Xics = np.concatenate((vesx[:,:,2902], vesy[:,:,2902]), axis=0)
X0 = torch.from_numpy(Xics).to(device)
# X0 = torch.concat((X0[:, 1:2], X0[:, 1:2]), dim=1)
# X0[:32, 1] += 0.15 + 0.2
# X0[:32, 0] +=  0.2
# X0[32:] += 0.25

if X0.shape[0] != 2*N:
    X0 = torch.concat((interpft(X0[:128], N), interpft(X0[128:], N)), dim=0)

X0 = X0.float()
nv = X0.shape[1]
area0, len0 = oc.geomProp(X0)[1:]
print(f"area0 is {area0}")
print(f"len0 is {len0}")
X = X0.clone().to(device)

# Ten_save = torch.zeros((num_steps, N, nv), dtype=torch.float64, device=device)

# %matplotlib inline

# plt.figure()
# plt.plot(X0[:32], X0[32:])
# plt.axis("scaled")
# plt.show()

# %%
print(f"We have {nv} vesicles")
Ten = torch.zeros((32,nv)).to(device)
# Ten = torch.from_numpy(np.load("output/Ten_save.npy")[3603,...]).to(device)

# Build MLARM class to take time steps using networks
# Load the normalization (mean, std) values for the networks
# ADV Net retrained in Oct 2024
adv_net_input_norm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/adv_fft_ds32/2024Oct_advfft_in_para_downsample_all_mode.npy")
adv_net_output_norm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/adv_fft_ds32/2024Oct_advfft_out_para_downsample_all_mode.npy")
# Relax Net
relax_net_input_norm = np.array([-1.5200416214611323e-07, 0.06278670579195023,
                            -2.5547041104800883e-07, 0.13339416682720184])
relax_net_output_norm = np.array([-2.329148207635967e-09, 0.00020403489179443568,
                            -1.5361016902915026e-09, 0.00017457373905926943])

nearNetInputNorm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/near_trained/in_param_downsample32_allmode.npy")
nearNetOutputNorm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/near_trained/out_param_downsample32_allmode.npy")
# inner Near Field
innerNearNetInputNorm = np.load("/work/09452/alberto47/ls6/vesicle_nearF2024/trained_disth_nocoords/inner_downsample32/inner_near_in_param_allmodes.npy")
innerNearNetOutputNorm = np.load("/work/09452/alberto47/ls6/vesicle_nearF2024/trained_disth_nocoords/inner_downsample32/inner_near_out_param_allmodes.npy")

# self ten network updated by using a 156k dataset
tenSelfNetInputNorm = np.array([0.00016914503066800535, 0.06278414279222488,
                            0.0020352655556052923,  0.13338139653205872])
tenSelfNetOutputNorm = np.array([337.7410888671875, 458.4122314453125 ])

tenAdvNetInputNorm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/advten_downsample32/2024Nov_advten_ds32_in_para_allmodes.npy")
tenAdvNetOutputNorm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/advten_downsample32/2024Nov_advten_ds32_out_para_allmodes.npy")

mlarm = MLARM_manyfree_py(dt, vinf, oc, False, 1e3,
                torch.from_numpy(adv_net_input_norm).to(cur_dtype), torch.from_numpy(adv_net_output_norm).to(cur_dtype),
                torch.from_numpy(relax_net_input_norm).to(cur_dtype), torch.from_numpy(relax_net_output_norm).to(cur_dtype),
                torch.from_numpy(nearNetInputNorm).to(cur_dtype), torch.from_numpy(nearNetOutputNorm).to(cur_dtype), 
                torch.from_numpy(innerNearNetInputNorm).to(cur_dtype), torch.from_numpy(innerNearNetOutputNorm).to(cur_dtype), 
                torch.from_numpy(tenSelfNetInputNorm).to(cur_dtype), torch.from_numpy(tenSelfNetOutputNorm).to(cur_dtype),
                torch.from_numpy(tenAdvNetInputNorm).to(cur_dtype), torch.from_numpy(tenAdvNetOutputNorm).to(cur_dtype), 
                device=device)

area0, len0 = oc.geomProp(X)[1:]
mlarm.area0 = area0
# mlarm.area0 = torch.ones((4), device=X.device, dtype=torch.float64) * 0.0524
mlarm.len0 = len0
# mlarm.len0 = torch.ones((4), device=X.device, dtype=torch.float64)
for _ in range(10):
    X, flag = oc.redistributeArcLength(X)
    if flag:
        break

# Save the initial data
# tosave[0] = X
with open(fileName, 'wb') as fid:
    np.array([N, nv]).flatten().astype('float64').tofile(fid)
    X.cpu().numpy().T.flatten().astype('float64').tofile(fid)

# Evolve in time
currtime = 0
# it = 0

# mlarm.Xold = X
# mlarm.tenOld = Ten
# while currtime < Th:
for it in tqdm(range(num_steps)): 
    # Take a time step
    tStart = time.time()
    
    X, Ten = mlarm.time_step_many_timing(X, Ten)
    # X, Ten = mlarm.time_step_many(X, Ten)
    # mlarm.i += 1
    # mlarm.time_step_many_self()
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
    print(f'{it+1}th time step for N=32, time: {currtime}')
    print(f'Solving with networks takes {tEnd - tStart} sec.')
    print(f'Error in area and length: {max(errArea, errLen)}')
    print('********************************************\n')

    # Save data
    output = np.concatenate(([currtime], X.cpu().numpy().T.flatten())).astype('float64')
    with open(fileName, 'ab') as fid:
        output.tofile(fid)

    # Ten_save[it+1] = Ten
    # if it % 50 == 0:
    # np.save("output/Ten_save_getIncAngle.npy", Ten_save.cpu().numpy())
