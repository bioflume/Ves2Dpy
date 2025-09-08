import numpy as np
import torch
torch.set_default_dtype(torch.float32)
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch._dynamo
torch._dynamo.reset()
from curve_batch_compile import Curve
from wrapper_MLARM_batch_compile_N32 import MLARM_manyfree_py
from math import sqrt
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.filter import filterShape, interpft_vec


def simulate(outfile, logging, resolution,
         flow, 
         rbf_params, 
         repulsion_params):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cur_dtype = torch.float32
    # Load curve_py
    oc = Curve(logger)

    # File name
    fileName = f'./output/{outfile}.bin'  # To save simulation data

    def set_bg_flow(bgFlow, speed, chanWidth=2.5, vortexSize=2.5):
        def get_flow(X):
            N = X.shape[0] // 2  # Assuming the input X is split into two halves
            if bgFlow == 'relax':
                return torch.zeros_like(X)  # Relaxation
            elif bgFlow == 'shear':
                return speed * torch.vstack((X[N:], torch.zeros_like(X[:N])))  # Shear
            elif bgFlow == 'tayGreen':
                return speed * torch.vstack((torch.sin(X[:N]) * torch.cos(X[N:]), -torch.cos(X[:N]) * torch.sin(X[N:])))  # Taylor-Green
            elif bgFlow == 'parabolic':
                chanWidth = 5
                return torch.vstack((speed * (1 - (X[N:] / chanWidth) ** 2), torch.zeros_like(X[:N])))  # Parabolic
            elif bgFlow == 'rotation':
                r = torch.sqrt(X[:N] ** 2 + X[N:] ** 2)
                theta = torch.atan2(X[N:], X[:N])
                return speed * torch.vstack((-torch.sin(theta) / r, torch.cos(theta) / r))  # Rotation
            elif bgFlow == 'vortex':
                return speed * torch.cat([
                    torch.sin(X[:X.shape[0]//2] / chanWidth * torch.pi) * torch.cos(X[X.shape[0]//2:] / chanWidth * torch.pi),
                    -torch.cos(X[:X.shape[0]//2] / chanWidth * torch.pi) * torch.sin(X[X.shape[0]//2:] / chanWidth * torch.pi)], dim=0)
            else:
                return torch.zeros_like(X)
        
        return get_flow

    # Flow specification
    bgFlow = flow['name']
    speed = flow['speed']
    vinf = set_bg_flow(bgFlow, speed, flow['chanWidth'], flow['vortexSize'])

    # Time stepping
    dt = 1e-5  # Time step size
    Th = num_steps * dt # Time horizon

    # Vesicle discretization
    N = resolution  # Number of points to discretize vesicle

    Xics = np.load(input) ### INIT SHAPES FROM THE DATA SET
    print(Xics.shape)
    X0 = torch.from_numpy(Xics).to(device)

    if X0.shape[0] != 2*N:
        X0 = torch.concat((interpft(X0[:128], N), interpft(X0[128:], N)), dim=0)

    X0 = X0.float()
    nv = X0.shape[1]
    area0, len0 = oc.geomProp(X0)[1:]
    logger.info(f"area0 is {area0}")
    logger.info(f"len0 is {len0}")
    X = X0.clone().to(device)

    ########## 就做到这里吧

    # %%
    logger.info(f"We have {nv} vesicles")
    Ten = torch.zeros((32,nv)).to(device)
    # Ten = torch.from_numpy(np.load("debug_last_tenNew.npy")).to(device)

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
    # tenSelfNetInputNorm = np.array([0.00016914503066800535, 0.06278414279222488,
    #                             0.0020352655556052923,  0.13338139653205872])
    # tenSelfNetOutputNorm = np.array([337.7410888671875, 458.4122314453125 ])
    # adding curvature
    tenSelfNetInputNorm = np.array([0.00016983709065243602, 0.06278808414936066,
                                0.0020364541560411453, 0.13337676227092743,
                                6.277393817901611, 9.243043899536133])
    tenSelfNetOutputNorm = np.array([337.7682800292969, 458.4842834472656])

    tenAdvNetInputNorm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/advten_downsample32/2024Nov_advten_ds32_in_para_allmodes.npy")
    tenAdvNetOutputNorm = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/advten_downsample32/2024Nov_advten_ds32_out_para_allmodes.npy")
    # adding curvature
    # tenAdvNetInputNorm = np.load("/work/09452/alberto47/ls6/vesicle_advten/norm_para/2025Feb_downsample_in_para.npy")
    # tenAdvNetOutputNorm = np.load("/work/09452/alberto47/ls6/vesicle_advten/norm_para/2025Feb_downsample_out_para.npy")

    mlarm = MLARM_manyfree_py(dt, vinf, oc, use_repulsion, repulsion_strength, eta,
                    rbf_upsample,
                    torch.from_numpy(adv_net_input_norm).to(cur_dtype), torch.from_numpy(adv_net_output_norm).to(cur_dtype),
                    torch.from_numpy(relax_net_input_norm).to(cur_dtype), torch.from_numpy(relax_net_output_norm).to(cur_dtype),
                    torch.from_numpy(nearNetInputNorm).to(cur_dtype), torch.from_numpy(nearNetOutputNorm).to(cur_dtype), 
                    torch.from_numpy(innerNearNetInputNorm).to(cur_dtype), torch.from_numpy(innerNearNetOutputNorm).to(cur_dtype), 
                    torch.from_numpy(tenSelfNetInputNorm).to(cur_dtype), torch.from_numpy(tenSelfNetOutputNorm).to(cur_dtype),
                    torch.from_numpy(tenAdvNetInputNorm).to(cur_dtype), torch.from_numpy(tenAdvNetOutputNorm).to(cur_dtype), 
                    device=device, logger=logger)

    area0, len0 = oc.geomProp(X)[1:]
    mlarm.area0 = area0
    # mlarm.area0 = torch.ones((128), device=X.device, dtype=torch.float64) * 0.0524
    mlarm.len0 = len0
    # mlarm.len0 = torch.ones((128), device=X.device, dtype=torch.float64)
    modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(X.device) #.float()
    for _ in range(10):
        X, flag = oc.redistributeArcLength(X, modes)
        if flag:
            break

    # ellipse = loadmat("../VF25_TG32Ves.mat").get('X')[:, 0:1]
    ellipse = np.load("relaxed_shape.npy")
    ellipse = torch.from_numpy(ellipse).float().to(device)
    center_ = oc.getPhysicalCenter(ellipse)
    ellipse[:32, :] -= center_[0]
    ellipse[32:, :] -= center_[1]
    mlarm.ellipse = ellipse
    logger.info(f"center is {oc.getPhysicalCenter(mlarm.ellipse)}")


    # Save the initial data
    # tosave[0] = X
    with open(fileName, 'wb') as fid:
        np.array([N, nv]).flatten().astype('float64').tofile(fid)
        X.cpu().numpy().T.flatten().astype('float64').tofile(fid)

    # Evolve in time
    currtime = 0
    # it = 0
    # mlarm.save_farFieldtracJump = torch.zeros((64, nv, num_steps), dtype=torch.float32, device=device)
    # mlarm.i = 0

    logger.info(f"using {nlayers} layers, {mlarm.rbf_upsample} upsampling, saved as {fileName}")
    # mlarm.Xold = X
    # mlarm.tenOld = Ten
    # while currtime < Th:
    for it in tqdm(range(num_steps)): 
        # Take a time step
        tStart = time.time()
        
        with torch.no_grad(): # comment this if carrying AugLag correctAreaAndLength
            X, Ten = mlarm.time_step_many_noinfo(X, Ten, nlayers=nlayers)
        # X, Ten = mlarm.time_step_many_timing_noinfo(X, Ten, nlayers=nlayers)
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
        logger.info('********************************************')
        logger.info(f'{it+1}th time step for N=32, time: {currtime}')
        logger.info(f'Solving with networks takes {tEnd - tStart} sec.')
        logger.info(f'Error in area and length: {max(errArea, errLen)}')
        logger.info('********************************************\n')

        # Save data
        output = np.concatenate(([currtime], X.cpu().numpy().T.flatten())).astype('float64')
        with open(fileName, 'ab') as fid:
            output.tofile(fid)

        # Ten_save[it+1] = Ten
        # if it % 50 == 0:
    # np.save("output/save_farFieldtracJump.npy", mlarm.save_farFieldtracJump.cpu().numpy())


# Set up the logger globally
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('try_repulse_' + args.filename + '.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
