# %%
import torch
torch.set_default_dtype(torch.float32)
import numpy as np
from curve_batch_compile import Curve
from capsules import capsules
import time
from tstep_biem import TStepBiem
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm
from filter import filterShape, interpft_vec

def initVes2D(options=None, prams=None):
    """
    Initialize vesicle simulation parameters and options.

    Args:
        options (dict): Simulation options.
        prams (dict): Physical and numerical parameters.

    Returns:
        options (dict), prams (dict): Fully populated with defaults.
    """

    if options is None:
        options = {}
    if prams is None:
        prams = {}

    # --- Default parameters ---
    defaultPram = {
        'N': 32,
        'nv': 1,
        'Nbd': 0,
        'nvbd': 0,
        'T': 1,
        'dt': 1e-5,
        'kappa': 1e-1,
        'viscCont': 1,
        'gmresTol': 1e-5,
        'gmresMaxIter': 1000,
        'areaLenTol': 1e-2,
        'repStrength': 900,
        'minDist': 0.4,
        'farFieldSpeed': 1000,
        'chanWidth': 2.5,
        'vortexSize': 2.5
    }

    defaultOpt = {
        'farField': 'shear',
        'repulsion': False,
        'correctShape': False,
        'reparameterization': False,
        'usePreco': True,
        'matFreeWalls': False,
        'confined': False
    }

    # --- Fill in missing prams ---
    for key, val in defaultPram.items():
        prams.setdefault(key, val)

    # --- Fill in missing options ---
    for key, val in defaultOpt.items():
        options.setdefault(key, val)

    # --- Geometry-dependent fix ---
    if not options['confined']:
        prams['Nbd'] = 0
        prams['nvbd'] = 0

    # --- Ensure viscCont is array of correct length ---
    if isinstance(prams['viscCont'], (int, float)):
        prams['viscCont'] = [prams['viscCont']] * prams['nv']

    return options, prams


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fileName = './output_BIEM/ls.bin'  # To save simulation data
fileName = './output_BIEM/linshi.bin'  # To save simulation data
# fileName = './output_BIEM/linshi_nv32.bin'  # To save simulation data
# fileName = './output_BIEM/ls_N128.bin'  # To save simulation data
# fileName = './output_BIEM/ls_N128_continued.bin'  # To save simulation data
# fileName = './output_BIEM/ls_N128_noNear.bin'  # To save simulation data
# fileName = './output_BIEM/shear.bin'  # To save simulation data
# fileName = './output_BIEM/TG_nv32_VF25.bin'  # To save simulation data
# fileName = './output_BIEM/db_N32.bin'  # To save simulation data

# Assume oc is your geometry utility class (like curve_py in MATLAB)
oc = Curve()  # You need to define this with required methods

# ------------------------------
# Create geometry for confinement
# ------------------------------
prams = {}
prams['Nbd'] = 32
# prams['nvbd'] = 2
prams['nvbd'] = 0

t = torch.linspace(0, 2 * torch.pi, prams['Nbd']) # end_point = False
rad1 = 1.0  # inner cylinder radius
rad2 = 2.0  # outer cylinder radius

# Outer and inner walls
x = torch.cat([rad2 * torch.cos(t), rad1 * torch.cos(-t)])
y = torch.cat([rad2 * torch.sin(t), rad1 * torch.sin(-t)])
# Xwalls = torch.vstack((x, y))
Xwalls = None

# ------------------------------
# Create vesicles
# ------------------------------

# Initial ellipse shape
# prams['N'] = 32
# X0 = oc.ellipse(prams['N'], torch.tensor([0.65]))
# _, _, length = oc.geomProp(X0)  # Get length
# X0 = X0 / length  # Normalize to unit length

# # # Position vesicles between cylinders
# X1 = torch.vstack((X0[:prams['N']], X0[prams['N']:]))       # shifted right
# X2 = torch.vstack((X0[:prams['N']]+0.22, X0[prams['N']:] + 0.15))       # shifted up
# # X3 = torch.vstack((X0[:prams['N']]-0.2, X0[prams['N']:] - 0.15))       # shifted up
# X = torch.cat((X1, X2), dim=1).to(device)

# Xics = np.load("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/shear_N32.npy") ### INIT SHAPES FROM THE DATA SET
# Xics = loadmat("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/ManyVesICsTaylorGreen/nv504IC.mat").get('X')
# Xics = loadmat("/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/ManyVesICsTaylorGreen/nv1020IC.mat").get('X')
Xics = loadmat("../VF25_TG32Ves.mat").get('X')[:, :3]
# Xics = loadmat("../2000vesShape8_VF30.mat").get('X')
# Xics = loadmat("../Nves_vs_dispersion_ICs/VF12_TG2220Ves.mat").get('X')[:, :2000]
# Xics = np.load("TG_N32_dilute_last100_nv128.npy")[:, :, 0]
# Xics = np.load("BIEM_TG_N128_last.npy")[:256, :]
# sigma = torch.from_numpy(np.load("BIEM_TG_N128_last.npy")[256:, :]).to(device)
# Xics = loadmat("../Nves_vs_dispersion_ICs/VF25_TG128Ves.mat").get('X')[:, :]
sigma = None
X = torch.from_numpy(Xics).float().to(device)
X = interpft_vec(X, 128).to(device)


# %matplotlib inline
# Plot initial vesicles
# plt.figure()
# # plt.plot(X[:32, :], X[32:, :], 'r', linewidth=2)
# plt.plot(X[:128, :], X[128:, :], 'r', linewidth=2)
# # for i in range(X.shape[0]//2):
# #     plt.text(X[i, 0], X[i+32, 0], str(i), fontsize=8, ha='center', va='bottom')
# #     plt.text(X[i, 1], X[i+32, 1], str(i), fontsize=8, ha='center', va='bottom')
#     # plt.text(X[i, 2], X[i+32, 2], str(i), fontsize=8, ha='center', va='bottom')

# plt.axis("scaled")
# plt.xlim([-0.3, 5.3])
# plt.ylim([-0.3, 5.3])
# plt.show()

# %%

# ------------------------------
# Simulation parameters and options
# ------------------------------
prams['N'] = X.shape[0]//2
prams['nv'] = X.shape[1]
prams['dt'] = 1e-5
prams['T'] = 50000 * prams['dt']
prams['kappa'] = 1.0
prams['viscCont'] = torch.ones(prams['nv'])
prams['gmresTol'] = 1e-10
prams['areaLenTol'] = 1e-2
prams['vortexSize'] = 2.5
prams['chanWidth'] = 2.5 
prams['farFieldSpeed'] = 400

prams['repStrength'] = 1e5
prams['minDist'] = 1./32

options = {
    'farField': 'vortex',
    'repulsion': False,
    'correctShape': True,
    'reparameterization': True,
    'usePreco': True,
    'matFreeWalls': False,
    'confined': False
}

# ------------------------------
# Initialize default values (if any missing)
# ------------------------------
options, prams = initVes2D(options, prams)

# ------------------------------
# Get original area and length
# ------------------------------
_, area0, len0 = oc.geomProp(X)
print("area0: ", area0)
print("len0: ", len0)

with open(fileName, 'wb') as fid:
    np.array([prams['N'], prams['nv']]).flatten().astype('float64').tofile(fid)
    X.cpu().numpy().T.flatten().astype('float64').tofile(fid)


# ------------------------------
# Time stepping object
# ------------------------------
print(prams)
print(options)

tt = TStepBiem(X, Xwalls, options, prams)  # You need to implement this class

if options['confined']:
    tt.initialConfined()

# ------------------------------
# Initialize variables
# ------------------------------
sigma = torch.zeros(prams['N'], prams['nv']) if sigma is None else sigma
eta = torch.zeros(2 * prams['Nbd'], prams['nvbd'])
RS = torch.zeros(3, prams['nvbd'])

# ------------------------------
# Display setup
# ------------------------------
print(f"{prams['nv']} vesicle(s) in {options['farField']} flow, dt: {prams['dt']}")
print(f"Vesicle(s) discretized with {prams['N']} points")
print(f"we are using {X.dtype}")
if options['confined']:
    print(f"Wall(s) discretized with {prams['Nbd']} points")

# ------------------------------
# Time loop
# ------------------------------
time_ = 0.0
modes = torch.concatenate((torch.arange(0, prams['N'] // 2), torch.arange(-prams['N'] // 2, 0))).to(X.device) #.double()

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)
# start.record()

# end.record()
# torch.cuda.synchronize()
# print(f'standardizationStep {start.elapsed_time(end)/1000} sec.')

for step in tqdm(range(int(prams['T'] / prams['dt']))):

    # t_start = time.time()

    # Perform time step
    # start.record()
    Xnew, sigma, eta, RS, iter_, iflag = tt.time_step(X, sigma, eta, RS)

    # t_end = time.time() - t_start

    if options['reparameterization']:
        # Redistribute arc-length
        XnewO = Xnew.clone()
        for _ in range(5):
            Xnew, allGood = oc.redistributeArcLength(Xnew, modes)
        X = oc.alignCenterAngle(XnewO, Xnew)
    else:
        X = Xnew
    
    # end.record()
    # torch.cuda.synchronize()
    # print(f'One Time Step {start.elapsed_time(end)/1000} sec.')

    # start.record()
    if options['correctShape']:
        X = oc.correctAreaAndLengthAugLag(X.float(), area0, len0)
        # X = X.double()
    
    # X = filterShape(X, modeCut=10)
        
    # end.record()
    # torch.cuda.synchronize()
    # print(f'correctAreaLength takes {start.elapsed_time(end)/1000} sec.')

    # Update simulation time
    time_ += prams['dt']

    # Display timestep info
    print("*****************************************************************")
    print(f"Time: {step} step, out of Tf: {prams['T']}")
    print(f"GMRES took {iter_} matvecs, successful {not iflag}")
    # print(f"Time step took {t_end:.4f} seconds")
    print("*****************************************************************")

    output = np.concatenate(([time_], X.cpu().numpy().T.flatten())).astype('float64')
    with open(fileName, 'ab') as fid:
        output.tofile(fid)


    # np.save("BIEM_TG_N128_last.npy", np.concatenate((X.cpu().numpy(), sigma.cpu().numpy()), axis=0))

    # Plot vesicles and walls
    # plt.figure(1); plt.clf()
    # plt.plot(Xwalls[0, :prams['Nbd']], Xwalls[1, :prams['Nbd']], 'k', linewidth=2)
    # plt.plot(Xwalls[0, prams['Nbd']:], Xwalls[1, prams['Nbd']:], 'k', linewidth=2)
    # plt.plot(X[0], X[1], 'r', linewidth=2)
    # plt.axis('equal')
    # plt.pause(0.1)
