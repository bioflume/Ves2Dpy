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
from filter import interpft

import json

# Load parameters from a JSON file
with open("vesnet_config.json", "r") as f:
    config = json.load(f)

# Set defaults (if not provided in JSON)
params = {
    "input": "/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/shear_N32.npy",
    "outfile": "linshi",
    "logging": False,
    "resolution": 32,
    "num_steps": 100000,
    "flow": {
        "name": "vortex",
        "speed": 400,
        "chanWidth": 2.5,
        "vortexSize": 2.5,
    },
    "rbf_params": {
        "nlayers": 5,
        "rbf_upsample": 4,
    },
    "repulsion_params": {
        "use_repulsion": False,
        "repulsion_strength": 1e4,
        "eta": 1/32,
    },
}

# Update defaults with values from config.json
params.update(config)

# Now you can access them like before:
input = params["input"]
outfile = params["outfile"]
logging = params["logging"]
resolution = params["resolution"]
num_steps = params["num_steps"]
flow = params["flow"]
rbf_params = params["rbf_params"]
repulsion_params = params["repulsion_params"]

simulate(
    input=input,
    outfile=outfile, 
    logging=logging, 
    resolution=resolution,
    num_steps=num_steps,
    bgFlow=flow, 
    rbf_params=rbf_params, 
    repulsion_params=repulsion_params)

