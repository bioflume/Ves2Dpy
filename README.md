# Ves2Dpy

**Two-dimensional vesicle dynamics simulation** from the VesNet paper.

The repository contains a **MATLAB reference implementation** (`matlab_version/`), **neural network architectures and loaders** (`model_zoo/`, `model_zoo_N32/`), and the **primary PyTorch package** (`torch_version/`). Trained weights are distributed via [Hugging Face](https://huggingface.co/) (see `trained_assets/manifest.json`).

---

## Requirements

### Core (simulation)

| Package | Used for |
|---------|----------|
| [PyTorch](https://pytorch.org/) | Tensors, networks, CUDA (recommended) |
| NumPy | Initial conditions, I/O |

### Automatic model download

| Package | Used for |
|---------|----------|
| [huggingface_hub](https://huggingface.co/docs/huggingface_hub) | Download trained `.pth` / `.pt` and normalization `.npy` files |

### Optional (examples post-processing)

| Package | Used for |
|---------|----------|
| matplotlib | Plot frames from `.bin` trajectories |
| opencv-python (`cv2`) | Encode `.mp4` videos |

**Python**: 3.10+ recommended.

**Hardware**: CUDA GPU strongly recommended for VesNet at useful problem sizes, but CPU is compatible.

---

## Installation

### 1. Clone and enter the repo

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install PyTorch

Install a build that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).


### 4. Install Python dependencies

```bash
pip install numpy scipy tqdm huggingface_hub torchsummary
```

Optional (plotting / video from examples):

```bash
pip install matplotlib opencv-python
```

### 5. PYTHONPATH for local imports

VesNet and BIEM code use flat imports (`from curve_batch_compile import …`). Add the package directory and the N=32 model zoo to `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd)/torch_version:$(pwd)/model_zoo_N32:${PYTHONPATH:-}"
```

For **N=128** runs, also include `model_zoo`:

```bash
export PYTHONPATH="$(pwd)/torch_version:$(pwd)/model_zoo_N32:$(pwd)/model_zoo:${PYTHONPATH:-}"
```

Example shell scripts under `examples/` set `PYTHONPATH` automatically.

### 6. Set the Hugging Face model repo (required for VesNet)

VesNet downloads trained weights from Hugging Face on first run. **Export the repo id before any vesnet / example run** (in every new shell, or add to your shell profile):

```bash
export VES2D_HF_REPO="shanzhong48/ves2d-trained"
```
---

## Quick start (VesNet examples)

Three ready-made cases at **N=32** are under [`examples/`](examples/). See [`examples/README.md`](examples/README.md) for details.

```bash
export VES2D_HF_REPO="shanzhong48/ves2d-trained"

# From repo root (downloads trained weights on first run)
./examples/ex1_one_ves_parabolic/run.sh

# Optional: fewer steps
./examples/ex1_one_ves_parabolic/run.sh --num-steps 100

# Optional: plot PNG frames and write MP4 (needs matplotlib + opencv-python)
./examples/ex1_one_ves_parabolic/run.sh --postprocess
```

| Example | Description |
|---------|-------------|
| `ex1_one_ves_parabolic/` | One vesicle, parabolic channel flow |
| `ex2_two_ves_shear/` | Two vesicles, shear flow (`shear_N32.npy` included) |
| `ex3_multi_ves_vortex/` | Eight of 48 vesicles, vortex flow (`48vesTG_N32.npy` included) |

**Outputs** (per example `config.json`): `output/<outfile>.bin` (binary trajectory), log file, and optionally frames / video.

### Run VesNet with your own config

```bash
export VES2D_HF_REPO="shanzhong48/ves2d-trained"
cd torch_version
export PYTHONPATH="$(pwd):$(pwd)/../model_zoo_N32:${PYTHONPATH:-}"
python entry_vesnet.py --config path/to/config.json --resolution 32
```

Minimal `config.json` fields: `input`, `outfile`, `output_dir`, `num_steps`, `dt`, `flow` (`name`, `speed`, …), `resolution` (32 or 128). Set `"use_hf_hub": true` (default) to auto-download weights, or set `trained_root` / `inner_near_root` and `"use_hf_hub": false` for a local copy.

**Background flows** (`flow.name`): `relax`, `shear`, `tayGreen`, `parabolic`, `rotation`, `vortex` (with optional `chanWidth`, `vortexSize`).

### Ground-truth BIEM

Edit simulation parameters at the bottom of `torch_version/driver_BIEM.py` (or import `initVes2D` / `TStepBiem` from your own script), then:

```bash
cd torch_version
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
python driver_BIEM.py
```

Writes a binary trajectory compatible with `torch_version/tools/load_ves2d_file.py` (same format as the MATLAB drivers).

---

## Trained models and cache

| Item | Location |
|------|----------|
| Asset manifest | `trained_assets/manifest.json` |
| Default HF repo (manifest) | `shanzhong48/ves2d-trained` in `trained_assets/manifest.json` |
| Repo you should export | `export VES2D_HF_REPO="shanzhong48/ves2d-trained"` (or your fork) |
| Local cache | `~/.cache/ves2d` (override with `VES2D_CACHE_DIR`) |

---

## Repository layout

```
Ves2Dpy/
├── README.md                 # This file
├── AGENTS.md                 # Detailed map for contributors / agents
│
├── torch_version/            # Main PyTorch package (geometry, BIEM, VesNet driver)
│   ├── curve_batch_compile.py
│   ├── capsules.py
│   ├── fft_tools.py
│   ├── poten.py              # Stokes potentials (BIEM)
│   ├── tstep_biem.py         # BIEM time stepper
│   ├── biem_support.py       # Exact Stokes layer kernels
│   ├── driver_BIEM.py        # Ground-truth simulation script
│   ├── driver_vesnet.py      # VesNet simulate()
│   ├── entry_vesnet.py       # CLI: JSON config → simulate
│   ├── vesnet_config.json    # Sample VesNet config
│   ├── wrapper_MLARM_batch_compile_N32.py
│   ├── wrapper_MLARM_batch_compile_N128.py
│   ├── TG_postprocess.py     # Plot frames from .bin
│   ├── create_video.py       # PNG sequence → MP4
│   └── tools/
│       ├── model_hub.py      # Hugging Face download / path resolution
│       ├── filter.py         # Spectral filters / interpolation
│       ├── load_ves2d_file.py
│       └── plot_ves2d_file.py
│
├── model_zoo/                # Network definitions + loaders (N=128)
├── model_zoo_N32/            # Network definitions + loaders (N=32)
│
├── trained_assets/
│   └── manifest.json         # HF bundle layout (weights + norm paths)
│
├── examples/                 # Runnable N=32 VesNet demos
│   ├── README.md
│   ├── _common.sh            # PYTHONPATH, HF download, run helper
│   ├── make_initial_conditions.py
│   └── ex1_…, ex2_…, ex3_…/  # config.json + run.sh per case
│
└── matlab_version/           # Original MATLAB drivers and shannets inference scripts
    ├── driver_confinedFlow.m
    ├── driver_manyVesicles.m
    ├── curve_py.m, capsules_py.m, poten_py.m
    ├── tstep_biem.m, MLARM_ManyFree_py.m
    └── shannets/             # Per-operator Python scripts called from MATLAB
```


### Resolution variants

| N | Wrapper | Model zoo |
|---|---------|-----------|
| 32 | `wrapper_MLARM_batch_compile_N32.py` | `model_zoo_N32/` |
| 128 | `wrapper_MLARM_batch_compile_N128.py` | `model_zoo/` | 

### Binary output format

Each `.bin` file: header `[N; nv; initial X flattened]`, then per time step `[time; X flattened column-major per vesicle]`. Load in Python with `tools/load_ves2d_file.py`; MATLAB drivers use the same layout.

---

## MATLAB ↔ Python correspondence

| MATLAB (`matlab_version/`) | Python (`torch_version/`) |
|----------------------------|---------------------------|
| `curve_py.m` | `curve_batch_compile.py` |
| `capsules_py.m` | `capsules.py` |
| `driver_confinedFlow.m` | `driver_BIEM.py` |
| `MLARM_ManyFree_py.m` | `wrapper_MLARM_batch_compile_N*.py` |
| `driver_manyVesicles.m` | `entry_vesnet.py` + config JSON |

---

## Citation and license

Add citation / license information here when publishing the project.

---

