"""Unified MLARM / vesnet simulation driver (N=32 and N=128)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from tqdm import tqdm

from curve_batch_compile import Curve
from tools.filter import interpft, interpft_vec

torch.set_default_dtype(torch.float32)
cudnn.benchmark = True
import torch._dynamo  # noqa: E402

torch._dynamo.reset()

DTYPE_BY_RESOLUTION = {32: torch.float32, 128: torch.float64}

RELAX_NORM = {
    32: {
        "input": np.array(
            [-1.5200416214611323e-07, 0.06278670579195023,
             -2.5547041104800883e-07, 0.13339416682720184]
        ),
        "output": np.array(
            [-2.329148207635967e-09, 0.00020403489179443568,
             -1.5361016902915026e-09, 0.00017457373905926943]
        ),
    },
    128: {
        "input": np.array(
            [-8.430413700466488e-09, 0.06278684735298157,
             6.290720477863943e-08, 0.13339413702487946]
        ),
        "output": np.array(
            [-2.884585348361668e-10, 0.00020574081281665713,
             -5.137390512999218e-10, 0.0001763451291481033]
        ),
    },
}

TEN_SELF_NORM = {
    32: {
        "input": np.array(
            [0.00016983709065243602, 0.06278808414936066,
             0.0020364541560411453, 0.13337676227092743,
             6.277393817901611, 9.243043899536133]
        ),
        "output": np.array([337.7682800292969, 458.4842834472656]),
    },
    128: {
        "input": np.array(
            [0.00017108717293012887, 0.06278623640537262,
             0.002038202714174986, 0.13337858021259308]
        ),
        "output": np.array([337.7627868652344, 466.6429138183594]),
    },
}

TRAINED_FILES = {
    32: {
        "adv_in": "adv_fft_ds32/2024Oct_advfft_in_para_downsample_all_mode.npy",
        "adv_out": "adv_fft_ds32/2024Oct_advfft_out_para_downsample_all_mode.npy",
        "near_in": "near_trained/in_param_downsample32_allmode.npy",
        "near_out": "near_trained/out_param_downsample32_allmode.npy",
        "ten_adv_in": "advten_downsample32/2024Nov_advten_ds32_in_para_allmodes.npy",
        "ten_adv_out": "advten_downsample32/2024Nov_advten_ds32_out_para_allmodes.npy",
    },
    128: {
        "adv_in": "2024Oct_adv_fft_tot_in_para.npy",
        "adv_out": "2024Oct_adv_fft_tot_out_para.npy",
        "near_in": "in_param_disth_allmode.npy",
        "near_out": "out_param_disth_allmode.npy",
        "ten_adv_in": "2024Oct_advten_in_para_allmodes.npy",
        "ten_adv_out": "2024Oct_advten_out_para_allmodes.npy",
    },
}

INNER_NEAR_FILES = {
    "in": "inner_near_in_param_allmodes.npy",
    "out": "inner_near_out_param_allmodes.npy",
}


def import_mlarm_class(resolution: int):
    if resolution == 32:
        from wrapper_MLARM_batch_compile_N32 import MLARM_manyfree_py
    elif resolution == 128:
        from wrapper_MLARM_batch_compile_N128 import MLARM_manyfree_py
    else:
        raise ValueError(f"resolution must be 32 or 128, got {resolution}")
    return MLARM_manyfree_py


def _resolve_path(base: str | Path, rel: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    return Path(base) / path


def load_network_norms(resolution: int, params: dict[str, Any]) -> dict[str, np.ndarray]:
    trained_root = Path(params["trained_root"])
    files = TRAINED_FILES[resolution]
    norms = {
        "adv_in": np.load(_resolve_path(trained_root, files["adv_in"])),
        "adv_out": np.load(_resolve_path(trained_root, files["adv_out"])),
        "near_in": np.load(_resolve_path(trained_root, files["near_in"])),
        "near_out": np.load(_resolve_path(trained_root, files["near_out"])),
        "ten_adv_in": np.load(_resolve_path(trained_root, files["ten_adv_in"])),
        "ten_adv_out": np.load(_resolve_path(trained_root, files["ten_adv_out"])),
        "relax_in": RELAX_NORM[resolution]["input"],
        "relax_out": RELAX_NORM[resolution]["output"],
        "ten_self_in": TEN_SELF_NORM[resolution]["input"],
        "ten_self_out": TEN_SELF_NORM[resolution]["output"],
    }
    if resolution == 32:
        inner_root = Path(params["inner_near_root"])
        norms["inner_near_in"] = np.load(_resolve_path(inner_root, INNER_NEAR_FILES["in"]))
        norms["inner_near_out"] = np.load(_resolve_path(inner_root, INNER_NEAR_FILES["out"]))
    return norms


def load_initial_shapes(params: dict[str, Any]) -> np.ndarray:
    path = Path(params["input"])
    fmt = params.get("input_format", "npy" if path.suffix == ".npy" else "mat")
    if fmt == "npy":
        return np.load(path)
    var = params.get("input_var", "X")
    data = loadmat(path)
    if var not in data:
        raise KeyError(f"Variable '{var}' not found in {path}")
    return data[var]


def resample_initial_positions(X0: torch.Tensor, resolution: int) -> torch.Tensor:
    n_pts = X0.shape[0] // 2
    if n_pts == resolution:
        return X0
    if resolution == 128:
        return interpft_vec(X0, resolution)
    # N=32 path: legacy entry used interpft on first 128 points
    return torch.concat(
        (interpft(X0[:128], resolution), interpft(X0[128:], resolution)),
        dim=0,
    )


def set_bg_flow(flow: dict[str, Any], resolution: int):
    name = flow["name"]
    speed = flow["speed"]
    chan_width = flow.get("chanWidth", 2.5)
    vortex_size = flow.get("vortexSize", chan_width)
    if resolution == 128 and name == "vortex":
        chan_width = chan_width * 2

    def get_flow(X):
        n = X.shape[0] // 2
        if name == "relax":
            return torch.zeros_like(X)
        if name == "shear":
            return speed * torch.vstack((X[n:], torch.zeros_like(X[:n])))
        if name == "tayGreen":
            return speed * torch.vstack(
                (torch.sin(X[:n]) * torch.cos(X[n:]),
                 -torch.cos(X[:n]) * torch.sin(X[n:]))
            )
        if name == "parabolic":
            width = 5 if resolution == 32 else 0.375
            return torch.vstack(
                (speed * (1 - (X[n:] / width) ** 2), torch.zeros_like(X[:n]))
            )
        if name == "rotation":
            r = torch.sqrt(X[:n] ** 2 + X[n:] ** 2)
            theta = torch.atan2(X[n:], X[:n])
            return speed * torch.vstack(
                (-torch.sin(theta) / r, torch.cos(theta) / r)
            )
        if name == "vortex":
            return speed * torch.cat(
                [
                    torch.sin(X[:n] / chan_width * torch.pi)
                    * torch.cos(X[n:] / chan_width * torch.pi),
                    -torch.cos(X[:n] / chan_width * torch.pi)
                    * torch.sin(X[n:] / chan_width * torch.pi),
                ],
                dim=0,
            )
        return torch.zeros_like(X)

    return get_flow


def prepare_ellipse(
    oc: Curve,
    path: str | Path,
    resolution: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ellipse = torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)
    center = oc.getPhysicalCenter(ellipse)
    n_ellipse = ellipse.shape[0] // 2
    ellipse[:n_ellipse, :] -= center[0]
    ellipse[n_ellipse:, :] -= center[1]
    if resolution == 128:
        return interpft_vec(ellipse, resolution)
    return ellipse


def build_mlarm(
    resolution: int,
    params: dict[str, Any],
    dt: float,
    vinf,
    oc: Curve,
    device: torch.device,
    logger: logging.Logger,
):
    MLARM_manyfree_py = import_mlarm_class(resolution)
    dtype = DTYPE_BY_RESOLUTION[resolution]
    norms = load_network_norms(resolution, params)
    rep = params["repulsion_params"]
    rbf = params["rbf_params"]

    def to_tensor(arr):
        return torch.from_numpy(arr).to(device=device, dtype=dtype)

    common = dict(
        dt=dt,
        vinf=vinf,
        oc=oc,
        use_repulsion=rep["use_repulsion"],
        repStrength=rep["repulsion_strength"],
        rbf_upsample=rbf["rbf_upsample"],
        advNetInputNorm=to_tensor(norms["adv_in"]),
        advNetOutputNorm=to_tensor(norms["adv_out"]),
        relaxNetInputNorm=to_tensor(norms["relax_in"]),
        relaxNetOutputNorm=to_tensor(norms["relax_out"]),
        nearNetInputNorm=to_tensor(norms["near_in"]),
        nearNetOutputNorm=to_tensor(norms["near_out"]),
        tenSelfNetInputNorm=to_tensor(norms["ten_self_in"]),
        tenSelfNetOutputNorm=to_tensor(norms["ten_self_out"]),
        tenAdvNetInputNorm=to_tensor(norms["ten_adv_in"]),
        tenAdvNetOutputNorm=to_tensor(norms["ten_adv_out"]),
        device=device,
        logger=logger,
    )

    if resolution == 32:
        return MLARM_manyfree_py(
            eta=rep["eta"],
            innerNearNetInputNorm=to_tensor(norms["inner_near_in"]),
            innerNearNetOutputNorm=to_tensor(norms["inner_near_out"]),
            **common,
        )
    return MLARM_manyfree_py(**common)


def simulate(params: dict[str, Any], logger: logging.Logger) -> None:
    resolution = int(params["resolution"])
    if resolution not in (32, 128):
        raise ValueError(f"resolution must be 32 or 128, got {resolution}")

    dtype = DTYPE_BY_RESOLUTION[resolution]
    torch.set_default_dtype(dtype)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    oc = Curve(logger)

    output_dir = Path(params.get("output_dir") or ("./output" if resolution == 32 else "./output_N128"))
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = params.get("outfile") or params.get("filename", "run")
    file_name = output_dir / f"{outfile}.bin"

    flow = params["flow"]
    rbf = params["rbf_params"]
    nlayers = rbf["nlayers"]
    dt = float(params.get("dt", 1e-5))
    num_steps = int(params["num_steps"])

    vinf = set_bg_flow(flow, resolution)

    xics = load_initial_shapes(params)
    logger.info("Initial shapes array shape: %s", xics.shape)

    x0 = torch.from_numpy(xics).to(device=device, dtype=dtype)
    x0 = resample_initial_positions(x0, resolution)
    if resolution == 32:
        x0 = x0.float()

    n = resolution
    nv = x0.shape[1]
    _, area0, len0 = oc.geomProp(x0)
    logger.info("area0: %s", area0)
    logger.info("len0: %s", len0)
    x = x0.clone()

    logger.info("We have %d vesicles at N=%d", nv, n)
    ten = torch.zeros((n, nv), device=device, dtype=dtype)

    mlarm = build_mlarm(resolution, params, dt, vinf, oc, device, logger)
    _, area0, len0 = oc.geomProp(x)
    mlarm.area0 = area0
    mlarm.len0 = len0

    if resolution == 128:
        from poten import Poten
        mlarm.op = Poten(n)

    modes = torch.concatenate(
        (torch.arange(0, n // 2), torch.arange(-n // 2, 0))
    ).to(device)
    for _ in range(10):
        x, flag = oc.redistributeArcLength(x, modes)
        if flag:
            break

    relaxed_path = params.get("relaxed_shape", "relaxed_shape.npy")
    if Path(relaxed_path).exists():
        mlarm.ellipse = prepare_ellipse(oc, relaxed_path, resolution, device, dtype)
        logger.info("ellipse center: %s", oc.getPhysicalCenter(mlarm.ellipse))

    with open(file_name, "wb") as fid:
        np.array([n, nv], dtype=np.float64).tofile(fid)
        x.detach().cpu().numpy().T.flatten().astype(np.float64).tofile(fid)

    logger.info(
        "N=%d, nlayers=%d, rbf_upsample=%s, output=%s",
        resolution,
        nlayers,
        mlarm.rbf_upsample,
        file_name,
    )

    currtime = 0.0
    for it in tqdm(range(num_steps)):
        # t_start = time.time()
        with torch.no_grad():
            x, ten = mlarm.time_step_many_noinfo(x, ten, nlayers=nlayers)
        # t_end = time.time()

        area, length = oc.geomProp(x)[1:]
        err_area = torch.max(torch.abs(area - mlarm.area0) / mlarm.area0)
        err_len = torch.max(torch.abs(length - mlarm.len0) / mlarm.len0)

        currtime += dt
        logger.info("********************************************")
        logger.info(
            "%d th time step for N=%d, time: %g",
            it + 1,
            resolution,
            currtime,
        )
        # logger.info("Solving with networks takes %.4f sec.", t_end - t_start)
        logger.info("Error in area and length: %s", max(err_area, err_len).item())
        logger.info("********************************************\n")

        output = np.concatenate(([currtime], x.detach().cpu().numpy().T.flatten()))
        with open(file_name, "ab") as fid:
            output.astype(np.float64).tofile(fid)
