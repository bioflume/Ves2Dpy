import numpy as np
import torch

torch.set_default_dtype(torch.float32)
# torch.set_default_device('cuda:0')
import sys

sys.path.append("..")
from collections import defaultdict
from capsules import capsules
from filter import (
    filterShape,
    filterTension,
    interpft,
    interpft_vec,
    upsample_fft,
    downsample_fft,
    gaussian_filter_shape,
    gaussian_filter_1d_energy_preserve,
)
from filter import rescale_outlier_vel, rescale_outlier_vel_abs, rescale_outlier_trans
from model_zoo.get_network_torch import (
    RelaxNetwork,
    TenSelfNetwork,
    MergedAdvNetwork,
    MergedTenAdvNetwork,
    MergedNearFourierNetwork,
)

from math import ceil, sqrt
from typing import List, Tuple
from biem_support import exactStokesSL_onlyself
from tools.compile_utils import compile_cudagraphs_if_cuda


def relax_solve(X, op):
    device = X.device
    N = X.shape[0] // 2
    nv = X.shape[1]
    vesicle = capsules(X, None, None, 1, 1)

    # Compute derivatives NOT BATCHED!!!!!
    Ben, Ten, Div = vesicle.computeDerivs()
    Ben = Ben.permute(2, 0, 1).to(device)
    Ten = Ten.permute(2, 0, 1).to(device)
    Div = Div.permute(2, 0, 1).to(device)

    G = op.stokesSLmatrix(vesicle)
    G = G.permute(2, 0, 1)

    dt = 1e-5

    # Compute M matrix
    M = (
        G
        @ Ten
        @ torch.linalg.solve(
            (Div @ G @ Ten), torch.eye(N, device=device).unsqueeze(0).repeat(nv, 1, 1)
        )
        @ Div
    )

    # Compute Left-Hand Side (LHS) matrix
    LHS = torch.eye(2 * N, device=device).unsqueeze(0).repeat(
        nv, 1, 1
    ) - vesicle.kappa * dt * (-G @ Ben + M @ G @ Ben)

    # Solve for Xnew
    Xnew = torch.linalg.solve(LHS, X.T)

    return Xnew.T


# @torch.jit.script
def allExactStokesSLTarget_broadcast(
    vesicleX, vesicle_sa, f, tarX, length: float = 1.0, offset: int = 0
):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """

    N, nv = vesicleX.shape[0] // 2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0] // 2, tarX.shape[1]
    stokesSLPtar = torch.zeros(
        (2 * Ntar, ntar), dtype=torch.float32, device=vesicleX.device
    )

    mask = ~torch.eye(nv, dtype=torch.bool)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = torch.arange(nv)[None,].expand(nv, -1)[mask].view(nv, nv - 1)
    indices = indices[offset : offset + ntar]

    den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
    deny = den[N:, indices].permute(0, 2, 1)

    diffx = (
        tarX[None, None, :Ntar, ...]
        - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None]
    )  # broadcasting, (N, (nv-1), Ntar, nv)
    diffy = (
        tarX[None, None, Ntar:, ...]
        - vesicleX[N:, indices].permute(0, 2, 1)[:, :, None]
    )

    dis2 = diffx**2 + diffy**2
    info = dis2 < (length / Ntar) ** 2
    # Compute the cell-level mask
    # cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
    # full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, ntar)
    full_mask = info.any(dim=0).unsqueeze(0)

    coeff = 0.5 * torch.log(dis2)
    coeff.masked_fill_(full_mask, 0)
    # coeff[full_mask.expand(N,-1,-1,-1)] = 0.
    col_indices = torch.arange(ntar)
    stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
    stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

    coeff = (diffx * denx.unsqueeze(2) + diffy * deny.unsqueeze(2)) / dis2
    coeff.masked_fill_(full_mask, 0)
    # coeff[full_mask.expand(N,-1,-1,-1)] = 0.
    stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0, 1])
    stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0, 1])

    return stokesSLPtar / (4 * torch.pi)


# @torch.compile(backend='cudagraphs')
@torch.jit.script
def allExactStokesSLTarget_compare1(
    vesicleX, vesicle_sa, f, tarX, length: float = 1.0, offset: int = 0
):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """

    N, nv = vesicleX.shape[0] // 2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0] // 2, tarX.shape[1]
    stokesSLPtar = torch.zeros(
        (2 * Ntar, ntar), dtype=tarX.dtype, device=vesicleX.device
    )

    mask = ~torch.eye(nv, dtype=torch.bool, device=vesicleX.device)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = (
        torch.arange(nv, device=vesicleX.device)[None,]
        .expand(nv, -1)[mask]
        .view(nv, nv - 1)
    )
    indices = indices[offset : offset + ntar]

    den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    denx = den[:N, indices].permute(0, 2, 1).unsqueeze(2)  # (N, (nv-1), 1, ntar)
    deny = den[N:, indices].permute(0, 2, 1).unsqueeze(2)
    
    # xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
    # ysou = vesicleX[N:, indices].permute(0, 2, 1)

    # if tarX is not None:
    # xtar = tarX[:Ntar]
    # ytar = tarX[Ntar:]
    # else:
    #     xtar = vesicleX[:N]
    #     ytar = vesicleX[N:]

    diffx = (
        tarX[None, None, :Ntar, ...]
        - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None]
    )  # broadcasting, (N, (nv-1), Ntar, ntar)
    diffy = (
        tarX[None, None, Ntar:, ...]
        - vesicleX[N:, indices].permute(0, 2, 1)[:, :, None]
    )

    # diff = tarX[None, None, ...] - vesicleX[:, indices].permute(0, 2, 1) [:, :, None]
    # diffx = diff[:N, :, :Ntar, :]
    # diffy = diff[N:, :, Ntar:, :]

    dis2 = diffx**2 + diffy**2
    # info = dis2 <= (1/Ntar)**2
    # Compute the cell-level mask
    # cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
    # full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, ntar)
    # full_mask = (dis2 <= (1/Ntar)**2).any(dim=0).unsqueeze(0).expand(N, -1, -1, -1)
    # ids_ = torch.unbind(full_mask.to_sparse().indices(), dim=0)

    ids = torch.where(
        torch.max((dis2.reshape(N, nv - 1, -1) < (length / Ntar) ** 2), dim=0)[0]
    )
    ids = (ids[0], ids[1] // ntar, ids[1] % ntar)

    l = len(ids[0])
    ids_ = (
        torch.arange(N, device=f.device)[:, None].expand(-1, l).reshape(-1),
        ids[0][None, :].expand(N, -1).reshape(-1),
        ids[1][None, :].expand(N, -1).reshape(-1),
        ids[2][None, :].expand(N, -1).reshape(-1),
    )

    coeff = (diffx * denx + diffy * deny) / dis2

    stokesSLPtar[:Ntar, :] = torch.sum(
        (coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )
    stokesSLPtar[Ntar:, :] = torch.sum(
        (coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )

    return stokesSLPtar / (4 * torch.pi), (ids[0], ids[1], ids[2] + offset)


# @torch.compile(backend='cudagraphs')
@torch.jit.script
def allExactStokesSLTarget_compare2(
    vesicleX,
    vesicle_sa,
    f,
    tarX,
    ids0,
    ids1,
    ids2,
    length: float = 1.0,
    offset: int = 0,
):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """

    N, nv = vesicleX.shape[0] // 2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0] // 2, tarX.shape[1]
    stokesSLPtar = torch.zeros(
        (2 * Ntar, ntar), dtype=vesicleX.dtype, device=vesicleX.device
    )

    mask = ~torch.eye(nv, dtype=torch.bool, device=vesicleX.device)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = (
        torch.arange(nv, device=vesicleX.device)[None,]
        .expand(nv, -1)[mask]
        .view(nv, nv - 1)
    )
    indices = indices[offset : offset + ntar]

    den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    denx = den[:N, indices].permute(0, 2, 1).unsqueeze(2)  # (N, (nv-1), nv)
    deny = den[N:, indices].permute(0, 2, 1).unsqueeze(2)

    # xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
    # ysou = vesicleX[N:, indices].permute(0, 2, 1)

    # if tarX is not None:
    # xtar = tarX[:Ntar]
    # ytar = tarX[Ntar:]
    # else:
    #     xtar = vesicleX[:N]
    #     ytar = vesicleX[N:]

    diffx = (
        tarX[None, None, :Ntar, ...]
        - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None]
    )  # broadcasting, (N, (nv-1), Ntar, nv)
    diffy = (
        tarX[None, None, Ntar:, ...]
        - vesicleX[N:, indices].permute(0, 2, 1)[:, :, None]
    )

    # diff = tarX[None, None, ...] - vesicleX[:, indices].permute(0, 2, 1) [:, :, None]
    # diffx = diff[:N, :, :Ntar, :]
    # diffy = diff[N:, :, Ntar:, :]

    dis2 = diffx**2 + diffy**2
    # info = dis2 <= (1/Ntar)**2
    # Compute the cell-level mask
    # cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
    # full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, ntar)
    # full_mask = (dis2 <= (1/Ntar)**2).any(dim=0).unsqueeze(0).expand(N, -1, -1, -1)
    # ids_ = torch.unbind(full_mask.to_sparse().indices(), dim=0)

    # ids = torch.where((dis2 <= (1/Ntar)**2).any(dim=0))
    # ids = torch.where(torch.sum((dis2 <= (1/Ntar)**2), dim=0))
    # ids = (ids0, ids1, ids2)
    # ids = torch.where((dis2 < (1/Ntar)**2 ).any(dim=0).unsqueeze(0).expand(N, -1, -1, -1))
    # ids = torch.unbind((dis2 <= (1/Ntar)**2).any(dim=0).to_sparse().indices(), dim=0)

    l = len(ids0)
    ids_ = (
        torch.arange(N, device=f.device)[:, None].expand(-1, l).reshape(-1),
        ids0[None, :].expand(N, -1).reshape(-1),
        ids1[None, :].expand(N, -1).reshape(-1),
        ids2[None, :].expand(N, -1).reshape(-1),
    )

    coeff = (diffx * denx + diffy * deny) / dis2

    stokesSLPtar[:Ntar, :] = torch.sum(
        (coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )
    stokesSLPtar[Ntar:, :] = torch.sum(
        (coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(
            ids_, torch.tensor([0.0], device=f.device, dtype=vesicleX.dtype)
        ),
        dim=[0, 1],
    )

    return stokesSLPtar / (4 * torch.pi)


class MLARM_manyfree_py(torch.jit.ScriptModule):
    def __init__(
        self,
        dt,
        vinf,
        oc,
        use_repulsion,
        repStrength,
        rbf_upsample: int,
        advNetInputNorm,
        advNetOutputNorm,
        relaxNetInputNorm,
        relaxNetOutputNorm,
        nearNetInputNorm,
        nearNetOutputNorm,
        tenSelfNetInputNorm,
        tenSelfNetOutputNorm,
        tenAdvNetInputNorm,
        tenAdvNetOutputNorm,
        device,
        logger,
    ):
        super().__init__()

        self.dt = dt  # time step size
        self.vinf = (
            vinf  # background flow (analytic -- itorchut as function of vesicle config)
        )
        self.oc = oc  # curve class
        self.kappa = 1  # bending stiffness is 1 for our simulations
        self.device = device
        self.logger = logger
        # Flag for repulsion
        self.use_repulsion = use_repulsion
        self.repStrength = repStrength
        self.rbf_upsample = rbf_upsample

        # Normalization values for advection (translation) networks
        self.advNetInputNorm = advNetInputNorm
        self.advNetOutputNorm = advNetOutputNorm
        self.mergedAdvNetwork = MergedAdvNetwork(
            self.advNetInputNorm.to(device),
            self.advNetOutputNorm.to(device),
            # model_path="../trained/2024Oct_ves_merged_adv.pth",
            model_path="../trained/torch_script_models/2024Oct_ves_merged_adv.pt",
            device=device,
        )

        # Normalization values for relaxation network
        self.relaxNetInputNorm = relaxNetInputNorm
        self.relaxNetOutputNorm = relaxNetOutputNorm
        self.relaxNetwork = RelaxNetwork(
            self.dt,
            self.relaxNetInputNorm.to(device),
            self.relaxNetOutputNorm.to(device),
            # model_path="../trained/ves_relax_DIFF_June8_625k_dt1e-5.pth",
            model_path="../trained/torch_script_models/ves_relax_DIFF_June8_625k_dt1e-5.pt",
            device=device,
        )

        # Normalization values for near field networks
        self.nearNetInputNorm = nearNetInputNorm
        self.nearNetOutputNorm = nearNetOutputNorm
        self.nearNetwork = MergedNearFourierNetwork(
            self.nearNetInputNorm.to(device),
            self.nearNetOutputNorm.to(device),
            # model_path="../trained/ves_merged_disth_nearFourier.pth",
            model_path="../trained/torch_script_models/ves_merged_disth_nearFourier.pt",
            device=device,
        )

        # Normalization values for tension-self network
        self.tenSelfNetInputNorm = tenSelfNetInputNorm
        self.tenSelfNetOutputNorm = tenSelfNetOutputNorm
        self.tenSelfNetwork = TenSelfNetwork(
            self.tenSelfNetInputNorm.to(device),
            self.tenSelfNetOutputNorm.to(device),
            # model_path = "../trained/Ves_2024Oct_selften_12blks_loss_0.00566cuda1.pth",
            model_path="../trained/torch_script_models/Ves_2024Oct_selften_12blks_loss_0.00566cuda1.pt",
            device=device,
        )

        # Normalization values for tension-advection networks
        self.tenAdvNetInputNorm = tenAdvNetInputNorm
        self.tenAdvNetOutputNorm = tenAdvNetOutputNorm
        self.tenAdvNetwork = MergedTenAdvNetwork(
            self.tenAdvNetInputNorm.to(device),
            self.tenAdvNetOutputNorm.to(device),
            # model_path="../trained/2024Oct_ves_merged_advten.pth",
            model_path="../trained/torch_script_models/2024Oct_ves_merged_advten.pt",
            device=device,
        )

    def time_step_many_noinfo(self, Xold, tenOld, nlayers=3):
        # oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]
        Nup = ceil(sqrt(N)) * N
        vesicleUp = capsules(upsample_fft(Xold, Nup), [], [], self.kappa, 1)

        # Compute velocity induced by repulsion force

        repForce = torch.zeros_like(Xold)
        if self.use_repulsion:
            repForce = vesicle.repulsionForce(Xold, self.repStrength)

        # Compute bending forces + old tension forces
        fTen = vesicle.tensionTerm(tenOld)
        fBend = vesicleUp.bendingTerm(vesicleUp.X)  # upsampled bending term
        fBend = downsample_fft(fBend, N)
        
        tracJump = fBend + fTen  # total elastic force

        Xstand, standardizationValues = self.standardizationStep(Xold)

        (
            velx_real,
            vely_real,
            velx_imag,
            vely_imag,
            xlayers,
            ylayers,
        ) = self.predictNearLayers(Xstand, standardizationValues, nlayers)
        
        info_rbf, info_stokes = None, None

        
        if self.rbf_upsample <= 2:
            # const = 0.495 * self.len0[0].item() * 4
            # const = 1.7 / 128
            const = 0.0132
        if self.rbf_upsample == 2:
            xlayers = interpft(xlayers.reshape(N, -1), N * 2)
            ylayers = interpft(ylayers.reshape(N, -1), N * 2)

        all_X = torch.concat(
            (xlayers.reshape(-1, 1, nv), ylayers.reshape(-1, 1, nv)), dim=1
        )  # (nlayers * N, 2, nv), 2 for x and y

        all_X = all_X / const
        matrices = torch.exp(
            -torch.sum((all_X[:, None] - all_X[None, ...]) ** 2, dim=-2)
        )
        matrices += (torch.eye(all_X.shape[0]).unsqueeze(-1) * 1e-6).expand(
            -1, -1, nv
        )  # (nlayers*N, nlayers*N, nv)

        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        
        farFieldtracJump, info_rbf, info_stokes = self.computeStokesInteractions(
            vesicle,
            vesicleUp,
            info_rbf,
            info_stokes,
            L,
            tracJump,
            repForce,
            velx_real,
            vely_real,
            velx_imag,
            vely_imag,
            xlayers,
            ylayers,
            standardizationValues,
            nlayers,
            first=True,
        )

        farFieldtracJump = filterShape(farFieldtracJump, 16)

        vBackSolve = self.invTenMatOnVback(
            Xstand, standardizationValues, vback + farFieldtracJump
        )

        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)

        tenNew = -(vBackSolve + selfBendSolve)
        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump, _, _ = self.computeStokesInteractions(
            vesicle,
            vesicleUp,
            info_rbf,
            info_stokes,
            L,
            tracJump,
            repForce,
            velx_real,
            vely_real,
            velx_imag,
            vely_imag,
            xlayers,
            ylayers,
            standardizationValues,
            nlayers,
            first=False,
        )

        farFieldtracJump = filterShape(farFieldtracJump, 16)

        if torch.any(torch.isnan(farFieldtracJump)) or torch.any(
            torch.isinf(farFieldtracJump)
        ):
            self.logger.warning("farFieldtracJump has nan or inf")
        # Total background velocity
        vbackTotal = vback + farFieldtracJump
        # self.save_farFieldtracJump[:, :, self.i] = farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)

        if torch.any(torch.isnan(Xadv)) or torch.any(torch.isinf(Xadv)):
            self.logger.warning("Xadv has nan or inf")
        # Xadv = rescale_outlier_trans(Xadv, Xold)

        Xadv = filterShape(Xadv, 16)
        # Xadv = gaussian_filter_shape(Xadv, sigma=3)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))

        # Compute the action of relax operator on Xold + Xadv
        Xnew = self.relaxWTorchNet(Xadv)

        modes = torch.concatenate(
            (torch.arange(0, N // 2), torch.arange(-N // 2, 0))
        ).to(
            Xold.device
        )  # .double()

        XnewC = Xnew.clone()
        for _ in range(5):
            Xnew, flag = self.oc.redistributeArcLength(Xnew, modes)
            if flag:
                break
        Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))
        with torch.enable_grad():
            Xnew = self.oc.correctAreaAndLengthAugLag(Xnew, self.area0, self.len0)

        Xnew = filterShape(Xnew.to(Xold.device), 16)

        return Xnew, tenNew

    # @torch.compile(backend='cudagraphs')
    def predictNearLayers(
        self,
        Xstand,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        nlayers: int = 3,
    ):
        # print('Near network predicting')
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]

        oc = self.oc

        # maxLayerDist = np.sqrt(1 / N)
        maxLayerDist = self.len0[0].item() / N  # length = 1, h = 1/N;

        # Create the layers around a vesicle on which velocity calculated
        tracersX_ = torch.zeros(
            (2 * N, nlayers, nv), dtype=Xstand.dtype, device=Xstand.device
        )
        if nlayers == 5:
            dlayer = torch.linspace(
                -maxLayerDist,
                maxLayerDist,
                nlayers,
                dtype=Xstand.dtype,
                device=Xstand.device,
            )
            tracersX_[:, 2] = Xstand
            _, tang = oc.diffProp_jac_tan(Xstand)
            rep_nx = tang[N:, :, None].expand(-1, -1, nlayers - 1)
            rep_ny = -tang[:N, :, None].expand(-1, -1, nlayers - 1)
            dx = rep_nx * dlayer[[0, 1, 3, 4]]  # (N, nv, nlayers-1)
            dy = rep_ny * dlayer[[0, 1, 3, 4]]
            tracersX_[:, [0, 1, 3, 4]] = torch.permute(
                torch.vstack(
                    [
                        torch.repeat_interleave(
                            Xstand[:N, :, None], nlayers - 1, dim=-1
                        )
                        + dx,
                        torch.repeat_interleave(
                            Xstand[N:, :, None], nlayers - 1, dim=-1
                        )
                        + dy,
                    ]
                ),
                (0, 2, 1),
            )
        else:
            dlayer = torch.linspace(
                0, maxLayerDist, nlayers, dtype=Xstand.dtype, device=Xstand.device
            )
            tracersX_[:, 0] = Xstand
            _, tang, _ = oc.diffProp(Xstand)
            rep_nx = torch.repeat_interleave(tang[N:, :, None], nlayers - 1, dim=-1)
            rep_ny = torch.repeat_interleave(-tang[:N, :, None], nlayers - 1, dim=-1)
            dx = rep_nx * dlayer[1:]  # (N, nv, nlayers-1)
            dy = rep_ny * dlayer[1:]
            tracersX_[:, 1:] = torch.permute(
                torch.vstack(
                    [
                        torch.repeat_interleave(
                            Xstand[:N, :, None], nlayers - 1, dim=-1
                        )
                        + dx,
                        torch.repeat_interleave(
                            Xstand[N:, :, None], nlayers - 1, dim=-1
                        )
                        + dy,
                    ]
                ),
                (0, 2, 1),
            )

        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(
            net_pred
        )

        if nlayers == 5:
            inner_input_net = self.innerNearNetwork.preProcess(Xstand)
            inner_net_pred = self.innerNearNetwork.forward(inner_input_net)
            (
                inner_velx_real,
                inner_vely_real,
                inner_velx_imag,
                inner_vely_imag,
            ) = self.innerNearNetwork.postProcess(inner_net_pred)

            velx_real = torch.concat((inner_velx_real, velx_real), dim=-1)
            vely_real = torch.concat((inner_vely_real, vely_real), dim=-1)
            velx_imag = torch.concat((inner_velx_imag, velx_imag), dim=-1)
            vely_imag = torch.concat((inner_vely_imag, vely_imag), dim=-1)

        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues
        Xl_ = self.destandardize(
            tracersX_.reshape(N * 2, -1),
            (
                scaling[None, :].expand(nlayers, -1).reshape(-1),
                rotate[None, :].expand(nlayers, -1).reshape(-1),
                rotCenter.tile((1, nlayers)),
                trans.tile((1, nlayers)),
                sortIdx.tile((nlayers, 1)),
            ),
        )

        xlayers_ = torch.zeros((N, nlayers, nv), dtype=Xstand.dtype)
        ylayers_ = torch.zeros((N, nlayers, nv), dtype=Xstand.dtype)
        xlayers_ = Xl_[
            :N, torch.arange(nlayers * nv, device=Xstand.device).reshape(nlayers, nv)
        ]
        ylayers_ = Xl_[
            N:, torch.arange(nlayers * nv, device=Xstand.device).reshape(nlayers, nv)
        ]

        return velx_real, vely_real, velx_imag, vely_imag, xlayers_, ylayers_

    # @torch.jit.script_method
    def buildVelocityInNear(
        self,
        tracJump,
        velx_real,
        vely_real,
        velx_imag,
        vely_imag,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        nlayers,
    ):
        nv = tracJump.shape[1]
        N = tracJump.shape[0] // 2
        _, rotate, _, _, sortIdx = standardizationValues

        fstand = self.standardize(
            tracJump,
            torch.zeros((2, nv), dtype=tracJump.dtype, device=tracJump.device),
            rotate,
            torch.zeros((2, nv), dtype=tracJump.dtype, device=tracJump.device),
            torch.tensor([1.0], device=tracJump.device),
            sortIdx,
        )
        z = fstand[:N] + 1.0j * fstand[N:]
        zh = torch.fft.fft(z, dim=0)
        fstandRe = torch.real(zh)
        fstandIm = torch.imag(zh)

        velx_stand_ = torch.einsum(
            "vnml, mv -> nvl", velx_real, fstandRe
        ) + torch.einsum("vnml, mv -> nvl", velx_imag, fstandIm)
        vely_stand_ = torch.einsum(
            "vnml, mv -> nvl", vely_real, fstandRe
        ) + torch.einsum("vnml, mv -> nvl", vely_imag, fstandIm)

        vx_ = torch.zeros(
            (nv, nlayers, N), device=tracJump.device, dtype=tracJump.dtype
        )
        vy_ = torch.zeros(
            (nv, nlayers, N), device=tracJump.device, dtype=tracJump.dtype
        )
        # Destandardize
        vx_[torch.arange(nv), :, sortIdx.T] = velx_stand_
        vy_[torch.arange(nv), :, sortIdx.T] = vely_stand_

        VelBefRot_ = torch.concat((vx_, vy_), dim=-1)  # (nv, nlayers, 2N)
        VelRot_ = self.rotationOperator(
            VelBefRot_.reshape(-1, 2 * N).T,
            torch.repeat_interleave(-rotate, nlayers, dim=0),
            torch.zeros(nv * nlayers),
        )
        VelRot_ = VelRot_.T.reshape(nv, nlayers, 2 * N).permute(2, 1, 0)
        velx_ = VelRot_[:N]  # (N, nlayers, nv)
        vely_ = VelRot_[N:]

        return velx_, vely_

    def naiveNearZoneInfo(self, vesicleX, vesicleUpX):
        """
        Naive way of doing range search by computing distances and creating masks.
        return a boolean nbrs_mask where (i,j)=True means i, j are close and are from different vesicles
        """
        N, nv = vesicleX.shape[0] // 2, vesicleX.shape[1]
        Nup = vesicleUpX.shape[0] // 2
        max_layer_dist = 1.0 / N

        all_points = torch.concat(
            (vesicleX[:N, :].T.reshape(-1, 1), vesicleX[N:, :].T.reshape(-1, 1)), dim=1
        )
        all_points_up = torch.concat(
            (
                vesicleUpX[:Nup, :].T.reshape(-1, 1),
                vesicleUpX[Nup:, :].T.reshape(-1, 1),
            ),
            dim=1,
        )

        sq_distances = torch.cdist(
            all_points.unsqueeze(0), all_points_up.unsqueeze(0)
        ).squeeze()
        dist_mask = sq_distances < max_layer_dist
        
        indices = (
            torch.arange(nv, device=dist_mask.device)
            .unsqueeze(-1)
            .expand(-1, N * Nup)
            .reshape(-1)
        )
        N_indices = (
            torch.arange(N, device=dist_mask.device)
            .unsqueeze(-1)
            .expand(-1, Nup)
            .reshape(-1)
        )
        N_indices = N_indices.unsqueeze(0).expand(nv, -1).reshape(-1)
        Nup_indices = (
            torch.arange(Nup, device=dist_mask.device)
            .unsqueeze(0)
            .expand(nv * N, -1)
            .reshape(-1)
        )

        nbrs_mask = dist_mask.reshape(nv, N, nv, Nup)
        nbrs_mask.index_put_(
            (indices, N_indices, indices, Nup_indices),
            torch.tensor(0.0, dtype=torch.bool, device=dist_mask.device),
        )

        rows_with_true = torch.max(nbrs_mask.reshape(nv * N, nv, Nup), dim=-1)[
            0
        ]  # (N*nv, nv)
        id1, id2 = torch.where(rows_with_true)
        ids1, ids2 = id1 % N, id1 // N
        ids0 = id2 - 1 * (ids2 <= id2)

        return (id1, id2), (ids0, ids1, ids2)  # for exactStokes

    def computeStokesInteractions(
        self,
        vesicle,
        vesicleUp,
        info_rbf,
        info_stokes,
        L,
        trac_jump,
        repForce,
        velx_real,
        vely_real,
        velx_imag,
        vely_imag,
        xlayers,
        ylayers,
        standardizationValues,
        nlayers,
        first: bool,
        upsample=True,
    ):
        # print('Near-singular interaction through interpolation and network')

        velx, vely = self.buildVelocityInNear(
            trac_jump + repForce,
            velx_real,
            vely_real,
            velx_imag,
            vely_imag,
            standardizationValues,
            nlayers,
        )
        rep_velx, rep_vely = self.buildVelocityInNear(
            repForce,
            velx_real[..., 2:3],
            vely_real[..., 2:3],
            velx_imag[..., 2:3],
            vely_imag[..., 2:3],
            standardizationValues,
            1,
        )

        totalForce = trac_jump + repForce
        # if upsample:
        N = vesicle.N
        nv = vesicle.nv
        Nup = ceil(sqrt(N)) * N
        length = 1.0
        totalForceUp = upsample_fft(totalForce, Nup)

        if first:
            fn = allExactStokesSLTarget_compare1
            if nv > 1048:
                num_parts = 10
                far_fields = []
                info_stokes_parts = [[], [], []]

                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (
                        (i + 1) * nv // num_parts if i < num_parts - 1 else None
                    )  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0  # Offset is None for the first call

                    far_field, info_stokes = fn(
                        vesicleUp.X,
                        vesicleUp.sa,
                        totalForceUp,
                        vesicle.X[:, start:end],
                        length,
                        offset=offset,
                    )

                    far_fields.append(far_field)
                    for j in range(3):
                        info_stokes_parts[j].append(info_stokes[j])

                far_field_1 = torch.concat(far_fields, dim=-1)
                info_stokes = tuple(
                    torch.cat(parts, dim=0) for parts in info_stokes_parts
                )

            elif nv > 504:
                far_field_1_1, info_stokes_1 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, : nv // 4],
                    length,
                )
                far_field_1_2, info_stokes_2 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, nv // 4 : nv // 2],
                    length,
                    offset=nv // 4,
                )
                far_field_1_3, info_stokes_3 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, nv // 2 : 3 * nv // 4],
                    length,
                    offset=nv // 2,
                )
                far_field_1_4, info_stokes_4 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, 3 * nv // 4 :],
                    length,
                    offset=3 * nv // 4,
                )
                far_field_1 = torch.concat(
                    (far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1
                )
                info_stokes = (
                    torch.cat(
                        (
                            info_stokes_1[0],
                            info_stokes_2[0],
                            info_stokes_3[0],
                            info_stokes_4[0],
                        ),
                        dim=0,
                    ),
                    torch.cat(
                        (
                            info_stokes_1[1],
                            info_stokes_2[1],
                            info_stokes_3[1],
                            info_stokes_4[1],
                        ),
                        dim=0,
                    ),
                    torch.cat(
                        (
                            info_stokes_1[2],
                            info_stokes_2[2],
                            info_stokes_3[2],
                            info_stokes_4[2],
                        ),
                        dim=0,
                    ),
                )
            elif nv > 100:
                far_field_1_1, info_stokes_1 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, : nv // 2],
                    length,
                )
                far_field_1_2, info_stokes_2 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X[:, nv // 2 :],
                    length,
                    offset=nv // 2,
                )
                far_field_1 = torch.concat((far_field_1_1, far_field_1_2), dim=-1)
                info_stokes = (
                    torch.cat((info_stokes_1[0], info_stokes_2[0]), dim=0),
                    torch.cat((info_stokes_1[1], info_stokes_2[1]), dim=0),
                    torch.cat((info_stokes_1[2], info_stokes_2[2]), dim=0),
                )
            else:
                far_field_1, info_stokes = fn(
                    vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, length
                )
            id1 = info_stokes[2] * N + info_stokes[1]
            id2 = info_stokes[0] + 1 * (info_stokes[0] >= info_stokes[2])
            info_rbf = (id1, id2)

        else:
            fn = allExactStokesSLTarget_compare2
            if nv > 1048:
                far_fields = []
                num_parts = 10
                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (
                        (i + 1) * nv // num_parts if i < num_parts - 1 else None
                    )  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0  # Offset is None for the first call

                    mask = (
                        (start <= info_stokes[2]) & (info_stokes[2] < end)
                        if i < num_parts - 1
                        else (start <= info_stokes[2])
                    )

                    far_field = fn(
                        vesicleUp.X,
                        vesicleUp.sa,
                        totalForceUp,
                        vesicle.X[:, start:end],
                        info_stokes[0][mask],
                        info_stokes[1][mask],
                        info_stokes[2][mask] - start,
                        offset=offset,
                    )

                    far_fields.append(far_field)

                far_field_1 = torch.concat(far_fields, dim=1)

            elif nv > 504:
                far_field_1 = torch.concat(
                    (
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, : nv // 4],
                            info_stokes[0][info_stokes[2] < nv // 4],
                            info_stokes[1][info_stokes[2] < nv // 4],
                            info_stokes[2][info_stokes[2] < nv // 4],
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, nv // 4 : nv // 2],
                            info_stokes[0][
                                (nv // 4 <= info_stokes[2]) & (info_stokes[2] < nv // 2)
                            ],
                            info_stokes[1][
                                (nv // 4 <= info_stokes[2]) & (info_stokes[2] < nv // 2)
                            ],
                            info_stokes[2][
                                (nv // 4 <= info_stokes[2]) & (info_stokes[2] < nv // 2)
                            ]
                            - nv // 4,
                            offset=nv // 4,
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, nv // 2 : 3 * nv // 4],
                            info_stokes[0][
                                (nv // 2 <= info_stokes[2])
                                & (info_stokes[2] < 3 * nv // 4)
                            ],
                            info_stokes[1][
                                (nv // 2 <= info_stokes[2])
                                & (info_stokes[2] < 3 * nv // 4)
                            ],
                            info_stokes[2][
                                (nv // 2 <= info_stokes[2])
                                & (info_stokes[2] < 3 * nv // 4)
                            ]
                            - nv // 2,
                            offset=nv // 2,
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, 3 * nv // 4 :],
                            info_stokes[0][3 * nv // 4 <= info_stokes[2]],
                            info_stokes[1][3 * nv // 4 <= info_stokes[2]],
                            info_stokes[2][3 * nv // 4 <= info_stokes[2]] - 3 * nv // 4,
                            offset=3 * nv // 4,
                        ),
                    ),
                    dim=1,
                )
            elif nv > 100:
                far_field_1 = torch.concat(
                    (
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, : nv // 2],
                            info_stokes[0][info_stokes[2] < nv // 2],
                            info_stokes[1][info_stokes[2] < nv // 2],
                            info_stokes[2][info_stokes[2] < nv // 2],
                        ),
                        fn(
                            vesicleUp.X,
                            vesicleUp.sa,
                            totalForceUp,
                            vesicle.X[:, nv // 2 :],
                            info_stokes[0][nv // 2 <= info_stokes[2]],
                            info_stokes[1][nv // 2 <= info_stokes[2]],
                            info_stokes[2][nv // 2 <= info_stokes[2]] - nv // 2,
                            offset=nv // 2,
                        ),
                    ),
                    dim=1,
                )

            else:
                far_field_1 = fn(
                    vesicleUp.X,
                    vesicleUp.sa,
                    totalForceUp,
                    vesicle.X,
                    info_stokes[0],
                    info_stokes[1],
                    info_stokes[2],
                )

        use_near_correct = False
        if use_near_correct:
            if self.rbf_upsample == 2:
                velx = interpft(velx.reshape(N, -1), N * 2)
                vely = interpft(vely.reshape(N, -1), N * 2)
            elif self.rbf_upsample == 4:
                velx = interpft(velx.reshape(N, -1), N * 4)
                vely = interpft(vely.reshape(N, -1), N * 4)

            self.nearFieldCorrectionUP_SOLVE(
                vesicle, info_rbf, L, far_field_1, velx, vely, xlayers, ylayers, nlayers
            )
        selfRepVel = torch.concat((rep_velx.squeeze(1), rep_vely.squeeze(1)), dim=0)
        return far_field_1 + selfRepVel, info_rbf, info_stokes

    
    def nearFieldCorrectionUP_SOLVE(
        self, vesicle, info, L, far_field, velx, vely, xlayers, ylayers, nlayers
    ):
        if len(info[0]) == 0 or len(info[1]) == 0:
            return

        N = vesicle.N
        nv = vesicle.nv

        all_points = torch.concat(
            (vesicle.X[:N, :].T.reshape(-1, 1), vesicle.X[N:, :].T.reshape(-1, 1)),
            dim=1,
        )

        if self.rbf_upsample <= 2:
            # const = 0.495 * self.len0[0].item() * 4
            # const = 1.7 / 128
            const = 0.0132

        all_X = torch.concat(
            (xlayers.reshape(-1, 1, nv), ylayers.reshape(-1, 1, nv)), dim=1
        )  # (3 * N, 2, nv), 2 for x and y
        all_X = all_X / const

        # matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) # (3*N, 3*N, nv)
        rhs = torch.concat(
            (velx.reshape(-1, 1, nv), vely.reshape(-1, 1, nv)), dim=1
        )  # (3 * N), 2, nv), 2 for x and y

        y = torch.linalg.solve_triangular(L, rhs.permute(2, 0, 1), upper=False)
        coeffs = torch.linalg.solve_triangular(L.permute(0, 2, 1), y, upper=True)

        
        id1_, id2_ = info
        if self.rbf_upsample <= 1:
            id2_ = id2_[:, None] + torch.arange(0, N * nlayers * nv, nv).to(id2_.device)
            id2_ = id2_.reshape(-1)
            id1_ = id1_[:, None].expand(-1, N * nlayers).reshape(-1)
            sp_matrix_ = torch.sparse_coo_tensor(
                torch.vstack((id1_, id2_)),
                torch.exp(
                    -torch.norm(
                        all_points[id1_] / const
                        - all_X.permute(0, 2, 1).reshape(-1, 2)[id2_, :],
                        dim=-1,
                    )
                    ** 2
                ),
                size=(N * nv, N * nlayers * nv),
            )
            correction = torch.sparse.mm(
                sp_matrix_, coeffs.permute(1, 0, 2).reshape(nv * N * nlayers, 2)
            )
        else:
            id2_ = id2_[:, None] + torch.arange(
                0, self.rbf_upsample * N * nlayers * nv, nv
            ).to(id2_.device)
            id2_ = id2_.reshape(-1)
            id1_ = id1_[:, None].expand(-1, self.rbf_upsample * N * nlayers).reshape(-1)
            sp_matrix_ = torch.sparse_coo_tensor(
                torch.vstack((id1_, id2_)),
                torch.exp(
                    -torch.norm(
                        all_points[id1_] / const
                        - all_X.permute(0, 2, 1).reshape(-1, 2)[id2_, :],
                        dim=-1,
                    )
                    ** 2
                ),
                size=(N * nv, self.rbf_upsample * N * nlayers * nv),
            )
            correction = torch.sparse.mm(
                sp_matrix_,
                coeffs.permute(1, 0, 2).reshape(
                    nv * self.rbf_upsample * N * nlayers, 2
                ),
            )

        correction = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)
        far_field += correction
        return

    # @torch.jit.script_method
    # @torch.compile(backend='cudagraphs')
    def translateVinfwTorch(
        self,
        Xold,
        Xstand,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        vinf,
    ):
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]

        _, rotate, _, _, sortIdx = standardizationValues

        Xpredict = self.mergedAdvNetwork.forward(Xstand)

        Z11r_ = torch.zeros((N, N, nv), dtype=Xstand.dtype)
        Z12r_ = torch.zeros_like(Z11r_)
        Z21r_ = torch.zeros_like(Z11r_)
        Z22r_ = torch.zeros_like(Z11r_)

        Z11r_[:, 1:] = torch.permute(Xpredict[:, :, 0, :N], (2, 0, 1))
        Z21r_[:, 1:] = torch.permute(Xpredict[:, :, 0, N:], (2, 0, 1))
        Z12r_[:, 1:] = torch.permute(Xpredict[:, :, 1, :N], (2, 0, 1))
        Z22r_[:, 1:] = torch.permute(Xpredict[:, :, 1, N:], (2, 0, 1))

        # Take fft of the velocity (should be standardized velocity)
        # only sort points and rotate to pi/2 (no translation, no scaling)
        vinf_stand = self.standardize(
            vinf,
            torch.zeros((2, nv), dtype=vinf.dtype),
            rotate,
            torch.zeros((2, nv), dtype=vinf.dtype),
            1,
            sortIdx,
        )
        z = vinf_stand[:N] + 1.0j * vinf_stand[N:]
        zh = torch.fft.fft(z, dim=0)
        V1, V2 = torch.real(zh), torch.imag(zh)
        MVinf_stand = torch.vstack(
            (
                torch.einsum("NiB,iB ->NB", Z11r_, V1)
                + torch.einsum("NiB,iB ->NB", Z12r_, V2),
                torch.einsum("NiB,iB ->NB", Z21r_, V1)
                + torch.einsum("NiB,iB ->NB", Z22r_, V2),
            )
        )

        Xnew = torch.zeros_like(Xold)
        MVinf = torch.zeros_like(MVinf_stand)
        idx = torch.vstack([sortIdx.T, sortIdx.T + N])
        MVinf[idx, torch.arange(nv, device=Xstand.device)] = MVinf_stand
        MVinf = self.rotationOperator(
            MVinf, -rotate, torch.zeros((2, nv), dtype=MVinf.dtype)
        )
        Xnew = Xold + self.dt * vinf - self.dt * MVinf

        return Xnew

    def relaxWTorchNet(self, Xmid):
        # RELAXATION w/ NETWORK
        Xin, standardizationValues = self.standardizationStep(Xmid)

        Xpred = self.relaxNetwork.forward(Xin)
        Xnew = self.destandardize(Xpred, standardizationValues)

        return Xnew

    # @torch.compile(backend='cudagraphs')
    def invTenMatOnVback(self, Xstand, standardizationValues, vinf):
        # Approximate inv(Div*G*Ten)*Div*vExt

        # number of vesicles
        nv = Xstand.shape[1]
        # number of points of exact solve
        N = Xstand.shape[0] // 2

        _, rotate, _, _, sortIdx = standardizationValues

        input = self.tenAdvNetwork.preProcess(Xstand)
        Xpredict = self.tenAdvNetwork.forward(input)
        out = self.tenAdvNetwork.postProcess(Xpredict)  # shape: (127, nv, 2, 128)

        # Approximate the multiplication Z = inv(DivGT)DivPhi_k
        Z1 = torch.zeros((N, N, nv), dtype=Xstand.dtype)
        Z2 = torch.zeros((N, N, nv), dtype=Xstand.dtype)

        Z1[:, 1:] = torch.permute(out[:, :, 0], (2, 0, 1))
        Z2[:, 1:] = torch.permute(out[:, :, 1], (2, 0, 1))

        vBackSolve = torch.zeros((N, nv), dtype=Xstand.dtype)
        vinfStand = self.standardize(
            vinf,
            torch.zeros((2, nv), dtype=Xstand.dtype),
            rotate,
            torch.zeros((2, nv), dtype=Xstand.dtype),
            1,
            sortIdx,
        )
        z = vinfStand[:N] + 1.0j * vinfStand[N:]
        zh = torch.fft.fft(z, dim=0)

        V1_ = torch.real(zh)
        V2_ = torch.imag(zh)
        # Compute the approximation to inv(Div*G*Ten)*Div*vExt
        MVinfStand = torch.einsum("NiB,iB ->NB", Z1, V1_) + torch.einsum(
            "NiB,iB ->NB", Z2, V2_
        )

        # Destandardize the multiplication
        vBackSolve[sortIdx.T, torch.arange(nv, device=Xstand.device)] = MVinfStand

        return vBackSolve

    # @torch.compile(backend='cudagraphs')
    def invTenMatOnSelfBend(self, Xstand, standardizationValues):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        nv = Xstand.shape[1]  # number of vesicles
        N = Xstand.shape[0] // 2

        scaling, _, _, _, sortIdx = standardizationValues

        tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        # tenPredictStand = self.tenSelfNetwork.forward_curv(Xstand)
        # tenPredictStand = tenPredictStand #.double()
        tenPred = torch.zeros((N, nv), dtype=Xstand.dtype, device=Xstand.device)

        tenPred[sortIdx.T, torch.arange(nv, device=Xstand.device)] = (
            tenPredictStand / scaling**2
        )

        return tenPred

    def invTenMatOnSelfBend_curv(self, Xstand, standardizationValues):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        nv = Xstand.shape[1]  # number of vesicles
        N = Xstand.shape[0] // 2

        # Xstand, scaling, _, _, _, sortIdx = self.standardizationStep(X)
        scaling, _, _, _, sortIdx = standardizationValues

        # tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        tenPredictStand = self.tenSelfNetwork_curv.forward_curv(Xstand, self.oc)
        tenPred = torch.zeros((N, nv), dtype=Xstand.dtype)

        tenPred[sortIdx.T, torch.arange(nv)] = tenPredictStand / scaling**2

        return tenPred

    # @torch.compile(backend='cudagraphs')
    def standardizationStep(self, Xin):
        # compatible with multi ves
        X = Xin.clone()
        N = X.shape[0] // 2
        modes = torch.concatenate(
            (torch.arange(0, N // 2), torch.arange(-N // 2, 0))
        ).to(
            X.device
        )
        for _ in range(5):
            X, flag = self.oc.redistributeArcLength(X, modes)

        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)

        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, (scaling, rotate, rotCenter, trans, multi_sortIdx)

    @compile_cudagraphs_if_cuda
    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        # compatible with multi ves
        N = len(multi_sortIdx[0])
        nv = X.shape[1]
        Xrotated = self.rotation_trans_Operator(X, rotation, rotCenter, translation)

        XrotSort = torch.vstack(
            (
                Xrotated[multi_sortIdx.T, torch.arange(nv, device=X.device)],
                Xrotated[multi_sortIdx.T + N, torch.arange(nv, device=X.device)],
            )
        )

        XrotSort = scaling * XrotSort
        return XrotSort

    @compile_cudagraphs_if_cuda
    def destandardize(
        self,
        XrotSort,
        standardizationValues: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """compatible with multiple ves"""
        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues

        N = len(sortIdx[0])
        nv = XrotSort.shape[1]

        # Scale back
        XrotSort = XrotSort / scaling

        # Change ordering back
        X = torch.zeros_like(XrotSort)
        X[sortIdx.T, torch.arange(nv, device=XrotSort.device)] = XrotSort[:N]
        X[sortIdx.T + N, torch.arange(nv, device=XrotSort.device)] = XrotSort[N:]

        # Take translation back
        X = self.translateOp(X, -trans)
        # Take rotation back
        X = self.rotationOperator(X, -rotate, rotCenter)

        return X

    def referenceValues(self, Xref):
        """Shan: compatible with multi ves"""

        oc = self.oc
        N = len(Xref) // 2
        tempX = Xref.clone()

        # Find the physical center
        rotCenter = oc.getPhysicalCenter(tempX)
        multi_V = oc.getPrincAxesGivenCentroid(tempX, rotCenter)
        rotation = torch.arctan2(multi_V[0], multi_V[1])

        Xref = self.rotationOperator(tempX, rotation, rotCenter)
        center_ = oc.getPhysicalCenter(Xref)  # redundant?
        translation = -center_

        Xref = self.translateOp(Xref, translation)

        theta = torch.arctan2(Xref[N:], Xref[:N])
        start_id = torch.argmin(torch.where(theta < 0, 100, theta), dim=0)
        multi_sortIdx = (
            start_id + torch.arange(N, device=Xref.device).unsqueeze(-1)
        ) % N
        multi_sortIdx = multi_sortIdx.int().T

        length = oc.geomProp_length(Xref)
        scaling = 1.0 / length

        return translation, rotation, rotCenter, scaling, multi_sortIdx

    def rotationOperator(self, X, theta, rotCent):
        """Shan: compatible with multi ves
        theta of shape (1,nv), rotCent of shape (2,nv)"""
        Xrot = torch.zeros_like(X)
        x = X[: len(X) // 2] - rotCent[0]
        y = X[len(X) // 2 :] - rotCent[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[: len(X) // 2] = xrot + rotCent[0]
        Xrot[len(X) // 2 :] = yrot + rotCent[1]
        return Xrot

    def translateOp(self, X, transXY):
        """Shan: compatible with multi ves
        transXY of shape (2,nv)"""
        Xnew = torch.zeros_like(X)
        Xnew[: len(X) // 2] = X[: len(X) // 2] + transXY[0]
        Xnew[len(X) // 2 :] = X[len(X) // 2 :] + transXY[1]
        return Xnew

    def rotation_trans_Operator(self, X, theta, rotCent, transXY):
        """
        combining rotate and trans
        """

        Xrot = torch.zeros_like(X, device=X.device)
        x = X[: len(X) // 2] - rotCent[0]
        y = X[len(X) // 2 :] - rotCent[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[: len(X) // 2] = xrot + rotCent[0] + transXY[0]
        Xrot[len(X) // 2 :] = yrot + rotCent[1] + transXY[1]
        return Xrot

    def trans_rotation_Operator(self, X, theta, rotCent, transXY):
        """
        combining rotate and trans
        """
        Xrot = torch.zeros_like(X)

        x = X[: len(X) // 2] - rotCent[0] - transXY[0]
        y = X[len(X) // 2 :] - rotCent[1] - transXY[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[: len(X) // 2] = xrot + rotCent[0]
        Xrot[len(X) // 2 :] = yrot + rotCent[1]
        return Xrot
