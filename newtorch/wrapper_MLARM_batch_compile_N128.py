import numpy as np
import torch
torch.set_default_dtype(torch.float32)
# torch.set_default_device('cuda:0')
import sys
sys.path.append("..")
import pdb
from collections import defaultdict
from capsules import capsules
from filter import filterShape, filterTension, interpft, interpft_vec, upsample_fft, downsample_fft, gaussian_filter_shape, gaussian_filter_1d_energy_preserve
from filter import rescale_outlier_vel, rescale_outlier_vel_abs, rescale_outlier_trans
from torch.profiler import profile, record_function, ProfilerActivity
from model_zoo.get_network_torch import RelaxNetwork, TenSelfNetwork, MergedAdvNetwork, MergedTenAdvNetwork, MergedNearFourierNetwork
# from relaxation_solve import relax_solve
from math import ceil, sqrt
from typing import List, Tuple
from biem_support import exactStokesSL_onlyself

def relax_solve(X,  op):
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
    M = G @ Ten @ torch.linalg.solve(
        (Div @ G @ Ten), 
        torch.eye(N, device=device).unsqueeze(0).repeat(nv, 1, 1)
    ) @ Div

    # Compute Left-Hand Side (LHS) matrix
    LHS = torch.eye(2 * N, device=device).unsqueeze(0).repeat(nv,1,1) - vesicle.kappa * dt * (
        -G @ Ben + M @ G @ Ben
    )

    # Solve for Xnew
    Xnew = torch.linalg.solve(LHS, X.T)

    return Xnew.T


# @torch.jit.script
# def allExactStokesSLTarget_1st_broadcast(vesicleX, vesicle_sa, f, tarX, length:float=1.0, offset: int = 0):
#     """
#     Computes the single-layer potential due to `f` around all vesicles except itself.
    
#     Parameters:
#     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
#     - f: Forcing term (2*N x nv).

#     Returns:
#     - stokesSLPtar: Single-layer potential at target points.
#     """
    
#     N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
#     Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
#     stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicleX.device)

#     mask = ~torch.eye(nv, dtype=torch.bool)
#     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
#     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
#     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
#     indices = indices[offset:offset+ntar]

#     den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
#     denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
#     deny = den[N:, indices].permute(0, 2, 1) 

        
#     xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
#     ysou = vesicleX[N:, indices].permute(0, 2, 1) 

#     # if tarX is not None:
#     xtar = tarX[:Ntar]
#     ytar = tarX[Ntar:]
#     # else:
#     #     xtar = vesicleX[:N]
#     #     ytar = vesicleX[N:]
    
#     diffx = xtar[None, None, ...] - xsou[:, :, None] # broadcasting, (N, (nv-1), Ntar, nv)
#     del xtar
#     del xsou
#     diffy = ytar[None, None, ...] - ysou[:, :, None]
#     del ytar
#     del ysou

#     dis2 = diffx**2 + diffy**2
#     info = dis2 < (length/Ntar)**2
#     # Compute the cell-level mask 
#     # cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
#     # full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, ntar)
#     full_mask = info.any(dim=0).unsqueeze(0) 
    
#     new_info = torch.zeros((ntar, Ntar,  nv, N), dtype=torch.bool, device=vesicleX.device)
#     new_info[torch.arange(ntar), :, indices.T, :] = info.permute(1,3,2,0)
#     # info = torch.concat((info, torch.zeros((N, 1, Ntar, ntar), dtype=torch.bool)), dim=1)

#     # start = torch.cuda.Event(enable_timing=True)
#     # end = torch.cuda.Event(enable_timing=True)
#     # start.record()
    
#     coeff = 0.5 * torch.log(dis2)
#     coeff.masked_fill_(full_mask, 0)
#     col_indices = torch.arange(ntar)
#     stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
#     stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

#     coeff = (diffx * denx.unsqueeze(2) + diffy * deny.unsqueeze(2)) / dis2
#     coeff.masked_fill_(full_mask, 0)
#     stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
#     stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])

    
#     # end.record()
#     # torch.cuda.synchronize()
#     # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

#     return stokesSLPtar / (4 * torch.pi), new_info


# @torch.jit.script
# def allExactStokesSLTarget_2nd_broadcast(vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask,  offset: int = 0):
#     """
#     Computes the single-layer potential due to `f` around all vesicles except itself.
    
#     Parameters:
#     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
#     - f: Forcing term (2*N x nv).

#     Returns:
#     - stokesSLPtar: Single-layer potential at target points.
#     """
    
#     N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
#     Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
#     stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicleX.device)

#     mask = ~torch.eye(nv, dtype=torch.bool)
#     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
#     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
#     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
#     indices = indices[offset:offset+ntar]

#     den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
#     denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
#     deny = den[N:, indices].permute(0, 2, 1) 

    
#     coeff = 0.5 * torch.log(dis2)
#     coeff.masked_fill_(full_mask, 0)
#     col_indices = torch.arange(ntar)
#     stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
#     stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

#     coeff = (diffx * denx.unsqueeze(2) + diffy * deny.unsqueeze(2)) / dis2
#     coeff.masked_fill_(full_mask, 0)
#     stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
#     stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])
    
#     # end.record()
#     # torch.cuda.synchronize()
#     # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

#     return stokesSLPtar / (4 * torch.pi) # info, dis2, diffx, diffy, full_mask


# @torch.jit.script
# def allExactStokesSLTarget_returninfo_broadcast(vesicleX, vesicle_sa, f, tarX, offset: int = 0, return_info: bool = False):
#     """
#     Computes the single-layer potential due to `f` around all vesicles except itself.
    
#     Parameters:
#     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
#     - f: Forcing term (2*N x nv).

#     Returns:
#     - stokesSLPtar: Single-layer potential at target points.
#     """
    
#     N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
#     Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
#     stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicleX.device)

#     mask = ~torch.eye(nv, dtype=torch.bool)
#     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
#     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
#     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
#     indices = indices[offset:offset+ntar]

#     den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
#     denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
#     deny = den[N:, indices].permute(0, 2, 1) 

        
#     xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
#     ysou = vesicleX[N:, indices].permute(0, 2, 1) 

#     # if tarX is not None:
#     xtar = tarX[:Ntar]
#     ytar = tarX[Ntar:]
#     # else:
#     #     xtar = vesicleX[:N]
#     #     ytar = vesicleX[N:]
    
#     diffx = xtar[None, None, ...] - xsou[:, :, None] # broadcasting, (N, (nv-1), Ntar, nv)
#     del xtar
#     del xsou
#     diffy = ytar[None, None, ...] - ysou[:, :, None]
#     del ytar
#     del ysou

#     dis2 = diffx**2 + diffy**2
#     info = dis2 <= (1/Ntar)**2
#     # Compute the cell-level mask 
#     # cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
#     # full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, ntar)
#     full_mask = info.any(dim=0).unsqueeze(0) 

#     # start = torch.cuda.Event(enable_timing=True)
#     # end = torch.cuda.Event(enable_timing=True)
#     # start.record()
    
#     coeff = 0.5 * torch.log(dis2)
#     coeff.masked_fill_(full_mask, 0)
#     col_indices = torch.arange(ntar)
#     stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
#     stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

#     coeff = (diffx * denx.unsqueeze(2) + diffy * deny.unsqueeze(2)) / dis2
#     coeff.masked_fill_(full_mask, 0)
#     stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
#     stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])

    
#     # end.record()
#     # torch.cuda.synchronize()
#     # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')
    
#     new_info = torch.zeros((ntar, Ntar,  nv, N), dtype=torch.bool, device=vesicleX.device)
#     new_info[torch.arange(ntar), :, indices.T, :] = info.permute(1,3,2,0)
#     return stokesSLPtar / (4 * torch.pi), new_info


# @torch.jit.script
def allExactStokesSLTarget_broadcast(vesicleX, vesicle_sa, f, tarX, length:float=1.0, offset: int = 0):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.
    
    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """
    
    N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
    stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicleX.device)

    mask = ~torch.eye(nv, dtype=torch.bool)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
    indices = indices[offset:offset+ntar]

    den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
    deny = den[N:, indices].permute(0, 2, 1) 

        
    # xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
    # ysou = vesicleX[N:, indices].permute(0, 2, 1) 

    # if tarX is not None:
    # xtar = tarX[:Ntar]
    # ytar = tarX[Ntar:]
    # else:
    #     xtar = vesicleX[:N]
    #     ytar = vesicleX[N:]
    
    diffx = tarX[None, None, :Ntar, ...] - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None] # broadcasting, (N, (nv-1), Ntar, nv)
    diffy = tarX[None, None, Ntar:, ...] - vesicleX[N:, indices].permute(0, 2, 1) [:, :, None]

    # diff = tarX[None, None, ...] - vesicleX[:, indices].permute(0, 2, 1) [:, :, None]
    # diffx = diff[:N, :, :Ntar, :]
    # diffy = diff[N:, :, Ntar:, :]

    dis2 = diffx**2 + diffy**2
    info = dis2 < (length /Ntar)**2
    # Compute the cell-level mask 
    # cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
    # full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, ntar)
    full_mask = info.any(dim=0).unsqueeze(0) 

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    
    coeff = 0.5 * torch.log(dis2)
    coeff.masked_fill_(full_mask, 0)
    # coeff[full_mask.expand(N,-1,-1,-1)] = 0.
    col_indices = torch.arange(ntar)
    stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
    stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

    coeff = (diffx * denx.unsqueeze(2) + diffy * deny.unsqueeze(2)) / dis2
    coeff.masked_fill_(full_mask, 0)
    # coeff[full_mask.expand(N,-1,-1,-1)] = 0.
    stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
    stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])

    
    # end.record()
    # torch.cuda.synchronize()
    # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

    return stokesSLPtar / (4 * torch.pi)

# @torch.jit.script
# def allExactStokesSLTarget_mat(vesicleX, vesicle_sa, f, tarX, offset: int = 0):
#     """
#     Computes the single-layer potential due to `f` around all vesicles except itself.
    
#     Parameters:
#     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
#     - f: Forcing term (2*N x nv).

#     Returns:
#     - stokesSLPtar: Single-layer potential at target points.
#     """
    
#     N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
#     Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
#     stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicleX.device)

#     mask = ~torch.eye(nv, dtype=torch.bool)
#     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
#     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
#     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
#     indices = indices[offset:offset+ntar]

#     den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
#     denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
#     deny = den[N:, indices].permute(0, 2, 1)

        
#     # xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
#     # ysou = vesicleX[N:, indices].permute(0, 2, 1) 

#     # if tarX is not None:
#     # xtar = tarX[:Ntar]
#     # ytar = tarX[Ntar:]
#     # else:
#     #     xtar = vesicleX[:N]
#     #     ytar = vesicleX[N:]
    
#     diffx2 = (tarX[None, None, :Ntar, ...] - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None])**2 # broadcasting, (N, (nv-1), Ntar, nv)
#     diffy2 = (tarX[None, None, Ntar:, ...] - vesicleX[N:, indices].permute(0, 2, 1) [:, :, None])**2

#     # diff = tarX[None, None, ...] - vesicleX[:, indices].permute(0, 2, 1) [:, :, None]
#     # diffx = diff[:N, :, :Ntar, :]
#     # diffy = diff[N:, :, Ntar:, :]

#     dis2 = diffx2 + diffy2
#     info = dis2 <= (1/Ntar)**2
#     # Compute the cell-level mask 
#     # cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
#     # full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, ntar)
#     full_mask = info.any(dim=0).unsqueeze(0) 

#     # cell_mask = info.any(dim=2)  # Shape: (N, nv-1, ntar)

#     # start = torch.cuda.Event(enable_timing=True)
#     # end = torch.cuda.Event(enable_timing=True)
#     # start.record()
#     diffxy = (tarX[None, None, :Ntar, ...] - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None]) * (tarX[None, None, Ntar:, ...] - vesicleX[N:, indices].permute(0, 2, 1)[:, :, None])
#     diffxy.masked_fill_(full_mask, 0)
#     diffxy = diffxy / dis2

#     coeff = 0.5 * torch.log(dis2)
#     coeff.masked_fill_(full_mask, 0)
#     col_indices = torch.arange(ntar)
#     # stokesSLPtar[:Ntar, col_indices] = torch.sum((diffx2.masked_fill_(full_mask, 0)/dis2 - coeff) * denx.unsqueeze(2) + \
#     #                                              diffxy * deny.unsqueeze(2), dim=[0, 1])
#     # stokesSLPtar[:Ntar, col_indices] = torch.einsum("abcd, abd -> cd", (diffx2.masked_fill_(full_mask, 0)/dis2 - coeff),  denx) + \
#     #                                              torch.einsum("abcd, abd -> cd", diffxy , deny)
#     # stokesSLPtar[Ntar:, col_indices] = torch.sum(diffxy * denx.unsqueeze(2) + \
#     #                                     (diffy2.masked_fill_(full_mask, 0)/dis2 - coeff) * deny.unsqueeze(2), dim=[0, 1])
#     stokesSLPtar[Ntar:, col_indices] = torch.einsum("abcd -> cd", diffxy * denx.unsqueeze(2) + \
#                                         (diffy2.masked_fill_(full_mask, 0)/dis2 - coeff) * deny.unsqueeze(2))


#     # stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
#     # stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

#     # coeff = (diffx * denx.unsqueeze(2) + diffy * deny.unsqueeze(2)) / dis2
#     # coeff.masked_fill_(full_mask, 0)
#     # stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
#     # stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])

#     # M = torch.concat(
                    # (torch.concat((diffx2.masked_fill_(full_mask, 0) /dis2 - coeff, diffxy/dis2), dim=0), 
                    # torch.concat((diffxy/dis2, diffy2.masked_fill_(full_mask, 0)/dis2 - coeff), dim=0))
                    # , dim=2)
        # M shape is (2N, nv-1, 2Ntar, nv)
#     # stokesSLPtar = torch.einsum("Nvnu, Nvu -> nu", M, torch.concat((denx, deny), dim=0))

    
#     # end.record()
#     # torch.cuda.synchronize()
#     # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

#     return stokesSLPtar / (4 * torch.pi)


# @torch.compile(backend='cudagraphs')
@torch.jit.script
def allExactStokesSLTarget_compare1(vesicleX, vesicle_sa, f, tarX, length:float=1.0, offset: int = 0):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.
    
    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """
    
    N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
    stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=tarX.dtype, device=vesicleX.device)

    mask = ~torch.eye(nv, dtype=torch.bool, device=vesicleX.device)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = torch.arange(nv, device=vesicleX.device)[None,].expand(nv,-1)[mask].view(nv, nv-1)
    indices = indices[offset:offset+ntar]

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    denx = den[:N, indices].permute(0, 2, 1).unsqueeze(2)  # (N, (nv-1), 1, ntar)
    deny = den[N:, indices].permute(0, 2, 1).unsqueeze(2)
    # end.record()
    # torch.cuda.synchronize()
    # print(f'inside ExactStokesSL, dup den {start.elapsed_time(end)/1000} sec.')
        
    # xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), nv)
    # ysou = vesicleX[N:, indices].permute(0, 2, 1) 

    # if tarX is not None:
    # xtar = tarX[:Ntar]
    # ytar = tarX[Ntar:]
    # else:
    #     xtar = vesicleX[:N]
    #     ytar = vesicleX[N:]
    
    diffx = tarX[None, None, :Ntar, ...] - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None] # broadcasting, (N, (nv-1), Ntar, ntar)
    diffy = tarX[None, None, Ntar:, ...] - vesicleX[N:, indices].permute(0, 2, 1) [:, :, None]

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
    ids = torch.where(torch.max((dis2.reshape(N, nv-1, -1) < (length/Ntar)**2), dim=0)[0])
    # ids = torch.where((dis2 < (1/Ntar)**2 ).any(dim=0).unsqueeze(0).expand(N, -1, -1, -1))
    # ids_true = torch.unbind(torch.max(dis2 < (1/Ntar)**2, dim=0)[0].to_sparse().indices(), dim=0)
    ids = (ids[0], ids[1]// ntar, ids[1] % ntar)

    l = len(ids[0])
    ids_ = (torch.arange(N, device=f.device)[:, None].expand(-1, l).reshape(-1), 
            ids[0][None,:].expand(N, -1).reshape(-1),
            ids[1][None,:].expand(N, -1).reshape(-1),
            ids[2][None,:].expand(N, -1).reshape(-1),
            )
    

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    
    # coeff = 0.5 * torch.log(dis2)
    # coeff.index_put_(ids, torch.tensor([0.], device=f.device))
    # c1 = 0.5 * torch.log(dis2).masked_fill(full_mask, 0)
    # col_indices = torch.arange(ntar)
    # stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
    # stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

    coeff = (diffx * denx + diffy * deny) / dis2
    # coeff.index_put_(ids, torch.tensor([0.], device=f.device))
    # stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
    # stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])

    # if not torch.allclose(diffy.index_put(ids, torch.tensor([0.])), diffy.index_put(ids_, torch.tensor([0.]))):
    #     np.save("ids_diff.npy", {'upX':vesicleX, 'X':tarX, 'input_ids':(ids0, ids1, ids2), 'ids':torch.where((dis2 <= (1/Ntar)**2).any(dim=0))})
    #     raise "ids err"
    
    # if not torch.allclose(diffx.index_put(ids, torch.tensor([0.])), diffx.index_put(ids_, torch.tensor([0.]))):
    #     raise "ids err"
    
    # if not torch.allclose(dis2.index_put(ids, torch.tensor([0.])), dis2.index_put(ids_, torch.tensor([0.]))):
    #     raise "ids err"

    stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(ids_, torch.tensor([0.], device=f.device, dtype=vesicleX.dtype)), dim=[0,1])
    stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(ids_, torch.tensor([0.], device=f.device, dtype=vesicleX.dtype)), dim=[0,1])

    # stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx), dim=[0,1])
    # stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny), dim=[0,1])

    # end.record()
    # torch.cuda.synchronize()
    # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')
    # pdb.set_trace()
    return stokesSLPtar / (4 * torch.pi), (ids[0], ids[1], ids[2]+offset)



# @torch.compile(backend='cudagraphs')
@torch.jit.script
def allExactStokesSLTarget_compare2(vesicleX, vesicle_sa, f, tarX, ids0, ids1, ids2, length:float=1.0, offset: int = 0):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.
    
    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """
    
    N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
    stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=vesicleX.dtype, device=vesicleX.device)

    mask = ~torch.eye(nv, dtype=torch.bool, device=vesicleX.device)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = torch.arange(nv, device=vesicleX.device)[None,].expand(nv,-1)[mask].view(nv, nv-1)
    indices = indices[offset:offset+ntar]

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
    
    diffx = tarX[None, None, :Ntar, ...] - vesicleX[:N, indices].permute(0, 2, 1)[:, :, None] # broadcasting, (N, (nv-1), Ntar, nv)
    diffy = tarX[None, None, Ntar:, ...] - vesicleX[N:, indices].permute(0, 2, 1) [:, :, None]

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
    ids_ = (torch.arange(N, device=f.device)[:, None].expand(-1, l).reshape(-1), 
            ids0[None,:].expand(N, -1).reshape(-1),
            ids1[None,:].expand(N, -1).reshape(-1),
            ids2[None,:].expand(N, -1).reshape(-1),
            )
    

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    
    # coeff = 0.5 * torch.log(dis2)
    # coeff.index_put_(ids, torch.tensor([0.], device=f.device))
    # c1 = 0.5 * torch.log(dis2).masked_fill(full_mask, 0)
    # col_indices = torch.arange(ntar)
    # stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
    # stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

    coeff = (diffx * denx + diffy * deny) / dis2
    # coeff.index_put_(ids, torch.tensor([0.], device=f.device))
    # stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
    # stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])

    # if not torch.allclose(diffy.index_put(ids, torch.tensor([0.])), diffy.index_put(ids_, torch.tensor([0.]))):
    #     np.save("ids_diff.npy", {'upX':vesicleX, 'X':tarX, 'input_ids':(ids0, ids1, ids2), 'ids':torch.where((dis2 <= (1/Ntar)**2).any(dim=0))})
    #     raise "ids err"
    
    # if not torch.allclose(diffx.index_put(ids, torch.tensor([0.])), diffx.index_put(ids_, torch.tensor([0.]))):
    #     raise "ids err"
    
    # if not torch.allclose(dis2.index_put(ids, torch.tensor([0.])), dis2.index_put(ids_, torch.tensor([0.]))):
    #     raise "ids err"

    stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(ids_, torch.tensor([0.], device=f.device, dtype=vesicleX.dtype)), dim=[0,1])
    stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(ids_, torch.tensor([0.], device=f.device, dtype=vesicleX.dtype)), dim=[0,1])

    # stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx), dim=[0,1])
    # stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny), dim=[0,1])

    # end.record()
    # torch.cuda.synchronize()
    # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

    return stokesSLPtar / (4 * torch.pi)



class MLARM_manyfree_py(torch.jit.ScriptModule):
    def __init__(self, dt, vinf, oc, use_repulsion, repStrength,
                 rbf_upsample : int,
                 advNetInputNorm, advNetOutputNorm,
                 relaxNetInputNorm, relaxNetOutputNorm, 
                 nearNetInputNorm, nearNetOutputNorm, 
                 tenSelfNetInputNorm, tenSelfNetOutputNorm,
                 tenAdvNetInputNorm, tenAdvNetOutputNorm, device, logger):        

        super().__init__()

        self.dt = dt  # time step size
        self.vinf = vinf  # background flow (analytic -- itorchut as function of vesicle config)
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
        self.mergedAdvNetwork = MergedAdvNetwork(self.advNetInputNorm.to(device), self.advNetOutputNorm.to(device), 
                                # model_path="../trained/2024Oct_ves_merged_adv.pth", 
                                model_path="../trained/torch_script_models/2024Oct_ves_merged_adv.pt", 
                                device = device)
                            
        # Normalization values for relaxation network
        self.relaxNetInputNorm = relaxNetInputNorm
        self.relaxNetOutputNorm = relaxNetOutputNorm
        self.relaxNetwork = RelaxNetwork(self.dt, self.relaxNetInputNorm.to(device), self.relaxNetOutputNorm.to(device), 
                                # model_path="../trained/ves_relax_DIFF_June8_625k_dt1e-5.pth", 
                                model_path="../trained/torch_script_models/ves_relax_DIFF_June8_625k_dt1e-5.pt", 
                                device = device)
        
        # Normalization values for near field networks
        self.nearNetInputNorm = nearNetInputNorm
        self.nearNetOutputNorm = nearNetOutputNorm
        self.nearNetwork = MergedNearFourierNetwork(self.nearNetInputNorm.to(device), self.nearNetOutputNorm.to(device),
                                # model_path="../trained/ves_merged_disth_nearFourier.pth",
                                model_path="../trained/torch_script_models/ves_merged_disth_nearFourier.pt",
                                device = device)
        
        # Normalization values for tension-self network
        self.tenSelfNetInputNorm = tenSelfNetInputNorm
        self.tenSelfNetOutputNorm = tenSelfNetOutputNorm
        self.tenSelfNetwork = TenSelfNetwork(self.tenSelfNetInputNorm.to(device), self.tenSelfNetOutputNorm.to(device), 
                                # model_path = "../trained/Ves_2024Oct_selften_12blks_loss_0.00566cuda1.pth",
                                model_path = "../trained/torch_script_models/Ves_2024Oct_selften_12blks_loss_0.00566cuda1.pt",
                                device = device)
        
        # Normalization values for tension-advection networks
        self.tenAdvNetInputNorm = tenAdvNetInputNorm
        self.tenAdvNetOutputNorm = tenAdvNetOutputNorm
        self.tenAdvNetwork = MergedTenAdvNetwork(self.tenAdvNetInputNorm.to(device), self.tenAdvNetOutputNorm.to(device), 
                                # model_path="../trained/2024Oct_ves_merged_advten.pth", 
                                model_path="../trained/torch_script_models/2024Oct_ves_merged_advten.pt", 
                                device = device)
        
    

    # def time_step_many(self, Xold, tenOld):
    #     # oc = self.oc
    #     torch.set_default_device(Xold.device)
    #     # background velocity on vesicles
    #     vback = self.vinf(Xold)
        

    #     # build vesicle class at the current step
    #     vesicle = capsules(Xold, [], [], self.kappa, 1)
    #     N = Xold.shape[0] // 2
    #     nv = Xold.shape[1]
    #     Nup = ceil(sqrt(N)) * N
        
    #     vesicleUp = capsules(upsample_fft(Xold, Nup), [],[], self.kappa, 1)

    #     # Compute velocity induced by repulsion force
    #     repForce = torch.zeros_like(Xold)
    #     # if self.use_repulsion:
    #     #     repForce = vesicle.repulsionForce(Xold, self.repStrength)

    #     # Compute bending forces + old tension forces
    #     fBend = vesicle.bendingTerm(Xold)
    #     fTen = vesicle.tensionTerm(tenOld)
    #     tracJump = fBend + fTen  # total elastic force

    #     Xstand, standardizationValues = self.standardizationStep(Xold)

    #     # Explicit Tension at the Current Step
    #     # Calculate velocity induced by vesicles on each other due to elastic force
    #     # use neural networks to calculate near-singular integrals
    #     # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    #     #     with record_function("predictNearLayers"):
    #     velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues)

    #     # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    #     # info = self.nearZoneInfo(vesicle)
    #     # info_rbf, info_stokes = self.naiveNearZoneInfo(vesicle.X, vesicleUp.X)
    #     info_rbf, info_stokes = None, None

    #     const = 0.672 * self.len0[0].item()
    #     all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
    #     all_X = all_X /const * N   
    #     matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) #+ 1e-6 * torch.eye(5*N).unsqueeze(-1) # (3*N, 3*N, nv)
    #     L = torch.linalg.cholesky(matrices.permute(2, 0, 1))

        
    #     farFieldtracJump, info_rbf, info_stokes = self.computeStokesInteractions(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
    #                                     xlayers, ylayers, standardizationValues, first=True)
    #     farFieldtracJump = filterShape(farFieldtracJump, 4)


    #     # if not torch.allclose(info, info_.reshape(nv, N, nv, Nup)):
    #     #     raise ValueError('info not equal')
        
    #     vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)

    #     selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)


    #     tenNew = -(vBackSolve + selfBendSolve)
    #     # tenNew = filterTension(tenNew, 4*N, 16)

    #     # update the elastic force with the new tension
    #     fTen_new = vesicle.tensionTerm(tenNew)
    #     tracJump = fBend + fTen_new

    #     # Calculate far-field again and correct near field before advection
    #     # use neural networks to calculate near-singular integrals
    #     farFieldtracJump, info_rbf, info_stokes = self.computeStokesInteractions(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
    #                                             xlayers, ylayers, standardizationValues, first=False)
    #     farFieldtracJump = filterShape(farFieldtracJump, 4)

    #     # Total background velocity
    #     vbackTotal = vback + farFieldtracJump

    #     # Compute the action of dt*(1-M) on Xold

    #     Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)


    #     Xadv = filterShape(Xadv, 8)
    #     # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
    #     # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
    #     # Compute the action of relax operator on Xold + Xadv
    #     Xnew = self.relaxWTorchNet(Xadv)

    #     modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(Xold.device)
    #     XnewC = Xnew.clone()
    #     for _ in range(5):
    #         Xnew, flag = self.oc.redistributeArcLength(Xnew, modes)
    #         if flag:
    #             break
    #     Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))

    #     Xnew = self.oc.correctAreaAndLength(Xnew, self.area0, self.len0)

    #     Xnew = filterShape(Xnew.to(Xold.device), 8)

    #     # MLARM_manyfree_py.info, MLARM_manyfree_py.dis2, MLARM_manyfree_py.diffx, MLARM_manyfree_py.diffy, MLARM_manyfree_py.full_mask = None, None, None, None, None

    #     return Xnew, tenNew
    

    
    def time_step_many_timing_noinfo(self, Xold, tenOld, nlayers=5):
        # oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]
        Nup = ceil(sqrt(N)) * N
        vesicleUp = capsules(upsample_fft(Xold, Nup), [],[], self.kappa, 1)

        # Compute velocity induced by repulsion force
        repForce = torch.zeros_like(Xold)
        # if self.use_repulsion:
        #     repForce = vesicle.repulsionForce(Xold, self.repStrength)

        # Compute bending forces + old tension forces
        # fBend = vesicle.bendingTerm(Xold)
        fBend = vesicleUp.bendingTerm(vesicleUp.X) # upsampled bending term
        fBend = downsample_fft(fBend, N)
        fTen = vesicleUp.tensionTerm(interpft(tenOld, Nup))
        fTen = downsample_fft(fTen, N)
        # fTen = vesicle.tensionTerm(tenOld)
        # tracJump = fBend + fTen  # total elastic force

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        Xstand, standardizationValues = self.standardizationStep(Xold)
        end.record()
        torch.cuda.synchronize()
        print(f'standardizationStep {start.elapsed_time(end)/1000} sec.')

        # Explicit Tension at the Current Step
        # Calculate velocity induced by vesicles on each other due to elastic force
        # use neural networks to calculate near-singular integrals
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        #     with record_function("predictNearLayers"):
        start.record()
        velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues, nlayers)
        end.record()
        torch.cuda.synchronize()
        print(f'predictNearLayers {start.elapsed_time(end)/1000} sec.')
        

        # start.record()
        # info_rbf, info_stokes = self.naiveNearZoneInfo(vesicle.X, vesicleUp.X)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'nearZoneInfo {start.elapsed_time(end)/1000} sec.')
        info_rbf, info_stokes = None, None

        start.record()
        if self.rbf_upsample <=1 :
            const = 0.672 * self.len0[0].item()
        elif self.rbf_upsample == 2:
            const = 0.566 * self.len0[0].item()
            xlayers = interpft(xlayers.reshape(N, -1), N * 2)
            ylayers = interpft(ylayers.reshape(N, -1), N * 2)
        elif self.rbf_upsample == 4:
            const = 0.495 * self.len0[0].item()
            xlayers = interpft(xlayers.reshape(N, -1), N * 4)
            ylayers = interpft(ylayers.reshape(N, -1), N * 4)

        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (nlayers * N, 2, nv), 2 for x and y
        all_X = all_X /const * N  
        matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) 
        if self.rbf_upsample == 4:
            matrices += (torch.eye(all_X.shape[0]).unsqueeze(-1) * 1e-6).expand(-1,-1,nv) # (nlayers*N, nlayers*N, nv)
        
        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        end.record()
        torch.cuda.synchronize()
        print(f'CHOLESKY {start.elapsed_time(end)/1000} sec.')
        

        start.record()
        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
        end.record()
        torch.cuda.synchronize()
        print(f'invTenMatOnSelfBend {start.elapsed_time(end)/1000} sec.')

        start.record()
        farFieldtracJump, info_rbf, info_stokes = self.computeStokesInteractions_timing_noinfo(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                        xlayers, ylayers, standardizationValues, nlayers, first=True)
        end.record()
        torch.cuda.synchronize()
        print(f'x1computeStokesInteractions first {start.elapsed_time(end)/1000} sec.')

        # farFieldtracJump = filterShape(farFieldtracJump, 4)
        
        start.record()
        vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
        end.record()
        torch.cuda.synchronize()
        print(f'invTenMatOnVback {start.elapsed_time(end)/1000} sec.')

        tenNew = -(vBackSolve + selfBendSolve)
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        start.record()
        farFieldtracJump, _, _ = self.computeStokesInteractions_timing_noinfo(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                                    xlayers, ylayers, standardizationValues, nlayers, first=False)
        end.record()
        torch.cuda.synchronize()
        print(f'x1computeStokesInteractions second {start.elapsed_time(end)/1000} sec.')

        
        # farFieldtracJump = filterShape(farFieldtracJump, 4)

        # Total background velocity
        vbackTotal = vback + farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        start.record()
        Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)
        end.record()
        torch.cuda.synchronize()
        print(f'translateVinfwTorch {start.elapsed_time(end)/1000} sec.')

        Xadv = filterShape(Xadv, 8)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
        # Compute the action of relax operator on Xold + Xadv
        start.record()
        Xnew = self.relaxWTorchNet(Xadv)
        end.record()
        torch.cuda.synchronize()
        print(f'relaxWTorchNet {start.elapsed_time(end)/1000} sec, containing standardization time.')

        modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(Xold.device) #.double()

        XnewC = Xnew.clone()
        start.record()
        for _ in range(5):
            Xnew, flag = self.oc.redistributeArcLength(Xnew, modes)
            # if flag:
            #     break
        Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))
        end.record()
        torch.cuda.synchronize()
        print(f'x5 redistributeArcLength and alignCenterAngle {start.elapsed_time(end)/1000} sec.')
        

        start.record()
        with torch.enable_grad():
            Xnew = self.oc.correctAreaAndLengthAugLag(Xnew, self.area0, self.len0)
        end.record()
        torch.cuda.synchronize()
        print(f'correctAreaLength {start.elapsed_time(end)/1000} sec.')

        Xnew = filterShape(Xnew.to(Xold.device), 8)

        return Xnew, tenNew
    

    
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
        vesicleUp = capsules(upsample_fft(Xold, Nup), [],[], self.kappa, 1)

        # Compute velocity induced by repulsion force
        
        repForce = torch.zeros_like(Xold)
        if self.use_repulsion:
            repForce = vesicle.repulsionForce(Xold, self.repStrength)

        # Compute bending forces + old tension forces
        # fBend = vesicle.bendingTerm(Xold)
        fTen = vesicle.tensionTerm(tenOld)
        fBend = vesicleUp.bendingTerm(vesicleUp.X) # upsampled bending term
        fBend = downsample_fft(fBend, N)
        # fTen = vesicleUp.tensionTerm(interpft(tenOld, N*6))
        # fTen = downsample_fft(fTen, N)
        
        tracJump = fBend + fTen  # total elastic force

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        
        Xstand, standardizationValues = self.standardizationStep(Xold)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'standardizationStep {start.elapsed_time(end)/1000} sec.')

        # Explicit Tension at the Current Step
        # Calculate velocity induced by vesicles on each other due to elastic force
        # use neural networks to calculate near-singular integrals
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        #     with record_function("predictNearLayers"):
        # start.record()
        velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues, nlayers)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'predictNearLayers {start.elapsed_time(end)/1000} sec.')
        

        # start.record()
        # info_rbf_true, info_stokes_true = self.naiveNearZoneInfo(vesicle.X, vesicleUp.X)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'nearZoneInfo {start.elapsed_time(end)/1000} sec.')
        info_rbf, info_stokes = None, None

        # start.record()
        # if self.rbf_upsample <=1 :
        #     const = 0.672 * self.len0[0].item()
        # elif self.rbf_upsample == 2:
        #     const = 0.566 * self.len0[0].item()
        #     xlayers = interpft(xlayers.reshape(N, -1), N * 2)
        #     ylayers = interpft(ylayers.reshape(N, -1), N * 2)
        # elif self.rbf_upsample == 4:
        #     const = 0.495 * self.len0[0].item()
        #     xlayers = interpft(xlayers.reshape(N, -1), N * 4)
        #     ylayers = interpft(ylayers.reshape(N, -1), N * 4)

        if self.rbf_upsample <= 2:
            # const = 0.495 * self.len0[0].item() * 4
            # const = 1.7 / 128
            const = 0.0132
        if self.rbf_upsample == 2:
            xlayers = interpft(xlayers.reshape(N, -1), N * 2)
            ylayers = interpft(ylayers.reshape(N, -1), N * 2)

        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (nlayers * N, 2, nv), 2 for x and y
        # all_X = all_X /const * N  
        all_X = all_X / const
        matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) 
        # if self.rbf_upsample > 1:
        matrices += (torch.eye(all_X.shape[0]).unsqueeze(-1) * 1e-6).expand(-1,-1,nv) # (nlayers*N, nlayers*N, nv)
        
        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        # end.record()
        # torch.cuda.synchronize()
        # print(f'CHOLESKY {start.elapsed_time(end)/1000} sec.')
        

        # start.record()
        farFieldtracJump, info_rbf, info_stokes = self.computeStokesInteractions_noinfo(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                        xlayers, ylayers, standardizationValues, nlayers, first=True)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x1computeStokesInteractions first {start.elapsed_time(end)/1000} sec.')
        # pdb.set_trace()

        farFieldtracJump = filterShape(farFieldtracJump, 16)
        # farFieldtracJump = rescale_outlier_vel_abs(farFieldtracJump, logger=self.logger)
        print(f"monitoring 1st farfieldtracjump magnitude: {torch.max(torch.abs(farFieldtracJump))}")

        # print(f"true rbf_info is {info_rbf_true}, true stokes_info is {info_stokes_true}")
        # print(f"my rbf_info is {info_rbf}, my stokes_info is {info_stokes}")
        
        # start.record()
        vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'invTenMatOnVback {start.elapsed_time(end)/1000} sec.')

        # start.record()
        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'invTenMatOnSelfBend {start.elapsed_time(end)/1000} sec.')

        tenNew = -(vBackSolve + selfBendSolve)
        # pdb.set_trace()
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new
        # pdb.set_trace()

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        # start.record()
        farFieldtracJump, _, _ = self.computeStokesInteractions_noinfo(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                                    xlayers, ylayers, standardizationValues, nlayers, first=False)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x1computeStokesInteractions second {start.elapsed_time(end)/1000} sec.')

        farFieldtracJump = filterShape(farFieldtracJump, 16)

        # farFieldtracJump = gaussian_filter_1d_energy_preserve(torch.concat((farFieldtracJump[:N, None],farFieldtracJump[N:, None]), dim=1) , sigma=7e-2)
        # farFieldtracJump = torch.concat((farFieldtracJump[:, 0], farFieldtracJump[:, 1]), dim=0)
        
        if torch.any(torch.isnan(farFieldtracJump)) or torch.any(torch.isinf(farFieldtracJump)):
            print("before_farFieldtacJump has nan or inf")
        # farFieldtracJump = rescale_outlier_vel(farFieldtracJump)
            
        # farFieldtracJump = rescale_outlier_vel_abs(farFieldtracJump, logger=self.logger)

        if torch.any(torch.isnan(farFieldtracJump)) or torch.any(torch.isinf(farFieldtracJump)):
            print("farFieldtacJump has nan or inf")
        # Total background velocity
        vbackTotal = vback + farFieldtracJump
        # self.save_farFieldtracJump[:, :, self.i] = farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        # start.record()
        Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'translateVinfwTorch {start.elapsed_time(end)/1000} sec.')

        if torch.any(torch.isnan(Xadv)) or torch.any(torch.isinf(Xadv)):
            print("Xadv input has nan or inf")
        # Xadv = rescale_outlier_trans(Xadv, Xold)


        Xadv = filterShape(Xadv, 16)
        # Xadv = gaussian_filter_shape(Xadv, sigma=3)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
        # Compute the action of relax operator on Xold + Xadv
        # start.record()
        Xnew = self.relaxWTorchNet(Xadv)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'relaxWTorchNet {start.elapsed_time(end)/1000} sec, containing standardization time.')

        modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(Xold.device) #.double()

        XnewC = Xnew.clone()
        # start.record()
        for _ in range(5):
            Xnew, flag = self.oc.redistributeArcLength(Xnew, modes)
            if flag:
                break
        Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x5 redistributeArcLength and alignCenterAngle {start.elapsed_time(end)/1000} sec.')
        

        # start.record()
        with torch.enable_grad():
            # Xnew = self.oc.correctAreaAndLength(Xnew, self.area0, self.len0)
            Xnew = self.oc.correctAreaAndLengthAugLag(Xnew, self.area0, self.len0)
        #     Xnew, mask_skip = self.oc.correctAreaAndLengthAugLag_replace(Xnew, self.area0, self.len0, self.oc)
        # num_skip = torch.sum(mask_skip)
        # if num_skip > 0:
        #     X_skipped = Xnew[:, mask_skip].reshape(2*N, -1)
        #     # _, standardizationValues = self.standardizationStep(X_skipped)
        #     # _, rotate, _, trans, _ = standardizationValues
        #     trans, rotate, _, _, _  = self.referenceValues(X_skipped)
        #     ellipses = torch.repeat_interleave(self.ellipse, num_skip, dim=-1).to(Xold.device)

        #     # Take rotation back
        #     ellipses = self.rotationOperator(ellipses, -rotate, torch.zeros(2, num_skip, device=Xold.device))
        #     # Take translation back
        #     ellipses = self.translateOp(ellipses, -trans)
        
        #     Xnew[:, mask_skip] = ellipses

        # end.record()
        # torch.cuda.synchronize()
        # print(f'correctAreaLength {start.elapsed_time(end)/1000} sec.')

        Xnew = filterShape(Xnew.to(Xold.device), 16)
        # Xnew = gaussian_filter_shape(Xnew.to(Xold.device), sigma=3)

        print(f"monitoring tenNew magnitude: {torch.max(torch.abs(tenNew))}")
        print(f"monitoring farfieldtracjump magnitude: {torch.max(torch.abs(farFieldtracJump))}")
        # np.save("debug_last_tenNew.npy", tenNew.cpu().numpy())
        return Xnew, tenNew
    


    def time_step_many_noinfo_exactVelLayer(self, Xold, tenOld, nlayers=4):
        # oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]
        Nup = ceil(sqrt(N)) * N
        vesicleUp = capsules(upsample_fft(Xold, Nup), [],[], self.kappa, 1)

        self.Galpert = self.op.stokesSLmatrix(vesicle)

        # Compute velocity induced by repulsion force
        
        repForce = torch.zeros_like(Xold)
        if self.use_repulsion:
            repForce = vesicle.repulsionForce(Xold, self.repStrength)

        # Compute bending forces + old tension forces
        # fBend = vesicle.bendingTerm(Xold)
        fTen = vesicle.tensionTerm(tenOld)
        fBend = vesicleUp.bendingTerm(vesicleUp.X) # upsampled bending term
        fBend = downsample_fft(fBend, N)
        # fTen = vesicleUp.tensionTerm(interpft(tenOld, N*6))
        # fTen = downsample_fft(fTen, N)
        
        tracJump = fBend + fTen  # total elastic force

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        
        Xstand, standardizationValues = self.standardizationStep(Xold)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'standardizationStep {start.elapsed_time(end)/1000} sec.')

        # Explicit Tension at the Current Step
        # Calculate velocity induced by vesicles on each other due to elastic force
        # use neural networks to calculate near-singular integrals
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        #     with record_function("predictNearLayers"):
        # start.record()
        # velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues, nlayers)
        velx_real, vely_real, velx_imag, vely_imag = None, None, None, None
        upsample = self.rbf_upsample
        if upsample > 0:
            Nup = N * upsample # is different from vesicleUp.N !!
            Xup = interpft_vec(Xold, Nup)  # Upsample source points
        else:
            Nup = N
            Xup = Xold

        # vesicleUp = capsules(Xup, None, None, vesicle.kappa, vesicle.viscCont)
        oc = self.oc
        dlayer = torch.linspace(0, 1/N, nlayers, dtype=torch.float64, device=Xold.device)
        _, tang = oc.diffProp_jac_tan(Xup)
        rep_nx = tang[Nup:, :, None].expand(-1,-1,nlayers-1)
        rep_ny = -tang[:Nup, :, None].expand(-1,-1,nlayers-1)
        dx =  rep_nx * dlayer[[1,2,3]] if nlayers == 4 else rep_nx * dlayer[[1,2]] # (N, nv, nlayers-1)
        dy =  rep_ny * dlayer[[1,2,3]] if nlayers == 4 else rep_ny * dlayer[[1,2]]
        tracers = torch.permute(
            torch.vstack([torch.repeat_interleave(Xup[:Nup, :, None], nlayers-1, dim=-1) + dx,
                        torch.repeat_interleave(Xup[Nup:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1)) # (2*N, nlayers-1, nv)
        tracers = torch.concat((Xup[:, None], tracers), dim=1)
        xlayers, ylayers = tracers[:Nup, ...], tracers[Nup:, ...]
        # np.save("layers.npy", { "xlayers": xlayers.cpu().numpy(), "ylayers": ylayers.cpu().numpy()})

        # end.record()
        # torch.cuda.synchronize()
        # print(f'predictNearLayers {start.elapsed_time(end)/1000} sec.')
        

        # start.record()
        # info_rbf_true, info_stokes_true = self.naiveNearZoneInfo(vesicle.X, vesicleUp.X)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'nearZoneInfo {start.elapsed_time(end)/1000} sec.')
        info_rbf, info_stokes = None, None

        # start.record()
        # if self.rbf_upsample <=1 :
        #     const = 0.672 * self.len0[0].item()
        # elif self.rbf_upsample == 2:
        #     const = 0.566 * self.len0[0].item()
        #     xlayers = interpft(xlayers.reshape(N, -1), N * 2)
        #     ylayers = interpft(ylayers.reshape(N, -1), N * 2)
        # elif self.rbf_upsample == 4:
        #     const = 0.495 * self.len0[0].item()
        #     xlayers = interpft(xlayers.reshape(N, -1), N * 4)
        #     ylayers = interpft(ylayers.reshape(N, -1), N * 4)

        if self.rbf_upsample <= 2:
            # const = 0.495 * self.len0[0].item() * 4
            # const = 1.7 / 128
            const = 0.0132

        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (nlayers * N, 2, nv), 2 for x and y
        # all_X = all_X /const * N  
        all_X = all_X / const
        matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) 
        # if self.rbf_upsample > 1:
        matrices += (torch.eye(all_X.shape[0]).unsqueeze(-1) * 1e-6).expand(-1,-1,nv) # (nlayers*N, nlayers*N, nv)
        
        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        # end.record()
        # torch.cuda.synchronize()
        # print(f'CHOLESKY {start.elapsed_time(end)/1000} sec.')
        

        # start.record()
        farFieldtracJump, info_rbf, info_stokes = self.computeStokesInteractions_noinfo_exactLayers(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                        xlayers, ylayers, standardizationValues, nlayers, first=True)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x1computeStokesInteractions first {start.elapsed_time(end)/1000} sec.')
        # pdb.set_trace()

        farFieldtracJump = filterShape(farFieldtracJump, 16)
        # farFieldtracJump = rescale_outlier_vel_abs(farFieldtracJump, logger=self.logger)
        print(f"monitoring 1st farfieldtracjump magnitude: {torch.max(torch.abs(farFieldtracJump))}")

        # print(f"true rbf_info is {info_rbf_true}, true stokes_info is {info_stokes_true}")
        # print(f"my rbf_info is {info_rbf}, my stokes_info is {info_stokes}")
        
        # start.record()
        vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'invTenMatOnVback {start.elapsed_time(end)/1000} sec.')

        # start.record()
        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'invTenMatOnSelfBend {start.elapsed_time(end)/1000} sec.')

        tenNew = -(vBackSolve + selfBendSolve)
        # pdb.set_trace()
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new
        # pdb.set_trace()

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        # start.record()
        farFieldtracJump, _, _ = self.computeStokesInteractions_noinfo_exactLayers(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                                    xlayers, ylayers, standardizationValues, nlayers, first=False)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x1computeStokesInteractions second {start.elapsed_time(end)/1000} sec.')

        farFieldtracJump = filterShape(farFieldtracJump, 16)

        # farFieldtracJump = gaussian_filter_1d_energy_preserve(torch.concat((farFieldtracJump[:N, None],farFieldtracJump[N:, None]), dim=1) , sigma=7e-2)
        # farFieldtracJump = torch.concat((farFieldtracJump[:, 0], farFieldtracJump[:, 1]), dim=0)
        
        if torch.any(torch.isnan(farFieldtracJump)) or torch.any(torch.isinf(farFieldtracJump)):
            print("before_farFieldtacJump has nan or inf")
        # farFieldtracJump = rescale_outlier_vel(farFieldtracJump)
            
        # farFieldtracJump = rescale_outlier_vel_abs(farFieldtracJump, logger=self.logger)

        if torch.any(torch.isnan(farFieldtracJump)) or torch.any(torch.isinf(farFieldtracJump)):
            print("farFieldtacJump has nan or inf")
        # Total background velocity
        vbackTotal = vback + farFieldtracJump
        # self.save_farFieldtracJump[:, :, self.i] = farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        # start.record()
        Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'translateVinfwTorch {start.elapsed_time(end)/1000} sec.')

        if torch.any(torch.isnan(Xadv)) or torch.any(torch.isinf(Xadv)):
            print("Xadv input has nan or inf")
        # Xadv = rescale_outlier_trans(Xadv, Xold)


        Xadv = filterShape(Xadv, 16)
        # Xadv = gaussian_filter_shape(Xadv, sigma=3)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
        # Compute the action of relax operator on Xold + Xadv
        # start.record()
        # Xnew = self.relaxWTorchNet(Xadv)
        Xnew = relax_solve(Xadv, self.op)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'relaxWTorchNet {start.elapsed_time(end)/1000} sec, containing standardization time.')

        modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(Xold.device) #.double()

        XnewC = Xnew.clone()
        # start.record()
        for _ in range(5):
            Xnew, flag = self.oc.redistributeArcLength(Xnew, modes)
            if flag:
                break
        Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x5 redistributeArcLength and alignCenterAngle {start.elapsed_time(end)/1000} sec.')
        

        # start.record()
        with torch.enable_grad():
            # Xnew = self.oc.correctAreaAndLength(Xnew, self.area0, self.len0)
            Xnew = self.oc.correctAreaAndLengthAugLag(Xnew, self.area0, self.len0)
        #     Xnew, mask_skip = self.oc.correctAreaAndLengthAugLag_replace(Xnew, self.area0, self.len0, self.oc)
        # num_skip = torch.sum(mask_skip)
        # if num_skip > 0:
        #     X_skipped = Xnew[:, mask_skip].reshape(2*N, -1)
        #     # _, standardizationValues = self.standardizationStep(X_skipped)
        #     # _, rotate, _, trans, _ = standardizationValues
        #     trans, rotate, _, _, _  = self.referenceValues(X_skipped)
        #     ellipses = torch.repeat_interleave(self.ellipse, num_skip, dim=-1).to(Xold.device)

        #     # Take rotation back
        #     ellipses = self.rotationOperator(ellipses, -rotate, torch.zeros(2, num_skip, device=Xold.device))
        #     # Take translation back
        #     ellipses = self.translateOp(ellipses, -trans)
        
        #     Xnew[:, mask_skip] = ellipses

        # end.record()
        # torch.cuda.synchronize()
        # print(f'correctAreaLength {start.elapsed_time(end)/1000} sec.')

        Xnew = filterShape(Xnew.to(Xold.device), 16)
        # Xnew = gaussian_filter_shape(Xnew.to(Xold.device), sigma=3)

        print(f"monitoring tenNew magnitude: {torch.max(torch.abs(tenNew))}")
        print(f"monitoring farfieldtracjump magnitude: {torch.max(torch.abs(farFieldtracJump))}")
        # np.save("debug_last_tenNew.npy", tenNew.cpu().numpy())
        return Xnew, tenNew
    

    # def time_step_many_timing(self, Xold, tenOld):
    #     # oc = self.oc
    #     torch.set_default_device(Xold.device)
    #     # background velocity on vesicles
    #     vback = self.vinf(Xold)

    #     # build vesicle class at the current step
    #     vesicle = capsules(Xold, [], [], self.kappa, 1)
    #     N = Xold.shape[0] // 2
    #     nv = Xold.shape[1]
    #     Nup = ceil(sqrt(N)) * N
    #     vesicleUp = capsules(upsample_fft(Xold, Nup), [],[], self.kappa, 1)

    #     # Compute velocity induced by repulsion force
    #     repForce = torch.zeros_like(Xold)
    #     # if self.use_repulsion:
    #     #     repForce = vesicle.repulsionForce(Xold, self.repStrength)

    #     # Compute bending forces + old tension forces
    #     fBend = vesicle.bendingTerm(Xold)
    #     fTen = vesicle.tensionTerm(tenOld)
    #     tracJump = fBend + fTen  # total elastic force

    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     Xstand, standardizationValues = self.standardizationStep(Xold)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'standardizationStep {start.elapsed_time(end)/1000} sec.')

    #     # Explicit Tension at the Current Step
    #     # Calculate velocity induced by vesicles on each other due to elastic force
    #     # use neural networks to calculate near-singular integrals
    #     # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    #     #     with record_function("predictNearLayers"):
    #     start.record()
    #     velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'predictNearLayers {start.elapsed_time(end)/1000} sec.')
    #     # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    #     start.record()
    #     info_rbf, info_stokes = self.naiveNearZoneInfo(vesicle.X, vesicleUp.X)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'nearZoneInfo {start.elapsed_time(end)/1000} sec.')

    #     start.record()
    #     const = 0.672
    #     all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
    #     all_X = all_X /const * N   
    #     matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) # (3*N, 3*N, nv)
    #     L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'CHOLESKY {start.elapsed_time(end)/1000} sec.')
        

    #     start.record()
    #     farFieldtracJump = self.computeStokesInteractions_timing(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
    #                                     xlayers, ylayers, standardizationValues)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'x1computeStokesInteractions first {start.elapsed_time(end)/1000} sec.')

    #     farFieldtracJump = filterShape(farFieldtracJump, 4)
        
    #     start.record()
    #     vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'invTenMatOnVback {start.elapsed_time(end)/1000} sec.')

    #     start.record()
    #     selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'invTenMatOnSelfBend {start.elapsed_time(end)/1000} sec.')

    #     tenNew = -(vBackSolve + selfBendSolve)
    #     # tenNew = filterTension(tenNew, 4*N, 16)

    #     # update the elastic force with the new tension
    #     fTen_new = vesicle.tensionTerm(tenNew)
    #     tracJump = fBend + fTen_new

    #     # Calculate far-field again and correct near field before advection
    #     # use neural networks to calculate near-singular integrals
    #     start.record()
    #     farFieldtracJump= self.computeStokesInteractions_timing(vesicle, vesicleUp, info_rbf, info_stokes, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
    #                                                 xlayers, ylayers, standardizationValues)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'x1computeStokesInteractions second {start.elapsed_time(end)/1000} sec.')

        
    #     farFieldtracJump = filterShape(farFieldtracJump, 4)

    #     # Total background velocity
    #     vbackTotal = vback + farFieldtracJump

    #     # Compute the action of dt*(1-M) on Xold
    #     start.record()
    #     Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'translateVinfwTorch {start.elapsed_time(end)/1000} sec.')

    #     Xadv = filterShape(Xadv, 8)
    #     # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
    #     # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
    #     # Compute the action of relax operator on Xold + Xadv
    #     start.record()
    #     Xnew = self.relaxWTorchNet(Xadv)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'relaxWTorchNet {start.elapsed_time(end)/1000} sec, containing standardization time.')

    #     modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(Xold.device) #.double()

    #     XnewC = Xnew.clone()
    #     start.record()
    #     for _ in range(5):
    #         Xnew, flag = self.oc.redistributeArcLength(Xnew, modes)
    #         if flag:
    #             break
    #     Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'x5 redistributeArcLength and alignCenterAngle {start.elapsed_time(end)/1000} sec.')
        

    #     start.record()
    #     Xnew = self.oc.correctAreaAndLength(Xnew, self.area0, self.len0)
    #     end.record()
    #     torch.cuda.synchronize()
    #     print(f'correctAreaLength {start.elapsed_time(end)/1000} sec.')

    #     Xnew = filterShape(Xnew.to(Xold.device), 8)

    #     # MLARM_manyfree_py.info, MLARM_manyfree_py.dis2, MLARM_manyfree_py.diffx, MLARM_manyfree_py.diffy, MLARM_manyfree_py.full_mask = None, None, None, None, None

    #     return Xnew, tenNew
    



    # def time_step_single(self, Xold):
        
    #     # % take a time step with neural networks
    #     oc = self.oc
    #     # background velocity on vesicles
    #     vback = torch.from_numpy(self.vinf(Xold))
    #     N = Xold.shape[0]//2

    #     # Compute the action of dt*(1-M) on Xold
    #     # tStart = time.time()
    #     Xadv = self.translateVinfwTorch(Xold, vback)
    #     # tEnd = time.time()
    #     # print(f'Solving ADV takes {tEnd - tStart} sec.')

    #     # Correct area and length
    #     # tStart = time.time()
    #     # Xadv = upsThenFilterShape(Xadv, 4*N, 16)
    #     # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
    #     # Xadv = oc.alignCenterAngle(Xadv, XadvC)
    #     # tEnd = time.time()
    #     # print(f'Solving correction takes {tEnd - tStart} sec.')

    #     # Compute the action of relax operator on Xold + Xadv
    #     # tStart = time.time()
    #     Xnew = self.relaxWTorchNet(Xadv)
    #     # tEnd = time.time()
    #     # print(f'Solving RELAX takes {tEnd - tStart} sec.')

    #     # Correct area and length
    #     # tStart = time.time()
    #     # Xnew = upsThenFilterShape(Xnew, 4*N, 16)
    #     for _ in range(5):
    #         Xnew, flag = oc.redistributeArcLength(Xnew)
    #         if flag:
    #             break
    #     XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
    #     Xnew = oc.alignCenterAngle(Xnew, XnewC)
        
    #     # tEnd = time.time()
    #     # print(f'Solving correction takes {tEnd - tStart} sec.')

    #     return Xnew

    # def time_step_many_order(self, Xold, tenOld):
    #     # oc = self.oc
    #     torch.set_default_device(Xold.device)
    #     # background velocity on vesicles
    #     vback = self.vinf(Xold)

    #     # build vesicle class at the current step
    #     vesicle = capsules(Xold, [], [], self.kappa, 1)

    #     # Compute velocity induced by repulsion force
    #     repForce = torch.zeros_like(Xold)
    #     # if self.use_repulsion:
    #     #     repForce = vesicle.repulsionForce(Xold, self.repStrength)

    #     # Compute bending forces + old tension forces
    #     fBend = vesicle.bendingTerm(Xold)
    #     fTen = vesicle.tensionTerm(tenOld)
    #     tracJump = fBend + fTen  # total elastic force

    #     Xstand, standardizationValues = self.standardizationStep(Xold)
    #     # Explicit Tension at the Current Step
    #     # Calculate velocity induced by vesicles on each other due to elastic force
    #     # use neural networks to calculate near-singular integrals
    #     velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues)
        
    #     # info = self.nearZoneInfo(vesicle)
    #     info = self.naiveNearZoneInfo(vesicle)

    #     farFieldtracJump = self.computeStokesInteractions(vesicle, info, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
    #                                     xlayers, ylayers, standardizationValues)

    #     farFieldtracJump = filterShape(farFieldtracJump, 16)
    #     # Solve for tension
    #     vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
    #     selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
    #     tenNew = -(vBackSolve + selfBendSolve)
    #     # tenNew = filterTension(tenNew, 4*N, 16)

    #     # update the elastic force with the new tension
    #     fTen_new = vesicle.tensionTerm(tenNew)
    #     tracJump = fBend + fTen_new

    #     # Calculate far-field again and correct near field before advection
    #     # use neural networks to calculate near-singular integrals
    #     farFieldtracJump = self.computeStokesInteractions(vesicle, info, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, standardizationValues)
    #     farFieldtracJump = filterShape(farFieldtracJump, 16)

    #     # Total background velocity
    #     vbackTotal = vback + farFieldtracJump

    #     # Compute the action of dt*(1-M) on Xold
    #     Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)
    #     Xadv = filterShape(Xadv, 16)
    #     # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
    #     # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
    #     # Compute the action of relax operator on Xold + Xadv
    #     Xnew = self.relaxWTorchNet(Xadv)
    #     # XnewC = Xnew.clone()
    #     for _ in range(5):
    #         Xnew, flag = self.oc.redistributeArcLength(Xnew)
    #         if flag:
    #             break

    #     tStart = time.time()
    #     XnewC = self.oc.correctAreaAndLength(Xnew, self.area0, self.len0)
    #     tEnd = time.time()
    #     print(f'correctAreaLength {tEnd - tStart} sec.')

    #     Xnew = self.oc.alignCenterAngle(Xnew, XnewC.to(Xold.device))
    #     Xnew = filterShape(Xnew.to(Xold.device), 16)

    #     return Xnew, tenNew
    
    
    # @torch.compile(backend='cudagraphs')
    def predictNearLayers(self, Xstand, standardizationValues: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                          nlayers: int = 3):
        # print('Near network predicting')
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]

        oc = self.oc

        # maxLayerDist = np.sqrt(1 / N) 
        maxLayerDist = (self.len0[0].item() / N) # length = 1, h = 1/N;
        # nlayers = 5 # three layers
        # dlayer = torch.linspace(-maxLayerDist, maxLayerDist, nlayers, dtype=torch.float32, device=Xstand.device)

        # Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)
        
        # Create the layers around a vesicle on which velocity calculated
        tracersX_ = torch.zeros((2 * N, nlayers, nv), dtype=Xstand.dtype, device=Xstand.device)
        if nlayers == 5:
            dlayer = torch.linspace(-maxLayerDist, maxLayerDist, nlayers, dtype=Xstand.dtype, device=Xstand.device)
            tracersX_[:, 2] = Xstand
            _, tang = oc.diffProp_jac_tan(Xstand)
            rep_nx = tang[N:, :, None].expand(-1,-1,nlayers-1)
            rep_ny = -tang[:N, :, None].expand(-1,-1,nlayers-1)
            dx =  rep_nx * dlayer[[0,1,3,4]] # (N, nv, nlayers-1)
            dy =  rep_ny * dlayer[[0,1,3,4]]
            tracersX_[:, [0,1,3,4]] = torch.permute(
                torch.vstack([torch.repeat_interleave(Xstand[:N, :, None], nlayers-1, dim=-1) + dx,
                            torch.repeat_interleave(Xstand[N:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1))
        else:
            dlayer = torch.linspace(0, maxLayerDist, nlayers, dtype=Xstand.dtype, device=Xstand.device)
            tracersX_[:, 0] = Xstand
            _, tang, _ = oc.diffProp(Xstand)
            rep_nx = torch.repeat_interleave(tang[N:, :, None], nlayers-1, dim=-1) 
            rep_ny = torch.repeat_interleave(-tang[:N, :, None], nlayers-1, dim=-1)
            dx =  rep_nx * dlayer[1:] # (N, nv, nlayers-1)
            dy =  rep_ny * dlayer[1:]
            tracersX_[:, 1:] = torch.permute(
                torch.vstack([torch.repeat_interleave(Xstand[:N, :, None], nlayers-1, dim=-1) + dx,
                            torch.repeat_interleave(Xstand[N:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1))
        
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        
        # start.record()
        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        # net_pred = self.nearNetwork.forward_half(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(net_pred)

        # print(f"------ rel err for half nearNetwork is {torch.norm(net_pred - net_pred_)/torch.norm(net_pred)}")

        if nlayers == 5:

            inner_input_net = self.innerNearNetwork.preProcess(Xstand)
            inner_net_pred = self.innerNearNetwork.forward(inner_input_net)
            inner_velx_real, inner_vely_real, inner_velx_imag, inner_vely_imag = self.innerNearNetwork.postProcess(inner_net_pred)

            # end.record()
            # torch.cuda.synchronize()
            # print(f'--------- inside predictNearLayers Networks {start.elapsed_time(end)/1000} sec.')

            velx_real = torch.concat((inner_velx_real, velx_real), dim=-1)
            vely_real = torch.concat((inner_vely_real, vely_real), dim=-1)
            velx_imag = torch.concat((inner_velx_imag, velx_imag), dim=-1)
            vely_imag = torch.concat((inner_vely_imag, vely_imag), dim=-1)
        

        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues
        Xl_ = self.destandardize(tracersX_.reshape(N*2, -1),  
            (scaling[None,:].expand(nlayers,-1).reshape(-1), 
             rotate[None,:].expand(nlayers,-1).reshape(-1), 
             rotCenter.tile((1,nlayers)), trans.tile((1,nlayers)), sortIdx.tile((nlayers,1))))
        
        xlayers_ = torch.zeros((N, nlayers, nv), dtype=Xstand.dtype)
        ylayers_ = torch.zeros((N, nlayers, nv), dtype=Xstand.dtype)
        xlayers_ = Xl_[:N, torch.arange(nlayers * nv, device=Xstand.device).reshape(nlayers, nv)]
        ylayers_ = Xl_[N:, torch.arange(nlayers * nv, device=Xstand.device).reshape(nlayers, nv)]

        # if not torch.allclose(xlayers, xlayers_):
        #     raise "batch err"
        # np.save("linshi_xlayers.npy", xlayers_.numpy())
        # np.save("linshi_ylayers.npy", ylayers_.numpy())
        # if not torch.allclose(ylayers, ylayers_):
        #     raise "batch err"

        return velx_real, vely_real, velx_imag, vely_imag, xlayers_, ylayers_


    # @torch.jit.script_method
    def buildVelocityInNear(self, tracJump, velx_real, vely_real, velx_imag, vely_imag,
                             standardizationValues: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], nlayers):
        
        nv = tracJump.shape[1]
        N = tracJump.shape[0]//2
        # nlayers = 5
        _, rotate, _, _, sortIdx = standardizationValues

        fstand = self.standardize(tracJump, torch.zeros((2,nv), dtype=tracJump.dtype, device=tracJump.device), rotate, 
                                  torch.zeros((2,nv), dtype=tracJump.dtype, device=tracJump.device), torch.tensor([1.0], device=tracJump.device), sortIdx)
        z = fstand[:N] + 1.0j * fstand[N:]
        zh = torch.fft.fft(z, dim=0)
        fstandRe = torch.real(zh)
        fstandIm = torch.imag(zh)

        velx_stand_ = torch.einsum('vnml, mv -> nvl', velx_real.double(), fstandRe) + torch.einsum('vnml, mv -> nvl', velx_imag.double(), fstandIm)
        vely_stand_ = torch.einsum('vnml, mv -> nvl', vely_real.double(), fstandRe) + torch.einsum('vnml, mv -> nvl', vely_imag.double(), fstandIm)
        
        vx_ = torch.zeros((nv, nlayers, N), device=tracJump.device, dtype=tracJump.dtype)
        vy_ = torch.zeros((nv, nlayers, N), device=tracJump.device, dtype=tracJump.dtype)
        # Destandardize
        vx_[torch.arange(nv), :, sortIdx.T] = velx_stand_
        vy_[torch.arange(nv), :, sortIdx.T] = vely_stand_

        VelBefRot_ = torch.concat((vx_, vy_), dim=-1) # (nv, nlayers, 2N)
        VelRot_ = self.rotationOperator(VelBefRot_.reshape(-1, 2*N).T, 
                        torch.repeat_interleave(-rotate, nlayers, dim=0), torch.zeros(nv * nlayers))
        VelRot_ = VelRot_.T.reshape(nv, nlayers, 2*N).permute(2,1,0)
        velx_ = VelRot_[:N] # (N, nlayers, nv)
        vely_ = VelRot_[N:]

        return velx_, vely_
    
    
    def naiveNearZoneInfo(self, vesicleX, vesicleUpX):
        '''
        Naive way of doing range search by computing distances and creating masks.
        return a boolean nbrs_mask where (i,j)=True means i, j are close and are from different vesicles
        '''
        N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
        Nup = vesicleUpX.shape[0]//2
        # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
        # max_layer_dist = vesicle.length.item() / vesicle.N
        max_layer_dist = 1./N

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        all_points =  torch.concat((vesicleX[:N, :].T.reshape(-1,1), vesicleX[N:, :].T.reshape(-1,1)), dim=1)
        all_points_up =  torch.concat((vesicleUpX[:Nup, :].T.reshape(-1,1), vesicleUpX[Nup:, :].T.reshape(-1,1)), dim=1)

        # if nv < 1600:
            # sq_distances  = torch.sum((all_points.unsqueeze(1) - all_points_up.unsqueeze(0))**2, dim=-1)  
            # sq_distances = torch.norm(all_points.unsqueeze(1) - all_points_up.unsqueeze(0), dim=-1)
        sq_distances = torch.cdist(all_points.unsqueeze(0), all_points_up.unsqueeze(0)).squeeze()
        dist_mask = sq_distances < max_layer_dist
            # sq_distances_  = torch.sum((all_points.half().unsqueeze(1) - all_points_up.half().unsqueeze(0))**2, dim=-1)  # Shape: (N, Nup)     
            # dist_mask_ = sq_distances_ <= max_layer_dist**2

            # if not torch.allclose(dist_mask, dist_mask_):
            #     raise "dist_mask err"

        # else:
        #     len0 = all_points.shape[0]
        #     sq_distances  = torch.norm(all_points[:len0//2].unsqueeze(1) - all_points_up.unsqueeze(0), dim=-1)  
        #     dist_mask1 = sq_distances <= max_layer_dist**2
        #     sq_distances  = torch.norm(all_points[len0//2:].unsqueeze(1) - all_points_up.unsqueeze(0), dim=-1)  
        #     dist_mask2 = sq_distances <= max_layer_dist**2
        #     dist_mask = torch.cat((dist_mask1, dist_mask2), dim=0)

        # if not torch.allclose(dist_mask, dist_mask_):
        #     raise "dist_mask err"   

        # id_mask = torch.ones((N*nv, Nup*nv), dtype=torch.bool, device=dist_mask.device)  # Initialize all True
        
        # indices = torch.arange(0, N*nv).reshape(nv, N)
        # indices_up = torch.arange(0, Nup*nv).reshape(nv, Nup)
        # # Use advanced indexing to set blocks to False
        # row_indices = indices.unsqueeze(2)  # Shape: (num_cells, points_per_cell, 1)
        # col_indices = indices_up.unsqueeze(1)  # Shape: (num_cells, 1, points_per_cell)
        # id_mask[row_indices, col_indices] = False

        # nbrs_mask = torch.logical_and(dist_mask, id_mask)

        indices = torch.arange(nv, device=dist_mask.device).unsqueeze(-1).expand(-1, N*Nup).reshape(-1)
        N_indices = torch.arange(N, device=dist_mask.device).unsqueeze(-1).expand(-1, Nup).reshape(-1)
        N_indices = N_indices.unsqueeze(0).expand(nv,-1).reshape(-1)
        Nup_indices = torch.arange(Nup, device=dist_mask.device).unsqueeze(0).expand(nv*N,-1).reshape(-1)

        nbrs_mask = dist_mask.reshape(nv, N, nv, Nup)
        nbrs_mask.index_put_((indices, N_indices, indices, Nup_indices), 
                             torch.tensor(0.0, dtype=torch.bool, device=dist_mask.device))


        # if not torch.allclose(nbrs_mask, dist_mask.reshape(nv*N, nv* Nup)):
        #     raise "nbrs_mask err"

        # end.record()
        # torch.cuda.synchronize()
        # print(f'----------- inside nearZoneinfo, creating mask  {start.elapsed_time(end)/1000} sec.')

        
        rows_with_true = torch.max(nbrs_mask.reshape(nv*N, nv, Nup), dim=-1)[0] # (N*nv, nv)
        id1, id2 = torch.where(rows_with_true)
        # id1, id2 = rows_with_true.to_sparse().indices() # for rbf solves
        ids1, ids2 = id1 % N, id1 // N
        ids0 = id2 - 1*(ids2 <= id2)
        # if torch.any(ids2 == id2):
        #     raise "unexpcted"

        return (id1, id2),  \
            (ids0, ids1, ids2)  # for exactStokes



    # def naiveNearZoneInfo_noidmask(self, vesicleX, vesicleUpX):
    #     '''
    #     Naive way of doing range search by computing distances and creating masks.
    #     return a boolean nbrs_mask where (i,j)=True means i, j are close and are from different vesicles
    #     '''
    #     N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
    #     Nup = vesicleUpX.shape[0]//2
    #     # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
    #     # max_layer_dist = vesicle.length.item() / vesicle.N
    #     max_layer_dist = 1./N

    #     all_points =  torch.concat((vesicleX[:N, :].T.reshape(-1,1), vesicleX[N:, :].T.reshape(-1,1)), dim=1)
    #     all_points_up =  torch.concat((vesicleUpX[:Nup, :].T.reshape(-1,1), vesicleUpX[Nup:, :].T.reshape(-1,1)), dim=1)

    #     # if nv < 1600:
    #         # sq_distances  = torch.sum((all_points.unsqueeze(1) - all_points_up.unsqueeze(0))**2, dim=-1)  
    #         # sq_distances = torch.norm(all_points.unsqueeze(1) - all_points_up.unsqueeze(0), dim=-1)
    #     sq_distances = torch.cdist(all_points.unsqueeze(0), all_points_up.unsqueeze(0)).squeeze()
    #     dist_mask = sq_distances < max_layer_dist
    #         # sq_distances_  = torch.sum((all_points.half().unsqueeze(1) - all_points_up.half().unsqueeze(0))**2, dim=-1)  # Shape: (N, Nup)     
    #         # dist_mask_ = sq_distances_ <= max_layer_dist**2

    #         # if not torch.allclose(dist_mask, dist_mask_):
    #         #     raise "dist_mask err"
    
    #     # else:
    #     #     len0 = all_points.shape[0]
    #     #     sq_distances  = torch.norm(all_points[:len0//2].unsqueeze(1) - all_points_up.unsqueeze(0), dim=-1)  
    #     #     dist_mask1 = sq_distances <= max_layer_dist**2
    #     #     sq_distances  = torch.norm(all_points[len0//2:].unsqueeze(1) - all_points_up.unsqueeze(0), dim=-1)  
    #     #     dist_mask2 = sq_distances <= max_layer_dist**2
    #     #     dist_mask = torch.cat((dist_mask1, dist_mask2), dim=0)

    #     # if not torch.allclose(dist_mask, dist_mask_):
    #     #     raise "dist_mask err"   

    #     # id_mask = torch.ones((N*nv, Nup*nv), dtype=torch.bool, device=dist_mask.device)  # Initialize all True
        
    #     # indices = torch.arange(0, N*nv).reshape(nv, N)
    #     # indices_up = torch.arange(0, Nup*nv).reshape(nv, Nup)
    #     # # Use advanced indexing to set blocks to False
    #     # row_indices = indices.unsqueeze(2)  # Shape: (num_cells, points_per_cell, 1)
    #     # col_indices = indices_up.unsqueeze(1)  # Shape: (num_cells, 1, points_per_cell)
    #     # id_mask[row_indices, col_indices] = False

    #     # nbrs_mask = torch.logical_and(dist_mask, id_mask)

    #     id_mask_ = ~torch.eye(nv, dtype=torch.bool)
    #     id_indices_ = torch.arange(nv)[None,].expand(nv,-1)[id_mask_].view(nv, nv-1)
    #     dist_mask_ = dist_mask.reshape(nv, N, nv, Nup)
    #     nbrs_mask_ = dist_mask_.gather(2, id_indices_[:, None, :, None].expand(-1, N, -1, Nup))  # (nv, N, nv-1, Nup)

    #     # if not torch.allclose(nbrs_mask.reshape(nv, N, nv, Nup), nbrs_mask_):
    #     #     raise "nbrs_mask err"

    #     rows_with_true = torch.max(nbrs_mask_.reshape(nv*N, nv, Nup), dim=-1)[0] # (N*nv, nv)
    #     id1, id2 = torch.where(rows_with_true)
    #     # id1, id2 = rows_with_true.to_sparse().indices() # for rbf solves
    #     ids1, ids2 = id1 % N, id1 // N
    #     ids0 = id2 - 1*(ids2 <= id2)
    #     # if torch.any(ids2 == id2):
    #     #     raise "unexpcted"

    #     return (id1, id2),  \
    #         (ids0, ids1, ids2)  # for exactStokes



    # def nearZoneInfo(self, vesicle, option='exact'):
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     xvesicle = vesicle.X[:N, :]
    #     yvesicle = vesicle.X[N:, :]
    #     # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
    #     max_layer_dist = vesicle.length.item() / vesicle.N

    #     i_call_near = [False]*nv
    #     # which of ves k's points are in others' near zone
    #     ids_in_store = defaultdict(list)
    #     # and their coords
    #     query_X = defaultdict(list)
    #     # k is in the near zone of j: near_ves_ids[k].add(j)
    #     near_ves_ids = defaultdict(set)

        
    #     if option == "kdtree":
    #         all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).cpu().numpy()
    #         tree = KDTree(all_points)
    #         all_nbrs = tree.query_ball_point(all_points, max_layer_dist, return_sorted=True)

    #         for j in range(nv):
    #             j_nbrs = all_nbrs[N*j : N*(j+1)]
    #             j_nbrs_flat = np.array(list(set(sum(j_nbrs, [])))) # flatten a list of lists and remove duplicates
    #             others = j_nbrs_flat[np.where((j_nbrs_flat >= N*(j+1)) | (j_nbrs_flat < N*j))]
    #             for k in range(nv):
    #                 if k == j:
    #                     continue
    #                 others_from_k = others[np.where((others>= N*k) & (others < N*(k+1)))]
    #                 if len(others_from_k) > 0:
    #                     # which of ves k's points are in others' near zone
    #                     ids_in_store[k] += list(others_from_k % N)
    #                     # and their coords
    #                     query_X[k].append(all_points[others_from_k])
    #                     # k is in the near zone of j
    #                     near_ves_ids[k].add(j)
    #                     i_call_near[k] = True


    #     elif option == 'faiss':
    #         # (npoints, 2)
    #         # all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).cpu().numpy().astype('float32')
    #         all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).float()
    #         res = faiss.StandardGpuResources()
    #         flat_config = faiss.GpuIndexFlatConfig()
    #         flat_config.device = 0
    #         index = faiss.GpuIndexFlatL2(res, 2, flat_config)

    #         # index = faiss.IndexFlatL2(2) # 2D
    #         index.add(all_points)
    #         lims, _, I = index.range_search(all_points, max_layer_dist**2)
    #         for j in range(nv):
    #             j_nbrs = I[lims[N*j] : lims[N*(j+1)]]
    #             others = j_nbrs[torch.where((j_nbrs >= N*(j+1)) | (j_nbrs < N*j))]
    #             for k in range(nv):
    #                 if k == j:
    #                     continue
    #                 others_from_k = others[torch.where((others>= N*k) & (others < N*(k+1)))]
    #                 if len(others_from_k) > 0:
    #                     # which of ves k's points are in others' near zone
    #                     ids_in_store[k] += list(others_from_k % N)
    #                     # and their coords
    #                     query_X[k].append(all_points[others_from_k])
    #                     # k is in the near zone of j
    #                     near_ves_ids[k].add(j)
    #                     i_call_near[k] = True

    #     return (i_call_near, query_X, ids_in_store, near_ves_ids)
    
    # def computeStokesInteractions_timing_noinfo_(self, vesicle, vesicleUp, info_rbf, info_stokes, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
    #                               xlayers, ylayers, standardizationValues, first: bool,  upsample=True):
    #     # print('Near-singular interaction through interpolation and network')

    #     velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
    #     # rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
    #     # Compute near/far hydro interactions without any correction
    #     # First calculate the far-field

    #     totalForce = trac_jump + repForce
    #     # if upsample:
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     Nup = ceil(sqrt(N)) * N
    #     # totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)
    #     totalForceUp = upsample_fft(totalForce, Nup)
        

    #     # start = torch.cuda.Event(enable_timing=True)
    #     # end = torch.cuda.Event(enable_timing=True)
        

    #     # if first_round:
    #     #     fn = allExactStokesSLTarget_broadcast
    #     #     if nv > 504:
    #     #         n_parts = 4
    #     #         far_field, info = [], []
    #     #         for i in range(n_parts):
    #     #             if i == 0:
    #     #                 out = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//n_parts], offset=0, return_info=True)
    #     #             else:
    #     #                 out = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, i*nv//n_parts:(i+1)*nv//n_parts], offset=i*nv//n_parts, return_info=True)
    #     #             far_field.append(out[0])
    #     #             info.append(out[1])
                
    #     #         del out
    #     #         far_field = torch.concat(far_field, dim=1)
    #     #         info = torch.concat(info, dim=0)

    #     #         # far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
    #     #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
    #     #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
    #     #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
    #     #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
    #     #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
    #     #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
    #     #     else:
    #     #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
    #     #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
    #     #         far_field, info = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, return_info=True)

    #     # else:
    #     #     fn = allExactStokesSLTarget_broadcast
    #     #     if nv > 504:
    #     #         far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], return_info=False)[0], 
    #     #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4, return_info=False)[0],
    #     #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2, return_info=False)[0],
    #     #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4, return_info=False)[0]), dim=1)
    #     #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
    #     #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
    #     #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
    #     #     else:
    #     #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
    #     #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
    #     #         far_field, _ = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, return_info=False)

    #     # start.record()
    #     # fn = allExactStokesSLTarget_broadcast
    #     # if nv > 504:
    #     #     far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
    #     #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
    #     #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
    #     #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
    #     #     # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
    #     #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
    #     #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
    #     # else:
    #     #     # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
    #     #     # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
    #     #     far_field = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)


    #     # end.record()
    #     # torch.cuda.synchronize()
    #     # print(f'stokes old EXACT {start.elapsed_time(end)/1000} sec.')
            

    #     # start.record()
    #     if first:
    #         fn = allExactStokesSLTarget_compare1
    #         if nv > 2:
    #             # far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3])
    #             # far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3)
    #             # far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)
    #             # far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3), dim=-1)
    #             # info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0]), dim=0),
    #             #                  torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1]), dim=0),
    #             #                  torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2]), dim=0)
    #             # )   
                
    #             far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4])
    #             far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4)
    #             far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4],  offset=nv//2)
    #             far_field_1_4, info_stokes_4 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)
    #             far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1)
    #             info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0], info_stokes_4[0]), dim=0),
    #                              torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1], info_stokes_4[1]), dim=0),
    #                              torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2], info_stokes_4[2]), dim=0)
    #             )
    #             # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
    #             #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
    #             #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
    #         else:
    #             # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
    #             # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
    #             far_field_1, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)
    #         id1 = info_stokes[2] * N + info_stokes[1]
    #         id2 = info_stokes[0] + 1*(info_stokes[0] >= info_stokes[2])
    #         info_rbf = (id1, id2)
        
    #     else:
    #         fn = allExactStokesSLTarget_compare2
    #         if nv > 504:
    #             # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
    #             #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
    #             #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
    #             far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
    #                                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
    #                                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
    #                                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=1)
    #             # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
    #             #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
    #             #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
    #         else:
    #             # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
    #             # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
    #             far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

    #     # end.record()
    #     # torch.cuda.synchronize()
    #     # # print(f'computeStokesInteractions EXACT {start.elapsed_time(end)/1000} sec, {torch.norm(far_field - far_field_1)/torch.norm(far_field)}')
    #     # print(f'computeStokesInteractions noinfo EXACT {start.elapsed_time(end)/1000} sec')

    #     # start.record()
    #     # fn = allExactStokesSLTarget_compare2
    #     # if nv > 504:
    #     #     # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
    #     #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
    #     #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
    #     #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=-1)
    #     #     # with torch.autocast(device_type='cuda', dtype=torch.float16):
    #     #     far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
    #     #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
    #     #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
            
    #     #     # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
    #     #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
    #     #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
    #     # else:
    #     #     # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
    #     #     # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
    #     #     far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

    #     # end.record()
    #     # torch.cuda.synchronize()
    #     # print(f'Stokes EXACT {start.elapsed_time(end)/1000} sec.')

    #     # print(f"--------- rel err of amp in stokes is {torch.norm(far_field - far_field_1)/torch.norm(far_field)}")

    #     # start.record()
    #     self.nearFieldCorrectionUP_SOLVE_timing(vesicle, info_rbf, L, far_field_1, velx, vely, xlayers, ylayers)
    #     # end.record()
    #     # torch.cuda.synchronize()
    #     # print(f'x1 nearFieldCorrection SOLVE {start.elapsed_time(end)/1000} sec.')


    #     return far_field_1, info_rbf, info_stokes
    
    
    def computeStokesInteractions_timing_noinfo(self, vesicle, vesicleUp, info_rbf, info_stokes, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
                                  xlayers, ylayers, standardizationValues, nlayers, first: bool, upsample=True):
        # print('Near-singular interaction through interpolation and network')

        velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues, nlayers)
        # rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
        # Compute near/far hydro interactions without any correction
        # First calculate the far-field

        totalForce = trac_jump + repForce
        # if upsample:
        N = vesicle.N
        nv = vesicle.nv
        Nup = ceil(sqrt(N)) * N
        # totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)
        totalForceUp = upsample_fft(totalForce, Nup)
        length = self.len0[0].item()
        

        start_cuda = torch.cuda.Event(enable_timing=True)
        end_cuda = torch.cuda.Event(enable_timing=True)
        

        # if first_round:
        #     fn = allExactStokesSLTarget_broadcast
        #     if nv > 504:
        #         n_parts = 4
        #         far_field, info = [], []
        #         for i in range(n_parts):
        #             if i == 0:
        #                 out = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//n_parts], offset=0, return_info=True)
        #             else:
        #                 out = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, i*nv//n_parts:(i+1)*nv//n_parts], offset=i*nv//n_parts, return_info=True)
        #             far_field.append(out[0])
        #             info.append(out[1])
                
        #         del out
        #         far_field = torch.concat(far_field, dim=1)
        #         info = torch.concat(info, dim=0)

        #         # far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
        #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
        #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
        #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field, info = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, return_info=True)

        # else:
        #     fn = allExactStokesSLTarget_broadcast
        #     if nv > 504:
        #         far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], return_info=False)[0], 
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4, return_info=False)[0],
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2, return_info=False)[0],
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4, return_info=False)[0]), dim=1)
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field, _ = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, return_info=False)

        # start.record()
        # fn = allExactStokesSLTarget_broadcast
        # if nv > 504:
        #     far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
        #     # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        # else:
        #     # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #     # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #     far_field = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)


        # end.record()
        # torch.cuda.synchronize()
        # print(f'stokes old EXACT {start.elapsed_time(end)/1000} sec.')
            

        start_cuda.record()
        if first:
            fn = allExactStokesSLTarget_compare1
            if nv > 1048:
                far_fields = []
                info_stokes_parts = [[], [], []]
                parts_ids = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
                num_parts = len(parts_ids) - 1

                for i in range(num_parts):
                    # start = i * nv // num_parts
                    # end = (i + 1) * nv // num_parts if i < num_parts-1 else None  # Ensure last slice goes to the end
                    start = parts_ids[i]
                    end = parts_ids[i+1]
                    offset = start if i > 0 else 0  # Offset is None for the first call

                    far_field, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X[:, start:end], length, offset=offset)

                    far_fields.append(far_field)
                    for j in range(3):
                        info_stokes_parts[j].append(info_stokes[j])

                far_field_1 = torch.concat(far_fields, dim=-1)
                info_stokes = tuple(torch.cat(parts, dim=0) for parts in info_stokes_parts)

                
            elif nv > 504:
                # far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3])
                # far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3)
                # far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)
                # far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3), dim=-1)
                # info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0]), dim=0),
                #                  torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1]), dim=0),
                #                  torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2]), dim=0)
                # )   
                
                far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], length)
                far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], length, offset=nv//4)
                far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], length, offset=nv//2)
                far_field_1_4, info_stokes_4 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], length, offset=3*nv//4)
                far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1)
                info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0], info_stokes_4[0]), dim=0),
                                 torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1], info_stokes_4[1]), dim=0),
                                 torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2], info_stokes_4[2]), dim=0)
                )
                # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
            else:
                # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
                # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
                far_field_1, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, length)
            id1 = info_stokes[2] * N + info_stokes[1]
            id2 = info_stokes[0] + 1*(info_stokes[0] >= info_stokes[2])
            info_rbf = (id1, id2)
        
        else:
            fn = allExactStokesSLTarget_compare2
            if nv > 1048:
                far_fields = []
                # parts_ids = [0, 256, 448, 640, 832, 1024, 1216, 1408, 1600, 1792, 2000]
                parts_ids = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
                num_parts = len(parts_ids) - 1
                
                    
                for i in range(num_parts):
                    # start = i * nv // num_parts
                    # end = (i + 1) * nv // num_parts if i < num_parts-1 else None  # Ensure last slice goes to the end
                    start = parts_ids[i]
                    end = parts_ids[i+1]
                    offset = start if i > 0 else 0   # Offset is None for the first call

                    mask = (start <= info_stokes[2]) & (info_stokes[2] < end) if i < num_parts-1 else (start <= info_stokes[2])
                    
                    far_field = fn(
                        vesicleUp.X, 
                        vesicleUp.sa, 
                        totalForceUp, 
                        vesicle.X[:, start:end], 
                        info_stokes[0][mask], 
                        info_stokes[1][mask], 
                        info_stokes[2][mask] - start, 
                        offset=offset
                    )
                    
                    far_fields.append(far_field)

                far_field_1 = torch.concat(far_fields, dim=1)


            elif nv > 504:
                # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
                #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
                #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
                far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=1)
                # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
            else:
                # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
                # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
                far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        end_cuda.record()
        torch.cuda.synchronize()
        # print(f'computeStokesInteractions EXACT {start.elapsed_time(end)/1000} sec, {torch.norm(far_field - far_field_1)/torch.norm(far_field)}')
        print(f'computeStokesInteractions noinfo EXACT {start_cuda.elapsed_time(end_cuda)/1000} sec')

        # start.record()
        # fn = allExactStokesSLTarget_compare2
        # if nv > 504:
        #     # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=-1)
        #     # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
            
        #     # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        # else:
        #     # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #     # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #     far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        # end.record()
        # torch.cuda.synchronize()
        # print(f'Stokes EXACT {start.elapsed_time(end)/1000} sec.')

        # print(f"--------- rel err of amp in stokes is {torch.norm(far_field - far_field_1)/torch.norm(far_field)}")

        start_cuda.record()
        if self.rbf_upsample == 2:
            velx = interpft(velx.reshape(N, -1), N * 2)
            vely = interpft(vely.reshape(N, -1), N * 2)
        elif self.rbf_upsample == 4:
            velx = interpft(velx.reshape(N, -1), N * 4)
            vely = interpft(vely.reshape(N, -1), N * 4)
        self.nearFieldCorrectionUP_SOLVE(vesicle, info_rbf, L, far_field_1, velx, vely, xlayers, ylayers, nlayers)
        end_cuda.record()
        torch.cuda.synchronize()
        print(f'x1 nearFieldCorrection SOLVE {start_cuda.elapsed_time(end_cuda)/1000} sec.')


        return far_field_1, info_rbf, info_stokes
    
    

    def computeStokesInteractions_noinfo(self, vesicle, vesicleUp, info_rbf, info_stokes, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
                                  xlayers, ylayers, standardizationValues, nlayers, first: bool,  upsample=True):
        # print('Near-singular interaction through interpolation and network')

        velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues, nlayers)
        rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real[..., 2:3], vely_real[..., 2:3], velx_imag[..., 2:3], vely_imag[..., 2:3], standardizationValues, 1)

        totalForce = trac_jump + repForce
        # if upsample:
        N = vesicle.N
        nv = vesicle.nv
        Nup = ceil(sqrt(N)) * N
        length = 1.0
        # totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)
        totalForceUp = upsample_fft(totalForce, Nup)
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
            

        # start.record()
        if first:
            fn = allExactStokesSLTarget_compare1
            if nv > 1048:
                num_parts = 10
                far_fields = []
                info_stokes_parts = [[], [], []]

                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (i + 1) * nv // num_parts if i < num_parts-1 else None  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0  # Offset is None for the first call

                    far_field, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X[:, start:end], length, offset=offset)

                    far_fields.append(far_field)
                    for j in range(3):
                        info_stokes_parts[j].append(info_stokes[j])

                far_field_1 = torch.concat(far_fields, dim=-1)
                info_stokes = tuple(torch.cat(parts, dim=0) for parts in info_stokes_parts)

                
            elif nv > 504:
                # far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3])
                # far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3)
                # far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)
                # far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3), dim=-1)
                # info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0]), dim=0),
                #                  torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1]), dim=0),
                #                  torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2]), dim=0)
                # )   
                
                far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], length)
                far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], length, offset=nv//4)
                far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], length, offset=nv//2)
                far_field_1_4, info_stokes_4 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], length, offset=3*nv//4)
                far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1)
                info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0], info_stokes_4[0]), dim=0),
                                 torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1], info_stokes_4[1]), dim=0),
                                 torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2], info_stokes_4[2]), dim=0)
                )
                # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
            elif nv > 100:
                far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//2], length)
                far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:], length, offset=nv//2)
                far_field_1 = torch.concat((far_field_1_1, far_field_1_2), dim=-1)
                info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0]), dim=0),
                                 torch.cat((info_stokes_1[1], info_stokes_2[1]), dim=0),
                                 torch.cat((info_stokes_1[2], info_stokes_2[2]), dim=0)
                )
            else:
                # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
                # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
                far_field_1, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, length)
            id1 = info_stokes[2] * N + info_stokes[1]
            id2 = info_stokes[0] + 1*(info_stokes[0] >= info_stokes[2])
            info_rbf = (id1, id2)
        
        else:
            fn = allExactStokesSLTarget_compare2
            if nv > 1048:
                far_fields = []
                num_parts = 10
                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (i + 1) * nv // num_parts if i < num_parts-1 else None  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0   # Offset is None for the first call

                    mask = (start <= info_stokes[2]) & (info_stokes[2] < end) if i < num_parts-1 else (start <= info_stokes[2])
                    
                    far_field = fn(
                        vesicleUp.X, 
                        vesicleUp.sa, 
                        totalForceUp, 
                        vesicle.X[:, start:end], 
                        info_stokes[0][mask], 
                        info_stokes[1][mask], 
                        info_stokes[2][mask] - start, 
                        offset=offset
                    )
                    
                    far_fields.append(far_field)

                far_field_1 = torch.concat(far_fields, dim=1)


            elif nv > 504:
                # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
                #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
                #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
                far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=1)
                # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
            elif nv > 100:
                far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//2], info_stokes[0][info_stokes[2]<nv//2], info_stokes[1][info_stokes[2]<nv//2], info_stokes[2][info_stokes[2]<nv//2]), 
                                          fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:], info_stokes[0][nv//2<=info_stokes[2]], info_stokes[1][nv//2<=info_stokes[2]], info_stokes[2][nv//2<=info_stokes[2]] - nv//2,  offset=nv//2)), dim=1)
                
            else:
                # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
                # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
                far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        # end.record()
        # torch.cuda.synchronize()
        # print(f'computeStokesInteractions EXACT {start.elapsed_time(end)/1000} sec, {torch.norm(far_field - far_field_1)/torch.norm(far_field)}')
        # print(f'computeStokesInteractions noinfo EXACT {start.elapsed_time(end)/1000} sec')

        # start.record()
        # fn = allExactStokesSLTarget_compare2
        # if nv > 504:
        #     # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=-1)
        #     # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
            
        #     # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        # else:
        #     # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #     # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #     far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        # end.record()
        # torch.cuda.synchronize()
        # print(f'Stokes EXACT {start.elapsed_time(end)/1000} sec.')

        # print(f"--------- rel err of amp in stokes is {torch.norm(far_field - far_field_1)/torch.norm(far_field)}")

        # start.record()

        # pdb.set_trace()
        use_near_correct = False
        if use_near_correct:
            if self.rbf_upsample == 2:
                velx = interpft(velx.reshape(N, -1), N * 2)
                vely = interpft(vely.reshape(N, -1), N * 2)
            elif self.rbf_upsample == 4:
                velx = interpft(velx.reshape(N, -1), N * 4)
                vely = interpft(vely.reshape(N, -1), N * 4)
                
            self.nearFieldCorrectionUP_SOLVE(vesicle, info_rbf, L, far_field_1, velx, vely, xlayers, ylayers, nlayers)
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x1 nearFieldCorrection SOLVE {start.elapsed_time(end)/1000} sec.')

        selfRepVel = torch.concat((rep_velx.squeeze(1), rep_vely.squeeze(1)), dim=0)
        return far_field_1 + selfRepVel, info_rbf, info_stokes
    
    
    def computeStokesInteractions_noinfo_exactLayers(self, vesicle, vesicleUp, info_rbf, info_stokes, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
                                  xlayers, ylayers, standardizationValues, nlayers, first: bool,  upsample=True):
        # print('Near-singular interaction through interpolation and network')

        # velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues, nlayers)
        # rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real[..., 2:3], vely_real[..., 2:3], velx_imag[..., 2:3], vely_imag[..., 2:3], standardizationValues, 1)
        nlayers=3
        Nsou = vesicle.N
        Xsou = vesicle.X
        Nup_for_layers = Nsou * (nlayers-1) * ceil(sqrt((nlayers-1)*Nsou))
        # Nup_for_layers = Nsou * 2 * math.ceil(math.sqrt(2*Nsou))
        Xup = interpft_vec(Xsou, Nup_for_layers)  # Upsample source points
        fup = interpft_vec(trac_jump, Nup_for_layers)  # Upsample density function
        vesicleUp_for_layers = capsules(Xup, None, None, kappa=1, viscCont=1)

        vself = torch.einsum('ijk,jk->ik', self.Galpert, trac_jump) 

        N = Xsou.shape[0]//2
        upsample = self.rbf_upsample
        if upsample > 0:
            Nup = N * upsample # is different from vesicleUp.N !!
            Xup = interpft_vec(Xsou, Nup)  # Upsample source points
        else:
            Nup = N
            Xup = Xsou

        # vesicleUp = capsules(Xup, None, None, vesicle.kappa, vesicle.viscCont)
        oc = self.oc
        dlayer = torch.linspace(0, 1/N, nlayers, dtype=torch.float64, device=Xsou.device)
        _, tang = oc.diffProp_jac_tan(Xup)
        rep_nx = tang[Nup:, :, None].expand(-1,-1,nlayers-1)
        rep_ny = -tang[:Nup, :, None].expand(-1,-1,nlayers-1)
        dx =  rep_nx * dlayer[[1,2,3]] if nlayers == 4 else rep_nx * dlayer[[1,2]] # (N, nv, nlayers-1)
        dy =  rep_ny * dlayer[[1,2,3]] if nlayers == 4 else rep_ny * dlayer[[1,2]]
        tracers = torch.permute(
            torch.vstack([torch.repeat_interleave(Xup[:Nup, :, None], nlayers-1, dim=-1) + dx,
                        torch.repeat_interleave(Xup[Nup:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1)) # (2*N, nlayers-1, nv)

        velx, vely, xlayers, ylayers = exactStokesSL_onlyself(vesicleUp_for_layers.X, vesicleUp_for_layers.sa, fup, Nup, Xup, vself, tracers)
        
        
        # Compute near/far hydro interactions without any correction
        # First calculate the far-field

        totalForce = trac_jump + repForce
        # if upsample:
        N = vesicle.N
        nv = vesicle.nv
        Nup = ceil(sqrt(N)) * N
        # totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)
        totalForceUp = upsample_fft(totalForce, Nup)
        length = self.len0[0].item()
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
            

        # start.record()
        if first:
            fn = allExactStokesSLTarget_compare1
            if nv > 1048:
                num_parts = 10
                far_fields = []
                info_stokes_parts = [[], [], []]

                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (i + 1) * nv // num_parts if i < num_parts-1 else None  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0  # Offset is None for the first call

                    far_field, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X[:, start:end], length, offset=offset)

                    far_fields.append(far_field)
                    for j in range(3):
                        info_stokes_parts[j].append(info_stokes[j])

                far_field_1 = torch.concat(far_fields, dim=-1)
                info_stokes = tuple(torch.cat(parts, dim=0) for parts in info_stokes_parts)

                
            elif nv > 504:
                # far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3])
                # far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3)
                # far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)
                # far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3), dim=-1)
                # info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0]), dim=0),
                #                  torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1]), dim=0),
                #                  torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2]), dim=0)
                # )   
                
                far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], length)
                far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], length, offset=nv//4)
                far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], length, offset=nv//2)
                far_field_1_4, info_stokes_4 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], length, offset=3*nv//4)
                far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1)
                info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0], info_stokes_4[0]), dim=0),
                                 torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1], info_stokes_4[1]), dim=0),
                                 torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2], info_stokes_4[2]), dim=0)
                )
                # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
            elif nv > 100:
                far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//2], length)
                far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:], length, offset=nv//2)
                far_field_1 = torch.concat((far_field_1_1, far_field_1_2), dim=-1)
                info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0]), dim=0),
                                 torch.cat((info_stokes_1[1], info_stokes_2[1]), dim=0),
                                 torch.cat((info_stokes_1[2], info_stokes_2[2]), dim=0)
                )
            else:
                # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
                # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
                far_field_1, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, length)
            id1 = info_stokes[2] * N + info_stokes[1]
            id2 = info_stokes[0] + 1*(info_stokes[0] >= info_stokes[2])
            info_rbf = (id1, id2)
        
        else:
            fn = allExactStokesSLTarget_compare2
            if nv > 1048:
                far_fields = []
                num_parts = 10
                for i in range(num_parts):
                    start = i * nv // num_parts
                    end = (i + 1) * nv // num_parts if i < num_parts-1 else None  # Ensure last slice goes to the end
                    offset = start if i > 0 else 0   # Offset is None for the first call

                    mask = (start <= info_stokes[2]) & (info_stokes[2] < end) if i < num_parts-1 else (start <= info_stokes[2])
                    
                    far_field = fn(
                        vesicleUp.X, 
                        vesicleUp.sa, 
                        totalForceUp, 
                        vesicle.X[:, start:end], 
                        info_stokes[0][mask], 
                        info_stokes[1][mask], 
                        info_stokes[2][mask] - start, 
                        offset=offset
                    )
                    
                    far_fields.append(far_field)

                far_field_1 = torch.concat(far_fields, dim=1)


            elif nv > 504:
                # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
                #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
                #                     fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
                far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
                                        fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=1)
                # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
                #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
            elif nv > 100:
                far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//2], info_stokes[0][info_stokes[2]<nv//2], info_stokes[1][info_stokes[2]<nv//2], info_stokes[2][info_stokes[2]<nv//2]), 
                                          fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:], info_stokes[0][nv//2<=info_stokes[2]], info_stokes[1][nv//2<=info_stokes[2]], info_stokes[2][nv//2<=info_stokes[2]] - nv//2,  offset=nv//2)), dim=1)
                
            else:
                # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
                # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
                far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        # end.record()
        # torch.cuda.synchronize()
        # print(f'computeStokesInteractions EXACT {start.elapsed_time(end)/1000} sec, {torch.norm(far_field - far_field_1)/torch.norm(far_field)}')
        # print(f'computeStokesInteractions noinfo EXACT {start.elapsed_time(end)/1000} sec')

        # start.record()
        # fn = allExactStokesSLTarget_compare2
        # if nv > 504:
        #     # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
        #     #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=-1)
        #     # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
            
        #     # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        # else:
        #     # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #     # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #     far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        # end.record()
        # torch.cuda.synchronize()
        # print(f'Stokes EXACT {start.elapsed_time(end)/1000} sec.')

        # print(f"--------- rel err of amp in stokes is {torch.norm(far_field - far_field_1)/torch.norm(far_field)}")

        # start.record()

        # pdb.set_trace()
        # if self.rbf_upsample == 2:
        #     velx = interpft(velx.reshape(N, -1), N * 2)
        #     vely = interpft(vely.reshape(N, -1), N * 2)
        # elif self.rbf_upsample == 4:
        #     velx = interpft(velx.reshape(N, -1), N * 4)
        #     vely = interpft(vely.reshape(N, -1), N * 4)
        self.nearFieldCorrectionUP_SOLVE(vesicle, info_rbf, L, far_field_1, velx, vely, xlayers, ylayers, nlayers)
        # np.save("vel_layers.npy", {"velx": velx.cpu().numpy(), "vely": vely.cpu().numpy(), "xlayers": xlayers.cpu().numpy(), "ylayers": ylayers.cpu().numpy()})
        # end.record()
        # torch.cuda.synchronize()
        # print(f'x1 nearFieldCorrection SOLVE {start.elapsed_time(end)/1000} sec.')

        # selfRepVel = torch.concat((rep_velx.squeeze(1), rep_vely.squeeze(1)), dim=0)
        return far_field_1, info_rbf, info_stokes
    
    

    def computeStokesInteractions_timing(self, vesicle, vesicleUp, info_rbf, info_stokes, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
                                  xlayers, ylayers, standardizationValues, upsample=True):
        # print('Near-singular interaction through interpolation and network')

        velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
        # rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
        # Compute near/far hydro interactions without any correction
        # First calculate the far-field

        totalForce = trac_jump + repForce
        # if upsample:
        N = vesicle.N
        nv = vesicle.nv
        Nup = ceil(sqrt(N)) * N
        # totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)
        totalForceUp = upsample_fft(totalForce, Nup)
        

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        

        # if first_round:
        #     fn = allExactStokesSLTarget_broadcast
        #     if nv > 504:
        #         n_parts = 4
        #         far_field, info = [], []
        #         for i in range(n_parts):
        #             if i == 0:
        #                 out = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//n_parts], offset=0, return_info=True)
        #             else:
        #                 out = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, i*nv//n_parts:(i+1)*nv//n_parts], offset=i*nv//n_parts, return_info=True)
        #             far_field.append(out[0])
        #             info.append(out[1])
                
        #         del out
        #         far_field = torch.concat(far_field, dim=1)
        #         info = torch.concat(info, dim=0)

        #         # far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
        #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
        #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
        #         #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field, info = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, return_info=True)

        # else:
        #     fn = allExactStokesSLTarget_broadcast
        #     if nv > 504:
        #         far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], return_info=False)[0], 
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4, return_info=False)[0],
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2, return_info=False)[0],
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4, return_info=False)[0]), dim=1)
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field, _ = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, return_info=False)

        # start.record()
        # fn = allExactStokesSLTarget_broadcast
        # if nv > 504:
        #     far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
        #                             fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
        #     # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #     #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        # else:
        #     # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #     # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #     far_field = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)


        # end.record()
        # torch.cuda.synchronize()
        # print(f'stokes old EXACT {start.elapsed_time(end)/1000} sec.')
            

        # start.record()
        # if first:
        #     fn = allExactStokesSLTarget_compare1
        #     if nv > 2:
        #         far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4])
        #         far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4)
        #         far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4],  offset=nv//2)
        #         far_field_1_4, info_stokes_4 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)
        #         far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1)
        #         info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0], info_stokes_4[0]), dim=0),
        #                          torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1], info_stokes_4[1]), dim=0),
        #                          torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2], info_stokes_4[2]), dim=0)
        #         )
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field_1, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)
        #     id1 = info_stokes[2] * N + info_stokes[1]
        #     id2 = info_stokes[0] + 1*(info_stokes[0] >= info_stokes[2])
        #     info_rbf = (id1, id2)
        
        # else:
        #     fn = allExactStokesSLTarget_compare2
        #     if nv > 504:
        #         far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=1)
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        # end.record()
        # torch.cuda.synchronize()
        # print(f'computeStokesInteractions EXACT {start.elapsed_time(end)/1000} sec, {torch.norm(far_field - far_field_1)/torch.norm(far_field)}')
        # print(f'computeStokesInteractions EXACT {start.elapsed_time(end)/1000} sec')

        start.record()
        fn = allExactStokesSLTarget_compare2
        if nv > 504:
            # far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
            #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
            #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
            #                         fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=-1)
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//3], info_stokes[0][info_stokes[2]<nv//3], info_stokes[1][info_stokes[2]<nv//3], info_stokes[2][info_stokes[2]<nv//3]), 
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//3:2*nv//3], info_stokes[0][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[1][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)], info_stokes[2][(nv//3<=info_stokes[2]) & (info_stokes[2]<2*nv//3)] - nv//3, offset=nv//3),
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 2*nv//3:], info_stokes[0][2*nv//3<=info_stokes[2]], info_stokes[1][2*nv//3<=info_stokes[2]], info_stokes[2][2*nv//3<=info_stokes[2]] - 2*nv//3,  offset=2*nv//3)), dim=-1)
            
            
            # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        else:
            # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
            # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
            far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        end.record()
        torch.cuda.synchronize()
        print(f'Stokes EXACT {start.elapsed_time(end)/1000} sec.')

        # print(f"--------- rel err of amp in stokes is {torch.norm(far_field - far_field_1)/torch.norm(far_field)}")

        start.record()
        self.nearFieldCorrectionUP_SOLVE_timing(vesicle, info_rbf, L, far_field_1, velx, vely, xlayers, ylayers)
        end.record()
        torch.cuda.synchronize()
        print(f'x1 nearFieldCorrection SOLVE {start.elapsed_time(end)/1000} sec.')


        return far_field_1
    
    
    def computeStokesInteractions(self, vesicle, vesicleUp, info_rbf, info_stokes, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
                                  xlayers, ylayers, standardizationValues, upsample=True):
        # print('Near-singular interaction through interpolation and network')

        velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
        # rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
        # Compute near/far hydro interactions without any correction
        # First calculate the far-field

        totalForce = trac_jump + repForce
        # if upsample:
        N = vesicle.N
        nv = vesicle.nv
        Nup = ceil(sqrt(N)) * N
        # totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)
        totalForceUp = upsample_fft(totalForce, Nup)

        
        fn = allExactStokesSLTarget_broadcast
        if nv > 504:
            far_field = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
            # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        else:
            # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
            # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
            far_field = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)

            # far_field_1 = allExactStokesSLTarget_compare(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])
        
        # if first:
        #     fn = allExactStokesSLTarget_compare1
        #     if nv > 2:
        #         far_field_1_1, info_stokes_1 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4])
        #         far_field_1_2, info_stokes_2 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4)
        #         far_field_1_3, info_stokes_3 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4],  offset=nv//2)
        #         far_field_1_4, info_stokes_4 = fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)
        #         far_field_1 = torch.concat((far_field_1_1, far_field_1_2, far_field_1_3, far_field_1_4), dim=-1)
        #         info_stokes = (torch.cat((info_stokes_1[0], info_stokes_2[0], info_stokes_3[0], info_stokes_4[0]), dim=0),
        #                          torch.cat((info_stokes_1[1], info_stokes_2[1], info_stokes_3[1], info_stokes_4[1]), dim=0),
        #                          torch.cat((info_stokes_1[2], info_stokes_2[2], info_stokes_3[2], info_stokes_4[2]), dim=0)
        #         )
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field_1, info_stokes = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)
        #     id1 = info_stokes[2] * N + info_stokes[1]
        #     id2 = info_stokes[0] + 1*(info_stokes[0] >= info_stokes[2])
        #     info_rbf = (id1, id2)
        
        # else:
        #     fn = allExactStokesSLTarget_compare2
        #     if nv > 504:
        #         far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
        #                                 fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=1)
        #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
        #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        #     else:
        #         # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
        #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
        #         far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        fn = allExactStokesSLTarget_compare2
        if nv > 504:
            far_field_1 = torch.concat((fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
                                    fn(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=-1)
            # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        else:
            # (vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask, offset: int = 0):
            # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
            far_field_1 = fn(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info_stokes[0], info_stokes[1], info_stokes[2])

        

        print(f"is close {torch.allclose(far_field, far_field_1)}, {torch.norm(far_field - far_field_1)/torch.norm(far_field)}")
        
        # (nv-1, Ntar, ntar)
        # rows_with_true = torch.max(nbrs_mask.reshape(nv*N, nv, Nup), dim=-1)[0] # (N*nv, nv)
        # id1, id2 = torch.where(rows_with_true)
        # # id1, id2 = rows_with_true.to_sparse().indices() # for rbf solves
        # ids1, ids2 = id1 % N, id1 // N
        # ids0 = id2 - 1*(ids2 <= id2)
        
        self.nearFieldCorrectionUP_SOLVE(vesicle, info_rbf, L, far_field, velx, vely, xlayers, ylayers)

        return far_field_1
    
    
    # def computeStokesInteractions(self, vesicle, vesicleUp, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
    #                               xlayers, ylayers, standardizationValues, info, dis2, diffx, diffy, full_mask, upsample=True):
    #     # print('Near-singular interaction through interpolation and network')

    #     velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
    #     # rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
    #     # Compute near/far hydro interactions without any correction
    #     # First calculate the far-field

    #     totalForce = trac_jump + repForce
    #     # if upsample:
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     Nup = ceil(sqrt(N)) * N
    #     totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)

    #     if nv > 504:
    #         far_field = torch.concat((allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
    #                                 allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
    #                                 allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
    #                                 allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
    #         # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
    #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
    #         #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
    #     else:
    #         # far_field = allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)
    #         # far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, self.info, self.dis2, self.diffx, self.diffy, self.full_mask)
    #         far_field, info, dis2, diffx, diffy, full_mask = self.allExactStokesSLTarget_broadcast_(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X, info, dis2, diffx, diffy, full_mask)


    #     # if not torch.allclose(far_field, far_field_):
    #     #     print(torch.norm(far_field - far_field_)/torch.norm(far_field))
    #     #     raise "haha"
        
    #     # if not torch.allclose(far_field, far_field2):
    #     #     print(torch.norm(far_field - far_field2)/torch.norm(far_field))
    #     #     raise "haha"
            
        
        
    #     self.nearFieldCorrectionUP_SOLVE(vesicle, vesicleUp, info, L, far_field, velx, vely, xlayers, ylayers)

    #     # if not torch.allclose(far_field, far_field2):
    #     #     print(f"farfield suanliangci {torch.norm(far_field - far_field2)/torch.norm(far_field)}")
    #         # raise "haha second"

    #     return far_field, info, dis2, diffx, diffy, full_mask
    
    
    
    def nearFieldCorrectionUP_SOLVE_timing(self, vesicle, info, L, far_field, velx, vely, xlayers, ylayers):
        if  len(info[0])==0 or len(info[1])==0:
            return
        
        N = vesicle.N
        nv = vesicle.nv
        # Nup = vesicleUp.N

        # nbrs_mask_reshaped = nbrs_mask.reshape(nv*N, nv, Nup)
        # rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1).unsqueeze(1) # (N*nv, 1, nv)
        # if not torch.any(rows_with_true):
        #     return
        
        all_points = torch.concat((vesicle.X[:N, :].T.reshape(-1,1), vesicle.X[N:, :].T.reshape(-1,1)), dim=1)
        correction = torch.zeros((N*nv, 2), dtype=far_field.dtype, device=far_field.device)
        
        const = 0.672 
        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
        all_X = all_X /const * N   

        
        # matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) # (3*N, 3*N, nv)
        rhs = torch.concat((velx.reshape(-1,1,nv), vely.reshape(-1,1,nv)), dim=1) # (3 * N), 2, nv), 2 for x and y
        
        # tStart = time.time()
        # coeffs = torch.linalg.solve(matrices.permute(2, 0, 1), rhs.permute(2, 0, 1))
        # coeffs = coeffs
        # print("coeffs solved")
        # tEnd = time.time()
        # print(f'x1 nearFieldCorrection linalg.SOLVE {tEnd - tStart} sec.')

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        # L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        y = torch.linalg.solve_triangular(L, rhs.permute(2, 0, 1), upper=False)
        coeffs = torch.linalg.solve_triangular(L.permute(0, 2, 1), y, upper=True)
        end.record()
        torch.cuda.synchronize()
        print(f'---- nearFieldCorrection TRIANGULAR SOLVES {start.elapsed_time(end)/1000} sec.')

        # if not torch.allclose(coeffs, coeffs_):
        #     print(f"coeffs solve vs cholesky {torch.norm(coeffs - coeffs_)/torch.norm(coeffs)}")
            

        # start.record()
        # for k in range(nv):
        #     rows_with_true = torch.any(nbrs_mask[:, k*Nup : (k+1)*Nup], dim=1)
        #     if not torch.any(rows_with_true):
        #         continue

        #     print(f'---- nearFieldCorrection for loop {k}')
        #     ids_in = torch.arange(N*nv)[rows_with_true] # which ves points are in k's near zone
        #     points_query = all_points[ids_in] # and their coords

        #     r2 = torch.sum((points_query[:, None]/const * N - all_X[None, ..., k])**2, dim=-1)
        #     matrix = torch.exp(- 1 * r2) 
        
        #     rbf_vel = matrix @ coeffs[k]

        #     correction[ids_in, 0] += rbf_vel[:, 0]
        #     correction[ids_in, 1] += rbf_vel[:, 1]

        # end.record()
        # torch.cuda.synchronize()
        # print(f'---- nearFieldCorrection for loop {start.elapsed_time(end)/1000} sec.')

        # start.record()
        # r = torch.norm(all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...], dim=-2)
        # r.masked_fill_(~rows_with_true, torch.inf)
        # matrix = torch.exp(- 1 * r**2) 

        # rr = torch.full((N*nv, N*5, nv), float('inf'))
        # matrix = torch.zeros((N*nv, N*5, nv), dtype=torch.float32, device=far_field.device)
        # id1, id2, id3 = torch.where(rows_with_true.expand(-1, N*5, -1))
        # matrix[id1, id2, id3] = torch.exp(- 1 *torch.norm(all_points[id1]/const*N - all_X[id2, :, id3], dim=-1)**2)
        # # matrix = torch.exp(- 1 * rr) 
        # end.record()
        # torch.cuda.synchronize()
        # t1 = start.elapsed_time(end)/1000


        # start.record()
        # correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs
        # end.record()
        # torch.cuda.synchronize()
        # t2 = start.elapsed_time(end)/1000
        # print(f'---- nearFieldCorrection masking {t1} and {t2} sec.')

        start.record()
        # id1, id2 = torch.where(rows_with_true.expand(-1, N*5, -1).reshape(N*nv, -1))
        id1_, id2_ = info
        id2_ = id2_[:, None] + torch.arange(0, N*5*nv, nv).to(id2_.device)
        id2_ = id2_.reshape(-1)
        id1_ = id1_[:, None].expand(-1, N*5).reshape(-1)
        sp_matrix = torch.sparse_coo_tensor(torch.vstack((id1_, id2_)), 
                        torch.exp(-torch.norm(all_points[id1_]/const*N - all_X.permute(0,2,1).reshape(-1, 2)[id2_, :], dim=-1)**2),
                        size=(N*nv, N*5 * nv))  
        correction = torch.sparse.mm(sp_matrix, coeffs.permute(1,0,2).reshape(nv * N*5, 2))
        end.record()
        torch.cuda.synchronize()
        t2 = start.elapsed_time(end)/1000
        print(f'---- nearFieldCorrection masking sparse {t2} sec.')

        
        correction = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)
        far_field += correction
        return 


    def nearFieldCorrectionUP_SOLVE(self, vesicle, info, L, far_field, velx, vely, xlayers, ylayers, nlayers):
        if  len(info[0])==0 or len(info[1])==0:
            return
        
        N = vesicle.N
        nv = vesicle.nv
        # Nup = vesicleUp.N

        all_points = torch.concat((vesicle.X[:N, :].T.reshape(-1,1), vesicle.X[N:, :].T.reshape(-1,1)), dim=1)
        # correction = torch.zeros((N*nv, 2), dtype=torch.float32, device=trac_jump.device)
        
        # if self.rbf_upsample <=0 :
        #     const = 0.672 * self.len0[0].item()
        # elif self.rbf_upsample == 2:
        #     const = 0.566 * self.len0[0].item()

        # elif self.rbf_upsample == 4:
        #     const = 0.495 * self.len0[0].item()
        if self.rbf_upsample <= 2:
            # const = 0.495 * self.len0[0].item() * 4
            # const = 1.7 / 128
            const = 0.0132

        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
        # all_X = all_X /const * N   
        all_X = all_X /const

        # matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) # (3*N, 3*N, nv)
        rhs = torch.concat((velx.reshape(-1,1,nv), vely.reshape(-1,1,nv)), dim=1) # (3 * N), 2, nv), 2 for x and y
        
        # tStart = time.time()
        # coeffs = torch.linalg.solve(matrices.permute(2, 0, 1), rhs.permute(2, 0, 1))
        # coeffs = coeffs
        # print("coeffs solved")
        # tEnd = time.time()
        # print(f'x1 nearFieldCorrection linalg.SOLVE {tEnd - tStart} sec.')

        # L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        y = torch.linalg.solve_triangular(L, rhs.permute(2, 0, 1), upper=False)
        coeffs = torch.linalg.solve_triangular(L.permute(0, 2, 1), y, upper=True)

        # if not torch.allclose(coeffs, coeffs_):
        #     print(f"coeffs solve vs cholesky {torch.norm(coeffs - coeffs_)/torch.norm(coeffs)}")
            

        # tStart = time.time()
        # for k in range(nv):
        #     rows_with_true = torch.any(nbrs_mask[:, k*Nup : (k+1)*Nup], dim=1)
        #     if not torch.any(rows_with_true):
        #         continue
        #     ids_in = torch.arange(N*nv)[rows_with_true] # which ves points are in k's near zone
            
        #     points_query = all_points[ids_in] # and their coords
        #     ves_id = torch.IntTensor([k])

        #     r2 = torch.sum((points_query[:, None]/const * N - all_X[None, ..., ves_id].squeeze(-1))**2, dim=-1)
        #     matrix = torch.exp(- 1 * r2) 
        
        #     rbf_vel = matrix @ coeffs[k]

        #     correction[ids_in, 0] += rbf_vel[:, 0]
        #     correction[ids_in, 1] += rbf_vel[:, 1]

        # tEnd = time.time()
        # print(f'x1 nearFieldCorrection for loop {tEnd - tStart} sec.')


        # nbrs_mask_reshaped = nbrs_mask.reshape(nv*N, nv, Nup)
        # rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1).unsqueeze(1) # (N*nv, 1, nv)
 
        # r = torch.norm(all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...], dim=-2) # (N*nv, N*nlayers, nv)
        # r.masked_fill_(~rows_with_true, torch.inf)
        # matrix = torch.exp(- 1 * r**2) 

        # rr = torch.full((N*nv, N*5, nv), float('inf'))
        # id1, id2, id3 = torch.where(rows_with_true.expand(-1, N*5, -1))
        # rr[id1, id2, id3] = torch.norm(all_points[id1]/const*N - all_X[id2, :, id3], dim=-1)
        # matrix = torch.exp(- 1 * rr**2) 

        # matrix = torch.zeros((N*nv, N*5, nv), dtype=torch.float32, device=far_field.device)
        # id1, id2, id3 = torch.where(rows_with_true.expand(-1, N*5, -1))
        # matrix[id1, id2, id3] = torch.exp(- torch.norm(all_points[id1]/const*N - all_X[id2, :, id3], dim=-1)**2)
        
        # if not torch.allclose(r, rr):
        #     raise "r and rr are not the same"

        # sp_matrix = matrix.reshape(N*nv, N*5 * nv).to_sparse()
        # c = torch.sparse.mm(sp_matrix, coeffs.permute(1,0,2).reshape(nv * N*5, 2))
        # correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs
        # correction_ = matrix.reshape(N*nv, N*5 * nv) @ coeffs.permute(1,0,2).reshape(nv * N*5, 2)

        # id1, id2 = torch.where(rows_with_true.expand(-1, N*5, -1).reshape(N*nv, -1))
        # sp_matrix = torch.sparse_coo_tensor(torch.vstack((id1, id2)), 
        #                 torch.exp(-torch.norm(all_points[id1]/const*N - all_X.permute(0,2,1).reshape(-1, 2)[id2, :], dim=-1)**2),
        #                 size=(N*nv, N*5 * nv))  
        # correction = torch.sparse.mm(sp_matrix, coeffs.permute(1,0,2).reshape(nv * N*5, 2))


        # id1_, id2_ = torch.where(rows_with_true.squeeze())
        id1_, id2_ = info
        if self.rbf_upsample <= 1:
            id2_ = id2_[:, None] + torch.arange(0, N*nlayers*nv, nv).to(id2_.device)
            id2_ = id2_.reshape(-1)
            id1_ = id1_[:, None].expand(-1, N*nlayers).reshape(-1)
            sp_matrix_ = torch.sparse_coo_tensor(torch.vstack((id1_, id2_)), 
                            torch.exp(-torch.norm(all_points[id1_]/const - all_X.permute(0,2,1).reshape(-1, 2)[id2_, :], dim=-1)**2),
                            size=(N*nv, N * nlayers * nv))
            correction = torch.sparse.mm(sp_matrix_, coeffs.permute(1,0,2).reshape(nv * N* nlayers, 2))
        else:
            id2_ = id2_[:, None] + torch.arange(0, self.rbf_upsample * N*nlayers*nv, nv).to(id2_.device)
            id2_ = id2_.reshape(-1)
            id1_ = id1_[:, None].expand(-1, self.rbf_upsample * N*nlayers).reshape(-1)
            sp_matrix_ = torch.sparse_coo_tensor(torch.vstack((id1_, id2_)), 
                            torch.exp(-torch.norm(all_points[id1_]/const - all_X.permute(0,2,1).reshape(-1, 2)[id2_, :], dim=-1)**2),
                            size=(N*nv, self.rbf_upsample  * N * nlayers * nv))
            correction = torch.sparse.mm(sp_matrix_, coeffs.permute(1,0,2).reshape(nv * self.rbf_upsample  * N* nlayers, 2))
        

        # if not torch.allclose(a, matrix.reshape(N*nv, N*5 * nv)):
        #     raise "a and matrix are not the same"
        
        # if not torch.allclose(cc, correction):
        #     raise "cc and correction are not the same"
        
        # if not torch.allclose(correction, correction_):
        #     raise "correction suanliangci"
        
        # else:
            
        #     nbrs_mask_reshaped = nbrs_mask.reshape(N*nv, nv, Nup)
        #     rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1).unsqueeze(1) # (N*nv, nv)

        #     r2 = torch.sum((all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...])**2, dim=-2)
        #     r2.masked_fill_(~rows_with_true, torch.inf)
        #     matrix = torch.exp(- 1 * r2) 
        #     end.record()
        #     torch.cuda.synchronize()
        #     t1 = start.elapsed_time(end)/1000

        #     start.record()
        #     correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs
        #     end.record()
        #     torch.cuda.synchronize()
        #     t2 = start.elapsed_time(end)/1000
        #     print(f'---- nearFieldCorrection masking {t1} and {t2} sec.')


        # if not torch.allclose(correction, correction_):
        #     print(f"correction suanliangci {torch.norm(correction - correction_)/torch.norm(correction)}")
        #     # raise "haha"
        
        correction = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)
        far_field += correction
        return 




    # def get_ns_coords(Xin, query):
    #     """
    #     Find n,s coords for scattered query.

    #     Args:
    #         Xin: (2, N)
    #         query: (2, M)

    #     Returns:
    #         torch.Tensor: Interpolated values at query points of shape (M,).
    #     """
    #     N = Xin.shape[-1] // 3  # 3 layers
    #     M = query.shape[-1]

    #     dist_sq = torch.sum((Xin[:, :, None] - query[:, None, :])**2, dim=0)  # (N, M)
        
    #     # Find the 4 nearest neighbors
    #     n_points = 4
    #     _, topk_indices = torch.topk(-dist_sq, n_points, dim=0)  # (4, M)

    #     # infer n-s indices from topk_indices
    #     # s1, s2 = torch.unique_consecutive(topk_indices % N)[:2] # cannot use dim for our purpose
    #     # n1, n2 = torch.unique_consecutive(topk_indices // N)[:2]         
    #     # s1, s2 = torch.sort(topk_indices % 128, dim=0)[0][[0, -1]]   
    #     # n1, n2 = torch.sort(topk_indices // 128, dim=0)[0][[0, -1]]   
    #     s1 = (topk_indices % N)[0, :]
    #     condition = (topk_indices % N) != s1
    #     s2 = (topk_indices % N)[torch.argmax(condition.int(), dim=0), torch.arange(M)]

    #     n1 = (topk_indices // N)[0, :]
    #     condition = (topk_indices // N) != n1
    #     n2 = (topk_indices // N)[torch.argmax(condition.int(), dim=0), torch.arange(M)]

    #     p1_id, p2_id, p3_id, p4_id = s1 + n1 * N, s2 + n1*N, s1 + n2*N, s2 + n2*N
    #     # print(f"top_indices are {topk_indices.squeeze()}, infer box is {(p1_id, p2_id, p3_id, p4_id)}")

        
    #     p1_x, p2_x, p3_x = Xin[0, p1_id], Xin[0, p2_id], Xin[0, p3_id]
    #     p1_y, p2_y, p3_y = Xin[1, p1_id], Xin[1, p2_id], Xin[1, p3_id]
    #     s_query = s1 + ((query[0] - p1_x)*(p2_x - p1_x) + (query[1] - p1_y)*(p2_y - p1_y))/((p2_x - p1_x)**2 + (p2_y - p1_y)**2) * (s2 - s1)
    #     n_query = n1 + ((query[0] - p1_x)*(p3_x - p1_x) + (query[1] - p1_y)*(p3_y - p1_y))/((p3_x - p1_x)**2 + (p3_y - p1_y)**2) * (n2 - n1)

    #     # print(f"known n-s indices are {(s1, s2, n1, n2)}, query is {s_query}, {n_query}")
    #     return s_query, n_query


    # def nearFieldCorrectionUP_ns_SOLVE(self, vesicle, vesicleUp, info, far_field, L, velx, vely, xlayers, ylayers):
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     Nup = vesicleUp.N
        
    #     nbrs_mask = info

    #     xvesicle = vesicle.X[:N, :]
    #     yvesicle = vesicle.X[N:, :]
    #     all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1)
        
        
    #     const = 1.69 # 1.2 * sqrt(2)

    #     all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
    #     # all_X = all_X /const * N   

    #     rhs = torch.concat((velx.reshape(-1,1,nv), vely.reshape(-1,1,nv)), dim=1) # (3 * N), 2, nv), 2 for x and y

    #     tStart = time.time()
    #     # L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
    #     y = torch.linalg.solve_triangular(L, rhs.permute(2, 0, 1), upper=False)
    #     coeffs = torch.linalg.solve_triangular(L.permute(0, 2, 1), y, upper=True)
    #     tEnd = time.time()
    #     print(f'x1 nearFieldCorrection ns CHOLESKY {tEnd - tStart} sec.')

    #     # if not torch.allclose(coeffs, coeffs_):
    #     #     print(f"coeffs solve vs cholesky {torch.norm(coeffs - coeffs_)/torch.norm(coeffs)}")

    #     tStart = time.time()
    #     nbrs_mask_reshaped = nbrs_mask.reshape(N*nv, nv, Nup)
    #     rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1).unsqueeze(1) # (N*nv, nv)

    #     r2 = torch.sum((all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...])**2, dim=-2)
    #     r2.masked_fill_(~rows_with_true, torch.inf)
    #     matrix = torch.exp(- 1 * r2) 
    #     correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs

    #     tEnd = time.time()
    #     print(f'x1 nearFieldCorrection masking {tEnd - tStart} sec.')

    #     # if not torch.allclose(correction, correction_):
    #     #     print(f"correction suanliangci {torch.norm(correction - correction_)/torch.norm(correction)}")
    #     #     # raise "haha"
        
    #     correction = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)
    #     far_field += correction
    #     return 

    
    # @torch.jit.script_method
    # @torch.compile(backend='cudagraphs')
    def translateVinfwTorch(self, Xold, Xstand, standardizationValues : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], vinf):
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]
        
        # Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(Xold)
        _, rotate, _, _, sortIdx = standardizationValues

        # Xinp = self.mergedAdvNetwork.preProcess(Xstand)
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
        vinf_stand = self.standardize(vinf, torch.zeros((2,nv), dtype=vinf.dtype), rotate, torch.zeros((2,nv), dtype=vinf.dtype), 1, sortIdx)
        z = vinf_stand[:N] + 1.0j * vinf_stand[N:]
        zh = torch.fft.fft(z, dim=0)
        V1, V2 = torch.real(zh), torch.imag(zh)
        MVinf_stand = torch.vstack((torch.einsum('NiB,iB ->NB', Z11r_, V1) + torch.einsum('NiB,iB ->NB', Z12r_, V2),
                               torch.einsum('NiB,iB ->NB', Z21r_, V1) + torch.einsum('NiB,iB ->NB', Z22r_, V2)))
        
        Xnew = torch.zeros_like(Xold)
        MVinf = torch.zeros_like(MVinf_stand)
        idx = torch.vstack([sortIdx.T, sortIdx.T + N])
        MVinf[idx, torch.arange(nv, device=Xstand.device)] = MVinf_stand
        MVinf = self.rotationOperator(MVinf, -rotate, torch.zeros((2, nv), dtype = MVinf.dtype))
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
        
        # Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(X)
        _, rotate, _, _, sortIdx = standardizationValues

        input = self.tenAdvNetwork.preProcess(Xstand)
        Xpredict = self.tenAdvNetwork.forward(input)
        # Xpredict_ = self.tenAdvNetwork.forward_half(input)
        # print(f"tenadv Xpredict and Xpredict_half are {torch.norm(Xpredict - Xpredict_) / torch.norm(Xpredict)}")
        out = self.tenAdvNetwork.postProcess(Xpredict) # shape: (127, nv, 2, 128)

        # Approximate the multiplication Z = inv(DivGT)DivPhi_k
        Z1 = torch.zeros((N, N, nv), dtype=Xstand.dtype)
        Z2 = torch.zeros((N, N, nv), dtype=Xstand.dtype)

        Z1[:, 1:] = torch.permute(out[:, :, 0], (2,0,1))
        Z2[:, 1:] = torch.permute(out[:, :, 1], (2,0,1))

        vBackSolve = torch.zeros((N, nv), dtype=Xstand.dtype)
        vinfStand = self.standardize(vinf, torch.zeros((2,nv), dtype=Xstand.dtype), rotate, torch.zeros((2,nv), dtype=Xstand.dtype), 1, sortIdx)
        z = vinfStand[:N] + 1.0j * vinfStand[N:]
        zh = torch.fft.fft(z, dim=0)
        
        V1_ = torch.real(zh)
        V2_ = torch.imag(zh)
        # Compute the approximation to inv(Div*G*Ten)*Div*vExt
        MVinfStand = torch.einsum('NiB,iB ->NB', Z1, V1_) + torch.einsum('NiB,iB ->NB', Z2, V2_)
                               
        # Destandardize the multiplication
        vBackSolve[sortIdx.T, torch.arange(nv, device=Xstand.device)] = MVinfStand

        return vBackSolve

    # @torch.compile(backend='cudagraphs')
    def invTenMatOnSelfBend(self, Xstand, standardizationValues):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        nv = Xstand.shape[1] # number of vesicles
        N = Xstand.shape[0] // 2

        # Xstand, scaling, _, _, _, sortIdx = self.standardizationStep(X)
        scaling, _, _, _, sortIdx = standardizationValues

        tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        # tenPredictStand = self.tenSelfNetwork.forward_curv(Xstand)
        # tenPredictStand = tenPredictStand #.double()
        tenPred = torch.zeros((N, nv), dtype=Xstand.dtype, device=Xstand.device)
        
        tenPred[sortIdx.T, torch.arange(nv, device=Xstand.device)] = tenPredictStand / scaling**2

        return tenPred

    
    def invTenMatOnSelfBend_curv(self, Xstand, standardizationValues):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        nv = Xstand.shape[1] # number of vesicles
        N = Xstand.shape[0] // 2

        # Xstand, scaling, _, _, _, sortIdx = self.standardizationStep(X)
        scaling, _, _, _, sortIdx = standardizationValues

        # tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        tenPredictStand = self.tenSelfNetwork_curv.forward_curv(Xstand, self.oc)
        # tenPredictStand = tenPredictStand #.double()
        tenPred = torch.zeros((N, nv), dtype=Xstand.dtype)
        
        tenPred[sortIdx.T, torch.arange(nv)] = tenPredictStand / scaling**2

        return tenPred

    
    # def exactStokesSL(self, vesicle, f, Xtar=None, K1=None):
    #     """
    #     Computes the single-layer potential due to `f` around all vesicles except itself.
    #     Also can pass a set of target points `Xtar` and a collection of vesicles `K1` 
    #     and the single-layer potential due to vesicles in `K1` will be evaluated at `Xtar`.

    #     Parameters:
    #     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    #     - f: Forcing term (2*N x nv).
    #     - Xtar: Target points (2*Ntar x ncol), optional.
    #     - K1: Collection of vesicles, optional.

    #     Returns:
    #     - stokesSLPtar: Single-layer potential at target points.
    #     """
        
        
    #     Ntar = Xtar.shape[0] // 2
    #     ncol = Xtar.shape[1]
    #     stokesSLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float32, device=vesicle.X.device)
        

    #     den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

    #     xsou = vesicle.X[:vesicle.N, K1].flatten()
    #     ysou = vesicle.X[vesicle.N:, K1].flatten()
    #     xsou = torch.tile(xsou, (Ntar, 1)).T    # (N*(nv-1), Ntar)
    #     ysou = torch.tile(ysou, (Ntar, 1)).T

    #     denx = den[:vesicle.N, K1].flatten()
    #     deny = den[vesicle.N:, K1].flatten()
    #     denx = torch.tile(denx, (Ntar, 1)).T    # (N*(nv-1), Ntar)
    #     deny = torch.tile(deny, (Ntar, 1)).T

    #     for k in range(ncol):  # Loop over columns of target points
    #         if ncol != 1:
    #             raise "ncol != 1"
    #         xtar = Xtar[:Ntar, k]
    #         ytar = Xtar[Ntar:, k]
    #         xtar = torch.tile(xtar, (vesicle.N * len(K1), 1))
    #         ytar = torch.tile(ytar, (vesicle.N * len(K1), 1))
            
    #         diffx = xtar - xsou
    #         diffy = ytar - ysou

    #         dis2 = diffx**2 + diffy**2

    #         coeff = 0.5 * torch.log(dis2)
    #         stokesSLPtar[:Ntar, k] = -torch.sum(coeff * denx, dim=0)
    #         stokesSLPtar[Ntar:, k] = -torch.sum(coeff * deny, dim=0)

    #         coeff = (diffx * denx + diffy * deny) / dis2
    #         stokesSLPtar[:Ntar, k] += torch.sum(coeff * diffx, dim=0)
    #         stokesSLPtar[Ntar:, k] += torch.sum(coeff * diffy, dim=0)


    #     return stokesSLPtar / (4 * torch.pi)
    
    
    # def exactStokesSL_expand(self, vesicle, f, Xtar=None, K1=None):
    #     """
    #     Computes the single-layer potential due to `f` around all vesicles except itself.
    #     Also can pass a set of target points `Xtar` and a collection of vesicles `K1` 
    #     and the single-layer potential due to vesicles in `K1` will be evaluated at `Xtar`.

    #     Parameters:
    #     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    #     - f: Forcing term (2*N x nv).
    #     - Xtar: Target points (2*Ntar x ncol), optional.
    #     - K1: Collection of vesicles, optional.

    #     Returns:
    #     - stokesSLPtar: Single-layer potential at target points.
    #     """
        
        
    #     Ntar = Xtar.shape[0] // 2
    #     ncol = Xtar.shape[1]
    #     stokesSLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float32, device=vesicle.X.device)
        

    #     den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

    #     xsou = vesicle.X[:vesicle.N, K1].flatten()
    #     ysou = vesicle.X[vesicle.N:, K1].flatten()
    #     # xsou = torch.tile(xsou, (Ntar, 1)).T    # (N*(nv-1), Ntar)
    #     # ysou = torch.tile(ysou, (Ntar, 1)).T
    #     xsou = xsou[None,:].expand(Ntar, -1).T
    #     ysou = ysou[None,:].expand(Ntar, -1).T

    #     denx = den[:vesicle.N, K1].flatten()
    #     deny = den[vesicle.N:, K1].flatten()
    #     # denx = torch.tile(denx, (Ntar, 1)).T    # (N*(nv-1), Ntar)
    #     # deny = torch.tile(deny, (Ntar, 1)).T
    #     deny = deny[None,:].expand(Ntar, -1).T
    #     denx = denx[None,:].expand(Ntar, -1).T

    #     for k in range(ncol):  # Loop over columns of target points
    #         if ncol != 1:
    #             raise "ncol != 1"
    #         xtar = Xtar[:Ntar, k]
    #         ytar = Xtar[Ntar:, k]
    #         # xtar = torch.tile(xtar, (vesicle.N * len(K1), 1))
    #         # ytar = torch.tile(ytar, (vesicle.N * len(K1), 1))
            
    #         # broadcasting
    #         diffx = xtar[None, :] - xsou
    #         diffy = ytar[None, :] - ysou

    #         dis2 = diffx**2 + diffy**2

    #         coeff = 0.5 * torch.log(dis2)
    #         stokesSLPtar[:Ntar, k] = -torch.sum(coeff * denx, dim=0)
    #         stokesSLPtar[Ntar:, k] = -torch.sum(coeff * deny, dim=0)

    #         coeff = (diffx * denx + diffy * deny) / dis2
    #         stokesSLPtar[:Ntar, k] += torch.sum(coeff * diffx, dim=0)
    #         stokesSLPtar[Ntar:, k] += torch.sum(coeff * diffy, dim=0)


    #     return stokesSLPtar / (4 * torch.pi)
    
    
    
    # def allExactStokesSL(self, vesicle, f, tarVes=None):
    #     """
    #     Computes the single-layer potential due to `f` around all vesicles except itself.
        
    #     Parameters:
    #     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    #     - f: Forcing term (2*N x nv).

    #     Returns:
    #     - stokesSLPtar: Single-layer potential at target points.
    #     """
        
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     stokesSLPtar = torch.zeros((2 * N, nv), dtype=torch.float32, device=vesicle.X.device)

    #     den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

    #     mask = ~torch.eye(nv).bool()
    #     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    #     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    #     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
        
    #     xsou = vesicle.X[:N, indices].permute(0, 2, 1).reshape(-1, nv)
    #     ysou = vesicle.X[N:, indices].permute(0, 2, 1).reshape(-1, nv)
    #     xsou = torch.tile(xsou, (N, 1, 1)).permute(1,0,2)    # (N*(nv-1), N, nv)
    #     ysou = torch.tile(ysou, (N, 1, 1)).permute(1,0,2)
    #     # xsou = xsou[None,].expand(N, -1, -1).permute(1,0,2)    # (N*(nv-1), N, nv)
    #     # ysou = ysou[None,].expand(N, -1, -1).permute(1,0,2)

    #     denx = den[:N, indices].permute(0, 2, 1).reshape(-1, nv)
    #     deny = den[N:, indices].permute(0, 2, 1).reshape(-1, nv)
    #     denx = torch.tile(denx, (N, 1, 1)).permute(1,0,2)    # (N*(nv-1), N)
    #     deny = torch.tile(deny, (N, 1, 1)).permute(1,0,2)

    #     if tarVes:
    #         xtar = tarVes.X[:tarVes.N]
    #         xtar = tarVes.X[tarVes.N:]
    #     else:
    #         xtar = vesicle.X[:N]
    #         ytar = vesicle.X[N:]
    #     # xtar = torch.tile(xtar, (N * (nv-1), 1, 1))
    #     # ytar = torch.tile(ytar, (N * (nv-1), 1, 1))
            
    #     diffx = xtar - xsou # broadcasting
    #     diffy = ytar - ysou

    #     dis2 = diffx**2 + diffy**2

    #     coeff = 0.5 * torch.log(dis2)
    #     stokesSLPtar[:N, torch.arange(nv)] = -torch.sum(coeff * denx, dim=0)
    #     stokesSLPtar[N:, torch.arange(nv)] = -torch.sum(coeff * deny, dim=0)

    #     coeff = (diffx * denx + diffy * deny) / dis2
    #     stokesSLPtar[:N, torch.arange(nv)] += torch.sum(coeff * diffx, dim=0)
    #     stokesSLPtar[N:, torch.arange(nv)] += torch.sum(coeff * diffy, dim=0)


    #     return stokesSLPtar / (4 * torch.pi)
    
    
    # def allExactStokesSLTarget(self, vesicle, f, tarVes=None):
    #     """
    #     Computes the single-layer potential due to `f` around all vesicles except itself.
        
    #     Parameters:
    #     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    #     - f: Forcing term (2*N x nv).

    #     Returns:
    #     - stokesSLPtar: Single-layer potential at target points.
    #     """
        
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     Ntar = tarVes.N
    #     ntar = tarVes.nv
    #     stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicle.X.device)

    #     den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

    #     mask = ~torch.eye(nv).bool()
    #     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    #     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    #     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
        
    #     xsou = vesicle.X[:N, indices].permute(0, 2, 1).reshape(-1, nv)
    #     ysou = vesicle.X[N:, indices].permute(0, 2, 1).reshape(-1, nv)
    #     xsou = torch.tile(xsou, (Ntar, 1, 1)).permute(1,0,2)    # (N*(nv-1), N, nv)
    #     ysou = torch.tile(ysou, (Ntar, 1, 1)).permute(1,0,2)
    #     # xsou = xsou[None,].expand(N, -1, -1).permute(1,0,2)    # (N*(nv-1), N, nv)
    #     # ysou = ysou[None,].expand(N, -1, -1).permute(1,0,2)

    #     denx = den[:N, indices].permute(0, 2, 1).reshape(-1, nv)
    #     deny = den[N:, indices].permute(0, 2, 1).reshape(-1, nv)
    #     denx = torch.tile(denx, (Ntar, 1, 1)).permute(1,0,2)    # (N*(nv-1), N)
    #     deny = torch.tile(deny, (Ntar, 1, 1)).permute(1,0,2)

    #     if tarVes:
    #         xtar = tarVes.X[:tarVes.N]
    #         ytar = tarVes.X[tarVes.N:]
    #     else:
    #         xtar = vesicle.X[:N]
    #         ytar = vesicle.X[N:]
    #     # xtar = torch.tile(xtar, (N * (nv-1), 1, 1))
    #     # ytar = torch.tile(ytar, (N * (nv-1), 1, 1))
            
    #     diffx = xtar - xsou # broadcasting
    #     diffy = ytar - ysou

    #     dis2 = diffx**2 + diffy**2

    #     coeff = 0.5 * torch.log(dis2)
    #     stokesSLPtar[:Ntar, torch.arange(ntar)] = -torch.sum(coeff * denx, dim=0)
    #     stokesSLPtar[Ntar:, torch.arange(ntar)] = -torch.sum(coeff * deny, dim=0)

    #     coeff = (diffx * denx + diffy * deny) / dis2
    #     stokesSLPtar[:Ntar, torch.arange(ntar)] += torch.sum(coeff * diffx, dim=0)
    #     stokesSLPtar[Ntar:, torch.arange(ntar)] += torch.sum(coeff * diffy, dim=0)


    #     return stokesSLPtar / (4 * torch.pi)
    
    
    # def allExactStokesSLTarget_expand(self, vesicle, f, tarVes=None):
    #     """
    #     Computes the single-layer potential due to `f` around all vesicles except itself.
        
    #     Parameters:
    #     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    #     - f: Forcing term (2*N x nv).

    #     Returns:
    #     - stokesSLPtar: Single-layer potential at target points.
    #     """
        
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     Ntar = tarVes.N
    #     ntar = tarVes.nv
    #     stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicle.X.device)

    #     den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

    #     mask = ~torch.eye(nv).bool()
    #     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    #     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    #     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
        
    #     xsou = vesicle.X[:N, indices].permute(0, 2, 1).reshape(-1, nv)
    #     ysou = vesicle.X[N:, indices].permute(0, 2, 1).reshape(-1, nv)
    #     # xsou = torch.tile(xsou, (Ntar, 1, 1)).permute(1,0,2)    
    #     # ysou = torch.tile(ysou, (Ntar, 1, 1)).permute(1,0,2)
    #     xsou = xsou[None,].expand(Ntar``, -1, -1).permute(1,0,2)    # (N*(nv-1), Ntar, nv)
    #     ysou = ysou[None,].expand(Ntar, -1, -1).permute(1,0,2)

    #     denx = den[:N, indices].permute(0, 2, 1).reshape(-1, nv)
    #     deny = den[N:, indices].permute(0, 2, 1).reshape(-1, nv)
    #     # denx = torch.tile(denx, (Ntar, 1, 1)).permute(1,0,2)   
    #     # deny = torch.tile(deny, (Ntar, 1, 1)).permute(1,0,2)
    #     denx = denx[None,].expand(Ntar, -1, -1).permute(1,0,2)    # (N*(nv-1), Ntar, nv)
    #     deny = deny[None,].expand(Ntar, -1, -1).permute(1,0,2)


    #     if tarVes:
    #         xtar = tarVes.X[:tarVes.N]
    #         ytar = tarVes.X[tarVes.N:]
    #     else:
    #         xtar = vesicle.X[:N]
    #         ytar = vesicle.X[N:]
    #     # xtar = torch.tile(xtar, (N * (nv-1), 1, 1))
    #     # ytar = torch.tile(ytar, (N * (nv-1), 1, 1))
        
    #     print(f"xtar shape {xtar.shape}, xsou shape {xsou.shape}")
    #     diffx = xtar - xsou # broadcasting
    #     diffy = ytar - ysou

    #     dis2 = diffx**2 + diffy**2

    #     coeff = 0.5 * torch.log(dis2)
    #     col_indices = torch.arange(ntar)
    #     stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx, dim=0)
    #     stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny, dim=0)

    #     coeff = (diffx * denx + diffy * deny) / dis2
    #     stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=0)
    #     stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=0)


    #     return stokesSLPtar / (4 * torch.pi)
    
    # @staticmethod
    # @torch.jit.script
    # @staticmethod
    # def allExactStokesSLTarget_broadcast(vesicleX, vesicle_sa, f, tarX, info, dis2, diffx, diffy, full_mask,  offset: int = 0):
    #     # , info, dis2, diffx, diffy, full_mask, 
    #     """
    #     Computes the single-layer potential due to `f` around all vesicles except itself.
        
    #     Parameters:
    #     - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    #     - f: Forcing term (2*N x nv).

    #     Returns:
    #     - stokesSLPtar: Single-layer potential at target points.
    #     """
        
    #     N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
    #     Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
    #     stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float32, device=vesicleX.device)

    #     mask = ~torch.eye(nv, dtype=torch.bool)
    #     # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    #     # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    #     indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
    #     indices = indices[offset:offset+ntar]

    #     den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N
    #     denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
    #     deny = den[N:, indices].permute(0, 2, 1) 

    #     if MLARM_manyfree_py.info is None:
            
    #         xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
    #         ysou = vesicleX[N:, indices].permute(0, 2, 1) 

    #         if tarX is not None:
    #             xtar = tarX[:Ntar]
    #             ytar = tarX[Ntar:]
    #         else:
    #             xtar = vesicleX[:N]
    #             ytar = vesicleX[N:]
            
    #         MLARM_manyfree_py.diffx = xtar[None, None, ...] - xsou[:, :, None] # broadcasting, (N, (nv-1), Ntar, nv)
    #         # del xtar
    #         # del xsou
    #         MLARM_manyfree_py.diffy = ytar[None, None, ...] - ysou[:, :, None]
    #         # del ytar
    #         # del ysou

    #         MLARM_manyfree_py.dis2 = MLARM_manyfree_py.diffx**2 + MLARM_manyfree_py.diffy**2
    #         info = MLARM_manyfree_py.dis2 <= (1/Ntar)**2
    #         # Compute the cell-level mask 
    #         cell_mask = info.any(dim=0)  # Shape: (nv-1, Ntar, ntar)
    #         MLARM_manyfree_py.full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, nv)
    #         MLARM_manyfree_py.info = torch.concat((info, torch.zeros((N, 1, Ntar, ntar), dtype=torch.bool)), dim=1)

    #         # start = torch.cuda.Event(enable_timing=True)
    #         # end = torch.cuda.Event(enable_timing=True)
    #         # start.record()
            
    #         coeff = 0.5 * torch.log(MLARM_manyfree_py.dis2)
    #         coeff.masked_fill_(MLARM_manyfree_py.full_mask, 0)
    #         col_indices = torch.arange(ntar)
    #         stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
    #         stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

    #         coeff = (MLARM_manyfree_py.diffx * denx.unsqueeze(2) + MLARM_manyfree_py.diffy * deny.unsqueeze(2)) / MLARM_manyfree_py.dis2
    #         coeff.masked_fill_(MLARM_manyfree_py.full_mask, 0)
    #         stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * MLARM_manyfree_py.diffx, dim=[0,1])
    #         stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * MLARM_manyfree_py.diffy, dim=[0,1])
        
    #     else:

    #         coeff = 0.5 * torch.log(MLARM_manyfree_py.dis2)
    #         coeff.masked_fill_(MLARM_manyfree_py.full_mask, 0)
    #         col_indices = torch.arange(ntar)
    #         stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
    #         stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

    #         coeff = (MLARM_manyfree_py.diffx * denx.unsqueeze(2) + MLARM_manyfree_py.diffy * deny.unsqueeze(2)) / MLARM_manyfree_py.dis2
    #         coeff.masked_fill_(MLARM_manyfree_py.full_mask, 0)
    #         stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * MLARM_manyfree_py.diffx, dim=[0,1])
    #         stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * MLARM_manyfree_py.diffy, dim=[0,1])
        
    #     # end.record()
    #     # torch.cuda.synchronize()
    #     # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

    #     return stokesSLPtar / (4 * torch.pi)


    # @staticmethod
    
    # @torch.compile(backend='cudagraphs')
    def standardizationStep(self, Xin):
        # compatible with multi ves
        X = Xin.clone()
        N = X.shape[0] // 2
        # % Equally distribute points in arc-length
        modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(X.device) #.double()
        for _ in range(5):
            X, flag = self.oc.redistributeArcLength(X, modes)
            # if flag:
            #     break
        
        # if X.device.type == 'cuda':
        #     start = torch.cuda.Event(enable_timing=True)
        #     end = torch.cuda.Event(enable_timing=True)
        #     start.record()
        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)

        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, (scaling, rotate, rotCenter, trans, multi_sortIdx)

    @torch.compile(backend='cudagraphs')
    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        # compatible with multi ves
        N = len(multi_sortIdx[0])
        nv = X.shape[1]
        # Xrotated = self.rotationOperator(X, rotation, rotCenter)
        # Xrotated = self.translateOp(Xrotated, translation)
        Xrotated = self.rotation_trans_Operator(X, rotation, rotCenter, translation)
        
        XrotSort = torch.vstack((Xrotated[multi_sortIdx.T, torch.arange(nv, device=X.device)], Xrotated[multi_sortIdx.T + N, torch.arange(nv, device=X.device)]))
        
        XrotSort = scaling * XrotSort
        return XrotSort

    @torch.compile(backend='cudagraphs')
    def destandardize(self, XrotSort, standardizationValues: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        ''' compatible with multiple ves'''
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

        # X = self.trans_rotation_Operator(X, -rotate, rotCenter, -trans)

        return X
    
    def referenceValues(self, Xref):
        ''' Shan: compatible with multi ves'''

        oc = self.oc
        N = len(Xref) // 2
        # nv = Xref.shape[1]
        tempX = Xref.clone()

        # Find the physical center
        rotCenter = oc.getPhysicalCenter(tempX)
        multi_V = oc.getPrincAxesGivenCentroid(tempX, rotCenter)
        # w = torch.tensor([0, 1]) # y-dim unit vector
        # rotation = torch.arctan2(w[1]*multi_V[0]-w[0]*multi_V[1], w[0]*multi_V[0]+w[1]*multi_V[1])
        rotation = torch.arctan2(multi_V[0], multi_V[1])
        
        Xref = self.rotationOperator(tempX, rotation, rotCenter)
        center_ = oc.getPhysicalCenter(Xref) # redundant?
        translation = -center_

        # if not torch.allclose(rotCenter, center_, rtol=1e-4):
        #     print(f"center {rotCenter} and center_{center_}")
            # raise "center different"
        
        Xref = self.translateOp(Xref, translation)
        
        # multi_sortIdx = torch.zeros((nv, N), dtype=torch.int32)
        # for k in range(nv):
        #     firstQuad = np.intersect1d(torch.where(Xref[:N,k] >= 0)[0].cpu(), torch.where(Xref[N:,k] >= 0)[0].cpu())
        #     theta = torch.arctan2(Xref[N:,k], Xref[:N,k])
        #     idx = torch.argmin(theta[firstQuad])
        #     sortIdx = torch.concatenate((torch.arange(firstQuad[idx],N), torch.arange(0, firstQuad[idx])))
        #     multi_sortIdx[k] = sortIdx
        
        theta = torch.arctan2(Xref[N:], Xref[:N])
        start_id = torch.argmin(torch.where(theta<0, 100, theta), dim=0)
        multi_sortIdx = (start_id + torch.arange(N, device=Xref.device).unsqueeze(-1)) % N
        multi_sortIdx = multi_sortIdx.int().T

        # if not torch.allclose(multi_sortIdx, multi_sortIdx_):
        #     raise "batch err"

        length = oc.geomProp_length(Xref)
        scaling = 1.0 / length
        
        return translation, rotation, rotCenter, scaling, multi_sortIdx

    
    def rotationOperator(self, X, theta, rotCent):
        ''' Shan: compatible with multi ves
        theta of shape (1,nv), rotCent of shape (2,nv)'''
        Xrot = torch.zeros_like(X)
        x = X[:len(X)//2] - rotCent[0]
        y = X[len(X)//2:] - rotCent[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[:len(X)//2] = xrot + rotCent[0]
        Xrot[len(X)//2:] = yrot + rotCent[1]
        return Xrot

    def translateOp(self, X, transXY):
        ''' Shan: compatible with multi ves
         transXY of shape (2,nv)'''
        Xnew = torch.zeros_like(X)
        Xnew[:len(X)//2] = X[:len(X)//2] + transXY[0]
        Xnew[len(X)//2:] = X[len(X)//2:] + transXY[1]
        return Xnew


    def rotation_trans_Operator(self,  X, theta, rotCent, transXY):
        '''
        combining rotate and trans
        '''

        Xrot = torch.zeros_like(X, device=X.device)
        x = X[:len(X)//2] - rotCent[0]
        y = X[len(X)//2:] - rotCent[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[:len(X)//2] = xrot + rotCent[0] + transXY[0]
        Xrot[len(X)//2:] = yrot + rotCent[1] + transXY[1]
        return Xrot

    def trans_rotation_Operator(self,  X, theta, rotCent, transXY):
        '''
        combining rotate and trans
        '''
        Xrot = torch.zeros_like(X)

        x = X[:len(X)//2] - rotCent[0] - transXY[0]
        y = X[len(X)//2:] - rotCent[1] - transXY[1]

        # Rotated shape
        xrot = x * torch.cos(theta) - y * torch.sin(theta)
        yrot = x * torch.sin(theta) + y * torch.cos(theta)

        Xrot[:len(X)//2] = xrot + rotCent[0] 
        Xrot[len(X)//2:] = yrot + rotCent[1] 
        return Xrot
        
