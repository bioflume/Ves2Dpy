import torch
torch.set_default_dtype(torch.float64)
from filter import interpft_vec
from capsules import capsules
from curve_batch_compile import Curve

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
    if len(vesicle_sa.shape) == 1:
        vesicle_sa = vesicle_sa.unsqueeze(1)
    if len(vesicleX.shape) == 1:
        vesicleX = vesicleX.unsqueeze(1)
    if len(tarX.shape) == 1:
        tarX = tarX.unsqueeze(1)
        
    N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
    Ntar, ntar = tarX.shape[0]//2, tarX.shape[1]
    stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float64, device=vesicleX.device)

    mask = ~torch.eye(nv, dtype=torch.bool)
    # When input is on CUDA, torch.nonzero() causes host-device synchronization.
    # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
    indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
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

    stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(ids_, torch.tensor([0.], device=f.device)), dim=[0,1])
    stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(ids_, torch.tensor([0.], device=f.device)), dim=[0,1])

    # stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx), dim=[0,1])
    # stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny), dim=[0,1])

    # end.record()
    # torch.cuda.synchronize()
    # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

    return stokesSLPtar / (4 * torch.pi), (ids[0], ids[1], ids[2]+offset)


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
    stokesSLPtar = torch.zeros((2 * Ntar, ntar), dtype=torch.float64, device=vesicleX.device)

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

    stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx).index_put_(ids_, torch.tensor([0.], device=f.device)), dim=[0,1])
    stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny).index_put_(ids_, torch.tensor([0.], device=f.device)), dim=[0,1])

    # stokesSLPtar[:Ntar, :] = torch.sum((coeff * diffx - 0.5 * torch.log(dis2) * denx), dim=[0,1])
    # stokesSLPtar[Ntar:, :] = torch.sum((coeff * diffy - 0.5 * torch.log(dis2) * deny), dim=[0,1])

    # end.record()
    # torch.cuda.synchronize()
    # print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')

    return stokesSLPtar / (4 * torch.pi)


def wrapper_allExactStokesSLTarget_compare2(vesicleX, vesicle_sa, fup, tarX, info_stokes):
    nv = tarX.shape[1]
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
                vesicleX, 
                vesicle_sa, 
                fup, 
                tarX[:, start:end], 
                info_stokes[0][mask], 
                info_stokes[1][mask], 
                info_stokes[2][mask] - start, 
                offset=offset
            )
            
            far_fields.append(far_field)

        far_field_1 = torch.concat(far_fields, dim=1)


    elif nv > 504:
        far_field_1 = torch.concat((fn(vesicleX, vesicle_sa,  fup, tarX[:, :nv//4], info_stokes[0][info_stokes[2]<nv//4], info_stokes[1][info_stokes[2]<nv//4], info_stokes[2][info_stokes[2]<nv//4]), 
                                fn(vesicleX, vesicle_sa,  fup, tarX[:, nv//4:nv//2], info_stokes[0][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[1][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)], info_stokes[2][(nv//4<=info_stokes[2]) & (info_stokes[2]<nv//2)] - nv//4, offset=nv//4),
                                fn(vesicleX, vesicle_sa,  fup, tarX[:, nv//2:3*nv//4], info_stokes[0][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[1][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)], info_stokes[2][(nv//2<=info_stokes[2]) & (info_stokes[2]<3*nv//4)] - nv//2, offset=nv//2),
                                fn(vesicleX, vesicle_sa,  fup, tarX[:, 3*nv//4:], info_stokes[0][3*nv//4<=info_stokes[2]], info_stokes[1][3*nv//4<=info_stokes[2]], info_stokes[2][3*nv//4<=info_stokes[2]] - 3*nv//4,  offset=3*nv//4)), dim=1)
       
    elif nv > 100:
        far_field_1 = torch.concat((fn(vesicleX, vesicle_sa, fup, tarX[:, :nv//2], info_stokes[0][info_stokes[2]<nv//2], info_stokes[1][info_stokes[2]<nv//2], info_stokes[2][info_stokes[2]<nv//2]), 
                                fn(vesicleX, vesicle_sa, fup, tarX[:, nv//2:], info_stokes[0][nv//2<=info_stokes[2]], info_stokes[1][nv//2<=info_stokes[2]], info_stokes[2][nv//2<=info_stokes[2]] - nv//2, offset=nv//2)), dim=1)
    else:
         far_field_1 = fn(vesicleX, vesicle_sa, fup, tarX, info_stokes[0], info_stokes[1], info_stokes[2])
    
    return far_field_1


# def exactStokesSL(vesicle, f, Xtar=None, K1=None):
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
#     stokesSLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float64, device=Xtar.device)
    

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
#         # if ncol != 1:
#         #     raise "ncol != 1"
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


def exactStokesSL_(vesicle, f, Xtar=None, K1=None):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.
    Also can pass a set of target points `Xtar` and a collection of vesicles `K1` 
    and the single-layer potential due to vesicles in `K1` will be evaluated at `Xtar`.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).
    - Xtar: Target points (2*Ntar x ncol), optional.
    - K1: Collection of vesicles, optional.

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """
    
    
    Ntar = Xtar.shape[0] // 2
    ncol = Xtar.shape[1]
    stokesSLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float64, device=Xtar.device)
    

    den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

    xsou = vesicle.X[:vesicle.N, K1].flatten()
    ysou = vesicle.X[vesicle.N:, K1].flatten()
    xsou = torch.tile(xsou, (Ntar, 1)).T    # (N*(nv-1), Ntar)
    ysou = torch.tile(ysou, (Ntar, 1)).T

    denx = den[:vesicle.N, K1]
    deny = den[vesicle.N:, K1]
    denx = torch.tile(denx, (Ntar, 1)).T    # (N*(nv-1), Ntar)
    deny = torch.tile(deny, (Ntar, 1)).T

    for k in range(ncol):  # Loop over columns of target points
        # if ncol != 1:
        #     raise "ncol != 1"
        xtar = Xtar[:Ntar, k]
        ytar = Xtar[Ntar:, k]
        xtar = torch.tile(xtar, (vesicle.N * len(K1), 1))
        ytar = torch.tile(ytar, (vesicle.N * len(K1), 1))
        
        diffx = xtar - xsou
        diffy = ytar - ysou

        dis2 = diffx**2 + diffy**2

        coeff = 0.5 * torch.log(dis2)
        stokesSLPtar[:Ntar, k] = -torch.sum(coeff * denx, dim=0)
        stokesSLPtar[Ntar:, k] = -torch.sum(coeff * deny, dim=0)

        coeff = (diffx * denx + diffy * deny) / dis2
        stokesSLPtar[:Ntar, k] += torch.sum(coeff * diffx, dim=0)
        stokesSLPtar[Ntar:, k] += torch.sum(coeff * diffy, dim=0)

    return stokesSLPtar / (4 * torch.pi)



def exactStokesSL_onlyself_old(vesicleUp, fup, X, vself, nlayers=3, upsample=-1):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.
    Also can pass a set of target points `Xtar` and a collection of vesicles `K1` 
    and the single-layer potential due to vesicles in `K1` will be evaluated at `Xtar`.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).
    - Xtar: Target points (2*Ntar x ncol), optional.
    - K1: Collection of vesicles, optional.

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """    
    
    
    N = X.shape[0]//2

    if upsample > 0:
        Nup = N * upsample # is different from vesicleUp.N !!
        Xup = interpft_vec(X, Nup)  # Upsample source points
    else:
        Nup = N
        Xup = X

    # vesicleUp = capsules(Xup, None, None, vesicle.kappa, vesicle.viscCont)
    oc = Curve()
    dlayer = torch.linspace(0, 1/N, nlayers, dtype=torch.float64, device=X.device)
    _, tang = oc.diffProp_jac_tan(Xup)
    rep_nx = tang[Nup:, :, None].expand(-1,-1,nlayers-1)
    rep_ny = -tang[:Nup, :, None].expand(-1,-1,nlayers-1)
    dx =  rep_nx * dlayer[[1,2,3]] # (N, nv, nlayers-1)
    dy =  rep_ny * dlayer[[1,2,3]]
    tracers = torch.permute(
        torch.vstack([torch.repeat_interleave(Xup[:Nup, :, None], nlayers-1, dim=-1) + dx,
                    torch.repeat_interleave(Xup[Nup:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1)) # (2*N, nlayers-1, nv)
    
    vel_layers = torch.zeros_like(tracers)
        
    den = fup * torch.tile(vesicleUp.sa, (2, 1)) * 2 * torch.pi / vesicleUp.N
    den = den[:, None, None] # (2*Nsou, 1, 1, nv)

    xsou = vesicleUp.X[:vesicleUp.N, :] # (Nsou, nv)
    ysou = vesicleUp.X[vesicleUp.N:, :]
    xtar = tracers[:Nup, :, :] # (Ntar, nlayers-1, nv)
    ytar = tracers[Nup:, :, :]
    denx = den[:vesicleUp.N, ...] # (Nsou, 1, 1, nv)
    deny = den[vesicleUp.N:, ...]
    
    diffx = xtar[None, :] - xsou[:, None, None] # (Nsou, Ntar, nlayers-1, nv)
    diffy = ytar[None, :] - ysou[:, None, None]

    dis2 = diffx**2 + diffy**2

    coeff = 0.5 * torch.log(dis2)
    vel_layers[:Nup, :] = -torch.sum(coeff * denx, dim=0)
    vel_layers[Nup:, :] = -torch.sum(coeff * deny, dim=0)

    coeff = (diffx * denx + diffy * deny) / dis2
    vel_layers[:Nup, :] += torch.sum(coeff * diffx, dim=0)
    vel_layers[Nup:, :] += torch.sum(coeff * diffy, dim=0)

    vel_layers /= (4*torch.pi) 

    vel_self_and_layer = torch.concat((vself[:, None], vel_layers), dim=1) # (2*N, nlayers, nv)
    tracers = torch.concat((Xup[:, None], tracers), dim=1)

    return vel_self_and_layer[:Nup], vel_self_and_layer[Nup:], tracers[:Nup], tracers[Nup:]



    
    

@torch.jit.script
def exactStokesSL_onlyself(vesicleUpX, vesicleUp_sa, fup, Nup: int, Xup, vself, tracers):
    """
    Computes the single-layer potential due to `f` around all vesicles except itself.
    Also can pass a set of target points `Xtar` and a collection of vesicles `K1` 
    and the single-layer potential due to vesicles in `K1` will be evaluated at `Xtar`.

    Parameters:
    - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
    - f: Forcing term (2*N x nv).
    - Xtar: Target points (2*Ntar x ncol), optional.
    - K1: Collection of vesicles, optional.

    Returns:
    - stokesSLPtar: Single-layer potential at target points.
    """    
  
    vel_layers = torch.zeros_like(tracers)

    vesicleUp_N = vesicleUpX.shape[0]//2    
    den = fup * torch.tile(vesicleUp_sa, (2, 1)) * 2 * torch.pi / vesicleUp_N
    den = den[:, None, None] # (2*Nsou, 1, 1, nv)

    xsou = vesicleUpX[:vesicleUp_N, :] # (Nsou, nv)
    ysou = vesicleUpX[vesicleUp_N:, :]
    xtar = tracers[:Nup, :, :] # (Ntar, nlayers-1, nv)
    ytar = tracers[Nup:, :, :]
    denx = den[:vesicleUp_N, ...] # (Nsou, 1, 1, nv)
    deny = den[vesicleUp_N:, ...]
    
    diffx = xtar[None, :] - xsou[:, None, None] # (Nsou, Ntar, nlayers-1, nv)
    diffy = ytar[None, :] - ysou[:, None, None]

    dis2 = diffx**2 + diffy**2

    coeff = 0.5 * torch.log(dis2)
    vel_layers[:Nup, :] = -torch.sum(coeff * denx, dim=0)
    vel_layers[Nup:, :] = -torch.sum(coeff * deny, dim=0)

    coeff = (diffx * denx + diffy * deny) / dis2
    vel_layers[:Nup, :] += torch.sum(coeff * diffx, dim=0)
    vel_layers[Nup:, :] += torch.sum(coeff * diffy, dim=0)

    vel_layers /= (4*torch.pi) 

    vel_self_and_layer = torch.concat((vself[:, None], vel_layers), dim=1) # (2*N, nlayers, nv)
    tracers = torch.concat((Xup[:, None], tracers), dim=1)

    return vel_self_and_layer[:Nup], vel_self_and_layer[Nup:], tracers[:Nup], tracers[Nup:]



def naiveNearZoneInfo(vesicleX, vesicleUpX, max_layer_dist=None):
    '''
    Naive way of doing range search by computing distances and creating masks.
    return a boolean nbrs_mask where (i,j)=True means i, j are close and are from different vesicles
    '''
    N, nv = vesicleX.shape[0]//2, vesicleX.shape[1]
    Nup = vesicleUpX.shape[0]//2
    # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
    # max_layer_dist = vesicle.length.item() / vesicle.N
    max_layer_dist = 1./N if max_layer_dist is None else max_layer_dist

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()

    all_points =  torch.concat((vesicleX[:N, :].T.reshape(-1,1), vesicleX[N:, :].T.reshape(-1,1)), dim=1)
    all_points_up =  torch.concat((vesicleUpX[:Nup, :].T.reshape(-1,1), vesicleUpX[Nup:, :].T.reshape(-1,1)), dim=1)

    # if nv < 1600:
        # sq_distances  = torch.sum((all_points.unsqueeze(1) - all_points_up.unsqueeze(0))**2, dim=-1)  
        # sq_distances = torch.norm(all_points.unsqueeze(1) - all_points_up.unsqueeze(0), dim=-1)
    distances = torch.cdist(all_points.unsqueeze(0), all_points_up.unsqueeze(0)).squeeze()
    dist_mask = distances < max_layer_dist
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

    import pdb; pdb.set_trace()
    
    rows_with_true = torch.max(nbrs_mask.reshape(nv*N, nv, Nup), dim=-1)[0] # (N*nv, nv)
    id1, id2 = torch.where(rows_with_true)
    # id1, id2 = rows_with_true.to_sparse().indices() # for rbf solves
    ids1, ids2 = id1 % N, id1 // N
    ids0 = id2 - 1*(ids2 <= id2)
    # if torch.any(ids2 == id2):
    #     raise "unexpcted"

    rows_with_true = torch.max(nbrs_mask.reshape(nv, N, nv * Nup), dim=1)[0] # (nv, nv * Nup)
    id2_hh, id1_hh = torch.where(rows_with_true) # for hedgehog


    # ind of points on vesicle k2 close to vesicle k1 : zone, can be computed by two indices, id1 is k2, id2 is k1
    # ind of closest point on vesicle k1 to each point on vesicle k2 that in k1's nearZone : icp
    # coord and dist of closest point on ves k1 to each point on vesicle k2 that in k1's nearZone: nearest, dist

    # for all points of k2, find the closest point on k1
    closest_dist, idx_cloest = torch.min(distances.reshape(nv, N, nv, Nup), dim=1) # (nv, nv, Nup)


    # three indices points to the shape: (nv-1, Ntar, ntar)     
    # two indices points to the shape: (N*nv, nv)
    return closest_dist, idx_cloest, \
        (id1, id2), \
        (id1_hh, id2_hh), \
        (ids0, ids1, ids2)  # for exactStokes


