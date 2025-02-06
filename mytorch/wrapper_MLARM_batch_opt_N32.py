import numpy as np
import torch
torch.set_default_dtype(torch.float32)
# print(torch.version.cuda)
import sys
sys.path.append("..")
from collections import defaultdict
from capsules import capsules
# from rayCasting import ray_casting
from filter import filterShape, filterTension, interpft
# from scipy.spatial import KDTree
# import faiss
# import cupy as cp
# from scipy.interpolate import RBFInterpolator as scipyinterp_cpu
# from cupyx.scipy.interpolate import RBFInterpolator as scipyinterp_gpu
from model_zoo_N32.get_network_torch_N32 import RelaxNetwork, TenSelfNetwork, MergedAdvNetwork, MergedTenAdvNetwork, MergedNearFourierNetwork, MergedInnerNearFourierNetwork
# from cuda_practice.my_cuda_matvec_numba import block_diag_matvec
# from cuda_practice.cuda_cg import solve_cg, solve_cg_onebyone
# from cuda_practice.minres_my_cuda_matvec_numba import block_diag_matvec
# from cupyx.scipy.sparse.linalg import minres
# from cupyx.scipy.sparse.linalg import LinearOperator
# from numba import cuda, float32
from math import ceil, sqrt
import time
import mat73
import scipy.io as scio


class MLARM_manyfree_py:
    def __init__(self, dt, vinf, oc, use_repulsion, repStrength,
                 advNetInputNorm, advNetOutputNorm,
                 relaxNetInputNorm, relaxNetOutputNorm, 
                 nearNetInputNorm, nearNetOutputNorm, 
                 innerNearNetInputNorm, innerNearNetOutputNorm, 
                 tenSelfNetInputNorm, tenSelfNetOutputNorm,
                 tenAdvNetInputNorm, tenAdvNetOutputNorm, device):
        self.dt = dt  # time step size
        self.vinf = vinf  # background flow (analytic -- itorchut as function of vesicle config)
        self.oc = oc  # curve class
        self.kappa = 1  # bending stiffness is 1 for our simulations
        self.device = device
        # Flag for repulsion
        self.use_repulsion = use_repulsion
        self.repStrength = repStrength
        
        # Normalization values for advection (translation) networks
        self.advNetInputNorm = advNetInputNorm
        self.advNetOutputNorm = advNetOutputNorm
        self.mergedAdvNetwork = MergedAdvNetwork(self.advNetInputNorm.to(device), self.advNetOutputNorm.to(device), 
                                model_path="/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/adv_fft_ds32/2024Oct_ves_merged_adv.pth", 
                                device = device)
        
        # Normalization values for relaxation network
        self.relaxNetInputNorm = relaxNetInputNorm
        self.relaxNetOutputNorm = relaxNetOutputNorm
        self.relaxNetwork = RelaxNetwork(self.dt, self.relaxNetInputNorm.to(device), self.relaxNetOutputNorm.to(device), 
                                model_path="/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/Ves_relax_downsample_DIFF.pth",
                                device = device)
        
        # # Normalization values for near field networks
        self.nearNetInputNorm = nearNetInputNorm
        self.nearNetOutputNorm = nearNetOutputNorm
        self.nearNetwork = MergedNearFourierNetwork(self.nearNetInputNorm.to(device), self.nearNetOutputNorm.to(device),
                                # model_path="../trained/ves_merged_nearFourier.pth",
                                model_path="/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/near_trained/ves_merged_disth_nearFourier.pth",
                                device = device)
        
        # # Normalization values for inner near field networks
        self.innerNearNetInputNorm = innerNearNetInputNorm
        self.innerNearNetOutputNorm = innerNearNetOutputNorm
        self.innerNearNetwork = MergedInnerNearFourierNetwork(self.innerNearNetInputNorm.to(device), self.innerNearNetOutputNorm.to(device),
                                model_path="/work/09452/alberto47/vista/Ves2Dpy/trained/2025ves_merged_disth_innerNearFourier.pth",
                                device = device)
        
        # Normalization values for tension-self network
        self.tenSelfNetInputNorm = tenSelfNetInputNorm
        self.tenSelfNetOutputNorm = tenSelfNetOutputNorm
        self.tenSelfNetwork = TenSelfNetwork(self.tenSelfNetInputNorm.to(device), self.tenSelfNetOutputNorm.to(device), 
                                model_path = "/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/ves_downsample_selften_zerolevel.pth",
                                device = device)
        
        # Normalization values for tension-advection networks
        self.tenAdvNetInputNorm = tenAdvNetInputNorm
        self.tenAdvNetOutputNorm = tenAdvNetOutputNorm
        self.tenAdvNetwork = MergedTenAdvNetwork(self.tenAdvNetInputNorm.to(device), self.tenAdvNetOutputNorm.to(device), 
                                model_path="/work/09452/alberto47/ls6/vesToPY/Ves2Dpy_N32/trained/advten_downsample32/2024Oct_merged_advten.pth", 
                                device = device)
    

    def time_step_many(self, Xold, tenOld):
        # oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]
        Nup = ceil(sqrt(N)) * N
        vesicleUp = capsules(torch.concat((interpft(Xold[:N, ], Nup),interpft(Xold[N:, ], Nup)), dim=0),
                                        [],[], self.kappa, 1)

        # Compute velocity induced by repulsion force
        repForce = torch.zeros_like(Xold)
        # if self.use_repulsion:
        #     repForce = vesicle.repulsionForce(Xold, self.repStrength)

        # Compute bending forces + old tension forces
        fBend = vesicle.bendingTerm(Xold)
        fTen = vesicle.tensionTerm(tenOld)
        tracJump = fBend + fTen  # total elastic force

        Xstand, standardizationValues = self.standardizationStep(Xold)

        # Explicit Tension at the Current Step
        # Calculate velocity induced by vesicles on each other due to elastic force
        # use neural networks to calculate near-singular integrals
        
        velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues)

        
        # info = self.nearZoneInfo(vesicle)
        info = self.naiveNearZoneInfo(vesicle, vesicleUp)

        const = 0.672 
        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
        all_X = all_X /const * N   
        matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) #+ 1e-6 * torch.eye(5*N).unsqueeze(-1) # (3*N, 3*N, nv)
        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))

        
        farFieldtracJump = self.computeStokesInteractions(vesicle, vesicleUp, info, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                        xlayers, ylayers, standardizationValues)
        farFieldtracJump = filterShape(farFieldtracJump, 4)
        
        
        vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)

        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)


        tenNew = -(vBackSolve + selfBendSolve)
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions(vesicle, vesicleUp, info, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, standardizationValues)
        farFieldtracJump = filterShape(farFieldtracJump, 4)

        # Total background velocity
        vbackTotal = vback + farFieldtracJump

        # Compute the action of dt*(1-M) on Xold

        Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)


        Xadv = filterShape(Xadv, 8)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
        # Compute the action of relax operator on Xold + Xadv
        Xnew = self.relaxWTorchNet(Xadv)

        XnewC = Xnew.clone()
        for _ in range(5):
            Xnew, flag = self.oc.redistributeArcLength(Xnew)
            if flag:
                break
        Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))

        Xnew = self.oc.correctAreaAndLength(Xnew, self.area0, self.len0)

        Xnew = filterShape(Xnew.to(Xold.device), 8)

        return Xnew, tenNew
    


    def time_step_many_timing(self, Xold, tenOld):
        # oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]
        Nup = ceil(sqrt(N)) * N
        vesicleUp = capsules(torch.concat((interpft(Xold[:N, ], Nup),interpft(Xold[N:, ], Nup)), dim=0),
                                        [],[], self.kappa, 1)

        # Compute velocity induced by repulsion force
        repForce = torch.zeros_like(Xold)
        # if self.use_repulsion:
        #     repForce = vesicle.repulsionForce(Xold, self.repStrength)

        # Compute bending forces + old tension forces
        fBend = vesicle.bendingTerm(Xold)
        fTen = vesicle.tensionTerm(tenOld)
        tracJump = fBend + fTen  # total elastic force

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
        start.record()
        velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers = self.predictNearLayers(Xstand, standardizationValues)
        end.record()
        torch.cuda.synchronize()
        print(f'predictNearLayers {start.elapsed_time(end)/1000} sec.')
        
        start.record()
        info = self.naiveNearZoneInfo(vesicle, vesicleUp)
        end.record()
        torch.cuda.synchronize()
        print(f'nearZoneInfo {start.elapsed_time(end)/1000} sec.')

        start.record()
        const = 0.672
        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
        all_X = all_X /const * N   
        matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) # (3*N, 3*N, nv)
        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        end.record()
        torch.cuda.synchronize()
        print(f'CHOLESKY {start.elapsed_time(end)/1000} sec.')
        

        start.record()
        farFieldtracJump = self.computeStokesInteractions_timing(vesicle, vesicleUp, info, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                        xlayers, ylayers, standardizationValues)
        end.record()
        torch.cuda.synchronize()
        print(f'x1computeStokesInteractions {start.elapsed_time(end)/1000} sec.')

        farFieldtracJump = filterShape(farFieldtracJump, 4)
        
        start.record()
        vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
        end.record()
        torch.cuda.synchronize()
        print(f'invTenMatOnVback {start.elapsed_time(end)/1000} sec.')

        start.record()
        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
        end.record()
        torch.cuda.synchronize()
        print(f'invTenMatOnSelfBend {start.elapsed_time(end)/1000} sec.')

        tenNew = -(vBackSolve + selfBendSolve)
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions_timing(vesicle, vesicleUp, info, L, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, standardizationValues)
        farFieldtracJump = filterShape(farFieldtracJump, 4)

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

        XnewC = Xnew.clone()
        start.record()
        for _ in range(5):
            Xnew, flag = self.oc.redistributeArcLength(Xnew)
            if flag:
                break
        Xnew = self.oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))
        end.record()
        torch.cuda.synchronize()
        print(f'x5 redistributeArcLength {start.elapsed_time(end)/1000} sec.')

        start.record()
        Xnew = self.oc.correctAreaAndLength(Xnew, self.area0, self.len0)
        end.record()
        torch.cuda.synchronize()
        print(f'correctAreaLength {start.elapsed_time(end)/1000} sec.')

        Xnew = filterShape(Xnew.to(Xold.device), 8)

        return Xnew, tenNew
    



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
    
    
    # @torch.jit.script
    def predictNearLayers(self, Xstand, standardizationValues):
        print('Near network predicting')
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]

        oc = self.oc

        # maxLayerDist = np.sqrt(1 / N) 
        maxLayerDist = (1 / N) # length = 1, h = 1/N;
        nlayers = 5 # three layers
        dlayer = torch.linspace(-maxLayerDist, maxLayerDist, nlayers, dtype=torch.float32)

        # Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)
        
        # Create the layers around a vesicle on which velocity calculated
        tracersX_ = torch.zeros((2 * N, nlayers, nv), dtype=torch.float32)
        tracersX_[:, 2] = Xstand
        _, tang, _ = oc.diffProp(Xstand)
        rep_nx = torch.repeat_interleave(tang[N:, :, None], nlayers-1, dim=-1) 
        rep_ny = torch.repeat_interleave(-tang[:N, :, None], nlayers-1, dim=-1)
        dx =  rep_nx * dlayer[[0,1,3,4]] # (N, nv, nlayers-1)
        dy =  rep_ny * dlayer[[0,1,3,4]]
        tracersX_[:, [0,1,3,4]] = torch.permute(
            torch.vstack([torch.repeat_interleave(Xstand[:N, :, None], nlayers-1, dim=-1) + dx,
                        torch.repeat_interleave(Xstand[N:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1))
        
        
        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(net_pred)

        inner_input_net = self.innerNearNetwork.preProcess(Xstand)
        inner_net_pred = self.innerNearNetwork.forward(inner_input_net)
        inner_velx_real, inner_vely_real, inner_velx_imag, inner_vely_imag = self.innerNearNetwork.postProcess(inner_net_pred)

        velx_real = torch.concat((torch.concat((inner_velx_real, torch.zeros(nv, 32, 1, 2)), dim=-2), velx_real), dim=-1)
        vely_real = torch.concat((torch.concat((inner_vely_real, torch.zeros(nv, 32, 1, 2)), dim=-2), vely_real), dim=-1)
        velx_imag = torch.concat((torch.concat((inner_velx_imag, torch.zeros(nv, 32, 1, 2)), dim=-2), velx_imag), dim=-1)
        vely_imag = torch.concat((torch.concat((inner_vely_imag, torch.zeros(nv, 32, 1, 2)), dim=-2), vely_imag), dim=-1)
        
        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues
        xlayers_ = torch.zeros((N, nlayers, nv), dtype=torch.float32)
        ylayers_ = torch.zeros((N, nlayers, nv), dtype=torch.float32)
        Xl_ = self.destandardize(tracersX_.reshape(N*2, -1),  
            (scaling.tile((nlayers)), rotate.tile((nlayers)), rotCenter.tile((1,nlayers)), trans.tile((1,nlayers)), sortIdx.tile((nlayers,1))))
        xlayers_ = Xl_[:N, np.arange(nlayers * nv).reshape(nlayers, nv)]
        ylayers_ = Xl_[N:, np.arange(nlayers * nv).reshape(nlayers, nv)]

        # if not torch.allclose(xlayers, xlayers_):
        #     raise "batch err"
        # np.save("linshi_xlayers.npy", xlayers_.numpy())
        # np.save("linshi_ylayers.npy", ylayers_.numpy())
        # if not torch.allclose(ylayers, ylayers_):
        #     raise "batch err"

        return velx_real, vely_real, velx_imag, vely_imag, xlayers_, ylayers_


    # @torch.jit.script
    def buildVelocityInNear(self, tracJump, velx_real, vely_real, velx_imag, vely_imag, standardizationValues):
        
        nv = tracJump.shape[1]
        N = tracJump.shape[0]//2
        nlayers = 5
        _, rotate, _, _, sortIdx = standardizationValues

        fstand = self.standardize(tracJump, torch.zeros((2,nv), dtype=torch.float32), rotate, torch.zeros((2,nv), dtype=torch.float32), 1, sortIdx)
        z = fstand[:N] + 1j * fstand[N:]
        zh = torch.fft.fft(z, dim=0)
        fstandRe = torch.real(zh)
        fstandIm = torch.imag(zh)
        
        velx_stand_ = torch.einsum('vnml, mv -> nvl', velx_real, fstandRe) + torch.einsum('vnml, mv -> nvl', velx_imag, fstandIm)
        vely_stand_ = torch.einsum('vnml, mv -> nvl', vely_real, fstandRe) + torch.einsum('vnml, mv -> nvl', vely_imag, fstandIm)
        
        vx_ = torch.zeros((nv, nlayers, N), dtype=torch.float32)
        vy_ = torch.zeros((nv, nlayers, N), dtype=torch.float32)
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
    
    def naiveNearZoneInfo(self, vesicle, vesicleUp):
        '''
        Naive way of doing range search by computing distances and creating masks.
        return a boolean nbrs_mask where (i,j)=True means i, j are close and are from different vesicles
        '''
        N = vesicle.N
        nv = vesicle.nv
        Nup = vesicleUp.N
        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
        max_layer_dist = vesicle.length.item() / vesicle.N

        all_points =  torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1)
        all_points_up =  torch.concat((vesicleUp.X[:Nup, :].T.reshape(-1,1), vesicleUp.X[Nup:, :].T.reshape(-1,1)), dim=1)

        if nv < 600:
            sq_distances  = torch.sum((all_points.unsqueeze(1) - all_points_up.unsqueeze(0))**2, dim=-1)  
            dist_mask = sq_distances <= max_layer_dist**2

            # sq_distances_  = torch.sum((all_points.half().unsqueeze(1) - all_points_up.half().unsqueeze(0))**2, dim=-1)  # Shape: (N, Nup)     
            # dist_mask_ = sq_distances_ <= max_layer_dist**2

            # if not torch.allclose(dist_mask, dist_mask_):
            #     raise "dist_mask err"

        else:
            len0 = all_points.shape[0]
            sq_distances  = torch.sum((all_points[:len0//2].unsqueeze(1) - all_points_up.unsqueeze(0))**2, dim=-1)  
            dist_mask1 = sq_distances <= max_layer_dist**2
            sq_distances  = torch.sum((all_points[len0//2:].unsqueeze(1) - all_points_up.unsqueeze(0))**2, dim=-1)  
            dist_mask2 = sq_distances <= max_layer_dist**2
            dist_mask = torch.cat((dist_mask1, dist_mask2), dim=0)

        # if not torch.allclose(dist_mask, dist_mask_):
        #     raise "dist_mask err"   

        id_mask = torch.ones((N*nv, Nup*nv), dtype=torch.bool)  # Initialize all True
        
        indices = torch.arange(0, N*nv).reshape(nv, N)
        indices_up = torch.arange(0, Nup*nv).reshape(nv, Nup)
        # Use advanced indexing to set blocks to False
        row_indices = indices.unsqueeze(2)  # Shape: (num_cells, points_per_cell, 1)
        col_indices = indices_up.unsqueeze(1)  # Shape: (num_cells, 1, points_per_cell)
        id_mask[row_indices, col_indices] = False

        nbrs_mask = torch.logical_and(dist_mask, id_mask)

        return nbrs_mask
    
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
    

    def computeStokesInteractions_timing(self, vesicle, vesicleUp, info, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
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
        totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if nv > 504:
            far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
                                    self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
                                    self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
                                    self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
            # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        else:
            far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)

        end.record()
        torch.cuda.synchronize()
        print(f'computeStokesInteractions EXACT {start.elapsed_time(end)/1000} sec.')

        # if not torch.allclose(far_field, far_field_):
        #     print(torch.norm(far_field - far_field_)/torch.norm(far_field))
        #     raise "haha"
        
        # if not torch.allclose(far_field, far_field2):
        #     print(torch.norm(far_field - far_field2)/torch.norm(far_field))
        #     raise "haha"
            
        
        start.record()
        self.nearFieldCorrectionUP_SOLVE_timing(vesicle, vesicleUp, info, L, far_field, velx, vely, xlayers, ylayers)
        end.record()
        torch.cuda.synchronize()
        print(f'x1 nearFieldCorrection SOLVE {start.elapsed_time(end)/1000} sec.')

        # if not torch.allclose(far_field, far_field2):
        #     print(f"farfield suanliangci {torch.norm(far_field - far_field2)/torch.norm(far_field)}")
            # raise "haha second"

        return far_field #+ torch.concat((rep_velx[:,0], rep_vely[:,0]), dim=0)
    
    
    def computeStokesInteractions(self, vesicle, vesicleUp, info, L, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
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
        totalForceUp = torch.concat((interpft(totalForce[:N], Nup),interpft(totalForce[N:], Nup)), dim=0)

        if nv > 504:
            far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, :nv//4]), 
                                    self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//4:nv//2], offset=nv//4),
                                    self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, nv//2:3*nv//4], offset=nv//2),
                                    self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa,  totalForceUp, vesicle.X[:, 3*nv//4:], offset=3*nv//4)), dim=1)
            # far_field = torch.concat((self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, :nv//3]), 
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, nv//3:2*nv//3], offset=nv//3),
            #                         self.allExactStokesSLTarget_broadcast(vesicleUp, totalForceUp, vesicle.X[:, 2*nv//3:], offset=2*nv//3)), dim=1)
        else:
            far_field = self.allExactStokesSLTarget_broadcast(vesicleUp.X, vesicleUp.sa, totalForceUp, vesicle.X)


        # if not torch.allclose(far_field, far_field_):
        #     print(torch.norm(far_field - far_field_)/torch.norm(far_field))
        #     raise "haha"
        
        # if not torch.allclose(far_field, far_field2):
        #     print(torch.norm(far_field - far_field2)/torch.norm(far_field))
        #     raise "haha"
            
        
        
        self.nearFieldCorrectionUP_SOLVE(vesicle, vesicleUp, info, L, far_field, velx, vely, xlayers, ylayers)

        # if not torch.allclose(far_field, far_field2):
        #     print(f"farfield suanliangci {torch.norm(far_field - far_field2)/torch.norm(far_field)}")
            # raise "haha second"

        return far_field #+ torch.concat((rep_velx[:,0], rep_vely[:,0]), dim=0)
    
    
    
    def nearFieldCorrectionUP_SOLVE_timing(self, vesicle, vesicleUp, info, L, far_field, velx, vely, xlayers, ylayers):
        N = vesicle.N
        nv = vesicle.nv
        Nup = vesicleUp.N
        
        nbrs_mask = info
        nbrs_mask_reshaped = nbrs_mask.reshape(N*nv, nv, Nup)
        rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1) # (N*nv, nv)
        if not torch.any(rows_with_true):
            return
        

        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1)
        correction = torch.zeros((N*nv, 2), dtype=torch.float32, device=far_field.device)
        
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


        start.record()

        r2 = torch.sum(torch.square(all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...]), dim=-2)
        r2.masked_fill_(~rows_with_true.unsqueeze(1), torch.inf)
        matrix = torch.exp(- 1 * r2) 
        end.record()
        torch.cuda.synchronize()
        t1 = start.elapsed_time(end)/1000

        start.record()
        correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs
        end.record()
        torch.cuda.synchronize()
        t2 = start.elapsed_time(end)/1000
        print(f'---- nearFieldCorrection masking {t1} and {t2} sec.')

        
        correction = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)
        far_field += correction
        return 


    def nearFieldCorrectionUP_SOLVE(self, vesicle, vesicleUp, info, L, far_field, velx, vely, xlayers, ylayers):
        N = vesicle.N
        nv = vesicle.nv
        Nup = vesicleUp.N
        
        nbrs_mask = info

        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1)
        # correction = torch.zeros((N*nv, 2), dtype=torch.float32, device=trac_jump.device)
        
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


        nbrs_mask_reshaped = nbrs_mask.reshape(N*nv, nv, Nup)
        rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1).unsqueeze(1) # (N*nv, 1, nv)

        r2 = torch.sum((all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...])**2, dim=-2) # (N*nv, N*nlayers, nv)
        r2.masked_fill_(~rows_with_true, torch.inf)
        matrix = torch.exp(- 1 * r2)   
        correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs

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




    def get_ns_coords(Xin, query):
        """
        Find n,s coords for scattered query.

        Args:
            Xin: (2, N)
            query: (2, M)

        Returns:
            torch.Tensor: Interpolated values at query points of shape (M,).
        """
        N = Xin.shape[-1] // 3  # 3 layers
        M = query.shape[-1]

        dist_sq = torch.sum((Xin[:, :, None] - query[:, None, :])**2, dim=0)  # (N, M)
        
        # Find the 4 nearest neighbors
        n_points = 4
        _, topk_indices = torch.topk(-dist_sq, n_points, dim=0)  # (4, M)

        # infer n-s indices from topk_indices
        # s1, s2 = torch.unique_consecutive(topk_indices % N)[:2] # cannot use dim for our purpose
        # n1, n2 = torch.unique_consecutive(topk_indices // N)[:2]         
        # s1, s2 = torch.sort(topk_indices % 128, dim=0)[0][[0, -1]]   
        # n1, n2 = torch.sort(topk_indices // 128, dim=0)[0][[0, -1]]   
        s1 = (topk_indices % N)[0, :]
        condition = (topk_indices % N) != s1
        s2 = (topk_indices % N)[torch.argmax(condition.int(), dim=0), torch.arange(M)]

        n1 = (topk_indices // N)[0, :]
        condition = (topk_indices // N) != n1
        n2 = (topk_indices // N)[torch.argmax(condition.int(), dim=0), torch.arange(M)]

        p1_id, p2_id, p3_id, p4_id = s1 + n1 * N, s2 + n1*N, s1 + n2*N, s2 + n2*N
        # print(f"top_indices are {topk_indices.squeeze()}, infer box is {(p1_id, p2_id, p3_id, p4_id)}")

        
        p1_x, p2_x, p3_x = Xin[0, p1_id], Xin[0, p2_id], Xin[0, p3_id]
        p1_y, p2_y, p3_y = Xin[1, p1_id], Xin[1, p2_id], Xin[1, p3_id]
        s_query = s1 + ((query[0] - p1_x)*(p2_x - p1_x) + (query[1] - p1_y)*(p2_y - p1_y))/((p2_x - p1_x)**2 + (p2_y - p1_y)**2) * (s2 - s1)
        n_query = n1 + ((query[0] - p1_x)*(p3_x - p1_x) + (query[1] - p1_y)*(p3_y - p1_y))/((p3_x - p1_x)**2 + (p3_y - p1_y)**2) * (n2 - n1)

        # print(f"known n-s indices are {(s1, s2, n1, n2)}, query is {s_query}, {n_query}")
        return s_query, n_query


    def nearFieldCorrectionUP_ns_SOLVE(self, vesicle, vesicleUp, info, far_field, L, velx, vely, xlayers, ylayers):
        N = vesicle.N
        nv = vesicle.nv
        Nup = vesicleUp.N
        
        nbrs_mask = info

        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1)
        
        
        const = 1.69 # 1.2 * sqrt(2)

        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
        # all_X = all_X /const * N   

        rhs = torch.concat((velx.reshape(-1,1,nv), vely.reshape(-1,1,nv)), dim=1) # (3 * N), 2, nv), 2 for x and y

        tStart = time.time()
        # L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        y = torch.linalg.solve_triangular(L, rhs.permute(2, 0, 1), upper=False)
        coeffs = torch.linalg.solve_triangular(L.permute(0, 2, 1), y, upper=True)
        tEnd = time.time()
        print(f'x1 nearFieldCorrection ns CHOLESKY {tEnd - tStart} sec.')

        # if not torch.allclose(coeffs, coeffs_):
        #     print(f"coeffs solve vs cholesky {torch.norm(coeffs - coeffs_)/torch.norm(coeffs)}")

        tStart = time.time()
        nbrs_mask_reshaped = nbrs_mask.reshape(N*nv, nv, Nup)
        rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1).unsqueeze(1) # (N*nv, nv)

        r2 = torch.sum((all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...])**2, dim=-2)
        r2.masked_fill_(~rows_with_true, torch.inf)
        matrix = torch.exp(- 1 * r2) 
        correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs

        tEnd = time.time()
        print(f'x1 nearFieldCorrection masking {tEnd - tStart} sec.')

        # if not torch.allclose(correction, correction_):
        #     print(f"correction suanliangci {torch.norm(correction - correction_)/torch.norm(correction)}")
        #     # raise "haha"
        
        correction = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)
        far_field += correction
        return 

    
    
    def translateVinfwTorch(self, Xold, Xstand, standardizationValues, vinf):
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]
        
        # Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(Xold)
        _, rotate, _, _, sortIdx = standardizationValues

        Xpredict = self.mergedAdvNetwork.forward(Xstand.to(self.device))
        
        Z11r_ = torch.zeros((N, N, nv), dtype=torch.float32)
        Z12r_ = torch.zeros_like(Z11r_)
        Z21r_ = torch.zeros_like(Z11r_)
        Z22r_ = torch.zeros_like(Z11r_)

        Z11r_[:, 1:] = torch.permute(Xpredict[:, :, 0, :N], (2, 0, 1))
        Z21r_[:, 1:] = torch.permute(Xpredict[:, :, 0, N:], (2, 0, 1))
        Z12r_[:, 1:] = torch.permute(Xpredict[:, :, 1, :N], (2, 0, 1))
        Z22r_[:, 1:] = torch.permute(Xpredict[:, :, 1, N:], (2, 0, 1))

        # Take fft of the velocity (should be standardized velocity)
        # only sort points and rotate to pi/2 (no translation, no scaling)
        vinf_stand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float32), rotate, torch.zeros((2,nv), dtype=torch.float32), 1, sortIdx)
        z = vinf_stand[:N] + 1j * vinf_stand[N:]
        zh = torch.fft.fft(z, dim=0)
        V1, V2 = torch.real(zh), torch.imag(zh)
        MVinf_stand = torch.vstack((torch.einsum('NiB,iB ->NB', Z11r_, V1) + torch.einsum('NiB,iB ->NB', Z12r_, V2),
                               torch.einsum('NiB,iB ->NB', Z21r_, V1) + torch.einsum('NiB,iB ->NB', Z22r_, V2)))
        
        Xnew = torch.zeros_like(Xold)
        MVinf = torch.zeros_like(MVinf_stand)
        idx = torch.vstack([sortIdx.T, sortIdx.T + N])
        MVinf[idx, torch.arange(nv)] = MVinf_stand
        MVinf = self.rotationOperator(MVinf, -rotate, torch.zeros((2, nv), dtype=torch.float32))
        Xnew = Xold + self.dt * vinf - self.dt * MVinf
        
        return Xnew

    def relaxWTorchNet(self, Xmid):
        # RELAXATION w/ NETWORK
        Xin, standardizationValues = self.standardizationStep(Xmid)

        Xpred = self.relaxNetwork.forward(Xin)
        Xnew = self.destandardize(Xpred, standardizationValues)

        return Xnew

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
        out = self.tenAdvNetwork.postProcess(Xpredict) # shape: (127, nv, 2, 128)

        # Approximate the multiplication Z = inv(DivGT)DivPhi_k
        Z1 = torch.zeros((N, N, nv), dtype=torch.float32)
        Z2 = torch.zeros((N, N, nv), dtype=torch.float32)

        Z1[:, 1:] = torch.permute(out[:, :, 0], (2,0,1))
        Z2[:, 1:] = torch.permute(out[:, :, 1], (2,0,1))

        vBackSolve = torch.zeros((N, nv), dtype=torch.float32)
        vinfStand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float32), rotate, torch.zeros((2,nv), dtype=torch.float32), 1, sortIdx)
        z = vinfStand[:N] + 1j * vinfStand[N:]
        zh = torch.fft.fft(z, dim=0)
        
        V1_ = torch.real(zh)
        V2_ = torch.imag(zh)
        # Compute the approximation to inv(Div*G*Ten)*Div*vExt
        MVinfStand = torch.einsum('NiB,iB ->NB', Z1, V1_) + torch.einsum('NiB,iB ->NB', Z2, V2_)
                               
        # Destandardize the multiplication
        vBackSolve[sortIdx.T, torch.arange(nv)] = MVinfStand

        return vBackSolve

    def invTenMatOnSelfBend(self, Xstand, standardizationValues):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        nv = Xstand.shape[1] # number of vesicles
        N = Xstand.shape[0] // 2

        # Xstand, scaling, _, _, _, sortIdx = self.standardizationStep(X)
        scaling, _, _, _, sortIdx = standardizationValues

        tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        tenPredictStand = tenPredictStand #.double()
        tenPred = torch.zeros((N, nv), dtype=torch.float32)
        
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
    #     xsou = xsou[None,].expand(Ntar, -1, -1).permute(1,0,2)    # (N*(nv-1), Ntar, nv)
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
    
    def allExactStokesSLTarget_broadcast(self, vesicleX, vesicle_sa, f, tarX=None, offset: int = 0):
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

        den = f * torch.tile(vesicle_sa, (2, 1)) * 2 * torch.pi / N

        mask = ~torch.eye(nv, dtype=torch.bool)
        # When input is on CUDA, torch.nonzero() causes host-device synchronization.
        # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
        indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
        indices = indices[offset:offset+ntar]
        
        xsou = vesicleX[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
        ysou = vesicleX[N:, indices].permute(0, 2, 1) 

        denx = den[:N, indices].permute(0, 2, 1)  # (N, (nv-1), Ntar, nv)
        deny = den[N:, indices].permute(0, 2, 1) 

        if tarX is not None:
            xtar = tarX[:Ntar]
            ytar = tarX[Ntar:]
        else:
            xtar = vesicleX[:N]
            ytar = vesicleX[N:]
        
        diffx = xtar[None, None, ...] - xsou[:, :, None] # broadcasting, (N, (nv-1), Ntar, nv)
        del xtar
        del xsou
        diffy = ytar[None, None, ...] - ysou[:, :, None]
        del ytar
        del ysou

        dis2 = diffx**2 + diffy**2
        # Compute the cell-level mask 
        cell_mask = (dis2 <= (1/Ntar)**2).any(dim=0)  # Shape: (nv-1, Ntar, ntar)
        full_mask = cell_mask.unsqueeze(0)  # Shape: (1, nv-1, Ntar, nv)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        coeff = 0.5 * torch.log(dis2)
        coeff.masked_fill_(full_mask, 0)
        col_indices = torch.arange(ntar)
        stokesSLPtar[:Ntar, col_indices] = -torch.sum(coeff * denx.unsqueeze(2), dim=[0, 1])
        stokesSLPtar[Ntar:, col_indices] = -torch.sum(coeff * deny.unsqueeze(2), dim=[0, 1])

        coeff = (diffx * denx.unsqueeze(2) + diffy * deny.unsqueeze(2)) / dis2
        coeff.masked_fill_(full_mask, 0)
        stokesSLPtar[:Ntar, col_indices] += torch.sum(coeff * diffx, dim=[0,1])
        stokesSLPtar[Ntar:, col_indices] += torch.sum(coeff * diffy, dim=[0,1])

        end.record()
        torch.cuda.synchronize()
        print(f'inside ExactStokesSL, last two steps {start.elapsed_time(end)/1000} sec.')


        return stokesSLPtar / (4 * torch.pi)



    def standardizationStep(self, Xin):
        # compatible with multi ves
        X = Xin.clone()
        # % Equally distribute points in arc-length
        for _ in range(5):
            X, flag = self.oc.redistributeArcLength(X)
            if flag:
                break
            
        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)
        
        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, (scaling, rotate, rotCenter, trans, multi_sortIdx)

    
    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        # compatible with multi ves
        N = len(multi_sortIdx[0])
        nv = X.shape[1]
        Xrotated = self.rotationOperator(X, rotation, rotCenter)
        Xrotated = self.translateOp(Xrotated, translation)
        
        XrotSort = torch.vstack((Xrotated[multi_sortIdx.T, torch.arange(nv)], Xrotated[multi_sortIdx.T + N, torch.arange(nv)]))
        
        XrotSort = scaling * XrotSort
        return XrotSort


    def destandardize(self, XrotSort, standardizationValues):
        ''' compatible with multiple ves'''
        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues
        
        N = len(sortIdx[0])
        nv = XrotSort.shape[1]

        # Scale back
        XrotSort = XrotSort / scaling

        # Change ordering back
        X = torch.zeros_like(XrotSort)
        X[sortIdx.T, torch.arange(nv)] = XrotSort[:N]
        X[sortIdx.T + N, torch.arange(nv)] = XrotSort[N:]

        # Take translation back
        X = self.translateOp(X, -trans)

        # Take rotation back
        X = self.rotationOperator(X, -rotate, rotCenter)

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
        w = torch.tensor([0, 1]) # y-dim unit vector
        rotation = torch.arctan2(w[1]*multi_V[0]-w[0]*multi_V[1], w[0]*multi_V[0]+w[1]*multi_V[1])
        
        Xref = self.rotationOperator(tempX, rotation, rotCenter)
        center_ = oc.getPhysicalCenter(Xref) # redundant?
        translation = -center_

        if not torch.allclose(rotCenter, center_):
            print(f"center {rotCenter} and center_{center_}")
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
        multi_sortIdx = (start_id + torch.arange(N).unsqueeze(-1)) % N
        multi_sortIdx = multi_sortIdx.int().T

        # if not torch.allclose(multi_sortIdx, multi_sortIdx_):
        #     raise "batch err"

        _, _, length = oc.geomProp(Xref)
        scaling = 1 / length
        
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


