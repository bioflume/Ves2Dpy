import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import sys
sys.path.append("..")
from collections import defaultdict
from capsules import capsules
# from rayCasting import ray_casting
from filter import filterShape, filterTension
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator as scipyinterp
from model_zoo.get_network_torch import RelaxNetwork, TenSelfNetwork, MergedAdvNetwork, MergedTenAdvNetwork, MergedNearFourierNetwork
import time
import mat73
import scipy.io as scio

class MLARM_manyfree_py:
    def __init__(self, dt, vinf, oc, advNetInputNorm, advNetOutputNorm,
                 relaxNetInputNorm, relaxNetOutputNorm, 
                 nearNetInputNorm, nearNetOutputNorm, 
                 tenSelfNetInputNorm, tenSelfNetOutputNorm,
                 tenAdvNetInputNorm, tenAdvNetOutputNorm, device):
        self.dt = dt  # time step size
        self.vinf = vinf  # background flow (analytic -- itorchut as function of vesicle config)
        self.oc = oc  # curve class
        self.kappa = 1  # bending stiffness is 1 for our simulations
        self.device = device
        
        # Normalization values for advection (translation) networks
        self.advNetInputNorm = advNetInputNorm
        self.advNetOutputNorm = advNetOutputNorm
        self.mergedAdvNetwork = MergedAdvNetwork(self.advNetInputNorm.to(device), self.advNetOutputNorm.to(device), 
                                model_path="../trained/2024Oct_ves_merged_adv.pth", 
                                device = device)
        
        # Normalization values for relaxation network
        self.relaxNetInputNorm = relaxNetInputNorm
        self.relaxNetOutputNorm = relaxNetOutputNorm
        self.relaxNetwork = RelaxNetwork(self.dt, self.relaxNetInputNorm.to(device), self.relaxNetOutputNorm.to(device), 
                                model_path="../trained/ves_relax_DIFF_June8_625k_dt1e-5.pth", 
                                device = device)
        
        # Normalization values for near field networks
        self.nearNetInputNorm = nearNetInputNorm
        self.nearNetOutputNorm = nearNetOutputNorm
        self.nearNetwork = MergedNearFourierNetwork(self.nearNetInputNorm.to(device), self.nearNetOutputNorm.to(device),
                                # model_path="../trained/ves_merged_nearFourier.pth",
                                model_path="../trained/ves_merged_disth_nearFourier.pth",
                                device = device)
        
        # Normalization values for tension-self network
        self.tenSelfNetInputNorm = tenSelfNetInputNorm
        self.tenSelfNetOutputNorm = tenSelfNetOutputNorm
        self.tenSelfNetwork = TenSelfNetwork(self.tenSelfNetInputNorm.to(device), self.tenSelfNetOutputNorm.to(device), 
                                model_path = "../trained/Ves_2024Oct_selften_12blks_loss_0.00566cuda1.pth",
                                # model_path="../trained/ves_selften_new.pth", #"../trained/ves_selften.pth", 
                                device = device)
        
        # Normalization values for tension-advection networks
        self.tenAdvNetInputNorm = tenAdvNetInputNorm
        self.tenAdvNetOutputNorm = tenAdvNetOutputNorm
        self.tenAdvNetwork = MergedTenAdvNetwork(self.tenAdvNetInputNorm.to(device), self.tenAdvNetOutputNorm.to(device), 
                                model_path="../trained/ves_merged_advten.pth", 
                                device = device)
    
    def time_step_many(self, Xold, tenOld):
        oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)

        # Compute bending forces + old tension forces
        fBend = vesicle.bendingTerm(Xold)
        fTen = vesicle.tensionTerm(tenOld)
        tracJump = fBend + fTen  # total elastic force

        # Explicit Tension at the Current Step
        # Calculate velocity induced by vesicles on each other due to elastic force
        # use neural networks to calculate near-singular integrals
        velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear, rotateNear,\
            rotCentNear, scalingNear, sortIdxNear = self.predictNearLayers(Xold)
        
        farFieldtracJump = self.computeStokesInteractions(vesicle, tracJump, velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear,\
            rotateNear, rotCentNear, scalingNear, sortIdxNear)

        # Solve for tension
        vBackSolve = self.invTenMatOnVback(Xold, vback + farFieldtracJump)
        selfBendSolve = self.invTenMatOnSelfBend(Xold)
        tenNew = -(vBackSolve + selfBendSolve)
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions(vesicle, tracJump, velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, transNear,\
            rotateNear, rotCentNear, scalingNear, sortIdxNear)

        # Total background velocity
        vbackTotal = vback + farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        Xadv = self.translateVinfwTorch(Xold, vbackTotal)
        Xadv = filterShape(Xadv, 16)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
        # Compute the action of relax operator on Xold + Xadv
        Xnew = self.relaxWTorchNet(Xadv)
        Xnew = filterShape(Xnew, 16)
        for _ in range(5):
            Xnew, flag = oc.redistributeArcLength(Xnew)
            if flag:
                break
        XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
        Xnew = oc.alignCenterAngle(Xnew, XnewC.to(Xold.device))
            
        return Xnew, tenNew
    
    def time_step_single(self, Xold):
        
        # % take a time step with neural networks
        oc = self.oc
        # background velocity on vesicles
        vback = torch.from_numpy(self.vinf(Xold))
        N = Xold.shape[0]//2

        # Compute the action of dt*(1-M) on Xold
        # tStart = time.time()
        Xadv = self.translateVinfwTorch(Xold, vback)
        # tEnd = time.time()
        # print(f'Solving ADV takes {tEnd - tStart} sec.')

        # Correct area and length
        # tStart = time.time()
        # Xadv = upsThenFilterShape(Xadv, 4*N, 16)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC)
        # tEnd = time.time()
        # print(f'Solving correction takes {tEnd - tStart} sec.')

        # Compute the action of relax operator on Xold + Xadv
        # tStart = time.time()
        Xnew = self.relaxWTorchNet(Xadv)
        # tEnd = time.time()
        # print(f'Solving RELAX takes {tEnd - tStart} sec.')

        # Correct area and length
        # tStart = time.time()
        # Xnew = upsThenFilterShape(Xnew, 4*N, 16)
        for _ in range(5):
            Xnew, flag = oc.redistributeArcLength(Xnew)
            if flag:
                break
        XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
        Xnew = oc.alignCenterAngle(Xnew, XnewC)
        
        # tEnd = time.time()
        # print(f'Solving correction takes {tEnd - tStart} sec.')

        return Xnew


    # def predictNearLayersWTorchNet(self, X, tracJump):
    #     N = X.shape[0] // 2
    #     nv = X.shape[1]

    #     oc = self.oc

    #     # maxLayerDist = np.sqrt(1 / N) # length = 1, h = 1/N;
    #     maxLayerDist = (1 / N) # length = 1, h = 1/N;
    #     nlayers = 3 # three layers
    #     dlayer = torch.linspace(0, maxLayerDist, nlayers, dtype=torch.float64)

    #     # Create the layers around a vesicle on which velocity calculated
    #     tracersX = torch.zeros((2 * N, nlayers, nv), dtype=torch.float64)
    #     # Standardize itorchut
    #     # Shan: standardizationStep is compatible with multiple ves
    #     Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)
    #     for k in range(nv):
    #         _, tang, _ = oc.diffProp(Xstand[:, [k]])
    #         nx = tang[N:].squeeze()
    #         ny = -tang[:N].squeeze()

    #         tracersX[:, 0, k] = Xstand[:, k]
    #         for il in range(1, nlayers):
    #             tracersX[:, il, k] = torch.hstack([Xstand[:N, k] + nx * dlayer[il], Xstand[N:, k] + ny * dlayer[il]])

    #     # How many modes to be used
    #     # MATLAB: modes = [(0:N/2-1) (-N/2:-1)]
    #     # modes = torch.concatenate((torch.arange(0,N/2), torch.arange(-N/2,0)))
    #     # modesInUse = 16
    #     # modeList = torch.where(torch.abs(modes) <= modesInUse)[0] # Shan: bug, creates 33 modes
    #     # modeList = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]

    #     # Normalize itorchut
    #     # itorchut_net = torch.zeros((modesInUse, nv, 2, N))
    #     # for imode in range(modesInUse):
    #     #     for k in range(nv):
    #     #         itorchut_net[imode, k, 0, :] = (Xstand[:N, k] - in_param[imode, 0]) / in_param[imode, 1]
    #     #         itorchut_net[imode, k, 1, :] = (Xstand[N:, k] - in_param[imode, 2]) / in_param[imode, 3]

    #     input_net = self.nearNetwork.preProcess(Xstand)
    #     net_pred = self.nearNetwork.forward(input_net)
    #     velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(net_pred)
        
    #     # Standardize tracJump
    #     # fstandRe = torch.zeros((N, nv))
    #     # fstandIm = torch.zeros((N, nv))
    #     # for k in range(nv):
    #     #     # fstand = self.standardize(tracJump[:, k], [0, 0], rotate[k], [0, 0], 1, sortIdx[k])
    #     #     z = fstand[:N] + 1j * fstand[N:]
    #     #     zh = torch.fft.fft(z)
    #     #     fstandRe[:, k] = torch.real(zh)
    #     #     fstandIm[:, k] = torch.imag(zh)
                
    #     fstand = self.standardize(tracJump, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
    #     z = fstand[:N] + 1j * fstand[N:]
    #     zh = torch.fft.fft(z, dim=0)
    #     fstandRe = torch.real(zh)
    #     fstandIm = torch.imag(zh)

    #     # Initialize outputs
    #     # velx_real = [torch.zeros((N, N, nlayers)) for _ in range(nv)]
    #     # vely_real = [torch.zeros((N, N, nlayers)) for _ in range(nv)]
    #     # velx_imag = [torch.zeros((N, N, nlayers)) for _ in range(nv)]
    #     # vely_imag = [torch.zeros((N, N, nlayers)) for _ in range(nv)]

    #     # Denormalize output
    #     # for ij, imode in enumerate(modeList):
    #     #     pred = Xpredict[ij]
    #     #     for k in range(nv):
    #     #         for ic in range(nlayers):
    #     #             velx_real[k][:, imode, ic] = (pred[k, ic] * out_param[imode, 1, ic]) + out_param[imode, 0, ic]
    #     #             vely_real[k][:, imode, ic] = (pred[k, nlayers + ic] * out_param[imode, 1, nlayers + ic]) + out_param[imode, 0, nlayers + ic]
    #     #             velx_imag[k][:, imode, ic] = (pred[k, 2 * nlayers + ic] * out_param[imode, 1, 2 * nlayers + ic]) + out_param[imode, 0, 2 * nlayers + ic]
    #     #             vely_imag[k][:, imode, ic] = (pred[k, 3 * nlayers + ic] * out_param[imode, 1, 3 * nlayers + ic]) + out_param[imode, 0, 3 * nlayers + ic]

    #     velx = torch.zeros((N, nlayers, nv), dtype=torch.float64)
    #     vely = torch.zeros((N, nlayers, nv), dtype=torch.float64)
    #     xlayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
    #     ylayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
    #     for k in range(nv):
    #         velx_stand = torch.zeros((N, nlayers), dtype=torch.float64)
    #         vely_stand = torch.zeros((N, nlayers), dtype=torch.float64)
    #         for il in range(nlayers):
    #             velx_stand[:, il] = velx_real[k][:, :, il] @ fstandRe[:, k] + velx_imag[k][:, :, il] @ fstandIm[:, k]
    #             vely_stand[:, il] = vely_real[k][:, :, il] @ fstandRe[:, k] + vely_imag[k][:, :, il] @ fstandIm[:, k]

    #             vx = torch.zeros(N, dtype=torch.float64)
    #             vy = torch.zeros(N, dtype=torch.float64)

    #             # Destandardize
    #             vx[sortIdx[k]] = velx_stand[:, il]
    #             vy[sortIdx[k]] = vely_stand[:, il]

    #             VelBefRot = torch.hstack([vx, vy])
    #             VelRot = self.rotationOperator(VelBefRot, -rotate[k], [0, 0])
    #             velx[:, il, k] = VelRot[:N]
    #             vely[:, il, k] = VelRot[N:]

    #     for il in range(nlayers):
    #         Xl = self.destandardize(tracersX[:, il], trans, rotate, rotCent, scaling, sortIdx)
    #         xlayers[:, il] = Xl[:N]
    #         ylayers[:, il] = Xl[N:]

    #     return xlayers, ylayers, velx, vely

    def predictNearLayers(self, X):
        print('Near network predicting')
        N = X.shape[0] // 2
        nv = X.shape[1]

        oc = self.oc

        # maxLayerDist = np.sqrt(1 / N) 
        maxLayerDist = (1 / N) # length = 1, h = 1/N;
        nlayers = 3 # three layers
        dlayer = torch.linspace(0, maxLayerDist, nlayers, dtype=torch.float64)

        # Create the layers around a vesicle on which velocity calculated
        tracersX = torch.zeros((2 * N, nlayers, nv), dtype=torch.float64)
        
        # Shan: standardizationStep is compatible with multiple ves
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)
        
        for k in range(nv):
            _, tang, _ = oc.diffProp(Xstand[:, [k]])
            nx = tang[N:].squeeze()
            ny = -tang[:N].squeeze()

            tracersX[:, 0, k] = Xstand[:, k]
            for il in range(1, nlayers):
                tracersX[:, il, k] = torch.hstack([Xstand[:N, k] + nx * dlayer[il], Xstand[N:, k] + ny * dlayer[il]])

        tracersX_ = torch.zeros((2 * N, nlayers, nv), dtype=torch.float64)
        tracersX_[:, 0] = Xstand
        _, tang, _ = oc.diffProp(Xstand)
        rep_nx = torch.repeat_interleave(tang[N:, :, None], nlayers-1, dim=-1) 
        rep_ny = torch.repeat_interleave(-tang[:N, :, None], nlayers-1, dim=-1)
        dx =  rep_nx * dlayer[1:] # (N, nv, nlayers-1)
        dy =  rep_ny * dlayer[1:]
        tracersX_[:, 1:] = torch.permute(
            torch.vstack([torch.repeat_interleave(Xstand[:N, :, None], nlayers-1, dim=-1) + dx,
                        torch.repeat_interleave(Xstand[N:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1))

        if not torch.allclose(tracersX, tracersX_):
            raise "batch op error"
        
        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(net_pred)
        
        xlayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        ylayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        
        for il in range(nlayers):
            Xl = self.destandardize(tracersX[:, il], trans, rotate, rotCent, scaling, sortIdx)
            xlayers[:, il] = Xl[:N]
            ylayers[:, il] = Xl[N:]

        return velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, trans, rotate, rotCent, scaling, sortIdx

    def buildVelocityInNear(self, tracJump, velx_real, vely_real, velx_imag, vely_imag, trans, rotate, rotCent, scaling, sortIdx):
        
        nv = tracJump.shape[1]
        N = tracJump.shape[0]//2
        nlayers = 3

        fstand = self.standardize(tracJump, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
        z = fstand[:N] + 1j * fstand[N:]
        zh = torch.fft.fft(z, dim=0)
        fstandRe = torch.real(zh)
        fstandIm = torch.imag(zh)

        velx = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        vely = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        for k in range(nv):
            velx_stand = torch.zeros((N, nlayers), dtype=torch.float64)
            vely_stand = torch.zeros((N, nlayers), dtype=torch.float64)
            for il in range(nlayers):
                velx_stand[:, il] = velx_real[k][:, :, il] @ fstandRe[:, k] + velx_imag[k][:, :, il] @ fstandIm[:, k]
                vely_stand[:, il] = vely_real[k][:, :, il] @ fstandRe[:, k] + vely_imag[k][:, :, il] @ fstandIm[:, k]

                vx = torch.zeros(N, dtype=torch.float64)
                vy = torch.zeros(N, dtype=torch.float64)

                # Destandardize
                vx[sortIdx[k]] = velx_stand[:, il]
                vy[sortIdx[k]] = vely_stand[:, il]

                VelBefRot = torch.hstack([vx, vy])
                VelRot = self.rotationOperator(VelBefRot, -rotate[k], [0, 0])
                velx[:, il, k] = VelRot[:N]
                vely[:, il, k] = VelRot[N:]
        
        velx_stand_ = torch.einsum('vnml, mv -> nvl', velx_real, fstandRe) + torch.einsum('vnml, mv -> nvl', velx_imag, fstandIm)
        vely_stand_ = torch.einsum('vnml, mv -> nvl', vely_real, fstandRe) + torch.einsum('vnml, mv -> nvl', vely_imag, fstandIm)
        
        vx_ = torch.zeros((nv, nlayers, N), dtype=torch.float64)
        vy_ = torch.zeros((nv, nlayers, N), dtype=torch.float64)
        # Destandardize
        vx_[torch.arange(nv), :, sortIdx.T] = velx_stand_
        vy_[torch.arange(nv), :, sortIdx.T] = vely_stand_

        VelBefRot_ = torch.concat((vx_, vy_), dim=-1) # (nv, nlayers, 2N)
        VelRot_ = self.rotationOperator(VelBefRot_.reshape(-1, 2*N).T, 
                        torch.repeat_interleave(-rotate, nlayers, dim=0), torch.zeros(nv * nlayers))
        VelRot_ = VelRot_.T.reshape(nv, nlayers, 2*N).permute(2,1,0)
        velx_ = VelRot_[:N] # (N, nlayers, nv)
        vely_ = VelRot_[N:]

        if (not torch.allclose(velx_, velx)) or (not torch.allclose(vely_, vely)):
            raise "batch err"
        
        return velx, vely
    
    # def computeStokesInteractions(self, vesicle, trac_jump, oc):
    #     print('Near-singular interaction through interpolation and network')
    #     N = vesicle.N
    #     nv = vesicle.nv

    #     # Compute near/far hydro interactions without any correction
    #     # First calculate the far-field
    #     far_field = torch.zeros((2 * N, nv), dtype=torch.float64)
    #     for k in range(nv):
    #         K = list(range(nv))
    #         K.remove(k)
    #         far_field[:, [k]] = self.exactStokesSL(vesicle, trac_jump, vesicle.X[:, [k]], K)

    #     self.nearFieldCorrection(oc, vesicle, trac_jump, far_field, option='kdtree')
        
    #     return far_field
    
    # def nearFieldCorrection(self, oc, vesicle, trac_jump, far_field,
    #                              option='kdtree'):
    #     N = vesicle.N
    #     nv = vesicle.nv
    #     xvesicle = vesicle.X[:N, :]
    #     yvesicle = vesicle.X[N:, :]
    #     # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
    #     max_layer_dist = vesicle.length.item() / vesicle.N

    #     i_call_near = [False]*nv
    #     query_X = defaultdict(list)
    #     ids_in_store = defaultdict(list)
    #     near_ves_ids = defaultdict(set)

    #     if option == "raycasting":
    #         # Ray Casting to find near field
    #         # Tangent
    #         _, tang, _ = oc.diffProp(vesicle.X)
    #         # Normal
    #         nx = tang[N:2*N, :]
    #         ny = -tang[:N, :]

    #         xvesicle = vesicle.X[:N, :]
    #         yvesicle = vesicle.X[N:, :]
    #         # Find the outermost layers of every vesicle
    #         Xlarge = torch.zeros((2 * N, nv), dtype=torch.float64)
    #         for k in range(nv):
    #             Xlarge[:, k] = torch.concatenate([xvesicle[:, k] + nx[:, k] * max_layer_dist, 
    #                                         yvesicle[:, k] + ny[:, k] * max_layer_dist])
    #         for j in range(nv):
    #             # Reorder coordinates, S is the shape of outmost layer of ves j
    #             S = torch.zeros(2 * N, dtype=torch.float64)
    #             S[0::2] = Xlarge[:N, j]
    #             S[1::2] = Xlarge[N:, j]

    #             for k in range(nv):
    #                 if k == j:
    #                     continue
    #                 for p in range(N): # loop over all points of ves k
    #                     flag = ray_casting([xvesicle[p, k].cpu().numpy(), yvesicle[p, k].cpu().numpy()], S.cpu().numpy())
    #                     if flag: # if inside S (outermost layer) of j
    #                         # indices
    #                         ids_in_store[k].append(p)
    #                         # % points where we need interpolation  
    #                         query_X[k].append([xvesicle[p, k], yvesicle[p, k]])
    #                         # k is in the near zone of j
    #                         near_ves_ids[k].add(j)
                            
    #                         i_call_near[k] = True

    #     elif option == "kdtree":
    #         all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).cpu().numpy()
    #         tree = KDTree(all_points)
    #         all_nbrs = tree.query_ball_point(all_points, max_layer_dist, return_sorted=True)
    #         # all_nbrs = np.array(all_nbrs)

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

    #     # If needed to call near-singular correction:
    #     if np.any(i_call_near):
    #         # TODO: here we could select true values only to pass into networks
    #         xlayers, ylayers, velx, vely = self.predictNearLayersWTorchNet(vesicle.X, trac_jump)

    #         for k in range(nv):
    #             if i_call_near[k]:
    #                 ids_in = ids_in_store[k] 
    #                 points_query = np.concatenate(query_X[k])
    #                 # print(ids_in)
    #                 ves_id = list(near_ves_ids[k])

    #                 n_points = N * len(ves_id)
    #                 Xin = torch.vstack([xlayers[:, :, ves_id].reshape(1, 3 * n_points), ylayers[:, :, ves_id].reshape(1, 3 * n_points)])
    #                 velXInput = velx[:, :, ves_id].reshape(1, 3 * n_points)
    #                 velYInput = vely[:, :, ves_id].reshape(1, 3 * n_points)

    #                 scipy_rbf =  scipyinterp(Xin.T.cpu(), torch.concatenate((velXInput.T, velYInput.T), dim=-1).cpu(), kernel='linear', degree = 1)
    #                 rbf_vel = torch.tensor(scipy_rbf(points_query))
                    
    #                 far_x = far_field[:N, k]
    #                 far_y = far_field[N:, k]
    #                 far_x[ids_in] = rbf_vel[:, 0]
    #                 far_y[ids_in] = rbf_vel[:, 1]
    #                 far_field[:, k] = torch.concatenate([far_x, far_y])
    #     return 

    def computeStokesInteractions(self, vesicle, trac_jump, velx_real, vely_real, velx_imag, vely_imag, \
                                  xlayers, ylayers, trans, rotate, rotCent, scaling, sortIdx):
        # print('Near-singular interaction through interpolation and network')
        N = vesicle.N
        nv = vesicle.nv

        velx, vely = self.buildVelocityInNear(trac_jump, velx_real, vely_real, velx_imag, vely_imag, trans, rotate, rotCent, scaling, sortIdx)
        # Compute near/far hydro interactions without any correction
        # First calculate the far-field
        far_field = torch.zeros((2 * N, nv), dtype=torch.float64)
        for k in range(nv):
            K = list(range(nv))
            K.remove(k)
            far_field[:, [k]] = self.exactStokesSL(vesicle, trac_jump, vesicle.X[:, [k]], K)

        # far_field_ = far_field.clone()
        self.nearFieldCorrection(vesicle, trac_jump, far_field, velx, vely, xlayers, ylayers, option='kdtree')
        # self.OLDnearFieldCorrection(vesicle, trac_jump, far_field_, velx, vely, xlayers, ylayers, option='kdtree')
        
        # if not torch.allclose(far_field_, far_field):
        #     raise "far field error"
        return far_field

    def nearFieldCorrection(self, vesicle, trac_jump, far_field, velx, vely, xlayers, ylayers,
                                 option='kdtree'):
        N = vesicle.N
        nv = vesicle.nv
        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
        max_layer_dist = vesicle.length.item() / vesicle.N

        i_call_near = [False]*nv
        query_X = defaultdict(list)
        ids_in_store = defaultdict(list)
        near_ves_ids = defaultdict(set)

        # if option == "raycasting":
        #     # Ray Casting to find near field
        #     # Tangent
        #     _, tang, _ = oc.diffProp(vesicle.X)
        #     # Normal
        #     nx = tang[N:2*N, :]
        #     ny = -tang[:N, :]

        #     xvesicle = vesicle.X[:N, :]
        #     yvesicle = vesicle.X[N:, :]
        #     # Find the outermost layers of every vesicle
        #     Xlarge = torch.zeros((2 * N, nv), dtype=torch.float64)
        #     for k in range(nv):
        #         Xlarge[:, k] = torch.concatenate([xvesicle[:, k] + nx[:, k] * max_layer_dist, 
        #                                     yvesicle[:, k] + ny[:, k] * max_layer_dist])
        #     for j in range(nv):
        #         # Reorder coordinates, S is the shape of outmost layer of ves j
        #         S = torch.zeros(2 * N, dtype=torch.float64)
        #         S[0::2] = Xlarge[:N, j]
        #         S[1::2] = Xlarge[N:, j]

        #         for k in range(nv):
        #             if k == j:
        #                 continue
        #             for p in range(N): # loop over all points of ves k
        #                 flag = ray_casting([xvesicle[p, k].cpu().numpy(), yvesicle[p, k].cpu().numpy()], S.cpu().numpy())
        #                 if flag: # if inside S (outermost layer) of j
        #                     # indices
        #                     ids_in_store[k].append(p)
        #                     # % points where we need interpolation  
        #                     query_X[k].append([xvesicle[p, k], yvesicle[p, k]])
        #                     # k is in the near zone of j
        #                     near_ves_ids[k].add(j)
                            
        #                     i_call_near[k] = True

        if option == "kdtree":
            all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).cpu().numpy()
            tree = KDTree(all_points)
            all_nbrs = tree.query_ball_point(all_points, max_layer_dist, return_sorted=True)

            for j in range(nv):
                j_nbrs = all_nbrs[N*j : N*(j+1)]
                j_nbrs_flat = np.array(list(set(sum(j_nbrs, [])))) # flatten a list of lists and remove duplicates
                others = j_nbrs_flat[np.where((j_nbrs_flat >= N*(j+1)) | (j_nbrs_flat < N*j))]
                for k in range(nv):
                    if k == j:
                        continue
                    others_from_k = others[np.where((others>= N*k) & (others < N*(k+1)))]
                    if len(others_from_k) > 0:
                        # which of ves k's points are in others' near zone
                        ids_in_store[k] += list(others_from_k % N)
                        # and their coords
                        query_X[k].append(all_points[others_from_k])
                        # k is in the near zone of j
                        near_ves_ids[k].add(j)
                        i_call_near[k] = True

        # If needed to call near-singular correction:
        if np.any(i_call_near):
            # TODO: here we could select true values only to pass into networks
            # xlayers, ylayers, velx, vely = self.predictNearLayersWTorchNet(vesicle.X, trac_jump)
            correction = torch.zeros_like(far_field)

            for k in range(nv):
                if i_call_near[k]: # ves k's points are in others' near zone
                    num = len(ids_in_store[k])
                    ids_in = torch.IntTensor(ids_in_store[k]) # which of ves k's points are in others' near zone
                    points_query = np.concatenate(query_X[k]) # and their coords
                    ves_id = list(near_ves_ids[k]) # k is in the near zone of who

                    xtar = torch.hstack((vesicle.X[ids_in, k], vesicle.X[ids_in + N, k])).unsqueeze(-1)
                    # shape of contribution_from_near: (2*num, 1)
                    contribution_from_near = self.exactStokesSL(vesicle, trac_jump, xtar, ves_id)
                    correction[ids_in, k] = - contribution_from_near[:num, 0]
                    correction[ids_in + N, k] = - contribution_from_near[num:, 0]

                    n_points = N * len(ves_id)
                    Xin = torch.vstack([xlayers[:, :, ves_id].reshape(1, 3 * n_points), ylayers[:, :, ves_id].reshape(1, 3 * n_points)])
                    velXInput = velx[:, :, ves_id].reshape(1, 3 * n_points)
                    velYInput = vely[:, :, ves_id].reshape(1, 3 * n_points)

                    scipy_rbf =  scipyinterp(Xin.T.cpu(), torch.concatenate((velXInput.T, velYInput.T), dim=-1).cpu(), kernel='linear', degree = 1)
                    rbf_vel = torch.tensor(scipy_rbf(points_query))
                    
                    correction[ids_in, k] += rbf_vel[:, 0]
                    correction[ids_in + N, k] += rbf_vel[:, 1]

            far_field += correction
        return 

    
    def OLDnearFieldCorrection(self, vesicle, trac_jump, far_field,  velx, vely, xlayers, ylayers,
                                 option='kdtree'):
        N = vesicle.N
        nv = vesicle.nv
        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
        max_layer_dist = vesicle.length.item() / vesicle.N

        i_call_near = [False]*nv
        query_X = defaultdict(list)
        ids_in_store = defaultdict(list)
        near_ves_ids = defaultdict(set)

        if option == "raycasting":
            # Ray Casting to find near field
            # Tangent
            _, tang, _ = oc.diffProp(vesicle.X)
            # Normal
            nx = tang[N:2*N, :]
            ny = -tang[:N, :]

            xvesicle = vesicle.X[:N, :]
            yvesicle = vesicle.X[N:, :]
            # Find the outermost layers of every vesicle
            Xlarge = torch.zeros((2 * N, nv), dtype=torch.float64)
            for k in range(nv):
                Xlarge[:, k] = torch.concatenate([xvesicle[:, k] + nx[:, k] * max_layer_dist, 
                                            yvesicle[:, k] + ny[:, k] * max_layer_dist])
            for j in range(nv):
                # Reorder coordinates, S is the shape of outmost layer of ves j
                S = torch.zeros(2 * N, dtype=torch.float64)
                S[0::2] = Xlarge[:N, j]
                S[1::2] = Xlarge[N:, j]

                for k in range(nv):
                    if k == j:
                        continue
                    for p in range(N): # loop over all points of ves k
                        flag = ray_casting([xvesicle[p, k].cpu().numpy(), yvesicle[p, k].cpu().numpy()], S.cpu().numpy())
                        if flag: # if inside S (outermost layer) of j
                            # indices
                            ids_in_store[k].append(p)
                            # % points where we need interpolation  
                            query_X[k].append([xvesicle[p, k], yvesicle[p, k]])
                            # k is in the near zone of j
                            near_ves_ids[k].add(j)
                            
                            i_call_near[k] = True

        elif option == "kdtree":
            all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).cpu().numpy()
            tree = KDTree(all_points)
            all_nbrs = tree.query_ball_point(all_points, max_layer_dist, return_sorted=True)
            # all_nbrs = np.array(all_nbrs)

            for j in range(nv):
                j_nbrs = all_nbrs[N*j : N*(j+1)]
                j_nbrs_flat = np.array(list(set(sum(j_nbrs, [])))) # flatten a list of lists and remove duplicates
                others = j_nbrs_flat[np.where((j_nbrs_flat >= N*(j+1)) | (j_nbrs_flat < N*j))]
                for k in range(nv):
                    if k == j:
                        continue
                    others_from_k = others[np.where((others>= N*k) & (others < N*(k+1)))]
                    if len(others_from_k) > 0:
                        # which of ves k's points are in others' near zone
                        ids_in_store[k] += list(others_from_k % N)
                        # and their coords
                        query_X[k].append(all_points[others_from_k])
                        # k is in the near zone of j
                        near_ves_ids[k].add(j)
                        i_call_near[k] = True

        # If needed to call near-singular correction:
        if np.any(i_call_near):
            # TODO: here we could select true values only to pass into networks
            # xlayers, ylayers, velx, vely = self.predictNearLayersWTorchNet(vesicle.X, trac_jump)

            for k in range(nv):
                if i_call_near[k]:
                    ids_in = ids_in_store[k] 
                    points_query = np.concatenate(query_X[k])
                    # print(ids_in)
                    ves_id = list(near_ves_ids[k])

                    n_points = N * len(ves_id)
                    Xin = torch.vstack([xlayers[:, :, ves_id].reshape(1, 3 * n_points), ylayers[:, :, ves_id].reshape(1, 3 * n_points)])
                    velXInput = velx[:, :, ves_id].reshape(1, 3 * n_points)
                    velYInput = vely[:, :, ves_id].reshape(1, 3 * n_points)

                    scipy_rbf =  scipyinterp(Xin.T.cpu(), torch.concatenate((velXInput.T, velYInput.T), dim=-1).cpu(), kernel='linear', degree = 1)
                    rbf_vel = torch.tensor(scipy_rbf(points_query))
                    
                    far_x = far_field[:N, k]
                    far_y = far_field[N:, k]
                    far_x[ids_in] = rbf_vel[:, 0]
                    far_y[ids_in] = rbf_vel[:, 1]
                    far_field[:, k] = torch.concatenate([far_x, far_y])
        return 

    
    def translateVinfwTorch(self, Xold, vinf):
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]

        mode_list = np.arange(128)
        
        Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(Xold)

        Xpredict = self.mergedAdvNetwork.forward(Xstand.to(self.device))
        
        # Z11r = torch.zeros((N, N, nv), dtype=torch.float64)
        # Z12r = torch.zeros_like(Z11r)
        # Z21r = torch.zeros_like(Z11r)
        # Z22r = torch.zeros_like(Z11r)

        # for ij in range(127):
        #     imode = mode_list[ij + 1] # imode skips 0 and starts with 1
        #     if imode == 64:
        #         continue
        #     pred = Xpredict[ij] # Xpredict is (127, nv, 2, 256)

        #     for k in range(nv):
        #         Z11r[:, imode, k] = pred[k, 0, :N]
        #         Z21r[:, imode, k] = pred[k, 0, N:]
        #         Z12r[:, imode, k] = pred[k, 1, :N]
        #         Z22r[:, imode, k] = pred[k, 1, N:]
        
        Z11r_ = torch.zeros((N, N, nv), dtype=torch.float64)
        Z12r_ = torch.zeros_like(Z11r_)
        Z21r_ = torch.zeros_like(Z11r_)
        Z22r_ = torch.zeros_like(Z11r_)

        Z11r_[:, 1:] = torch.permute(Xpredict[:, :, 0, :N], (2, 0, 1))
        Z21r_[:, 1:] = torch.permute(Xpredict[:, :, 0, N:], (2, 0, 1))
        Z12r_[:, 1:] = torch.permute(Xpredict[:, :, 1, :N], (2, 0, 1))
        Z22r_[:, 1:] = torch.permute(Xpredict[:, :, 1, N:], (2, 0, 1))

        # if not (torch.allclose(Z11r, Z11r_) and torch.allclose(Z21r, Z21r_) and\
        #         torch.allclose(Z12r, Z12r_) and torch.allclose(Z22r, Z22r_)):
        #     raise "batch op error"
        

        # Take fft of the velocity (should be standardized velocity)
        # only sort points and rotate to pi/2 (no translation, no scaling)
        Xnew = torch.zeros_like(Xold)
        vinf_stand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
        z = vinf_stand[:N] + 1j * vinf_stand[N:]
        zh = torch.fft.fft(z, dim=0)
        V1, V2 = torch.real(zh), torch.imag(zh)
        MVinf_stand = torch.vstack((torch.einsum('NiB,iB ->NB', Z11r_, V1) + torch.einsum('NiB,iB ->NB', Z12r_, V2),
                               torch.einsum('NiB,iB ->NB', Z21r_, V1) + torch.einsum('NiB,iB ->NB', Z22r_, V2)))
           
        for k in range(nv):
            # vinf_stand = self.standardize(vinf[:, k], torch.tensor([0, 0]), rotate[k], torch.tensor([0, 0]), 1, sortIdx[k])
            # z = vinf_stand[:N] + 1j * vinf_stand[N:]

            # zh = torch.fft(z, dim=0)
            # V1, V2 = torch.real(zh[:, k]), torch.imag(zh[:, k])
            # Compute the approximate value of the term M*vinf
            # MVinf_stand = torch.hstack([Z11r[:, :, k] @ V1 + Z12r[:, :, k] @ V2, 
            #                         Z21r[:, :, k] @ V1 + Z22r[:, :, k] @ V2])
            
            # Need to destandardize MVinf (take sorting and rotation back)
            MVinf = torch.zeros_like(MVinf_stand[:, k])
            idx = torch.concatenate([sortIdx[k], sortIdx[k] + N])
            MVinf[idx] = MVinf_stand[:, k]
            MVinf = self.rotationOperator(MVinf, -rotate[k], [0, 0])

            Xnew[:, k] = Xold[:, k] + self.dt * vinf[:, k] - self.dt * MVinf

        Xnew_ = torch.zeros_like(Xold)
        MVinf = torch.zeros_like(MVinf_stand)
        idx = torch.vstack([sortIdx.T, sortIdx.T + N])
        MVinf[idx, torch.arange(nv)] = MVinf_stand
        MVinf = self.rotationOperator(MVinf, -rotate, torch.zeros((nv, 2), dtype=torch.float64))
        Xnew_ = Xold + self.dt * vinf - self.dt * MVinf

        if not torch.allclose(Xnew, Xnew_):
            raise "batch op error"
        
        return Xnew

    def relaxWTorchNet(self, Xmid):
        # RELAXATION w/ NETWORK
        Xin, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(Xmid)

        Xpred = self.relaxNetwork.forward(Xin)
        Xnew = self.destandardize(Xpred, trans, rotate, rotCent, scaling, sortIdx)

        return Xnew

    def invTenMatOnVback(self, X, vinf):
        # Approximate inv(Div*G*Ten)*Div*vExt 
        
        # number of vesicles
        nv = X.shape[1]
        # number of points of exact solve
        N = X.shape[0] // 2
        
        # Modes to be called
        modeList = np.arange(128)
        Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(X)

        input = self.tenAdvNetwork.preProcess(Xstand)
        Xpredict = self.tenAdvNetwork.forward(input)
        out = self.tenAdvNetwork.postProcess(Xpredict) # shape: (127, nv, 2, 128)

        # Approximate the multiplication Z = inv(DivGT)DivPhi_k
        Z1 = torch.zeros((N, N, nv), dtype=torch.float64)
        Z2 = torch.zeros((N, N, nv), dtype=torch.float64)

        for ij in range(len(modeList) - 1):
            imode = modeList[ij + 1]  # mode index, skipping the first mode
            pred = out[ij]  # size(pred) = [nv 2 128]

            for k in range(nv):
                Z1[:, imode, k] = pred[k, 0, :]
                Z2[:, imode, k] = pred[k, 1, :]
        
        Z1_ = torch.zeros((N, N, nv), dtype=torch.float64)
        Z2_ = torch.zeros((N, N, nv), dtype=torch.float64)

        Z1_[:, 1:] = torch.permute(out[:, :, 0], (2,0,1))
        Z2_[:, 1:] = torch.permute(out[:, :, 1], (2,0,1))

        if not (torch.allclose(Z1, Z1_) and torch.allclose(Z2, Z2_)):
            raise "batch op error"

        vBackSolve = torch.zeros((N, nv), dtype=torch.float64)
        vinfStand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
        z = vinfStand[:N] + 1j * vinfStand[N:]
        zh = torch.fft.fft(z, dim=0)
        for k in range(nv):
            # Take fft of the velocity, standardize velocity
            V1 = torch.real(zh[:, k])
            V2 = torch.imag(zh[:, k])

            # Compute the approximation to inv(Div*G*Ten)*Div*vExt
            MVinfStand = Z1[:, :, k] @ V1 + Z2[:, :, k] @ V2

            # Destandardize the multiplication
            MVinf = torch.zeros_like(MVinfStand)
            MVinf[sortIdx[k]] = MVinfStand 
            # vBackSolve[:, k] = self.rotationOperator(MVinf, -rotate[k], [0, 0])
            vBackSolve[:, k] = MVinf
        
        vBackSolve_ = torch.zeros((N, nv), dtype=torch.float64)
        V1_ = torch.real(zh)
        V2_ = torch.imag(zh)
        # Compute the approximation to inv(Div*G*Ten)*Div*vExt
        MVinfStand_ = torch.einsum('NiB,iB ->NB', Z1, V1_) + torch.einsum('NiB,iB ->NB', Z2, V2_)
                               
        # Destandardize the multiplication
        vBackSolve_[sortIdx.T, torch.arange(nv)] = MVinfStand_ 

        if not torch.allclose(vBackSolve, vBackSolve_):
            raise "batch op error"

        return vBackSolve

    def invTenMatOnSelfBend(self, X):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        nv = X.shape[1] # number of vesicles
        N = X.shape[0] // 2

        Xstand, scaling, _, _, _, sortIdx = self.standardizationStep(X)

        tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        tenPredictStand = tenPredictStand.double()
        tenPred = torch.zeros((N, nv), dtype=torch.float64)
        for k in range(nv):
            # also destandardize
            tenPred[sortIdx[k], k] = tenPredictStand[:,k]/ scaling[k]**2
        
        tenPred_ = torch.zeros((N, nv), dtype=torch.float64)
        tenPred_[sortIdx.T, torch.arange(nv)] = tenPredictStand / scaling**2

        if not torch.allclose(tenPred, tenPred_):
            raise "batch op error"

        return tenPred

    def exactStokesSL(self, vesicle, f, Xtar=None, K1=None):
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
        
        if Xtar is not None and K1 is not None:
            Ntar = Xtar.shape[0] // 2
            ncol = Xtar.shape[1]
            stokesSLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float64, device=vesicle.X.device)
        else:
            K1 = []
            Ntar = 0
            stokesSLPtar = None
            ncol = 0

        den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

        xsou = vesicle.X[:vesicle.N, K1].flatten()
        ysou = vesicle.X[vesicle.N:, K1].flatten()
        xsou = torch.tile(xsou, (Ntar, 1)).T
        ysou = torch.tile(ysou, (Ntar, 1)).T

        denx = den[:vesicle.N, K1].flatten()
        deny = den[vesicle.N:, K1].flatten()
        denx = torch.tile(denx, (Ntar, 1)).T
        deny = torch.tile(deny, (Ntar, 1)).T

        for k in range(ncol):  # Loop over columns of target points
            if ncol != 1:
                raise "ncol != 1"
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

        stokesSLPtar = stokesSLPtar / (4 * torch.pi)
        return stokesSLPtar

    # def standardizationStep(self, Xin, Nnet):
    #     oc = self.oc
    #     N = len(Xin) // 2

    #     if Nnet != N:
    #         Xin = torch.vstack([
    #             interpft(Xin[:N], Nnet),
    #             interpft(Xin[N:], Nnet)
    #         ])

    #     # Equally distribute points in arc-length
    #     for _ in range(10):
    #         Xin, _, _ = oc.redistributeArcLength(Xin)

    #     X = Xin
    #     trans, rotate, rotCent, scaling, sortIdx = self.referenceValues(X)

    #     # Standardize angle, center, scaling, and point order
    #     X = self.standardize(X, trans, rotate, rotCent, scaling, sortIdx)
    #     return X, scaling, rotate, rotCent, trans, sortIdx
    
    def standardizationStep(self, Xin):
        # compatible with multi ves
        X = Xin[:]
        # % Equally distribute points in arc-length
        for _ in range(5):
            X, flag = self.oc.redistributeArcLength(X)
            if flag:
                break
            
        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)
        
        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, scaling, rotate, rotCenter, trans, multi_sortIdx

    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        # compatible with multi ves
        N = len(multi_sortIdx[0])
        nv = X.shape[1]
        Xrotated = self.rotationOperator(X, rotation, rotCenter)
        Xrotated = self.translateOp(Xrotated, translation)
        XrotSort = torch.zeros_like(Xrotated)
        for i in range(nv):
            XrotSort[:,i] = torch.concatenate((Xrotated[multi_sortIdx[i], i], Xrotated[multi_sortIdx[i] + N, i]))
        
        X_ = torch.vstack((Xrotated[multi_sortIdx.T, torch.arange(nv)], Xrotated[multi_sortIdx.T + N, torch.arange(nv)]))
        if not torch.allclose(XrotSort, X_):
            raise "batch op wrong"
        
        XrotSort = scaling*XrotSort
        return XrotSort


    def destandardize(self, XrotSort, translation, rotation, rotCent, scaling, sortIdx):
        ''' compatible with multiple ves'''
        N = len(sortIdx[0])
        nv = XrotSort.shape[1]

        # Scale back
        XrotSort = XrotSort / scaling

        # Change ordering back
        X = torch.zeros_like(XrotSort)
        for i in range(nv):
            X[sortIdx[i], i] = XrotSort[:N, i]
            X[sortIdx[i] + N, i] = XrotSort[N:, i]
        
        X_ = torch.zeros_like(XrotSort)
        X_[sortIdx.T, torch.arange(nv)] = XrotSort[:N]
        X_[sortIdx.T + N, torch.arange(nv)] = XrotSort[N:]

        if not torch.allclose(X, X_):
            raise "batch op wrong"

        # Take translation back
        X = self.translateOp(X, -translation)

        # Take rotation back
        X = self.rotationOperator(X, -rotation, rotCent)

        return X
    
    def referenceValues(self, Xref):
        ''' Shan: compatible with multi ves'''

        oc = self.oc
        N = len(Xref) // 2
        nv = Xref.shape[1]
        tempX = torch.zeros_like(Xref)
        tempX = Xref[:]

        # Find the physical center
        center = oc.getPhysicalCenter(tempX)
        multi_V = oc.getPrincAxesGivenCentroid(tempX,center)
        w = torch.tensor([0, 1]) # y-dim unit vector
        rotation = torch.arctan2(w[1]*multi_V[0]-w[0]*multi_V[1], w[0]*multi_V[0]+w[1]*multi_V[1])
        
        rotCenter = center # the point around which the frame is rotated
        Xref = self.rotationOperator(tempX, rotation, rotCenter)
        center_ = oc.getPhysicalCenter(Xref) # redundant?
        translation = -center_

        if not torch.allclose(center, center_):
            print(f"center {center} and center_{center_}")
            # raise "center different"
        
        Xref = self.translateOp(Xref, translation)
        
        multi_sortIdx = torch.zeros((nv, N), dtype=torch.int32)
        for k in range(nv):
            firstQuad = np.intersect1d(torch.where(Xref[:N,k] >= 0)[0].cpu(), torch.where(Xref[N:,k] >= 0)[0].cpu())
            theta = torch.arctan2(Xref[N:,k], Xref[:N,k])
            idx = torch.argmin(theta[firstQuad])
            sortIdx = torch.concatenate((torch.arange(firstQuad[idx],N), torch.arange(0, firstQuad[idx])))
            multi_sortIdx[k] = sortIdx
        
        theta = torch.arctan2(Xref[N:], Xref[:N])
        start_id = torch.argmin(torch.where(theta<0, 100, theta), dim=0)
        multi_sortIdx_ = (start_id + torch.arange(N).unsqueeze(-1)) % N
        multi_sortIdx_ = multi_sortIdx_.int().T

        if not torch.allclose(multi_sortIdx, multi_sortIdx_):
            raise "batch err"

        _, _, length = oc.geomProp(Xref)
        scaling = 1 / length
        
        return translation, rotation, rotCenter, scaling, multi_sortIdx

    # def referenceValues(self, Xref):
    #     oc = self.oc
    #     N = len(Xref) // 2

    #     # Find translation, rotation and scaling
    #     center = oc.getPhysicalCenter(Xref)
    #     V = oc.getPrincAxesGivenCentroid(Xref, center)
        
    #     # Find rotation angle
    #     w = torch.array([0, 1])  # y-dim
    #     rotation = torch.arctan2(w[1] * V[0] - w[0] * V[1], w[0] * V[0] + w[1] * V[1])

    #     # Find the ordering of the points
    #     rotCent = center
    #     Xref = self.rotationOperator(Xref, rotation, center)
    #     center = oc.getPhysicalCenter(Xref)
    #     translation = -center

    #     Xref = self.translateOp(Xref, translation)

    #     firstQuad = torch.where((Xref[:N] >= 0) & (Xref[N:] >= 0))[0]
    #     theta = torch.arctan2(Xref[N:], Xref[:N])
    #     idx = torch.argmin(theta[firstQuad])
    #     sortIdx = torch.concatenate((torch.arange(firstQuad[idx], N), torch.arange(firstQuad[idx])))

    #     # Amount of scaling
    #     _, _, length = oc.geomProp(Xref)
    #     scaling = 1 / length
        
    #     return translation, rotation, rotCent, scaling, sortIdx

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


