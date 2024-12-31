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
# import faiss
import cupy as cp
from scipy.interpolate import RBFInterpolator as scipyinterp_cpu
from cupyx.scipy.interpolate import RBFInterpolator as scipyinterp_gpu
from model_zoo.get_network_torch import RelaxNetwork, TenSelfNetwork, MergedAdvNetwork, MergedTenAdvNetwork, MergedNearFourierNetwork
import time
import mat73
import scipy.io as scio

class MLARM_manyfree_py:
    def __init__(self, dt, vinf, oc, use_repulsion, repStrength,
                 advNetInputNorm, advNetOutputNorm,
                 relaxNetInputNorm, relaxNetOutputNorm, 
                 nearNetInputNorm, nearNetOutputNorm, 
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
                                model_path="../trained/2024Oct_ves_merged_advten.pth", 
                                device = device)
    
    def time_step_many(self, Xold, tenOld):
        oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)

        # Compute velocity induced by repulsion force
        repForce = torch.zeros_like(Xold)
        if self.use_repulsion:
            repForce = vesicle.repulsionForce(Xold, self.repStrength)

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
        info = self.naiveNearZoneInfo(vesicle)

        farFieldtracJump = self.computeStokesInteractions(vesicle, info, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                        xlayers, ylayers, standardizationValues)

        farFieldtracJump = filterShape(farFieldtracJump, 16)
        # Solve for tension
        vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
        tenNew = -(vBackSolve + selfBendSolve)
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions(vesicle, info, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, standardizationValues)
        farFieldtracJump = filterShape(farFieldtracJump, 16)

        # Total background velocity
        vbackTotal = vback + farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)
        Xadv = filterShape(Xadv, 16)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
        # Compute the action of relax operator on Xold + Xadv
        Xnew = self.relaxWTorchNet(Xadv)
        XnewC = Xnew.clone()
        for _ in range(5):
            Xnew, flag = oc.redistributeArcLength(Xnew)
            if flag:
                break
        Xnew = oc.alignCenterAngle(XnewC, Xnew.to(Xold.device))

        tStart = time.time()
        Xnew = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
        tEnd = time.time()
        print(f'correctAreaLength {tEnd - tStart} sec.')

        Xnew = filterShape(Xnew.to(Xold.device), 16)

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

    def time_step_many_order(self, Xold, tenOld):
        oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)

        # Compute velocity induced by repulsion force
        repForce = torch.zeros_like(Xold)
        if self.use_repulsion:
            repForce = vesicle.repulsionForce(Xold, self.repStrength)

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
        info = self.naiveNearZoneInfo(vesicle)

        farFieldtracJump = self.computeStokesInteractions(vesicle, info, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, 
                                        xlayers, ylayers, standardizationValues)

        farFieldtracJump = filterShape(farFieldtracJump, 16)
        # Solve for tension
        vBackSolve = self.invTenMatOnVback(Xstand, standardizationValues, vback + farFieldtracJump)
        selfBendSolve = self.invTenMatOnSelfBend(Xstand, standardizationValues)
        tenNew = -(vBackSolve + selfBendSolve)
        # tenNew = filterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions(vesicle, info, tracJump, repForce, velx_real, vely_real, velx_imag, vely_imag, xlayers, ylayers, standardizationValues)
        farFieldtracJump = filterShape(farFieldtracJump, 16)

        # Total background velocity
        vbackTotal = vback + farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        Xadv = self.translateVinfwTorch(Xold, Xstand, standardizationValues, vbackTotal)
        Xadv = filterShape(Xadv, 16)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        
        # Compute the action of relax operator on Xold + Xadv
        Xnew = self.relaxWTorchNet(Xadv)
        # XnewC = Xnew.clone()
        for _ in range(5):
            Xnew, flag = oc.redistributeArcLength(Xnew)
            if flag:
                break

        tStart = time.time()
        XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
        tEnd = time.time()
        print(f'correctAreaLength {tEnd - tStart} sec.')

        Xnew = oc.alignCenterAngle(Xnew, XnewC.to(Xold.device))
        Xnew = filterShape(Xnew.to(Xold.device), 16)

        return Xnew, tenNew
    
    

    def predictNearLayers(self, Xstand, standardizationValues):
        print('Near network predicting')
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]

        oc = self.oc

        # maxLayerDist = np.sqrt(1 / N) 
        maxLayerDist = (1 / N) # length = 1, h = 1/N;
        nlayers = 3 # three layers
        dlayer = torch.linspace(0, maxLayerDist, nlayers, dtype=torch.float64)

        # Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)
        
        # Create the layers around a vesicle on which velocity calculated
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
        
        
        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(net_pred)
        
        # xlayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        # ylayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        # for il in range(nlayers):
        #     Xl = self.destandardize(tracersX_[:, il], standardizationValues)
        #     xlayers[:, il] = Xl[:N]
        #     ylayers[:, il] = Xl[N:]
        
        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues
        xlayers_ = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        ylayers_ = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        Xl_ = self.destandardize(torch.concat((tracersX_[:, 0], tracersX_[:, 1], tracersX_[:, 2]), dim=-1), 
            (scaling.tile((nlayers)), rotate.tile((nlayers)), rotCenter.tile((1,nlayers)), trans.tile((1,nlayers)), sortIdx.tile((nlayers,1))))
        xlayers_ = Xl_[:N, np.arange(3*nv).reshape(3, nv)]
        ylayers_ = Xl_[N:, np.arange(3*nv).reshape(3, nv)]

        # if not torch.allclose(xlayers, xlayers_):
        #     raise "batch err"

        # if not torch.allclose(ylayers, ylayers_):
        #     raise "batch err"

        return velx_real, vely_real, velx_imag, vely_imag, xlayers_, ylayers_

    def buildVelocityInNear(self, tracJump, velx_real, vely_real, velx_imag, vely_imag, standardizationValues):
        
        nv = tracJump.shape[1]
        N = tracJump.shape[0]//2
        nlayers = 3
        scaling, rotate, rotCenter, trans, sortIdx = standardizationValues

        fstand = self.standardize(tracJump, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
        z = fstand[:N] + 1j * fstand[N:]
        zh = torch.fft.fft(z, dim=0)
        fstandRe = torch.real(zh)
        fstandIm = torch.imag(zh)
        
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

        return velx_, vely_
    
    def naiveNearZoneInfo(self, vesicle):
        '''
        Naive way of doing range search by computing distances and creating masks.
        return a boolean nbrs_mask where (i,j)=True means i, j are close and are from different vesicles
        '''
        N = vesicle.N
        nv = vesicle.nv
        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
        max_layer_dist = vesicle.length.item() / vesicle.N

        all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1)

        # Compute squared pairwise distances between all points
        diff = all_points.unsqueeze(1) - all_points.unsqueeze(0)  # Shape: (N, N, 2)
        sq_distances = torch.sum(diff**2, dim=-1)         # Shape: (N, N)
        
        dist_mask = sq_distances <= max_layer_dist**2

        id_mask = torch.ones((N*nv, N*nv), dtype=torch.bool)  # Initialize all True
        
        indices = torch.arange(0, N*nv).reshape(nv, N)
        # Use advanced indexing to set blocks to False
        row_indices = indices.unsqueeze(2)  # Shape: (num_cells, points_per_cell, 1)
        col_indices = indices.unsqueeze(1)  # Shape: (num_cells, 1, points_per_cell)
        id_mask[row_indices, col_indices] = False

        nbrs_mask = torch.logical_and(dist_mask, id_mask)

        return nbrs_mask
    
    def nearZoneInfo(self, vesicle, option='exact'):
        N = vesicle.N
        nv = vesicle.nv
        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]
        # max_layer_dist = np.sqrt(vesicle.length.item() / vesicle.N)
        max_layer_dist = vesicle.length.item() / vesicle.N

        i_call_near = [False]*nv
        # which of ves k's points are in others' near zone
        ids_in_store = defaultdict(list)
        # and their coords
        query_X = defaultdict(list)
        # k is in the near zone of j: near_ves_ids[k].add(j)
        near_ves_ids = defaultdict(set)

        
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


        elif option == 'faiss':
            # (npoints, 2)
            # all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).cpu().numpy().astype('float32')
            all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1).float()
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0
            index = faiss.GpuIndexFlatL2(res, 2, flat_config)

            # index = faiss.IndexFlatL2(2) # 2D
            index.add(all_points)
            lims, _, I = index.range_search(all_points, max_layer_dist**2)
            for j in range(nv):
                j_nbrs = I[lims[N*j] : lims[N*(j+1)]]
                others = j_nbrs[torch.where((j_nbrs >= N*(j+1)) | (j_nbrs < N*j))]
                for k in range(nv):
                    if k == j:
                        continue
                    others_from_k = others[torch.where((others>= N*k) & (others < N*(k+1)))]
                    if len(others_from_k) > 0:
                        # which of ves k's points are in others' near zone
                        ids_in_store[k] += list(others_from_k % N)
                        # and their coords
                        query_X[k].append(all_points[others_from_k])
                        # k is in the near zone of j
                        near_ves_ids[k].add(j)
                        i_call_near[k] = True

        return (i_call_near, query_X, ids_in_store, near_ves_ids)
    

    def computeStokesInteractions(self, vesicle, info, trac_jump, repForce, velx_real, vely_real, velx_imag, vely_imag, \
                                  xlayers, ylayers, standardizationValues):
        # print('Near-singular interaction through interpolation and network')

        velx, vely = self.buildVelocityInNear(trac_jump + repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
        rep_velx, rep_vely = self.buildVelocityInNear(repForce, velx_real, vely_real, velx_imag, vely_imag, standardizationValues)
        # Compute near/far hydro interactions without any correction
        # First calculate the far-field
        # far_field = torch.zeros((2 * N, nv), dtype=torch.float64)
        # for k in range(nv):
        #     K = list(range(nv))
        #     K.remove(k)
        #     far_field[:, [k]] = self.exactStokesSL(vesicle, trac_jump, vesicle.X[:, [k]], K)

        far_field_ = self.allExactStokesSL(vesicle, trac_jump + repForce)

        # if not torch.allclose(far_field, far_field_):
        #     raise "batch op"

        self.nearFieldCorrection(vesicle, info, trac_jump + repForce, far_field_, velx, vely, xlayers, ylayers)
        return far_field_ + torch.concat((rep_velx[:,0], rep_vely[:,0]), dim=0)
    

    def nearFieldCorrection(self, vesicle, info, trac_jump, far_field, velx, vely, xlayers, ylayers):
        N = vesicle.N
        nv = vesicle.nv
        

        if isinstance(info, tuple):
            i_call_near, query_X, ids_in_store, near_ves_ids = info

            # If needed to call near-singular correction:
            if np.any(i_call_near):
                correction = torch.zeros_like(far_field)

                # TODO: here we could select true values only to pass into networks
                # xlayers, ylayers, velx, vely = self.predictNearLayersWTorchNet(vesicle.X, trac_jump)
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

                        scipy_rbf = scipyinterp_cpu(Xin.T.cpu(), torch.concatenate((velXInput.T, velYInput.T), dim=-1).cpu(), kernel='linear', degree = 1)
                        rbf_vel = torch.tensor(scipy_rbf(points_query))
                        
                        correction[ids_in, k] += rbf_vel[:, 0]
                        correction[ids_in + N, k] += rbf_vel[:, 1]
        else:
            nbrs_mask = info

            xvesicle = vesicle.X[:N, :]
            yvesicle = vesicle.X[N:, :]
            all_points = torch.concat((xvesicle.T.reshape(-1,1), yvesicle.T.reshape(-1,1)), dim=1)
            correction = torch.zeros((N*nv, 2), dtype=torch.float64, device=trac_jump.device)

            for k in range(nv):
                rows_with_true = torch.any(nbrs_mask[:, k*N : (k+1)*N], dim=1)
                if not torch.any(rows_with_true):
                    continue
                ids_in = torch.arange(N*nv)[rows_with_true] # which ves points are in k's near zone
                num = len(ids_in)
                # print(ids_in)
                points_query = all_points[ids_in] # and their coords
                # cols_with_true = torch.any(nbrs_mask[k*N : (k+1)*N], dim=0)
                ves_id = torch.IntTensor([k])

                xtar = torch.concat((all_points[ids_in, 0], all_points[ids_in, 1])).unsqueeze(-1)
                # shape of contribution_from_near: (2*num, 1)
                contribution_from_near = self.exactStokesSL(vesicle, trac_jump, xtar, ves_id)
                correction[ids_in, 0] = - contribution_from_near[:num, 0]
                correction[ids_in, 1] = - contribution_from_near[num:, 0]

                # n_points = N * len(ves_id)
                Xin = torch.vstack([xlayers[:, :, ves_id].reshape(1, 3 * N), ylayers[:, :, ves_id].reshape(1, 3 * N)])
                velXInput = velx[:, :, ves_id].reshape(1, 3 * N)
                velYInput = vely[:, :, ves_id].reshape(1, 3 * N)

                if Xin.is_cuda:
                    scipy_rbf =  scipyinterp_gpu(cp.asarray(Xin.T), cp.asarray(torch.concatenate((velXInput.T, velYInput.T), dim=-1)), kernel='linear', degree = 1)
                    ans = scipy_rbf(points_query)
                    rbf_vel = torch.as_tensor(ans)
                else:
                    scipy_rbf =  scipyinterp_cpu(Xin.T, torch.concatenate((velXInput.T, velYInput.T), dim=-1), kernel='linear', degree = 1)
                    rbf_vel = torch.from_numpy(scipy_rbf(points_query))

                correction[ids_in, 0] += rbf_vel[:, 0]
                correction[ids_in, 1] += rbf_vel[:, 1]
            correction_ = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)

        far_field += correction_
        return 

    
    def translateVinfwTorch(self, Xold, Xstand, standardizationValues, vinf):
        N = Xstand.shape[0] // 2
        nv = Xstand.shape[1]
        
        # Xstand, _, rotate, _, _, sortIdx = self.standardizationStep(Xold)
        _, rotate, _, _, sortIdx = standardizationValues

        Xpredict = self.mergedAdvNetwork.forward(Xstand.to(self.device))
        
        Z11r_ = torch.zeros((N, N, nv), dtype=torch.float64)
        Z12r_ = torch.zeros_like(Z11r_)
        Z21r_ = torch.zeros_like(Z11r_)
        Z22r_ = torch.zeros_like(Z11r_)

        Z11r_[:, 1:] = torch.permute(Xpredict[:, :, 0, :N], (2, 0, 1))
        Z21r_[:, 1:] = torch.permute(Xpredict[:, :, 0, N:], (2, 0, 1))
        Z12r_[:, 1:] = torch.permute(Xpredict[:, :, 1, :N], (2, 0, 1))
        Z22r_[:, 1:] = torch.permute(Xpredict[:, :, 1, N:], (2, 0, 1))

        # Take fft of the velocity (should be standardized velocity)
        # only sort points and rotate to pi/2 (no translation, no scaling)
        vinf_stand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
        z = vinf_stand[:N] + 1j * vinf_stand[N:]
        zh = torch.fft.fft(z, dim=0)
        V1, V2 = torch.real(zh), torch.imag(zh)
        MVinf_stand = torch.vstack((torch.einsum('NiB,iB ->NB', Z11r_, V1) + torch.einsum('NiB,iB ->NB', Z12r_, V2),
                               torch.einsum('NiB,iB ->NB', Z21r_, V1) + torch.einsum('NiB,iB ->NB', Z22r_, V2)))
        
        Xnew = torch.zeros_like(Xold)
        MVinf = torch.zeros_like(MVinf_stand)
        idx = torch.vstack([sortIdx.T, sortIdx.T + N])
        MVinf[idx, torch.arange(nv)] = MVinf_stand
        MVinf = self.rotationOperator(MVinf, -rotate, torch.zeros((2, nv), dtype=torch.float64))
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
        Z1 = torch.zeros((N, N, nv), dtype=torch.float64)
        Z2 = torch.zeros((N, N, nv), dtype=torch.float64)

        Z1[:, 1:] = torch.permute(out[:, :, 0], (2,0,1))
        Z2[:, 1:] = torch.permute(out[:, :, 1], (2,0,1))

        vBackSolve = torch.zeros((N, nv), dtype=torch.float64)
        vinfStand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
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
        tenPredictStand = tenPredictStand.double()
        tenPred = torch.zeros((N, nv), dtype=torch.float64)
        
        tenPred[sortIdx.T, torch.arange(nv)] = tenPredictStand / scaling**2

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
        
        
        Ntar = Xtar.shape[0] // 2
        ncol = Xtar.shape[1]
        stokesSLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float64, device=vesicle.X.device)
        

        den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

        xsou = vesicle.X[:vesicle.N, K1].flatten()
        ysou = vesicle.X[vesicle.N:, K1].flatten()
        xsou = torch.tile(xsou, (Ntar, 1)).T    # (N*(nv-1), Ntar)
        ysou = torch.tile(ysou, (Ntar, 1)).T

        denx = den[:vesicle.N, K1].flatten()
        deny = den[vesicle.N:, K1].flatten()
        denx = torch.tile(denx, (Ntar, 1)).T    # (N*(nv-1), Ntar)
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


        return stokesSLPtar / (4 * torch.pi)
    
    
    def allExactStokesSL(self, vesicle, f):
        """
        Computes the single-layer potential due to `f` around all vesicles except itself.
        
        Parameters:
        - vesicle: Vesicle object with attributes `sa`, `N`, and `X`.
        - f: Forcing term (2*N x nv).

        Returns:
        - stokesSLPtar: Single-layer potential at target points.
        """
        
        N = vesicle.N
        nv = vesicle.nv
        stokesSLPtar = torch.zeros((2 * N, nv), dtype=torch.float64, device=vesicle.X.device)

        den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

        mask = ~torch.eye(nv).bool()
        # When input is on CUDA, torch.nonzero() causes host-device synchronization.
        # indices = mask.nonzero(as_tuple=True)[1].view(nv, nv-1)
        indices = torch.arange(nv)[None,].expand(nv,-1)[mask].view(nv, nv-1)
        
        xsou = vesicle.X[:N, indices].permute(0, 2, 1).reshape(-1, nv)
        ysou = vesicle.X[N:, indices].permute(0, 2, 1).reshape(-1, nv)
        xsou = torch.tile(xsou, (N, 1, 1)).permute(1,0,2)    # (N*(nv-1), N, nv)
        ysou = torch.tile(ysou, (N, 1, 1)).permute(1,0,2)
        # xsou = xsou[None,].expand(N, -1, -1).permute(1,0,2)    # (N*(nv-1), N, nv)
        # ysou = ysou[None,].expand(N, -1, -1).permute(1,0,2)

        denx = den[:N, indices].permute(0, 2, 1).reshape(-1, nv)
        deny = den[N:, indices].permute(0, 2, 1).reshape(-1, nv)
        denx = torch.tile(denx, (N, 1, 1)).permute(1,0,2)    # (N*(nv-1), N)
        deny = torch.tile(deny, (N, 1, 1)).permute(1,0,2)

        xtar = vesicle.X[:N]
        ytar = vesicle.X[N:]
        # xtar = torch.tile(xtar, (N * (nv-1), 1, 1))
        # ytar = torch.tile(ytar, (N * (nv-1), 1, 1))
            
        diffx = xtar - xsou # broadcasting
        diffy = ytar - ysou

        dis2 = diffx**2 + diffy**2

        coeff = 0.5 * torch.log(dis2)
        stokesSLPtar[:N, torch.arange(nv)] = -torch.sum(coeff * denx, dim=0)
        stokesSLPtar[N:, torch.arange(nv)] = -torch.sum(coeff * deny, dim=0)

        coeff = (diffx * denx + diffy * deny) / dis2
        stokesSLPtar[:N, torch.arange(nv)] += torch.sum(coeff * diffx, dim=0)
        stokesSLPtar[N:, torch.arange(nv)] += torch.sum(coeff * diffy, dim=0)


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
        tempX = torch.zeros_like(Xref)
        tempX = Xref[:]

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


