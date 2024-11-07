import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import sys
sys.path.append("..")
from collections import defaultdict
from capsules import capsules
# from rayCasting import ray_casting
from filter import upsThenFilterShape, upsThenFilterTension
from scipy.spatial import KDTree
from scipy.interpolate import RBFInterpolator as scipyinterp
from model_zoo.get_network_torch import RelaxNetwork, TenSelfNetwork, MergedAdvNetwork, MergedTenAdvNetwork, MergedNearFourierNetwork
import time
import mat73
import scipy.io as scio

# class MLARM_py:
#     def __init__(self, dt, vinf, oc, advNetInputNorm, advNetOutputNorm,
#                  relaxNetInputNorm, relaxNetOutputNorm, device):
#         self.dt = dt  # time step size
#         self.vinf = vinf  # background flow (analytic -- itorchut as function of vesicle config)
#         self.oc = oc  # curve class
#         self.kappa = 1  # bending stiffness is 1 for our simulations
#         self.device = device
        
#         # Normalization values for advection (translation) networks
#         self.advNetInputNorm = advNetInputNorm
#         self.advNetOutputNorm = advNetOutputNorm
#         self.mergedAdvNetwork = MergedAdvNetwork(self.advNetInputNorm.to(device), self.advNetOutputNorm.to(device), 
#                                 model_path="/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/trained/ves_merged_adv.pth", 
#                                 device = device)
        
#         # Normalization values for relaxation network
#         self.relaxNetInputNorm = relaxNetInputNorm
#         self.relaxNetOutputNorm = relaxNetOutputNorm
#         self.relaxNetwork = RelaxNetwork(self.dt, self.relaxNetInputNorm.to(device), self.relaxNetOutputNorm.to(device), 
#                                 model_path="/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/trained/ves_relax_DIFF_June8_625k_dt1e-5.pth", 
#                                 device = device)
    
#     def time_step(self, Xold):
#         # % take a time step with neural networks
#         oc = self.oc
#         # background velocity on vesicles
#         vback = torch.from_numpy(self.vinf(Xold))

#         # Compute the action of dt*(1-M) on Xold
#         # tStart = time.time()
#         Xadv = self.translateVinfwTorch(Xold, vback)
#         # tEnd = time.time()
#         # print(f'Solving ADV takes {tEnd - tStart} sec.')

#         # Correct area and length
#         # tStart = time.time()
#         XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
#         Xadv = oc.alignCenterAngle(Xadv, XadvC)
#         # tEnd = time.time()
#         # print(f'Solving correction takes {tEnd - tStart} sec.')

#         # Compute the action of relax operator on Xold + Xadv
#         # tStart = time.time()
#         Xnew = self.relaxWTorchNet(Xadv)
#         # tEnd = time.time()
#         # print(f'Solving RELAX takes {tEnd - tStart} sec.')

#         # Correct area and length
#         # tStart = time.time()
#         XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
#         Xnew = oc.alignCenterAngle(Xnew, XnewC)
#         # tEnd = time.time()
#         # print(f'Solving correction takes {tEnd - tStart} sec.')

#         return Xnew
    
#     def translateVinfwTorch(self, Xold, vinf):
#         # Xitorchut is equally distributed in arc-length
#         # Xold as well. So, we add up coordinates of the same points.
#         N = Xold.shape[0] // 2
#         nv = Xold.shape[1]

#         # If we only use some modes
#         # modes = torch.concatenate((torch.arange(0, N//2), torch.arange(-N//2, 0)))
#         # modesInUse = 16
#         # mode_list = torch.where(torch.abs(modes) <= modes_in_use)[0]
#         # mode_list = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]
#         mode_list = [i for i in range(128)]
#         # Standardize itorchut
#         # Xstand = torch.zeros_like(Xold)
#         # scaling = torch.zeros(nv)
#         # rotate = torch.zeros(nv)
#         # rot_cent = torch.zeros((2, nv))
#         # trans = torch.zeros((2, nv))
#         # sort_idx = torch.zeros((N, nv), dtype=int)
#         # for k in range(nv):
#         #     (Xstand[:, k], scaling[k], rotate[k], 
#         #     rot_cent[:, k], trans[:, k], sort_idx[:, k]) = self.standardization_step(Xold[:, k], N)
#         Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(Xold)

#         # Normalize itorchut
#         # itorchut_list = []
#         # for imode in mode_list:
#         #     if imode != 0:
#         #         # itorchut_net = torch.zeros((nv, 2, 128)) # Shan: should be (nv, 2, 256)
#         #         x_mean, x_std, y_mean, y_std = in_param[imode-1]
#         #         # for k in range(nv):
#         #         #     itorchut_net[k, 0, :] = (Xstand[:N, k] - x_mean) / x_std
#         #         #     itorchut_net[k, 1, :] = (Xstand[N:, k] - y_mean) / y_std
#         #         itorchut_net = torch.concatenate(((Xstand[:N, None] - x_mean)/x_std, (Xstand[N:, None] - y_mean) / y_std), dim=0).T
#         #         # prepare fourier basis to be combined into itorchut
#         #         theta = torch.arange(N)/N*2*torch.pi
#         #         theta = theta.reshape(N,1)
#         #         bases = 1/N*torch.exp(1j*theta*torch.arange(N).reshape(1,N))
#         #         rr, ii = torch.real(bases[:, imode]), torch.imag(bases[:, imode])
#         #         basis = torch.concatenate((rr,ii)).reshape(1,1,256).repeat(nv, dim=0)
#         #         itorchut_net = torch.concatenate((itorchut_net, basis), dim=1)
#         #         itorchut_list.append(itorchut_net)


#         # Xpredict = pyrunfile("advect_predict.py", "output_list", itorchut_shape=itorchut_list, num_ves=nv)
        
#         Xpredict = self.mergedAdvNetwork.forward(Xstand.to(self.device))
#         Xpredict = Xpredict.cpu()
#         # Approximate the multiplication M*(FFTBasis)
#         Z11r = torch.zeros((N, N, nv), dtype=torch.float64)
#         Z12r = torch.zeros_like(Z11r)
#         Z21r = torch.zeros_like(Z11r)
#         Z22r = torch.zeros_like(Z11r)

#         for ij in range(len(mode_list) - 1):
#             imode = mode_list[ij + 1]
#             pred = Xpredict[ij]

#             for k in range(nv):
#                 Z11r[:, imode, k] = pred[k, 0, :N]
#                 Z21r[:, imode, k] = pred[k, 0, N:]
#                 Z12r[:, imode, k] = pred[k, 1, :N]
#                 Z22r[:, imode, k] = pred[k, 1, N:]

#         # Take fft of the velocity (should be standardized velocity)
#         # only sort points and rotate to pi/2 (no translation, no scaling)
#         Xnew = torch.zeros_like(Xold)
#         vinf_stand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
#         z = vinf_stand[:N] + 1j * vinf_stand[N:]
#         zh = torch.fft.fft(z, dim=0)
#         V1, V2 = torch.real(zh), torch.imag(zh)
#         MVinf_stand = torch.vstack((torch.einsum('NiB,iB ->NB', Z11r, V1) + torch.einsum('NiB,iB ->NB', Z12r, V2),
#                                torch.einsum('NiB,iB ->NB', Z21r, V1) + torch.einsum('NiB,iB ->NB', Z22r, V2)))
            
#         for k in range(nv):
#             # vinf_stand = self.standardize(vinf[:, k], torch.array([0, 0]), rotate[k], torch.array([0, 0]), 1, sort_idx[:, k])
#             # z = vinf_stand[:N] + 1j * vinf_stand[N:]

#             # zh = torch.fft(z)
#             # V1, V2 = torch.real(zh[:, k]), torch.imag(zh[:, k])
#             # Compute the approximate value of the term M*vinf
#             # MVinf_stand = torch.vstack([Z11r[:, :, k] @ V1 + Z12r[:, :, k] @ V2, 
#             #                         Z21r[:, :, k] @ V1 + Z22r[:, :, k] @ V2])
            
#             # Need to destandardize MVinf (take sorting and rotation back)
#             MVinf = torch.zeros_like(MVinf_stand[:,k])
#             idx = torch.concatenate([sortIdx[k], sortIdx[k] + N])
#             MVinf[idx] = MVinf_stand[:,k]
#             MVinf = self.rotationOperator(MVinf, -rotate[k], [0, 0])

#             Xnew[:, k] = Xold[:, k] + self.dt * vinf[:, k] - self.dt * MVinf

#         return Xnew

#     def relaxWTorchNet(self, Xmid):
       
#         Xin, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(Xmid)
        
#         Xpred = self.relaxNetwork.forward(Xin)
#         Xnew = self.destandardize(Xpred, trans, rotate, rotCent, scaling, sortIdx)

#         return Xnew
    
#     def standardizationStep(self, Xin):
#         # compatible with multi ves
#         oc = self.oc
#         X = Xin[:]
#         # % Equally distribute points in arc-length
#         for w in range(10):
#             X, _, _ = oc.redistributeArcLength(X)
#         # % standardize angle, center, scaling and point order
#         trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)
        
#         X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
#         return X, scaling, rotate, rotCenter, trans, multi_sortIdx

#     def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
#         # compatible with multi ves
#         N = len(multi_sortIdx[0])
#         Xrotated = self.rotationOperator(X, rotation, rotCenter)
#         Xrotated = self.translateOp(Xrotated, translation)
#         XrotSort = torch.zeros_like(Xrotated)
#         for i in range(X.shape[1]):
#             XrotSort[:,i] = torch.concatenate((Xrotated[multi_sortIdx[i], i], Xrotated[multi_sortIdx[i] + N, i]))
#         XrotSort = scaling*XrotSort
#         return XrotSort


#     def destandardize(self, XrotSort, translation, rotation, rotCent, scaling, sortIdx):
#         ''' compatible with multiple ves'''
#         N = len(sortIdx[0])
#         nv = XrotSort.shape[1]

#         # Scale back
#         XrotSort = XrotSort / scaling

#         # Change ordering back
#         X = torch.zeros_like(XrotSort)
#         for i in range(nv):
#             X[sortIdx[i], i] = XrotSort[:N, i]
#             X[sortIdx[i] + N, i] = XrotSort[N:, i]

#         # Take translation back
#         X = self.translateOp(X, -translation)

#         # Take rotation back
#         X = self.rotationOperator(X, -rotation, rotCent)

#         return X
    
    
#     def referenceValues(self, Xref):
#         ''' Shan: compatible with multi ves'''

#         oc = self.oc
#         N = len(Xref) // 2
#         nv = Xref.shape[1]
#         tempX = torch.zeros_like(Xref)
#         tempX = Xref[:]

#         # Find the physical center
#         center = oc.getPhysicalCenter(tempX)
#         multi_V = oc.getPrincAxesGivenCentroid(tempX,center)
#         w = torch.tensor([0, 1]) # y-dim unit vector
#         rotation = torch.arctan2(w[1]*multi_V[0]-w[0]*multi_V[1], w[0]*multi_V[0]+w[1]*multi_V[1])
        
#         rotCenter = center # the point around which the frame is rotated
#         Xref = self.rotationOperator(tempX, rotation, rotCenter)
#         center = oc.getPhysicalCenter(Xref) # redundant?
#         translation = -center
        
#         Xref = self.translateOp(Xref, translation)
        
#         multi_sortIdx = []
#         for k in range(nv):
#         # Shan: This for loop can be avoided but will be less readable
#             firstQuad = np.intersect1d(torch.where(Xref[:N,k] >= 0)[0], torch.where(Xref[N:,k] >= 0)[0])
#             theta = torch.arctan2(Xref[N:,k], Xref[:N,k])
#             idx = torch.argmin(theta[firstQuad])
#             sortIdx = torch.concatenate((torch.arange(firstQuad[idx],N), torch.arange(0, firstQuad[idx])))
#             multi_sortIdx.append(sortIdx)

#         _, _, length = oc.geomProp(Xref)
#         scaling = 1 / length
        
#         return translation, rotation, rotCenter, scaling, multi_sortIdx

    
#     def rotationOperator(self, X, theta, rotCent):
#         ''' Shan: compatible with multi ves
#         theta of shape (1,nv), rotCent of shape (2,nv)'''
#         Xrot = torch.zeros_like(X)
#         x = X[:len(X)//2] - rotCent[0]
#         y = X[len(X)//2:] - rotCent[1]

#         # Rotated shape
#         xrot = x * torch.cos(theta) - y * torch.sin(theta)
#         yrot = x * torch.sin(theta) + y * torch.cos(theta)

#         Xrot[:len(X)//2] = xrot + rotCent[0]
#         Xrot[len(X)//2:] = yrot + rotCent[1]
#         return Xrot

#     def translateOp(self, X, transXY):
#         ''' Shan: compatible with multi ves
#          transXY of shape (2,nv)'''
#         Xnew = torch.zeros_like(X)
#         Xnew[:len(X)//2] = X[:len(X)//2] + transXY[0]
#         Xnew[len(X)//2:] = X[len(X)//2:] + transXY[1]
#         return Xnew


    
# class MLARM_py:
#     def __init__(self, dt, vinf, oc, advNetItorchutNorm, advNetOutputNorm, relaxNetItorchutNorm, relaxNetOutputNorm):
#         self.dt = dt
#         self.vinf = vinf # background flow (analytic -- itorchut as function of vesicle config)
#         self.oc = oc # curve class
#         # % Normalization values for advection (translation) networks
#         self.advNetItorchutNorm = advNetItorchutNorm
#         self.advNetOutputNorm = advNetOutputNorm
#         # % Normalization values for relaxation network
#         self.relaxNetItorchutNorm = relaxNetItorchutNorm
#         self.relaxNetOutputNorm = relaxNetOutputNorm
#         self.area0 = None  # initial area of vesicle
#         self.len0 = None  # initial length of vesicle

#     def time_step(self, X):
#         # % take a time step with neural networks
#         oc = self.oc
#         vback = self.vinf(X)

#         # 1) Translate vesicle with network
#         # Xadv = self.translateVinfNet(X, vback)
#         Xadv = self.translateVinfMergeNet(X, vback, 12)

#         # Correct area and length
#         XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
#         Xadv = oc.alignCenterAngle(Xadv, XadvC)

#         # 2) Relax vesicle with network
#         Xnew = self.relaxNet(Xadv)

#         # Correct area and length
#         XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
#         Xnew = oc.alignCenterAngle(Xnew, XnewC)

#         return Xnew

#     def translateVinfNet(self, X, vback):
#         # Translate vesicle using networks
#         N = X.shape[0]//2
#         nv = X.shape[1]
#         # % Standardize vesicle (zero center, pi/2 inclination angle, equil dist)
#         Xstand, scaling, rotate, rotCenter, trans, sortIdx = self.standardizationStep(X)
#         device = torch.device("cpu")
#         Xpredict = torch.zeros(127, nv, 2, 256).to(device)
#         # Normalize itorchut
#         coords = torch.zeros((nv, 2, 128)).to(device)
#         for imode in range(2, 129):
#             x_mean = self.advNetItorchutNorm[imode - 2][0]
#             x_std = self.advNetItorchutNorm[imode - 2][1]
#             y_mean = self.advNetItorchutNorm[imode - 2][2]
#             y_std = self.advNetItorchutNorm[imode - 2][3]

#             coords[:, 0, :] = torch.from_numpy((Xstand[:N].T - x_mean) / x_std).float()
#             coords[:, 1, :] = torch.from_numpy((Xstand[N:].T - y_mean) / y_std).float()

#             # coords (N,2,128) -> (N,1,256)
#             itorchut_net = torch.concat((coords[:,0], coords[:,1]), dim=1)[:,None,:]
#             # specify which mode, imode=2,3,...,128
#             theta = torch.arange(N)/N*2*torch.pi
#             theta = theta.reshape(N,1)
#             bases = 1/N*torch.exp(1j*theta*torch.arange(N).reshape(1,N))
#             rr, ii = torch.real(bases[:,imode-1]), torch.imag(bases[:,imode-1])
#             basis = torch.from_numpy(torch.concatenate((rr,ii))).float().reshape(1,1,256).to(device)
#             # add the channel of fourier basis
#             itorchut_net = torch.concat((itorchut_net, basis.repeat(nv,1,1)), dim=1).to(device)

#             # Predict using neural networks
#             model = Net_ves_adv_fft(12,1.7,20)
#             model.load_state_dict(torch.load(f"../ves_adv_trained/ves_fft_mode{imode}.pth", map_location=device))
#             model.eval()
#             with torch.no_grad():
#                 Xpredict[imode - 2] = model(itorchut_net)

#         # % Above line approximates multiplication M*(FFTBasis) 
#         # % Now, reconstruct Mvinf = (M*FFTBasis) * vinf_hat
#         Z11 = torch.zeros((128, 128))
#         Z12 = torch.zeros((128, 128))
#         Z21 = torch.zeros((128, 128))
#         Z22 = torch.zeros((128, 128))

#         for imode in range(2, 129): # the first mode is zero
#             pred = Xpredict[imode - 2]

#             real_mean = self.advNetOutputNorm[imode - 2][0]
#             real_std = self.advNetOutputNorm[imode - 2][1]
#             imag_mean = self.advNetOutputNorm[imode - 2][2]
#             imag_std = self.advNetOutputNorm[imode - 2][3]

#             # % first channel is real
#             pred[:, 0, :] = (pred[:, 0, :] * real_std) + real_mean
#             # % second channel is imaginary
#             pred[:, 1, :] = (pred[:, 1, :] * imag_std) + imag_mean

#             Z11[:, imode-1] = pred[0, 0, :][:N]
#             Z12[:, imode-1] = pred[0, 1, :][:N] 
#             Z21[:, imode-1] = pred[0, 0, :][N:]
#             Z22[:, imode-1] = pred[0, 1, :][N:]

#         # % Take fft of the velocity (should be standardized velocity)
#         # % only sort points and rotate to pi/2 (no translation, no scaling)
#         vinfStand = self.standardize(vback, [0, 0], rotate, [0, 0], 1, sortIdx)
#         z = vinfStand[:N] + 1j * vinfStand[N:]

#         zh = torch.fft.fft(z)
#         V1 = zh.real
#         V2 = zh.imag
#         # % Compute the approximate value of the term M*vinf
#         MVinf = torch.vstack((torch.dot(Z11, V1) + torch.dot(Z12, V2), torch.dot(Z21, V1) + torch.dot(Z22, V2)))
#         # % update the standardized shape
#         XnewStand = self.dt * vinfStand - self.dt * MVinf   
#         # % destandardize
#         Xadv = self.destandardize(XnewStand, trans, rotate, rotCenter, scaling, sortIdx)
#         # % add the initial since solving dX/dt = (I-M)vinf
#         Xadv = X + Xadv

#         return Xadv

#     def translateVinfMergeNet(self, X, vback, num_modes):
#         # Translate vesicle using networks
#         N = X.shape[0]//2
#         nv = X.shape[1]
#         # % Standardize vesicle (zero center, pi/2 inclination angle, equil dist)
#         Xstand, scaling, rotate, rotCenter, trans, multi_sortIdx = self.standardizationStep(X)
#         device = torch.device("cpu")
#         Xpredict = torch.zeros(127, nv, 2, 256).to(device)
#         # prepare fourier basis
#         theta = torch.arange(N)/N*2*torch.pi
#         theta = theta.reshape(N,1)
#         bases = 1/N*torch.exp(1j*theta*torch.arange(N).reshape(1,N))

#         for i in range(127//num_modes+1):
#             # from s mode to t mode, both end included
#             s = 2 + i*num_modes
#             t = min(2 + (i+1)*num_modes -1, 128)
#             print(f"from mode {s} to mode {t}")
#             rep = t - s + 1 # number of repetitions
#             Xstand = Xstand.reshape(2*N, nv, 1)
#             multiX = torch.from_numpy(Xstand).float().repeat(1,1,rep)

#             x_mean = self.advNetItorchutNorm[s-2:t-1][:,0]
#             x_std = self.advNetItorchutNorm[s-2:t-1][:,1]
#             y_mean = self.advNetItorchutNorm[s-2:t-1][:,2]
#             y_std = self.advNetItorchutNorm[s-2:t-1][:,3]

#             coords = torch.zeros((nv, 2*rep, 128)).to(device)
#             coords[:, :rep, :] = ((multiX[:N] - x_mean) / x_std).permute(1,2,0)
#             coords[:, rep:, :] = ((multiX[N:] - y_mean) / y_std).permute(1,2,0)

#             # coords (N,2*rep,128) -> (N,rep,256)
#             itorchut_coords = torch.concat((coords[:,:rep], coords[:,rep:]), dim=-1)
#             # specify which mode
#             rr, ii = torch.real(bases[:,s-1:t]), torch.imag(bases[:,s-1:t])
#             basis = torch.from_numpy(torch.concatenate((rr,ii),dim=-1)).float().reshape(1,rep,256).to(device)
#             # add the channel of fourier basis
#             one_mode_itorchuts = [torch.concat((itorchut_coords[:, [k]], basis.repeat(nv,1,1)[:,[k]]), dim=1) for k in range(rep)]
#             itorchut_net = torch.concat(tuple(one_mode_itorchuts), dim=1).to(device)

#             # prepare the network
#             model = Net_merge_advection(12, 1.7, 20, rep=rep)
#             dicts = []
#             models = []
#             for l in range(s, t+1):
#                 # path = "/work/09452/alberto47/ls6/vesicle/save_models/ves_fft_models/ves_fft_mode"+str(i)+".pth"
#                 path = "/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/ves_adv_trained/ves_fft_mode"+str(l)+".pth"
#                 dicts.append(torch.load(path, map_location=device))
#                 subnet = Net_ves_adv_fft(12, 1.7, 20)
#                 subnet.load_state_dict(dicts[-1])
#                 models.append(subnet.to(device))

#             # organize and match trained weights
#             dict_keys = dicts[-1].keys()
#             new_weights = {}
#             for key in dict_keys:
#                 key_comps = key.split('.')
#                 if key_comps[-1][0:3] =='num':
#                     continue
#                 params = []
#                 for dict in dicts:
#                     params.append(dict[key])
#                 new_weights[key] = torch.concat(tuple(params),dim=0)
#             model.load_state_dict(new_weights, strict=True)
#             model.eval()
#             model.to(device)

#             # Predict using neural networks
#             with torch.no_grad():
#                 Xpredict[s-2:t-1] = model(itorchut_net).reshape(-1,rep,2,256).transpose(0,1)

#         # % Above line approximates multiplication M*(FFTBasis) 
#         # % Now, reconstruct Mvinf = (M*FFTBasis) * vinf_hat
#         Z11 = torch.zeros((nv, 128, 128))
#         Z12 = torch.zeros((nv, 128, 128))
#         Z21 = torch.zeros((nv, 128, 128))
#         Z22 = torch.zeros((nv, 128, 128))
#         # Z11 = torch.zeros((128, 128))
#         # Z12 = torch.zeros((128, 128))
#         # Z21 = torch.zeros((128, 128))
#         # Z22 = torch.zeros((128, 128))

#         for imode in range(2, 129): # the first mode is zero
#             pred = Xpredict[imode - 2]

#             real_mean = self.advNetOutputNorm[imode - 2][0]
#             real_std = self.advNetOutputNorm[imode - 2][1]
#             imag_mean = self.advNetOutputNorm[imode - 2][2]
#             imag_std = self.advNetOutputNorm[imode - 2][3]

#             # % first channel is real
#             pred[:, 0, :] = (pred[:, 0, :] * real_std) + real_mean
#             # % second channel is imaginary
#             pred[:, 1, :] = (pred[:, 1, :] * imag_std) + imag_mean

#             # pred shape: (nv, 2, 256)
#             Z11[:, :, imode-1] = pred[:, 0, :N]
#             Z12[:, :, imode-1] = pred[:, 1, :N] 
#             Z21[:, :, imode-1] = pred[:, 0, N:]
#             Z22[:, :, imode-1] = pred[:, 1, N:]
#             # Z11[:, imode-1] = pred[0, 0, :][:N]
#             # Z12[:, imode-1] = pred[0, 1, :][:N] 
#             # Z21[:, imode-1] = pred[0, 0, :][N:]
#             # Z22[:, imode-1] = pred[0, 1, :][N:]

#         # % Take fft of the velocity (should be standardized velocity)
#         # % only sort points and rotate to pi/2 (no translation, no scaling)
#         vinfStand = self.standardize(vback, [0, 0], rotate, [0, 0], 1, multi_sortIdx)
#         z = vinfStand[:N] + 1j * vinfStand[N:]

#         zh = torch.fft.fft(z, dim=0)
#         V1 = zh.real
#         V2 = zh.imag
#         # % Compute the approximate value of the term M*vinf
#         MVinf = torch.hstack((torch.einsum('BNi,Bi ->BN', Z11, V1.T) + torch.einsum('BNi,Bi ->BN', Z12, V2.T),
#                             torch.einsum('BNi,Bi ->BN', Z21, V1.T) + torch.einsum('BNi,Bi ->BN', Z22, V2.T))).T
#         # MVinf = torch.vstack((torch.dot(Z11, V1) + torch.dot(Z12, V2), torch.dot(Z21, V1) + torch.dot(Z22, V2)))
#         # % update the standardized shape
#         XnewStand = self.dt * vinfStand - self.dt * MVinf   
#         # % destandardize
#         Xadv = self.destandardize(XnewStand, trans, rotate, rotCenter, scaling, multi_sortIdx)
#         # % add the initial since solving dX/dt = (I-M)vinf
#         Xadv = X + Xadv

#         return Xadv


#     def relaxNet(self, X):
#         N = X.shape[0]//2
#         nv = X.shape[1]

#         # % Standardize vesicle
#         Xin, scaling, rotate, rotCenter, trans, multi_sortIdx = self.standardizationStep(X)
#         # % Normalize itorchut
#         x_mean = self.relaxNetItorchutNorm[0]
#         x_std = self.relaxNetItorchutNorm[1]
#         y_mean = self.relaxNetItorchutNorm[2]
#         y_std = self.relaxNetItorchutNorm[3]
        
#         Xstand = torch.copy(Xin)
#         Xin[:N] = (Xin[:N] - x_mean) / x_std
#         Xin[N:] = (Xin[N:] - y_mean) / y_std

#         XinitShape = torch.zeros((nv, 2, 128))
#         XinitShape[:, 0, :] = Xin[:N].T
#         XinitShape[:, 1, :] = Xin[N:].T
#         XinitConv = torch.tensor(XinitShape).float()

#         # Make prediction -- needs to be adjusted for python
#         device = torch.device("cpu")
#         # model = pdeNet_Ves_factor_periodic(14, 2.9)
#         # model.load_state_dict(torch.load("../ves_relax_DIFF_June8_625k_dt1e-5.pth", map_location=device))
#         model = pdeNet_Ves_factor_periodic(14, 2.7)
#         model.load_state_dict(torch.load("../ves_relax_DIFF_IT3_625k_dt1e-5.pth", map_location=device))
        
#         model.eval()
#         with torch.no_grad():
#             DXpredictStand = model(XinitConv)

#         # Denormalize output
#         DXpred = torch.zeros_like(Xin)
#         DXpredictStand = DXpredictStand.numpy()

#         out_x_mean = self.relaxNetOutputNorm[0]
#         out_x_std = self.relaxNetOutputNorm[1]
#         out_y_mean = self.relaxNetOutputNorm[2]
#         out_y_std = self.relaxNetOutputNorm[3]

#         DXpred[:N] = (DXpredictStand[:, 0, :] * out_x_std + out_x_mean).T
#         DXpred[N:] = (DXpredictStand[:, 1, :] * out_y_std + out_y_mean).T

#         # Difference between two time steps predicted, update the configuration
#         Xpred = Xstand + DXpred
#         Xnew = self.destandardize(Xpred, trans, rotate, rotCenter, scaling, multi_sortIdx)
#         return Xnew

#     def standardizationStep(self, Xin):
#         oc = self.oc
#         X = Xin[:]
#         # % Equally distribute points in arc-length
#         for w in range(5):
#             X, _, _ = oc.redistributeArcLength(X)
#         # % standardize angle, center, scaling and point order
#         trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)
        
#         X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
#         return X, scaling, rotate, rotCenter, trans, multi_sortIdx

#     def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
#         N = len(multi_sortIdx[0])
#         Xrotated = self.rotationOperator(X, rotation, rotCenter)
#         Xrotated = self.translateOp(Xrotated, translation)
#         XrotSort = torch.zeros_like(Xrotated)
#         for i in range(X.shape[1]):
#             XrotSort[:,i] = torch.concatenate((Xrotated[multi_sortIdx[i], i], Xrotated[multi_sortIdx[i] + N, i]))
#         XrotSort = scaling*XrotSort
#         return XrotSort

#     def destandardize(self, XrotSort, translation, rotation, rotCenter, scaling, multi_sortIdx):
#         N = len(multi_sortIdx[0])
        
#         XrotSort = XrotSort / scaling
        
#         X = torch.zeros_like(XrotSort)
#         for i in range(len(multi_sortIdx)):
#             X[multi_sortIdx[i], i] = XrotSort[:N,i]
#             X[multi_sortIdx[i] + N, i] = XrotSort[N:,i]
        
#         X = self.translateOp(X, -1*torch.array(translation))
        
#         X = self.rotationOperator(X, -rotation, rotCenter)

#         return X

#     def referenceValues(self, Xref):
#         oc = self.oc
#         N = len(Xref) // 2
#         nv = Xref.shape[1]
#         tempX = torch.zeros_like(Xref)
#         tempX = Xref[:]

#         # Find the physical center
#         center = oc.getPhysicalCenter(tempX)
#         multi_V = oc.getPrincAxesGivenCentroid(tempX,center)
#         w = torch.array([0, 1]) # y-dim unit vector
#         rotation = torch.arctan2(w[1]*multi_V[0]-w[0]*multi_V[1], w[0]*multi_V[0]+w[1]*multi_V[1])
        
#         rotCenter = center # the point around which the frame is rotated
#         Xref = self.rotationOperator(tempX, rotation, rotCenter)
#         center = oc.getPhysicalCenter(Xref) # redundant?
#         translation = -center
        
#         Xref = self.translateOp(Xref, translation)
        
#         multi_sortIdx = []
#         for k in range(nv):
#         # Shan: This for loop can be avoided but will be less readable
#             firstQuad = torch.intersect1d(torch.where(Xref[:N,k] >= 0)[0], torch.where(Xref[N:,k] >= 0)[0])
#             theta = torch.arctan2(Xref[N:,k], Xref[:N,k])
#             idx = torch.argmin(theta[firstQuad])
#             sortIdx = torch.concatenate((torch.arange(firstQuad[idx],N), torch.arange(0, firstQuad[idx])))
#             multi_sortIdx.append(sortIdx)

#         _, _, length = oc.geomProp(Xref)
#         scaling = 1 / length
        
#         return translation, rotation, rotCenter, scaling, multi_sortIdx

#     def rotationOperator(self, X, theta, rotCenter):
#         Xrot = torch.zeros_like(X)
#         x = X[:len(X) // 2]
#         y = X[len(X) // 2:]

#         xrot = (x-rotCenter[0]) * torch.cos(theta) - (y-rotCenter[1]) * torch.sin(theta) + rotCenter[0]
#         yrot = (x-rotCenter[0]) * torch.sin(theta) + (y-rotCenter[1]) * torch.cos(theta) + rotCenter[1]

#         Xrot[:len(X) // 2] = xrot
#         Xrot[len(X) // 2:] = yrot
#         return Xrot

#     def translateOp(self, X, transXY):
#         Xnew = torch.zeros_like(X)
#         Xnew[:len(X) // 2] = X[:len(X) // 2] + transXY[0]
#         Xnew[len(X) // 2:] = X[len(X) // 2:] + transXY[1]
#         return Xnew


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
    
    def many_time_step(self, Xold, tenOld):
        oc = self.oc
        torch.set_default_device(Xold.device)
        # background velocity on vesicles
        vback = self.vinf(Xold)

        # build vesicle class at the current step
        vesicle = capsules(Xold, [], [], self.kappa, 1)
        nv = vesicle.nv
        N = vesicle.N

        # Compute bending forces + old tension forces
        fBend = vesicle.bendingTerm(Xold)
        fTen = vesicle.tensionTerm(tenOld)
        tracJump = fBend + fTen  # total elastic force

        # Explicit Tension at the Current Step
        # Calculate velocity induced by vesicles on each other due to elastic force
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions(vesicle, tracJump, oc)

        # Solve for tension
        vBackSolve = self.invTenMatOnVback(Xold, vback + farFieldtracJump)
        selfBendSolve = self.invTenMatOnSelfBend(Xold)
        tenNew = -(vBackSolve + selfBendSolve)
        tenNew = upsThenFilterTension(tenNew, 4*N, 16)

        # update the elastic force with the new tension
        fTen_new = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen_new

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions(vesicle, tracJump, oc)

        # Total background velocity
        vbackTotal = vback + farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        Xadv = self.translateVinfwTorch(Xold, vbackTotal)
        Xadv = upsThenFilterShape(Xadv, 4*N, 16)
        # XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        # Xadv = oc.alignCenterAngle(Xadv, XadvC.to(Xold.device))
        

        # Compute the action of relax operator on Xold + Xadv
        Xnew = self.relaxWTorchNet(Xadv)
        Xnew = upsThenFilterShape(Xnew, 4*N, 16)
        for _ in range(5):
            Xnew, flag = oc.redistributeArcLength(Xnew)
            if flag:
                break
        XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
        Xnew = oc.alignCenterAngle(Xnew, XnewC.to(Xold.device))
            
        return Xnew, tenNew
    
    def single_time_step(self, Xold):
        
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


    def predictNearLayersWTorchNet(self, X, tracJump):
        N = X.shape[0] // 2
        nv = X.shape[1]

        oc = self.oc

        # maxLayerDist = np.sqrt(1 / N) # length = 1, h = 1/N;
        maxLayerDist = (1 / N) # length = 1, h = 1/N;
        nlayers = 3 # three layers
        dlayer = torch.linspace(0, maxLayerDist, nlayers, dtype=torch.float64)

        # Create the layers around a vesicle on which velocity calculated
        tracersX = torch.zeros((2 * N, nlayers, nv), dtype=torch.float64)
        # Standardize itorchut
        # Shan: standardizationStep is compatible with multiple ves
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)
        for k in range(nv):
            _, tang, _ = oc.diffProp(Xstand[:, [k]])
            nx = tang[N:].squeeze()
            ny = -tang[:N].squeeze()

            tracersX[:, 0, k] = Xstand[:, k]
            for il in range(1, nlayers):
                tracersX[:, il, k] = torch.hstack([Xstand[:N, k] + nx * dlayer[il], Xstand[N:, k] + ny * dlayer[il]])

        # How many modes to be used
        # MATLAB: modes = [(0:N/2-1) (-N/2:-1)]
        # modes = torch.concatenate((torch.arange(0,N/2), torch.arange(-N/2,0)))
        # modesInUse = 16
        # modeList = torch.where(torch.abs(modes) <= modesInUse)[0] # Shan: bug, creates 33 modes
        # modeList = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]

        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(net_pred)
        
        # Standardize tracJump
        # fstandRe = torch.zeros((N, nv))
        # fstandIm = torch.zeros((N, nv))
        # for k in range(nv):
        #     # fstand = self.standardize(tracJump[:, k], [0, 0], rotate[k], [0, 0], 1, sortIdx[k])
        #     z = fstand[:N] + 1j * fstand[N:]
        #     zh = torch.fft.fft(z)
        #     fstandRe[:, k] = torch.real(zh)
        #     fstandIm[:, k] = torch.imag(zh)
                
        fstand = self.standardize(tracJump, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
        z = fstand[:N, :] + 1j * fstand[N:, :]
        zh = torch.fft.fft(z, dim=0)
        fstandRe = torch.real(zh)
        fstandIm = torch.imag(zh)

        # Initialize outputs
        # velx_real = [torch.zeros((N, N, nlayers)) for _ in range(nv)]
        # vely_real = [torch.zeros((N, N, nlayers)) for _ in range(nv)]
        # velx_imag = [torch.zeros((N, N, nlayers)) for _ in range(nv)]
        # vely_imag = [torch.zeros((N, N, nlayers)) for _ in range(nv)]

        # Denormalize output
        # for ij, imode in enumerate(modeList):
        #     pred = Xpredict[ij]
        #     for k in range(nv):
        #         for ic in range(nlayers):
        #             velx_real[k][:, imode, ic] = (pred[k, ic] * out_param[imode, 1, ic]) + out_param[imode, 0, ic]
        #             vely_real[k][:, imode, ic] = (pred[k, nlayers + ic] * out_param[imode, 1, nlayers + ic]) + out_param[imode, 0, nlayers + ic]
        #             velx_imag[k][:, imode, ic] = (pred[k, 2 * nlayers + ic] * out_param[imode, 1, 2 * nlayers + ic]) + out_param[imode, 0, 2 * nlayers + ic]
        #             vely_imag[k][:, imode, ic] = (pred[k, 3 * nlayers + ic] * out_param[imode, 1, 3 * nlayers + ic]) + out_param[imode, 0, 3 * nlayers + ic]

        velx = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        vely = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        xlayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
        ylayers = torch.zeros((N, nlayers, nv), dtype=torch.float64)
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

        for il in range(nlayers):
            Xl = self.destandardize(tracersX[:, il], trans, rotate, rotCent, scaling, sortIdx)
            xlayers[:, il] = Xl[:N]
            ylayers[:, il] = Xl[N:]

        return xlayers, ylayers, velx, vely

    def computeStokesInteractions(self, vesicle, trac_jump, oc):
        print('Near-singular interaction through interpolation and network')
        N = vesicle.N
        nv = vesicle.nv

        # Compute near/far hydro interactions without any correction
        # First calculate the far-field
        far_field = torch.zeros((2 * N, nv), dtype=torch.float64)
        for k in range(nv):
            K = list(range(nv))
            K.remove(k)
            far_field[:, [k]] = self.exactStokesSL(vesicle, trac_jump, vesicle.X[:, [k]], K)

        self.nearFieldCorrection(oc, vesicle, trac_jump, far_field, option='kdtree')
        
        return far_field
    
    def nearFieldCorrection(self, oc, vesicle, trac_jump, far_field,
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
            xlayers, ylayers, velx, vely = self.predictNearLayersWTorchNet(vesicle.X, trac_jump)

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

        # If we only use some modes
        # modes = torch.concatenate((torch.arange(0, N//2), torch.arange(-N//2, 0)))
        # modesInUse = 16
        # mode_list = torch.where(torch.abs(modes) <= modes_in_use)[0]
        # mode_list = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]
        mode_list = np.arange(128)
        # Standardize itorchut
        # Xstand = torch.zeros_like(Xold)
        # scaling = torch.zeros(nv)
        # rotate = torch.zeros(nv)
        # rot_cent = torch.zeros((2, nv))
        # trans = torch.zeros((2, nv))
        # sort_idx = torch.zeros((N, nv), dtype=int)
        # for k in range(nv):
        #     (Xstand[:, k], scaling[k], rotate[k], 
        #     rot_cent[:, k], trans[:, k], sort_idx[:, k]) = self.standardization_step(Xold[:, k], N)
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(Xold)

        # Normalize itorchut
        # itorchut_list = []
        # for imode in mode_list:
        #     if imode != 0:
        #         # itorchut_net = torch.zeros((nv, 2, 128)) # Shan: should be (nv, 2, 256)
        #         x_mean, x_std, y_mean, y_std = in_param[imode-1]
        #         # for k in range(nv):
        #         #     itorchut_net[k, 0, :] = (Xstand[:N, k] - x_mean) / x_std
        #         #     itorchut_net[k, 1, :] = (Xstand[N:, k] - y_mean) / y_std
        #         itorchut_net = torch.concatenate(((Xstand[:N, None] - x_mean)/x_std, (Xstand[N:, None] - y_mean) / y_std), dim=0).T
        #         # prepare fourier basis to be combined into itorchut
        #         theta = torch.arange(N)/N*2*torch.pi
        #         theta = theta.reshape(N,1)
        #         bases = 1/N*torch.exp(1j*theta*torch.arange(N).reshape(1,N))
        #         rr, ii = torch.real(bases[:, imode]), torch.imag(bases[:, imode])
        #         basis = torch.concatenate((rr,ii)).reshape(1,1,256).repeat(nv, dim=0)
        #         itorchut_net = torch.concatenate((itorchut_net, basis), dim=1)
        #         itorchut_list.append(itorchut_net)

        # Xpredict = pyrunfile("advect_predict.py", "output_list", itorchut_shape=itorchut_list, num_ves=nv)
        # data = scio.loadmat("../advectionNetCheck.mat")

        # X = torch.tensor(data['X'])
        # # # Xstand = torch.tensor(data['Xstand'])
        # MVinf = torch.tensor(data['MVinf'])
        # # # MVinfNN = torch.tensor(data['MVinfNN'])
        # vback = torch.tensor(data['vback'])
        # # # vinfstand = torch.tensor(data['vinfStand'])
        # trueZ11 = torch.tensor(data['Z11true'])
        # trueZ21 = torch.tensor(data['Z21true'])
        # trueZ22 = torch.tensor(data['Z22true'])
        # trueZ12 = torch.tensor(data['Z12true'])
        
        Xpredict = self.mergedAdvNetwork.forward(Xstand.to(self.device))
        Xpredict = Xpredict.cpu()
        # Xpredict = torch.from_numpy(np.load("singles_adv.npy"))
        # np.save("Xpredict_adv.npy", Xpredict.numpy())
        # Xpredict = torch.ones(127, nv, 2, 256)
        # Approximate the multiplication M*(FFTBasis)
        Z11r = torch.zeros((N, N, nv), dtype=torch.float64)
        Z12r = torch.zeros_like(Z11r)
        Z21r = torch.zeros_like(Z11r)
        Z22r = torch.zeros_like(Z11r)

        # for ij in range(len(mode_list) - 1):
        for ij in range(127):
            imode = mode_list[ij + 1] # imode skips 0 and starts with 1
            if imode == 64:
                continue
            pred = Xpredict[ij] # Xpredict is (127, nv, 2, 256)

            for k in range(nv):
                Z11r[:, imode, k] = pred[k, 0, :N]
                Z21r[:, imode, k] = pred[k, 0, N:]
                Z12r[:, imode, k] = pred[k, 1, :N]
                Z22r[:, imode, k] = pred[k, 1, N:]

        # print(torch.norm(trueZ11[:,mode-1] - out[0,0,:128])/torch.norm(trueZ11[:,1]))
        # print(torch.norm(trueZ21[:,mode-1] - out[0,0,128:])/torch.norm(trueZ21[:,1]))
        # print(torch.norm(trueZ12[:,mode-1] - out[0,1,:128])/torch.norm(trueZ12[:,1]))
        # print(torch.norm(trueZ22[:,mode-1] - out[0,1,128:])/torch.norm(trueZ22[:,1]))
        
        # Take fft of the velocity (should be standardized velocity)
        # only sort points and rotate to pi/2 (no translation, no scaling)
        Xnew = torch.zeros_like(Xold)
        vinf_stand = self.standardize(vinf, torch.zeros((2,nv), dtype=torch.float64), rotate, torch.zeros((2,nv), dtype=torch.float64), 1, sortIdx)
        z = vinf_stand[:N] + 1j * vinf_stand[N:]
        zh = torch.fft.fft(z, dim=0)
        V1, V2 = torch.real(zh), torch.imag(zh)
        MVinf_stand = torch.vstack((torch.einsum('NiB,iB ->NB', Z11r, V1) + torch.einsum('NiB,iB ->NB', Z12r, V2),
                               torch.einsum('NiB,iB ->NB', Z21r, V1) + torch.einsum('NiB,iB ->NB', Z22r, V2)))
           
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

        return Xnew

    def relaxWTorchNet(self, Xmid):
        # 1) RELAXATION w/ NETWORK
        # Standardize vesicle Xmid
        # nv = Xmid.shape[1]
        # N = Xmid.shape[0] // 2

        Xin, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(Xmid)

        # # INPUT NORMALIZATION INFO
        # # x_mean, x_std, y_mean, y_std = self.relaxNetItorchutNorm

        # # INPUT NORMALIZING
        # # REAL SPACE
        # Xstand = Xin.copy()  # before normalization
        # Xin[:N, :] = (Xin[:N, :] - x_mean) / x_std
        # Xin[N:, :] = (Xin[N:, :] - y_mean) / y_std
        # XinitShape = torch.zeros((nv, 2, 128))
        # for k in range(nv):
        #     XinitShape[k, 0, :] = Xin[:N, k]
        #     XinitShape[k, 1, :] = Xin[N:, k]
        # XinitConv = torch.tensor(XinitShape)

        # # OUTPUT
        # # June8 - Dt1E5
        # # DXpredictStand = pyrunfile("relax_predict_DIFF_June8_dt1E5.py", "predicted_shape", itorchut_shape=XinitConv)
        # DXpredictStand = torch.random.rand(nv, 2, 128)
        
        # # For the 625k - June8 - Dt = 1E-5 data
        # x_mean, x_std, y_mean, y_std = self.relaxNetOutputNorm

        # DXpred = torch.zeros_like(Xin)
        # DXpredictStand = torch.array(DXpredictStand)
        # # Xnew = torch.zeros_like(Xmid)

        # # for k in range(nv):
        # #     # normalize output
        # #     DXpred[:N, k] = DXpredictStand[k, 0, :] * x_std + x_mean
        # #     DXpred[N:, k] = DXpredictStand[k, 1, :] * y_std + y_mean

        # #     DXpred[:, k] = DXpred[:, k] / 1E-5 * self.dt  # scale the output if dt is other than 1E-5
        # #     Xpred = Xstand[:, k] + DXpred[:, k]

        # DXpred[:N] = (DXpredictStand[0, :] * x_std + x_mean).T
        # DXpred[N:] = (DXpredictStand[1, :] * y_std + y_mean).T
        # DXpred = DXpred / 1E-5 * self.dt
        # Xpred = Xstand + DXpred

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
        # modes = torch.concatenate((torch.arange(0, N//2), torch.arange(-N//2, 0)))
        # modesInUse = 16
        # modeList = torch.where(torch.abs(modes) <= modesInUse)[0]
        # modeList = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]
        modeList = np.arange(128)
        # for k in range(nv):
        #     Xstand[:, k], scaling[k], rotate[k], rotCent[:, k], trans[:, k], sortIdx[:, k] = \
        #         self.standardizationStep(X[:, k], 128)
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)

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

        return vBackSolve

    def invTenMatOnSelfBend(self, X):
        # Approximate inv(Div*G*Ten)*G*(-Ben)*x

        # number of vesicles
        nv = X.shape[1]
        N = X.shape[0] // 2

        # for k in range(nv):
        #     Xstand[:, k], scaling[k], rotate[k], rotCent[:, k], trans[:, k], sortIdx[:, k] = \
        #         self.standardizationStep(X[:, k], 128)
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)

        # Normalize itorchut
        # x_mean, x_std, y_mean, y_std = self.tenSelfNetItorchutNorm

        # Adjust the itorchut shape for the network
        # XinitShape = torch.zeros((nv, 2, 128))
        # for k in range(nv):
        #     XinitShape[k, 0, :] = (Xstand[:N, k] - x_mean) / x_std
        #     XinitShape[k, 1, :] = (Xstand[N:, k] - y_mean) / y_std
        # XinitConv = torch.tensor(XinitShape)

        # Make prediction -- needs to be adjusted for python
        # tenPredictStand = pyrunfile("tension_self_network.py", "predicted_shape", itorchut_shape=XinitConv)
        # tenPredictStand = torch.random.rand(nv, 1, 128)

        # Denormalize output
        # out_mean, out_std = self.tenSelfNetOutputNorm

        # tenPred = torch.zeros((N, nv))
        # tenPredictStand = torch.array(tenPredictStand, dtype=float)

        tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        tenPredictStand = tenPredictStand.double()
        tenPred = torch.zeros((N, nv), dtype=torch.float64)
        for k in range(nv):
            # also destandardize
            tenPred[sortIdx[k], k] = tenPredictStand[:,k]/ scaling[k]**2

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
    

    # def standardize(self, X, translation, rotation, rotCent, scaling, sortIdx):
    #     N = len(sortIdx)

    #     # Translate, rotate and scale configuration
    #     Xrotated = self.rotationOperator(X, rotation, rotCent)
    #     Xrotated = self.translateOp(Xrotated, translation)

    #     # Now order the points
    #     XrotSort = torch.concatenate([Xrotated[sortIdx], Xrotated[sortIdx + N]])

    #     XrotSort = scaling * XrotSort

    #     return XrotSort
    
    def standardizationStep(self, Xin):
        # compatible with multi ves
        oc = self.oc
        X = Xin[:]
        # % Equally distribute points in arc-length
        for _ in range(5):
            X, flag = oc.redistributeArcLength(X)
            if flag:
                break
            
        # X = oc.alignCenterAngle(Xin,X)
        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)
        
        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, scaling, rotate, rotCenter, trans, multi_sortIdx

    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        # compatible with multi ves
        N = len(multi_sortIdx[0])
        Xrotated = self.rotationOperator(X, rotation, rotCenter)
        Xrotated = self.translateOp(Xrotated, translation)
        XrotSort = torch.zeros_like(Xrotated)
        for i in range(X.shape[1]):
            XrotSort[:,i] = torch.concatenate((Xrotated[multi_sortIdx[i], i], Xrotated[multi_sortIdx[i] + N, i]))
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
        center = oc.getPhysicalCenter(Xref) # redundant?
        translation = -center
        
        Xref = self.translateOp(Xref, translation)
        
        multi_sortIdx = []
        for k in range(nv):
        # Shan: This for loop can be avoided but will be less readable
            firstQuad = np.intersect1d(torch.where(Xref[:N,k] >= 0)[0].cpu(), torch.where(Xref[N:,k] >= 0)[0].cpu())
            theta = torch.arctan2(Xref[N:,k], Xref[:N,k])
            idx = torch.argmin(theta[firstQuad])
            sortIdx = torch.concatenate((torch.arange(firstQuad[idx],N), torch.arange(0, firstQuad[idx])))
            multi_sortIdx.append(sortIdx)

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


