import numpy as np
import torch
import sys
sys.path.append("..")
from capsules import capsules
from rayCasting import ray_casting
from rbf_create import rbfcreate
from rbf_interp import rbfinterp
from model_zoo.get_network import RelaxNetwork, TenSelfNetwork, MergedAdvNetwork, MergedTenAdvNetwork, MergedNearFourierNetwork


class MLARM_py:
    def __init__(self, dt, vinf, oc, advNetInputNorm, advNetOutputNorm, relaxNetInputNorm, relaxNetOutputNorm):
        self.dt = dt
        self.vinf = vinf # background flow (analytic -- input as function of vesicle config)
        self.oc = oc # curve class
        # % Normalization values for advection (translation) networks
        self.advNetInputNorm = advNetInputNorm
        self.advNetOutputNorm = advNetOutputNorm
        # % Normalization values for relaxation network
        self.relaxNetInputNorm = relaxNetInputNorm
        self.relaxNetOutputNorm = relaxNetOutputNorm
        self.area0 = None  # initial area of vesicle
        self.len0 = None  # initial length of vesicle

    def time_step(self, X):
        # % take a time step with neural networks
        oc = self.oc
        vback = self.vinf(X)

        # 1) Translate vesicle with network
        # Xadv = self.translateVinfNet(X, vback)
        Xadv = self.translateVinfMergeNet(X, vback, 12)

        # Correct area and length
        XadvC = oc.correctAreaAndLength(Xadv, self.area0, self.len0)
        Xadv = oc.alignCenterAngle(Xadv, XadvC)

        # 2) Relax vesicle with network
        Xnew = self.relaxNet(Xadv)

        # Correct area and length
        XnewC = oc.correctAreaAndLength(Xnew, self.area0, self.len0)
        Xnew = oc.alignCenterAngle(Xnew, XnewC)

        return Xnew

    def translateVinfNet(self, X, vback):
        # Translate vesicle using networks
        N = X.shape[0]//2
        nv = X.shape[1]
        # % Standardize vesicle (zero center, pi/2 inclination angle, equil dist)
        Xstand, scaling, rotate, rotCenter, trans, sortIdx = self.standardizationStep(X)
        device = torch.device("cpu")
        Xpredict = torch.zeros(127, nv, 2, 256).to(device)
        # Normalize input
        coords = torch.zeros((nv, 2, 128)).to(device)
        for imode in range(2, 129):
            x_mean = self.advNetInputNorm[imode - 2][0]
            x_std = self.advNetInputNorm[imode - 2][1]
            y_mean = self.advNetInputNorm[imode - 2][2]
            y_std = self.advNetInputNorm[imode - 2][3]

            coords[:, 0, :] = torch.from_numpy((Xstand[:N].T - x_mean) / x_std).float()
            coords[:, 1, :] = torch.from_numpy((Xstand[N:].T - y_mean) / y_std).float()

            # coords (N,2,128) -> (N,1,256)
            input_net = torch.concat((coords[:,0], coords[:,1]), dim=1)[:,None,:]
            # specify which mode, imode=2,3,...,128
            theta = np.arange(N)/N*2*np.pi
            theta = theta.reshape(N,1)
            bases = 1/N*np.exp(1j*theta*np.arange(N).reshape(1,N))
            rr, ii = np.real(bases[:,imode-1]), np.imag(bases[:,imode-1])
            basis = torch.from_numpy(np.concatenate((rr,ii))).float().reshape(1,1,256).to(device)
            # add the channel of fourier basis
            input_net = torch.concat((input_net, basis.repeat(nv,1,1)), dim=1).to(device)

            # Predict using neural networks
            model = Net_ves_adv_fft(12,1.7,20)
            model.load_state_dict(torch.load(f"../ves_adv_trained/ves_fft_mode{imode}.pth", map_location=device))
            model.eval()
            with torch.no_grad():
                Xpredict[imode - 2] = model(input_net)

        # % Above line approximates multiplication M*(FFTBasis) 
        # % Now, reconstruct Mvinf = (M*FFTBasis) * vinf_hat
        Z11 = np.zeros((128, 128))
        Z12 = np.zeros((128, 128))
        Z21 = np.zeros((128, 128))
        Z22 = np.zeros((128, 128))

        for imode in range(2, 129): # the first mode is zero
            pred = Xpredict[imode - 2]

            real_mean = self.advNetOutputNorm[imode - 2][0]
            real_std = self.advNetOutputNorm[imode - 2][1]
            imag_mean = self.advNetOutputNorm[imode - 2][2]
            imag_std = self.advNetOutputNorm[imode - 2][3]

            # % first channel is real
            pred[:, 0, :] = (pred[:, 0, :] * real_std) + real_mean
            # % second channel is imaginary
            pred[:, 1, :] = (pred[:, 1, :] * imag_std) + imag_mean

            Z11[:, imode-1] = pred[0, 0, :][:N]
            Z12[:, imode-1] = pred[0, 1, :][:N] 
            Z21[:, imode-1] = pred[0, 0, :][N:]
            Z22[:, imode-1] = pred[0, 1, :][N:]

        # % Take fft of the velocity (should be standardized velocity)
        # % only sort points and rotate to pi/2 (no translation, no scaling)
        vinfStand = self.standardize(vback, [0, 0], rotate, [0, 0], 1, sortIdx)
        z = vinfStand[:N] + 1j * vinfStand[N:]

        zh = np.fft.fft(z)
        V1 = zh.real
        V2 = zh.imag
        # % Compute the approximate value of the term M*vinf
        MVinf = np.vstack((np.dot(Z11, V1) + np.dot(Z12, V2), np.dot(Z21, V1) + np.dot(Z22, V2)))
        # % update the standardized shape
        XnewStand = self.dt * vinfStand - self.dt * MVinf   
        # % destandardize
        Xadv = self.destandardize(XnewStand, trans, rotate, rotCenter, scaling, sortIdx)
        # % add the initial since solving dX/dt = (I-M)vinf
        Xadv = X + Xadv

        return Xadv

    def translateVinfMergeNet(self, X, vback, num_modes):
        # Translate vesicle using networks
        N = X.shape[0]//2
        nv = X.shape[1]
        # % Standardize vesicle (zero center, pi/2 inclination angle, equil dist)
        Xstand, scaling, rotate, rotCenter, trans, multi_sortIdx = self.standardizationStep(X)
        device = torch.device("cpu")
        Xpredict = torch.zeros(127, nv, 2, 256).to(device)
        # prepare fourier basis
        theta = np.arange(N)/N*2*np.pi
        theta = theta.reshape(N,1)
        bases = 1/N*np.exp(1j*theta*np.arange(N).reshape(1,N))

        for i in range(127//num_modes+1):
            # from s mode to t mode, both end included
            s = 2 + i*num_modes
            t = min(2 + (i+1)*num_modes -1, 128)
            print(f"from mode {s} to mode {t}")
            rep = t - s + 1 # number of repetitions
            Xstand = Xstand.reshape(2*N, nv, 1)
            multiX = torch.from_numpy(Xstand).float().repeat(1,1,rep)

            x_mean = self.advNetInputNorm[s-2:t-1][:,0]
            x_std = self.advNetInputNorm[s-2:t-1][:,1]
            y_mean = self.advNetInputNorm[s-2:t-1][:,2]
            y_std = self.advNetInputNorm[s-2:t-1][:,3]

            coords = torch.zeros((nv, 2*rep, 128)).to(device)
            coords[:, :rep, :] = ((multiX[:N] - x_mean) / x_std).permute(1,2,0)
            coords[:, rep:, :] = ((multiX[N:] - y_mean) / y_std).permute(1,2,0)

            # coords (N,2*rep,128) -> (N,rep,256)
            input_coords = torch.concat((coords[:,:rep], coords[:,rep:]), dim=-1)
            # specify which mode
            rr, ii = np.real(bases[:,s-1:t]), np.imag(bases[:,s-1:t])
            basis = torch.from_numpy(np.concatenate((rr,ii),axis=-1)).float().reshape(1,rep,256).to(device)
            # add the channel of fourier basis
            one_mode_inputs = [torch.concat((input_coords[:, [k]], basis.repeat(nv,1,1)[:,[k]]), dim=1) for k in range(rep)]
            input_net = torch.concat(tuple(one_mode_inputs), dim=1).to(device)

            # prepare the network
            model = Net_merge_advection(12, 1.7, 20, rep=rep)
            dicts = []
            models = []
            for l in range(s, t+1):
                # path = "/work/09452/alberto47/ls6/vesicle/save_models/ves_fft_models/ves_fft_mode"+str(i)+".pth"
                path = "/work/09452/alberto47/ls6/vesToPY/Ves2Dpy/ves_adv_trained/ves_fft_mode"+str(l)+".pth"
                dicts.append(torch.load(path, map_location=device))
                subnet = Net_ves_adv_fft(12, 1.7, 20)
                subnet.load_state_dict(dicts[-1])
                models.append(subnet.to(device))

            # organize and match trained weights
            dict_keys = dicts[-1].keys()
            new_weights = {}
            for key in dict_keys:
                key_comps = key.split('.')
                if key_comps[-1][0:3] =='num':
                    continue
                params = []
                for dict in dicts:
                    params.append(dict[key])
                new_weights[key] = torch.concat(tuple(params),dim=0)
            model.load_state_dict(new_weights, strict=True)
            model.eval()
            model.to(device)

            # Predict using neural networks
            with torch.no_grad():
                Xpredict[s-2:t-1] = model(input_net).reshape(-1,rep,2,256).transpose(0,1)

        # % Above line approximates multiplication M*(FFTBasis) 
        # % Now, reconstruct Mvinf = (M*FFTBasis) * vinf_hat
        Z11 = np.zeros((nv, 128, 128))
        Z12 = np.zeros((nv, 128, 128))
        Z21 = np.zeros((nv, 128, 128))
        Z22 = np.zeros((nv, 128, 128))
        # Z11 = np.zeros((128, 128))
        # Z12 = np.zeros((128, 128))
        # Z21 = np.zeros((128, 128))
        # Z22 = np.zeros((128, 128))

        for imode in range(2, 129): # the first mode is zero
            pred = Xpredict[imode - 2]

            real_mean = self.advNetOutputNorm[imode - 2][0]
            real_std = self.advNetOutputNorm[imode - 2][1]
            imag_mean = self.advNetOutputNorm[imode - 2][2]
            imag_std = self.advNetOutputNorm[imode - 2][3]

            # % first channel is real
            pred[:, 0, :] = (pred[:, 0, :] * real_std) + real_mean
            # % second channel is imaginary
            pred[:, 1, :] = (pred[:, 1, :] * imag_std) + imag_mean

            # pred shape: (nv, 2, 256)
            Z11[:, :, imode-1] = pred[:, 0, :N]
            Z12[:, :, imode-1] = pred[:, 1, :N] 
            Z21[:, :, imode-1] = pred[:, 0, N:]
            Z22[:, :, imode-1] = pred[:, 1, N:]
            # Z11[:, imode-1] = pred[0, 0, :][:N]
            # Z12[:, imode-1] = pred[0, 1, :][:N] 
            # Z21[:, imode-1] = pred[0, 0, :][N:]
            # Z22[:, imode-1] = pred[0, 1, :][N:]

        # % Take fft of the velocity (should be standardized velocity)
        # % only sort points and rotate to pi/2 (no translation, no scaling)
        vinfStand = self.standardize(vback, [0, 0], rotate, [0, 0], 1, multi_sortIdx)
        z = vinfStand[:N] + 1j * vinfStand[N:]

        zh = np.fft.fft(z, axis=0)
        V1 = zh.real
        V2 = zh.imag
        # % Compute the approximate value of the term M*vinf
        MVinf = np.hstack((np.einsum('BNi,Bi ->BN', Z11, V1.T) + np.einsum('BNi,Bi ->BN', Z12, V2.T),
                            np.einsum('BNi,Bi ->BN', Z21, V1.T) + np.einsum('BNi,Bi ->BN', Z22, V2.T))).T
        # MVinf = np.vstack((np.dot(Z11, V1) + np.dot(Z12, V2), np.dot(Z21, V1) + np.dot(Z22, V2)))
        # % update the standardized shape
        XnewStand = self.dt * vinfStand - self.dt * MVinf   
        # % destandardize
        Xadv = self.destandardize(XnewStand, trans, rotate, rotCenter, scaling, multi_sortIdx)
        # % add the initial since solving dX/dt = (I-M)vinf
        Xadv = X + Xadv

        return Xadv


    def relaxNet(self, X):
        N = X.shape[0]//2
        nv = X.shape[1]

        # % Standardize vesicle
        Xin, scaling, rotate, rotCenter, trans, multi_sortIdx = self.standardizationStep(X)
        # % Normalize input
        x_mean = self.relaxNetInputNorm[0]
        x_std = self.relaxNetInputNorm[1]
        y_mean = self.relaxNetInputNorm[2]
        y_std = self.relaxNetInputNorm[3]
        
        Xstand = np.copy(Xin)
        Xin[:N] = (Xin[:N] - x_mean) / x_std
        Xin[N:] = (Xin[N:] - y_mean) / y_std

        XinitShape = np.zeros((nv, 2, 128))
        XinitShape[:, 0, :] = Xin[:N].T
        XinitShape[:, 1, :] = Xin[N:].T
        XinitConv = torch.tensor(XinitShape).float()

        # Make prediction -- needs to be adjusted for python
        device = torch.device("cpu")
        # model = pdeNet_Ves_factor_periodic(14, 2.9)
        # model.load_state_dict(torch.load("../ves_relax_DIFF_June8_625k_dt1e-5.pth", map_location=device))
        model = pdeNet_Ves_factor_periodic(14, 2.7)
        model.load_state_dict(torch.load("../ves_relax_DIFF_IT3_625k_dt1e-5.pth", map_location=device))
        
        model.eval()
        with torch.no_grad():
            DXpredictStand = model(XinitConv)

        # Denormalize output
        DXpred = np.zeros_like(Xin)
        DXpredictStand = DXpredictStand.numpy()

        out_x_mean = self.relaxNetOutputNorm[0]
        out_x_std = self.relaxNetOutputNorm[1]
        out_y_mean = self.relaxNetOutputNorm[2]
        out_y_std = self.relaxNetOutputNorm[3]

        DXpred[:N] = (DXpredictStand[:, 0, :] * out_x_std + out_x_mean).T
        DXpred[N:] = (DXpredictStand[:, 1, :] * out_y_std + out_y_mean).T

        # Difference between two time steps predicted, update the configuration
        Xpred = Xstand + DXpred
        Xnew = self.destandardize(Xpred, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return Xnew

    def standardizationStep(self, Xin):
        oc = self.oc
        X = Xin[:]
        # % Equally distribute points in arc-length
        for w in range(5):
            X, _, _ = oc.redistributeArcLength(X)
        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)
        
        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, scaling, rotate, rotCenter, trans, multi_sortIdx

    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        N = len(multi_sortIdx[0])
        Xrotated = self.rotationOperator(X, rotation, rotCenter)
        Xrotated = self.translateOp(Xrotated, translation)
        XrotSort = np.zeros_like(Xrotated)
        for i in range(X.shape[1]):
            XrotSort[:,i] = np.concatenate((Xrotated[multi_sortIdx[i], i], Xrotated[multi_sortIdx[i] + N, i]))
        XrotSort = scaling*XrotSort
        return XrotSort

    def destandardize(self, XrotSort, translation, rotation, rotCenter, scaling, multi_sortIdx):
        N = len(multi_sortIdx[0])
        
        XrotSort = XrotSort / scaling
        
        X = np.zeros_like(XrotSort)
        for i in range(len(multi_sortIdx)):
            X[multi_sortIdx[i], i] = XrotSort[:N,i]
            X[multi_sortIdx[i] + N, i] = XrotSort[N:,i]
        
        X = self.translateOp(X, -1*np.array(translation))
        
        X = self.rotationOperator(X, -rotation, rotCenter)

        return X

    def referenceValues(self, Xref):
        oc = self.oc
        N = len(Xref) // 2
        nv = Xref.shape[1]
        tempX = np.zeros_like(Xref)
        tempX = Xref[:]

        # Find the physical center
        center = oc.getPhysicalCenter(tempX)
        multi_V = oc.getPrincAxesGivenCentroid(tempX,center)
        w = np.array([0, 1]) # y-axis unit vector
        rotation = np.arctan2(w[1]*multi_V[0]-w[0]*multi_V[1], w[0]*multi_V[0]+w[1]*multi_V[1])
        
        rotCenter = center # the point around which the frame is rotated
        Xref = self.rotationOperator(tempX, rotation, rotCenter)
        center = oc.getPhysicalCenter(Xref) # redundant?
        translation = -center
        
        Xref = self.translateOp(Xref, translation)
        
        multi_sortIdx = []
        for k in range(nv):
        # Shan: This for loop can be avoided but will be less readable
            firstQuad = np.intersect1d(np.where(Xref[:N,k] >= 0)[0], np.where(Xref[N:,k] >= 0)[0])
            theta = np.arctan2(Xref[N:,k], Xref[:N,k])
            idx = np.argmin(theta[firstQuad])
            sortIdx = np.concatenate((np.arange(firstQuad[idx],N), np.arange(0, firstQuad[idx])))
            multi_sortIdx.append(sortIdx)

        _, _, length = oc.geomProp(Xref)
        scaling = 1 / length
        
        return translation, rotation, rotCenter, scaling, multi_sortIdx

    def rotationOperator(self, X, theta, rotCenter):
        Xrot = np.zeros_like(X)
        x = X[:len(X) // 2]
        y = X[len(X) // 2:]

        xrot = (x-rotCenter[0]) * np.cos(theta) - (y-rotCenter[1]) * np.sin(theta) + rotCenter[0]
        yrot = (x-rotCenter[0]) * np.sin(theta) + (y-rotCenter[1]) * np.cos(theta) + rotCenter[1]

        Xrot[:len(X) // 2] = xrot
        Xrot[len(X) // 2:] = yrot
        return Xrot

    def translateOp(self, X, transXY):
        Xnew = np.zeros_like(X)
        Xnew[:len(X) // 2] = X[:len(X) // 2] + transXY[0]
        Xnew[len(X) // 2:] = X[len(X) // 2:] + transXY[1]
        return Xnew


class MLARM_manyfree_py:
    def __init__(self, dt, vinf, oc, advNetInputNorm, advNetOutputNorm,
                 relaxNetInputNorm, relaxNetOutputNorm, nearNetInputNorm,
                 nearNetOutputNorm, tenSelfNetInputNorm, tenSelfNetOutputNorm,
                 tenAdvNetInputNorm, tenAdvNetOutputNorm, device):
        self.dt = dt  # time step size
        self.vinf = vinf  # background flow (analytic -- input as function of vesicle config)
        self.oc = oc  # curve class
        self.kappa = 1  # bending stiffness is 1 for our simulations
        self.device = device
        
        # Normalization values for advection (translation) networks
        self.advNetInputNorm = advNetInputNorm
        self.advNetOutputNorm = advNetOutputNorm
        self.mergedAdvNetwork = MergedAdvNetwork(self.advNetInputNorm, self.advNetOutputNorm, 
                                model_path="../trained/ves_adv_trained", 
                                device = device)
        
        # Normalization values for relaxation network
        self.relaxNetInputNorm = relaxNetInputNorm
        self.relaxNetOutputNorm = relaxNetOutputNorm
        self.relaxNetwork = RelaxNetwork(self.dt, self.relaxNetInputNorm, self.relaxNetOutputNorm, 
                                model_path="../trained/ves_relax_DIFF_June8_625k_dt1e-5.pth", 
                                device = device)
        
        # Normalization values for near field networks
        self.nearNetInputNorm = nearNetInputNorm
        self.nearNetOutputNorm = nearNetOutputNorm
        self.nearNetwork = MergedNearFourierNetwork(self.nearNetInputNorm, self.nearNetOutputNorm,
                                model_path="",
                                device = device)
        
        # Normalization values for tension-self network
        self.tenSelfNetInputNorm = tenSelfNetInputNorm
        self.tenSelfNetOutputNorm = tenSelfNetOutputNorm
        self.tenSelfNetwork = TenSelfNetwork(self.tenSelfNetInputNorm, self.tenSelfNetOutputNorm, 
                                model_path="../trained/ves_selften.pth", 
                                device = device)
        
        # Normalization values for tension-advection networks
        self.tenAdvNetInputNorm = tenAdvNetInputNorm
        self.tenAdvNetOutputNorm = tenAdvNetOutputNorm
        self.tenAdvNetwork = MergedTenAdvNetwork(self.tenAdvNetInputNorm, self.tenAdvNetOutputNorm, 
                                model_path="../trained/ves_advten_models", 
                                device = device)
    
    def time_step(self, Xold, tenOld):
        oc = self.oc

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

        # update the elastic force with the new tension
        # fTen = vesicle.tracJump(np.zeros((2 * N, nv)), tenNew)
        fTen = vesicle.tensionTerm(tenNew)
        tracJump = fBend + fTen

        # Calculate far-field again and correct near field before advection
        # use neural networks to calculate near-singular integrals
        farFieldtracJump = self.computeStokesInteractions(vesicle, tracJump, oc)

        # Total background velocity
        vbackTotal = vback + farFieldtracJump

        # Compute the action of dt*(1-M) on Xold
        Xadv = self.translateVinfwTorch(Xold, vbackTotal)

        # Compute the action of relax operator on Xold + Xadv
        Xnew = self.relaxWTorchNet(Xadv)

        return Xnew, tenNew

    def predictNearLayersWTorchNet(self, X, tracJump):
        N = X.shape[0] // 2
        nv = X.shape[1]

        oc = self.oc

        in_param = self.nearNetInputNorm
        out_param = self.nearNetOutputNorm

        maxLayerDist = np.sqrt(1 / N) # length = 1, h = 1/N;
        nlayers = 3 # three layers
        dlayer = np.linspace(0, maxLayerDist, nlayers)

        # Create the layers around a vesicle on which velocity calculated
        tracersX = np.zeros((2 * N, nlayers, nv))
        # Standardize input
        # Shan: standardizationStep is compatible with multiple ves
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)
        for k in range(nv):
            _, tang, _ = oc.diffProp(Xstand[:, [k]])
            nx = tang[N:].squeeze()
            ny = -tang[:N].squeeze()

            tracersX[:, 0, k] = Xstand[:, k]
            for il in range(1, nlayers):
                tracersX[:, il, k] = np.hstack([Xstand[:N, k] + nx * dlayer[il], Xstand[N:, k] + ny * dlayer[il]])

        # How many modes to be used
        # MATLAB: modes = [(0:N/2-1) (-N/2:-1)]
        # modes = np.concatenate((np.arange(0,N/2), np.arange(-N/2,0)))
        # modesInUse = 16
        # modeList = np.where(np.abs(modes) <= modesInUse)[0] # Shan: bug, creates 33 modes
        # modeList = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]

        # Normalize input
        # input_net = np.zeros((modesInUse, nv, 2, N))
        # for imode in range(modesInUse):
        #     for k in range(nv):
        #         input_net[imode, k, 0, :] = (Xstand[:N, k] - in_param[imode, 0]) / in_param[imode, 1]
        #         input_net[imode, k, 1, :] = (Xstand[N:, k] - in_param[imode, 2]) / in_param[imode, 3]

        input_net = self.nearNetwork.preProcess(Xstand)
        net_pred = self.nearNetwork.forward(input_net)
        velx_real, vely_real, velx_imag, vely_imag = self.nearNetwork.postProcess(net_pred)

        # Standardize tracJump
        # fstandRe = np.zeros((N, nv))
        # fstandIm = np.zeros((N, nv))
        # for k in range(nv):
        #     # fstand = self.standardize(tracJump[:, k], [0, 0], rotate[k], [0, 0], 1, sortIdx[k])
        #     z = fstand[:N] + 1j * fstand[N:]
        #     zh = np.fft.fft(z)
        #     fstandRe[:, k] = np.real(zh)
        #     fstandIm[:, k] = np.imag(zh)
                
        fstand = self.standardize(tracJump, np.zeros((2,nv)), rotate, np.zeros((2,nv)), 1, sortIdx)
        z = fstand[:N] + 1j * fstand[N:]
        zh = np.fft.fft(z, axis=0)
        fstandRe = np.real(zh)
        fstandIm = np.imag(zh)

        # Initialize outputs
        # velx_real = [np.zeros((N, N, nlayers)) for _ in range(nv)]
        # vely_real = [np.zeros((N, N, nlayers)) for _ in range(nv)]
        # velx_imag = [np.zeros((N, N, nlayers)) for _ in range(nv)]
        # vely_imag = [np.zeros((N, N, nlayers)) for _ in range(nv)]

        # Denormalize output
        # for ij, imode in enumerate(modeList):
        #     pred = Xpredict[ij]
        #     for k in range(nv):
        #         for ic in range(nlayers):
        #             velx_real[k][:, imode, ic] = (pred[k, ic] * out_param[imode, 1, ic]) + out_param[imode, 0, ic]
        #             vely_real[k][:, imode, ic] = (pred[k, nlayers + ic] * out_param[imode, 1, nlayers + ic]) + out_param[imode, 0, nlayers + ic]
        #             velx_imag[k][:, imode, ic] = (pred[k, 2 * nlayers + ic] * out_param[imode, 1, 2 * nlayers + ic]) + out_param[imode, 0, 2 * nlayers + ic]
        #             vely_imag[k][:, imode, ic] = (pred[k, 3 * nlayers + ic] * out_param[imode, 1, 3 * nlayers + ic]) + out_param[imode, 0, 3 * nlayers + ic]

        velx = np.zeros((N, nlayers, nv))
        vely = np.zeros((N, nlayers, nv))
        xlayers = np.zeros((N, nlayers, nv))
        ylayers = np.zeros((N, nlayers, nv))
        for k in range(nv):
            velx_stand = np.zeros((N, nlayers))
            vely_stand = np.zeros((N, nlayers))
            for il in range(nlayers):
                velx_stand[:, il] = velx_real[k][:, :, il] @ fstandRe[:, k] + velx_imag[k][:, :, il] @ fstandIm[:, k]
                vely_stand[:, il] = vely_real[k][:, :, il] @ fstandRe[:, k] + vely_imag[k][:, :, il] @ fstandIm[:, k]

                vx = np.zeros(N)
                vy = np.zeros(N)

                # Destandardize
                vx[sortIdx[k]] = velx_stand[:, il]
                vy[sortIdx[k]] = vely_stand[:, il]

                VelBefRot = np.hstack([vx, vy])
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
        max_layer_dist = np.sqrt(vesicle.length / vesicle.N)

        # Tangent
        _, tang, _ = oc.diffProp(vesicle.X)
        # Normal
        nx = tang[N:2*N, :]
        ny = -tang[:N, :]

        xvesicle = vesicle.X[:N, :]
        yvesicle = vesicle.X[N:, :]

        # Compute near/far hydro interactions without any correction
        # First calculate the far-field
        far_field = np.zeros((2 * N, nv))
        for k in range(nv):
            K = list(range(nv))
            K.remove(k)
            far_field[:, [k]] = self.exactStokesSL(vesicle, trac_jump, vesicle.X[:, [k]], K)

        # Find the outermost layers of every vesicle, then perform Laplace kernel
        Xlarge = np.zeros((2 * N, nv))
        for k in range(nv):
            Xlarge[:, k] = np.concatenate([xvesicle[:, k] + nx[:, k] * max_layer_dist, 
                                        yvesicle[:, k] + ny[:, k] * max_layer_dist])

        # Ray Casting to find near field
        i_call_near = np.zeros(nv, dtype=int)# which one needs to call near-singular correction
        query_X = {}
        ids_in_store = {}
        near_ves_ids = {}

        for j in range(nv):
            K = list(range(nv))
            K.remove(j)
            # Reorder coordinates, S is the shape of outmost layer of ves j
            S = np.zeros(2 * N)
            S[0::2] = Xlarge[:N, j]
            S[1::2] = Xlarge[N:, j]

            for k in K:
                query_X[k] = []
                ids_in_store[k] = []
                near_ves_ids[k] = []

                for p in range(N): # loop over all points of ves k
                    flag = ray_casting([xvesicle[p, k], yvesicle[p, k]], S)
                    if flag: # if inside S
                        # indices
                        ids_in_store[k].append(p)
                        # % points where we need interpolation  
                        query_X[k].append([xvesicle[p, k], yvesicle[p, k]])
                        # who is near k
                        near_ves_ids[k].append(j) #可以小改一下
                        
                        i_call_near[k] = 1

        # If needed to call near-singular correction:
        if np.any(i_call_near):
            xlayers, ylayers, velx, vely = self.predictNearLayersWTorchNet(vesicle.X, trac_jump)

            for k in range(nv):
                if i_call_near[k]:
                    ids_in = ids_in_store[k] # possible duplicates?
                    points_in = np.array(query_X[k]).T # possible duplicates?
                    ves_id = np.unique(near_ves_ids[k])

                    Xin = np.vstack([xlayers[:, :, ves_id].reshape(1, 3 * N), ylayers[:, :, ves_id].reshape(1, 3 * N)])
                    velXInput = velx[:, :, ves_id].reshape(1, 3 * N)
                    velYInput = vely[:, :, ves_id].reshape(1, 3 * N)

                    rbf_vel_x = rbfinterp(points_in, options = rbfcreate(Xin, velXInput, RBFFunction='linear'))
                    rbf_vel_y = rbfinterp(points_in, options = rbfcreate(Xin, velYInput, RBFFunction='linear'))

                    far_x = far_field[:N, k] #
                    far_y = far_field[N:, k]
                    far_x[ids_in] = rbf_vel_x
                    far_y[ids_in] = rbf_vel_y
                    far_field[:, k] = np.concatenate([far_x, far_y])

        return far_field

    def translateVinfwTorch(self, Xold, vinf):
        # Xinput is equally distributed in arc-length
        # Xold as well. So, we add up coordinates of the same points.
        N = Xold.shape[0] // 2
        nv = Xold.shape[1]

        # If we only use some modes
        # modes = np.concatenate((np.arange(0, N//2), np.arange(-N//2, 0)))
        # modesInUse = 16
        # mode_list = np.where(np.abs(modes) <= modes_in_use)[0]
        # mode_list = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]
        mode_list = [i for i in range(128)]
        # Standardize input
        # Xstand = np.zeros_like(Xold)
        # scaling = np.zeros(nv)
        # rotate = np.zeros(nv)
        # rot_cent = np.zeros((2, nv))
        # trans = np.zeros((2, nv))
        # sort_idx = np.zeros((N, nv), dtype=int)
        # for k in range(nv):
        #     (Xstand[:, k], scaling[k], rotate[k], 
        #     rot_cent[:, k], trans[:, k], sort_idx[:, k]) = self.standardization_step(Xold[:, k], N)
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(Xold)

        # Normalize input
        # input_list = []
        # for imode in mode_list:
        #     if imode != 0:
        #         # input_net = np.zeros((nv, 2, 128)) # Shan: should be (nv, 2, 256)
        #         x_mean, x_std, y_mean, y_std = in_param[imode-1]
        #         # for k in range(nv):
        #         #     input_net[k, 0, :] = (Xstand[:N, k] - x_mean) / x_std
        #         #     input_net[k, 1, :] = (Xstand[N:, k] - y_mean) / y_std
        #         input_net = np.concatenate(((Xstand[:N, None] - x_mean)/x_std, (Xstand[N:, None] - y_mean) / y_std), axis=0).T
        #         # prepare fourier basis to be combined into input
        #         theta = np.arange(N)/N*2*np.pi
        #         theta = theta.reshape(N,1)
        #         bases = 1/N*np.exp(1j*theta*np.arange(N).reshape(1,N))
        #         rr, ii = np.real(bases[:, imode]), np.imag(bases[:, imode])
        #         basis = np.concatenate((rr,ii)).reshape(1,1,256).repeat(nv, axis=0)
        #         input_net = np.concatenate((input_net, basis), axis=1)
        #         input_list.append(input_net)


        # Xpredict = pyrunfile("advect_predict.py", "output_list", input_shape=input_list, num_ves=nv)
        
        Xpredict = self.mergedAdvNetwork.forward(Xstand)
        # Approximate the multiplication M*(FFTBasis)
        Z11r = np.zeros((N, N, nv))
        Z12r = np.zeros_like(Z11r)
        Z21r = np.zeros_like(Z11r)
        Z22r = np.zeros_like(Z11r)

        for ij in range(len(mode_list) - 1):
            imode = mode_list[ij + 1]
            pred = np.array(Xpredict[ij])

            for k in range(nv):
                Z11r[:, imode, k] = pred[k, 0, :N]
                Z21r[:, imode, k] = pred[k, 0, N:]
                Z12r[:, imode, k] = pred[k, 1, :N]
                Z22r[:, imode, k] = pred[k, 1, N:]


        # Take fft of the velocity (should be standardized velocity)
        # only sort points and rotate to pi/2 (no translation, no scaling)
        Xnew = np.zeros_like(Xold)
        vinf_stand = self.standardize(vinf, np.zeros((2,nv)), rotate, np.zeros((2,nv)), 1, sortIdx)
        z = vinf_stand[:N] + 1j * vinf_stand[N:]
        zh = np.fft.fft(z, axis=0)
        V1, V2 = np.real(zh), np.imag(zh)
        MVinf_stand = np.vstack((np.einsum('NiB,iB ->NB', Z11r, V1) + np.einsum('NiB,iB ->NB', Z12r, V2),
                               np.einsum('NiB,iB ->NB', Z21r, V1) + np.einsum('NiB,iB ->NB', Z22r, V2)))
            
        for k in range(nv):
            # vinf_stand = self.standardize(vinf[:, k], np.array([0, 0]), rotate[k], np.array([0, 0]), 1, sort_idx[:, k])
            # z = vinf_stand[:N] + 1j * vinf_stand[N:]

            # zh = np.fft(z)
            # V1, V2 = np.real(zh[:, k]), np.imag(zh[:, k])
            # Compute the approximate value of the term M*vinf
            # MVinf_stand = np.vstack([Z11r[:, :, k] @ V1 + Z12r[:, :, k] @ V2, 
            #                         Z21r[:, :, k] @ V1 + Z22r[:, :, k] @ V2])
            
            # Need to destandardize MVinf (take sorting and rotation back)
            MVinf = np.zeros_like(MVinf_stand[:,k])
            idx = np.concatenate([sortIdx[k], sortIdx[k] + N])
            MVinf[idx] = MVinf_stand[:,k]
            MVinf = self.rotationOperator(MVinf, -rotate[k], np.array([0, 0]))

            Xnew[:, k] = Xold[:, k] + self.dt * vinf[:, k] - self.dt * MVinf

        return Xnew

    def relaxWTorchNet(self, Xmid):
        # 1) RELAXATION w/ NETWORK
        # Standardize vesicle Xmid
        # nv = Xmid.shape[1]
        # N = Xmid.shape[0] // 2

        Xin, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(Xmid)

        # # INPUT NORMALIZATION INFO
        # # x_mean, x_std, y_mean, y_std = self.relaxNetInputNorm

        # # INPUT NORMALIZING
        # # REAL SPACE
        # Xstand = Xin.copy()  # before normalization
        # Xin[:N, :] = (Xin[:N, :] - x_mean) / x_std
        # Xin[N:, :] = (Xin[N:, :] - y_mean) / y_std
        # XinitShape = np.zeros((nv, 2, 128))
        # for k in range(nv):
        #     XinitShape[k, 0, :] = Xin[:N, k]
        #     XinitShape[k, 1, :] = Xin[N:, k]
        # XinitConv = torch.tensor(XinitShape)

        # # OUTPUT
        # # June8 - Dt1E5
        # # DXpredictStand = pyrunfile("relax_predict_DIFF_June8_dt1E5.py", "predicted_shape", input_shape=XinitConv)
        # DXpredictStand = np.random.rand(nv, 2, 128)
        
        # # For the 625k - June8 - Dt = 1E-5 data
        # x_mean, x_std, y_mean, y_std = self.relaxNetOutputNorm

        # DXpred = np.zeros_like(Xin)
        # DXpredictStand = np.array(DXpredictStand)
        # # Xnew = np.zeros_like(Xmid)

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
        # input X is non-standardized
        
        # number of vesicles
        nv = X.shape[1]
        # number of points of exact solve
        N = X.shape[0] // 2
        
        # Modes to be called
        # modes = np.concatenate((np.arange(0, N//2), np.arange(-N//2, 0)))
        modesInUse = 16
        # modeList = np.where(np.abs(modes) <= modesInUse)[0]
        modeList = [i for i in range(modesInUse)] + [128-i for i in range(modesInUse, 0, -1)]

        # for k in range(nv):
        #     Xstand[:, k], scaling[k], rotate[k], rotCent[:, k], trans[:, k], sortIdx[:, k] = \
        #         self.standardizationStep(X[:, k], 128)
        Xstand, scaling, rotate, rotCent, trans, sortIdx = self.standardizationStep(X)

        input = self.tenAdvNetwork.preProcess(Xstand)
        Xpredict = self.tenAdvNetwork.forward(input)
        out = self.tenAdvNetwork.postProcess(Xpredict)

        # Approximate the multiplication Z = inv(DivGT)DivPhi_k
        Z1 = np.zeros((N, N, nv))
        Z2 = np.zeros((N, N, nv))

        for ij in range(len(modeList) - 1):
            imode = modeList[ij + 1]  # mode index, skipping the first mode
            pred = np.array(out[ij], dtype=float)  # size(pred) = [1 2 128]

            for k in range(nv):
                Z1[:, imode, k] = pred[k, 0, :]
                Z2[:, imode, k] = pred[k, 1, :]

        vBackSolve = np.zeros((N, nv))
        vinfStand = self.standardize(vinf, np.zeros((2,nv)), rotate, np.zeros((2,nv)), 1, sortIdx)
        z = vinfStand[:N] + 1j * vinfStand[N:]
        zh = np.fft.fft(z, axis=0)
        for k in range(nv):
            # Take fft of the velocity, standardize velocity
            V1 = np.real(zh[:, k])
            V2 = np.imag(zh[:, k])

            # Compute the approximation to inv(Div*G*Ten)*Div*vExt
            MVinfStand = Z1[:, :, k] @ V1 + Z2[:, :, k] @ V2

            # Destandardize the multiplication
            MVinf = np.zeros_like(MVinfStand)
            MVinf[sortIdx[k]] = MVinfStand 
            vBackSolve[:, k] = self.rotationOperator(MVinf, -rotate[k], [0, 0])

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

        # Normalize input
        # x_mean, x_std, y_mean, y_std = self.tenSelfNetInputNorm

        # Adjust the input shape for the network
        # XinitShape = np.zeros((nv, 2, 128))
        # for k in range(nv):
        #     XinitShape[k, 0, :] = (Xstand[:N, k] - x_mean) / x_std
        #     XinitShape[k, 1, :] = (Xstand[N:, k] - y_mean) / y_std
        # XinitConv = torch.tensor(XinitShape)

        # Make prediction -- needs to be adjusted for python
        # tenPredictStand = pyrunfile("tension_self_network.py", "predicted_shape", input_shape=XinitConv)
        # tenPredictStand = np.random.rand(nv, 1, 128)

        # Denormalize output
        # out_mean, out_std = self.tenSelfNetOutputNorm

        # tenPred = np.zeros((N, nv))
        # tenPredictStand = np.array(tenPredictStand, dtype=float)

        tenPredictStand = self.tenSelfNetwork.forward(Xstand)
        tenPred = np.zeros((N, nv))
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
            stokesSLPtar = np.zeros((2 * Ntar, ncol))
        else:
            K1 = []
            Ntar = 0
            stokesSLPtar = None
            ncol = 0

        den = f * np.tile(vesicle.sa, (2, 1)) * 2 * np.pi / vesicle.N

        xsou = vesicle.X[:vesicle.N, K1].flatten()
        ysou = vesicle.X[vesicle.N:, K1].flatten()
        xsou = np.tile(xsou, (Ntar, 1)).T
        ysou = np.tile(ysou, (Ntar, 1)).T

        denx = den[:vesicle.N, K1].flatten()
        deny = den[vesicle.N:, K1].flatten()
        denx = np.tile(denx, (Ntar, 1)).T
        deny = np.tile(deny, (Ntar, 1)).T

        for k in range(ncol):  # Loop over columns of target points
            xtar = Xtar[:Ntar, k]
            ytar = Xtar[Ntar:, k]
            xtar = np.tile(xtar, (vesicle.N * len(K1), 1))
            ytar = np.tile(ytar, (vesicle.N * len(K1), 1))
            
            diffx = xtar - xsou
            diffy = ytar - ysou

            dis2 = diffx**2 + diffy**2

            coeff = 0.5 * np.log(dis2)
            stokesSLPtar[:Ntar, k] = -np.sum(coeff * denx, axis=0)
            stokesSLPtar[Ntar:, k] = -np.sum(coeff * deny, axis=0)

            coeff = (diffx * denx + diffy * deny) / dis2
            stokesSLPtar[:Ntar, k] += np.sum(coeff * diffx, axis=0)
            stokesSLPtar[Ntar:, k] += np.sum(coeff * diffy, axis=0)

        stokesSLPtar = stokesSLPtar / (4 * np.pi)
        return stokesSLPtar

    # def standardizationStep(self, Xin, Nnet):
    #     oc = self.oc
    #     N = len(Xin) // 2

    #     if Nnet != N:
    #         Xin = np.vstack([
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
    #     XrotSort = np.concatenate([Xrotated[sortIdx], Xrotated[sortIdx + N]])

    #     XrotSort = scaling * XrotSort

    #     return XrotSort
    
    def standardizationStep(self, Xin):
        # compatible with multi ves
        oc = self.oc
        X = Xin[:]
        # % Equally distribute points in arc-length
        for w in range(10):
            X, _, _ = oc.redistributeArcLength(X)
        # % standardize angle, center, scaling and point order
        trans, rotate, rotCenter, scaling, multi_sortIdx = self.referenceValues(X)
        
        X = self.standardize(X, trans, rotate, rotCenter, scaling, multi_sortIdx)
        return X, scaling, rotate, rotCenter, trans, multi_sortIdx

    def standardize(self, X, translation, rotation, rotCenter, scaling, multi_sortIdx):
        # compatible with multi ves
        N = len(multi_sortIdx[0])
        Xrotated = self.rotationOperator(X, rotation, rotCenter)
        Xrotated = self.translateOp(Xrotated, translation)
        XrotSort = np.zeros_like(Xrotated)
        for i in range(X.shape[1]):
            XrotSort[:,i] = np.concatenate((Xrotated[multi_sortIdx[i], i], Xrotated[multi_sortIdx[i] + N, i]))
        XrotSort = scaling*XrotSort
        return XrotSort


    def destandardize(self, XrotSort, translation, rotation, rotCent, scaling, sortIdx):
        ''' compatible with multiple ves'''
        N = len(sortIdx[0])
        nv = XrotSort.shape[1]

        # Scale back
        XrotSort = XrotSort / scaling

        # Change ordering back
        X = np.zeros_like(XrotSort)
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
        tempX = np.zeros_like(Xref)
        tempX = Xref[:]

        # Find the physical center
        center = oc.getPhysicalCenter(tempX)
        multi_V = oc.getPrincAxesGivenCentroid(tempX,center)
        w = np.array([0, 1]) # y-axis unit vector
        rotation = np.arctan2(w[1]*multi_V[0]-w[0]*multi_V[1], w[0]*multi_V[0]+w[1]*multi_V[1])
        
        rotCenter = center # the point around which the frame is rotated
        Xref = self.rotationOperator(tempX, rotation, rotCenter)
        center = oc.getPhysicalCenter(Xref) # redundant?
        translation = -center
        
        Xref = self.translateOp(Xref, translation)
        
        multi_sortIdx = []
        for k in range(nv):
        # Shan: This for loop can be avoided but will be less readable
            firstQuad = np.intersect1d(np.where(Xref[:N,k] >= 0)[0], np.where(Xref[N:,k] >= 0)[0])
            theta = np.arctan2(Xref[N:,k], Xref[:N,k])
            idx = np.argmin(theta[firstQuad])
            sortIdx = np.concatenate((np.arange(firstQuad[idx],N), np.arange(0, firstQuad[idx])))
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
    #     w = np.array([0, 1])  # y-axis
    #     rotation = np.arctan2(w[1] * V[0] - w[0] * V[1], w[0] * V[0] + w[1] * V[1])

    #     # Find the ordering of the points
    #     rotCent = center
    #     Xref = self.rotationOperator(Xref, rotation, center)
    #     center = oc.getPhysicalCenter(Xref)
    #     translation = -center

    #     Xref = self.translateOp(Xref, translation)

    #     firstQuad = np.where((Xref[:N] >= 0) & (Xref[N:] >= 0))[0]
    #     theta = np.arctan2(Xref[N:], Xref[:N])
    #     idx = np.argmin(theta[firstQuad])
    #     sortIdx = np.concatenate((np.arange(firstQuad[idx], N), np.arange(firstQuad[idx])))

    #     # Amount of scaling
    #     _, _, length = oc.geomProp(Xref)
    #     scaling = 1 / length
        
    #     return translation, rotation, rotCent, scaling, sortIdx

    def rotationOperator(self, X, theta, rotCent):
        ''' Shan: compatible with multi ves
        theta of shape (1,nv), rotCent of shape (2,nv)'''
        Xrot = np.zeros_like(X)
        x = X[:len(X)//2] - rotCent[0]
        y = X[len(X)//2:] - rotCent[1]

        # Rotated shape
        xrot = x * np.cos(theta) - y * np.sin(theta)
        yrot = x * np.sin(theta) + y * np.cos(theta)

        Xrot[:len(X)//2] = xrot + rotCent[0]
        Xrot[len(X)//2:] = yrot + rotCent[1]
        return Xrot

    def translateOp(self, X, transXY):
        ''' Shan: compatible with multi ves
         transXY of shape (2,nv)'''
        Xnew = np.zeros_like(X)
        Xnew[:len(X)//2] = X[:len(X)//2] + transXY[0]
        Xnew[len(X)//2:] = X[len(X)//2:] + transXY[1]
        return Xnew


