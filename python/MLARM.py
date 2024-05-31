import numpy as np
import torch
import sys
sys.path.append("..")
from model_zoo.Net_ves_relax_midfat import Net_ves_midfat
from model_zoo.Net_ves_adv_fft import Net_ves_adv_fft
from model_zoo.Net_ves_merge_adv import Net_merge_advection

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
        Xstand, scaling, rotate, trans, sortIdx = self.standardizationStep(X)
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
        vinfStand = self.standardize(vback, [0, 0], rotate, 1, sortIdx)
        z = vinfStand[:N] + 1j * vinfStand[N:]

        zh = np.fft.fft(z)
        V1 = zh.real
        V2 = zh.imag
        # % Compute the approximate value of the term M*vinf
        MVinf = np.vstack((np.dot(Z11, V1) + np.dot(Z12, V2), np.dot(Z21, V1) + np.dot(Z22, V2)))
        # % update the standardized shape
        XnewStand = self.dt * vinfStand - self.dt * MVinf   
        # % destandardize
        Xadv = self.destandardize(XnewStand, trans, rotate, scaling, sortIdx)
        # % add the initial since solving dX/dt = (I-M)vinf
        Xadv = X + Xadv

        return Xadv

    def translateVinfMergeNet(self, X, vback, num_modes):
            # Translate vesicle using networks
            N = X.shape[0]//2
            nv = X.shape[1]
            # % Standardize vesicle (zero center, pi/2 inclination angle, equil dist)
            Xstand, scaling, rotate, trans, sortIdx = self.standardizationStep(X)
            device = torch.device("cpu")
            Xpredict = torch.zeros(127, nv, 2, 256).to(device)
            # prepare fourier basis
            theta = np.arange(N)/N*2*np.pi
            theta = theta.reshape(N,1)
            bases = 1/N*np.exp(1j*theta*np.arange(N).reshape(1,N))

            for i in range(127//num_modes):
                # from s mode to t mode, both end included
                s = 2 + i*num_modes + i
                t = min(2 + (i+1)*num_modes + i, 128)
                print(f"from mode {s} to mode {t}")
                rep = t - s + 1
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
                one_mode_inputs = [torch.concat((input_coords[:, [i]], basis.repeat(nv,1,1)[:,[i]]), dim=1) for i in range(rep)]
                input_net = torch.concat(tuple(one_mode_inputs), dim=1).to(device)

                # prepare the network
                model = Net_merge_advection(12, 1.7, 20, rep=rep)
                dicts = []
                models = []
                for i in range(s, t+1):
                    path = "/work/09452/alberto47/ls6/vesicle/save_models/ves_fft_models/ves_fft_mode"+str(i)+".pth"
                    dicts.append(torch.load(path, map_location=device))
                    subnet = Net_ves_adv_fft(12, 1.7, 20)
                    subnet.load_state_dict(dicts[-1])
                    models.append(subnet.to(device))

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
            vinfStand = self.standardize(vback, [0, 0], rotate, 1, sortIdx)
            z = vinfStand[:N] + 1j * vinfStand[N:]

            zh = np.fft.fft(z)
            V1 = zh.real
            V2 = zh.imag
            # % Compute the approximate value of the term M*vinf
            MVinf = np.vstack((np.dot(Z11, V1) + np.dot(Z12, V2), np.dot(Z21, V1) + np.dot(Z22, V2)))
            # % update the standardized shape
            XnewStand = self.dt * vinfStand - self.dt * MVinf   
            # % destandardize
            Xadv = self.destandardize(XnewStand, trans, rotate, scaling, sortIdx)
            # % add the initial since solving dX/dt = (I-M)vinf
            Xadv = X + Xadv

            return Xadv


    def relaxNet(self, X):
        N = X.shape[0]//2

        # % Standardize vesicle
        Xin, scaling, rotate, trans, sortIdx = self.standardizationStep(X)
        # % Normalize input
        x_mean = self.relaxNetInputNorm[0]
        x_std = self.relaxNetInputNorm[1]
        y_mean = self.relaxNetInputNorm[2]
        y_std = self.relaxNetInputNorm[3]

        Xin[:N] = (Xin[:N] - x_mean) / x_std
        Xin[N:] = (Xin[N:] - y_mean) / y_std

        XinitShape = np.zeros((1, 2, 128))
        XinitShape[0, 0, :] = Xin[:N,0]
        XinitShape[0, 1, :] = Xin[N:,0]
        XinitConv = torch.tensor(XinitShape).float()

        # Make prediction -- needs to be adjusted for python
        device = torch.device("cpu")
        model = Net_ves_midfat(num_blocks=16)
        model.load_state_dict(torch.load("../ves_relax619k_mirr_dt1e-5.pth", map_location=device))
        model.eval()
        with torch.no_grad():
            XpredictStand = model(XinitConv)

        # Denormalize output
        Xpred = np.zeros_like(Xin)
        XpredictStand = XpredictStand.numpy()

        out_x_mean = self.relaxNetOutputNorm[0]
        out_x_std = self.relaxNetOutputNorm[1]
        out_y_mean = self.relaxNetOutputNorm[2]
        out_y_std = self.relaxNetOutputNorm[3]

        # nv=1 case
        Xpred[:N,0] = XpredictStand[0, 0, :] * out_x_std + out_x_mean
        Xpred[N:,0] = XpredictStand[0, 1, :] * out_y_std + out_y_mean

        Xnew = self.destandardize(Xpred, trans, rotate, scaling, sortIdx)
        return Xnew

    def standardizationStep(self, Xin):
        oc = self.oc
        X = Xin[:]
        # % Equally distribute points in arc-length
        for w in range(5):
            X, _, _ = oc.redistributeArcLength(X)
        # % standardize angle, center, scaling and point order
        trans, rotate, scaling, sortIdx = self.referenceValues(X)
        
        X = self.standardize(X, trans, rotate, scaling, sortIdx)
        return X, scaling, rotate, trans, sortIdx

    def standardize(self, X, translation, rotation, scaling, sortIdx):
        N = len(sortIdx)
        Xrotated = self.rotationOperator(X, rotation)
        Xrotated = self.translateOp(Xrotated, translation)
        XrotSort = np.concatenate((Xrotated[sortIdx], Xrotated[sortIdx + N]))
        XrotSort = scaling*XrotSort
        return XrotSort

    def destandardize(self, XrotSort, translation, rotation, scaling, sortIdx):
        N = len(sortIdx)
        X = np.zeros_like(XrotSort)
        X[sortIdx] = XrotSort[:N]
        X[sortIdx + N] = XrotSort[N:]

        X = X / scaling
        X = self.translateOp(X, -1*np.array(translation))
        
        cx = np.mean(X[:N])
        cy = np.mean(X[N:])
        
        X = self.rotationOperator(np.concatenate([X[:N] - cx, X[N:] - cy]), -rotation)
        X = np.concatenate([X[:N] + cx, X[N:] + cy])

        return X

    def referenceValues(self, Xref):
        oc = self.oc
        N = len(Xref) // 2
        tempX = np.zeros_like(Xref)
        tempX = Xref[:]

        # Find the physical center
        center = oc.getPhysicalCenter(tempX)
        V = oc.getPrincAxesGivenCentroid(tempX,center)
        w = np.array([0, 1]) # y-axis unit vector
        rotation = np.arctan2(w[1]*V[0]-w[0]*V[1], w[0]*V[0]+w[1]*V[1])
        
        #translation = [-np.mean(Xref[:N]), -np.mean(Xref[N:])]
        #rotation = np.pi / 2 - oc.getIncAngle(tempX)
        Xref = self.rotationOperator(tempX, rotation)
        center = oc.getPhysicalCenter(Xref)
        translation = -center
        
        _, _, length = oc.geomProp(tempX)
        scaling = 1 / length
        
        tempX = scaling*self.translateOp(Xref, translation)
        
        #tempX = scaling * self.rotationOperator(self.translateOp(tempX, translation), rotation)
        
        firstQuad = np.intersect1d(np.where(tempX[:N] >= 0)[0], np.where(tempX[N:] >= 0)[0])
        theta = np.arctan2(tempX[N:], tempX[:N])
        idx = np.argmin(theta[firstQuad])
        sortIdx = np.concatenate((np.arange(firstQuad[idx],N), np.arange(0, firstQuad[idx])))

        return translation, rotation, scaling, sortIdx

    def rotationOperator(self, X, theta):
        Xrot = np.zeros_like(X)
        x = X[:len(X) // 2]
        y = X[len(X) // 2:]

        xrot = (x-np.mean(x)) * np.cos(theta) - (y-np.mean(y)) * np.sin(theta) + np.mean(x)
        yrot = (x-np.mean(x)) * np.sin(theta) + (y-np.mean(y)) * np.cos(theta) + np.mean(y)

        Xrot[:len(X) // 2] = xrot
        Xrot[len(X) // 2:] = yrot
        return Xrot

    def translateOp(self, X, transXY):
        Xnew = np.zeros_like(X)
        Xnew[:len(X) // 2] = X[:len(X) // 2] + transXY[0]
        Xnew[len(X) // 2:] = X[len(X) // 2:] + transXY[1]
        return Xnew
