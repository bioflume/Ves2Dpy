import numpy as np
import torch
import sys
sys.path.append("..")
from model_zoo.Net_ves_relax_midfat import Net_ves_midfat

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
        Xadv = self.translateVinfNet(X, vback)

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

        # Normalize input
        input_net = np.zeros((nv, 2, 128))
        for imode in range(2, 128):
            x_mean = self.advNetInputNorm[imode - 1][0]
            x_std = self.advNetInputNorm[imode - 1][1]
            y_mean = self.advNetInputNorm[imode - 1][2]
            y_std = self.advNetInputNorm[imode - 1][3]

            input_net[:, 0, :] = (Xstand[:N] - x_mean) / x_std
            input_net[:, 1, :] = (Xstand[N:] - y_mean) / y_std

        # Predict using neural networks
        # Adjust this part for Python implementation
        Xpredict = ...

        # % Above line approximates multiplication M*(FFTBasis) 
        # % Now, reconstruct Mvinf = (M*FFTBasis) * vinf_hat
        Z11 = np.zeros((128, 128))
        Z12 = np.zeros((128, 128))
        Z21 = np.zeros((128, 128))
        Z22 = np.zeros((128, 128))

        for imode in range(2, 128): # the first mode is zero
            pred = Xpredict[imode - 1]

            real_mean = self.advNetOutputNorm[imode - 1][0]
            real_std = self.advNetOutputNorm[imode - 1][1]
            imag_mean = self.advNetOutputNorm[imode - 1][2]
            imag_std = self.advNetOutputNorm[imode - 1][3]

            pred[0, 0, :] = (pred[0, 0, :] * real_std) + real_mean
            pred[0, 1, :] = (pred[0, 1, :] * imag_std) + imag_mean

            Z11[:, imode, 0] = pred[0, 0, :][:N]
            Z12[:, imode, 0] = pred[0, 1, :][:N] 
            Z21[:, imode, 0] = pred[0, 0, :][N:]
            Z22[:, imode, 0] = pred[0, 1, :][N:]

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
        # % Fix misalignment in center and angle due to reparametrization
        X = oc.alignCenterAngle(Xin, X)
        # % standardize angle, center, scaling and point order
        trans, rotate, scaling, sortIdx = self.referenceValues(X)
        
        X = self.standardize(X, trans, rotate, scaling, sortIdx)
        return X, scaling, rotate, trans, sortIdx

    def standardize(self, X, translation, rotation, scaling, sortIdx):
        N = len(sortIdx)
        Xrotated = scaling * self.rotationOperator(self.translateOp(X, translation), rotation)
        XrotSort = np.concatenate((Xrotated[sortIdx], Xrotated[sortIdx + N]))
        return XrotSort

    def destandardize(self, XrotSort, translation, rotation, scaling, sortIdx):
        N = len(sortIdx)
        X = np.zeros_like(XrotSort)
        X[sortIdx] = XrotSort[:N]
        X[sortIdx + N] = XrotSort[N:]

        X = X / scaling

        cx = np.mean(X[:N])
        cy = np.mean(X[N:])
        
        X = self.rotationOperator(np.concatenate([X[:N] - cx, X[N:] - cy]), -rotation)
        X = np.concatenate([X[:N] + cx, X[N:] + cy])

        X = self.translateOp(X, -1*np.array(translation))

        return X

    def referenceValues(self, Xref):
        oc = self.oc
        N = len(Xref) // 2
        tempX = np.zeros_like(Xref)
        tempX = Xref[:]

        translation = [-np.mean(Xref[:N]), -np.mean(Xref[N:])]
        rotation = np.pi / 2 - oc.getIncAngle2(tempX)
        _, _, length = oc.geomProp(tempX)
        scaling = 1 / length
        
        tempX = scaling * self.rotationOperator(self.translateOp(tempX, translation), rotation)
        
        firstQuad = np.intersect1d(np.where(tempX[:N] >= 0)[0], np.where(tempX[N:] >= 0)[0])
        theta = np.arctan2(tempX[N:], tempX[:N])
        idx = np.argmin(theta[firstQuad])
        sortIdx = np.concatenate((np.arange(firstQuad[idx],N), np.arange(0, firstQuad[idx])))

        return translation, rotation, scaling, sortIdx

    def rotationOperator(self, X, theta):
        Xrot = np.zeros_like(X)
        x = X[:len(X) // 2]
        y = X[len(X) // 2:]

        xrot = x * np.cos(theta) - y * np.sin(theta)
        yrot = x * np.sin(theta) + y * np.cos(theta)

        Xrot[:len(X) // 2] = xrot
        Xrot[len(X) // 2:] = yrot
        return Xrot

    def translateOp(self, X, transXY):
        Xnew = np.zeros_like(X)
        Xnew[:len(X) // 2] = X[:len(X) // 2] + transXY[0]
        Xnew[len(X) // 2:] = X[len(X) // 2:] + transXY[1]
        return Xnew
