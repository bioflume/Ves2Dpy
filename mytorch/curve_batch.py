import torch
import numpy as np
torch.set_default_dtype(torch.float64)
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from fft1 import fft1
from scipy.interpolate import interp1d
import torch.nn as nn
import time
# import matlab.engine
# eng = matlab.engine.start_matlab()

class Curve:
    '''
    % This class implements that basic calculus on the curve.
    % The basic data structure is a matrix X in which the columns 
    % represent periodic C^{\infty} closed curves with N points, 
    % X(1:n,j) is the x-coordinate of the j_th curve and X(n+1:N,j) 
    % is the y-coordinate of the j_th curve; here n=N/2
    % X coordinates do not have to be periodic, but the curvature,
    % normals, etc that they compute will be garbage.  This allows
    % us to store target points for tracers or the pressure using
    % this class and then using near-singular integration is easy
    % to implement
    '''
    def getXY(self, X):
        """Get the [x,y] component of curves X."""
        N = X.shape[0] // 2
        return X[:N, :], X[N:, :]

    def setXY(self, x, y):
        """Set the [x,y] component of vector V on the curve."""
        N = x.shape[0]
        V = torch.zeros(2 * N, x.shape[1], dtype=torch.float64, device=x.device)
        V[:N, :] = x
        V[N:, :] = y
        return V

    # def getCenter(self, X):
    #     """Find the center of each capsule."""
    #     center = torch.sqrt(torch.mean(X[:X.shape[0] // 2], dim=0) ** 2 +
    #                             torch.mean(X[X.shape[0] // 2:], dim=0) ** 2)
    #     return center
    
    def getPhysicalCenter(self, X):
        """Fin the physical center of each capsule. Compatible with multi ves.
        returns center in shape (2, nv)
        """
        nv = X.shape[1]
        # Compute the differential properties of X
        jac, tan, _ = self.diffProp(X)
        # Assign the normal as well
        nx, ny = tan[tan.shape[0] // 2:,:], -tan[:tan.shape[0] // 2,:] 
        x, y = X[:X.shape[0] // 2, :], X[X.shape[0] // 2:, :]
      
        center = torch.zeros((2, nv), dtype=torch.float64)
        xdotn = x * nx
        ydotn = y * ny
        xdotn_sum = torch.sum(xdotn * jac, dim=0)
        ydotn_sum = torch.sum(ydotn * jac, dim=0)
        # x-component of the center
        center[0] = 0.5 * torch.sum(x * xdotn * jac, dim=0) / xdotn_sum
        # y-component of the center
        center[1] = 0.5 * torch.sum(y * ydotn * jac, dim=0) / ydotn_sum

        return center 

    def getIncAngle(self, X):
        """Find the inclination angle of each capsule.
        % GK: THIS IS NEEDED IN STANDARDIZING VESICLE SHAPES 
        % WE NEED TO KNOW THE INCLINATION ANGLE AND ROTATE THE VESICLE TO pi/2
        % IA = getIncAngle(o,X) finds the inclination angle of each capsule
        % The inclination angle (IA) is the angle between the x-dim and the 
        % principal dim corresponding to the smallest principal moment of inertia
        """
        nv = X.shape[1]
        # IA = torch.zeros(nv, dtype=torch.float64)
        # % compute inclination angle on an upsampled grid
        N = X.shape[0] // 2
        # modes = torch.concatenate((torch.arange(0, N // 2), torch.tensor([0]), torch.arange(-N // 2 + 1, 0))).double()
        center = self.getPhysicalCenter(X)
        
        # tempX = torch.zeros_like(X)
        # # tempX[:X.shape[0] // 2] = X[:X.shape[0] // 2] - torch.mean(X[:X.shape[0] // 2], dim=0)
        # # tempX[X.shape[0] // 2:] = X[X.shape[0] // 2:] - torch.mean(X[X.shape[0] // 2:], dim=0)
        # tempX[:X.shape[0] // 2] = X[:X.shape[0] // 2] - center[0]
        # tempX[X.shape[0] // 2:] = X[X.shape[0] // 2:] - center[1]
        
        # for k in range(nv):
        #     x = tempX[:N, k]
        #     y = tempX[N:, k]
            
        #     Dx = torch.real(torch.fft.ifft(1j * modes * torch.fft.fft(x)))
        #     Dy = torch.real(torch.fft.ifft(1j * modes * torch.fft.fft(y)))
        #     jac = torch.sqrt(Dx ** 2 + Dy ** 2)
        #     tx = Dx / jac
        #     ty = Dy / jac
        #     nx = ty
        #     ny = -tx #Shan: n is the right hand side of t
        #     rdotn = x * nx + y * ny
        #     rho2 = x ** 2 + y ** 2

        #     J11 = 0.25 * torch.sum(rdotn * (rho2 - x * x) * jac) * 2 * torch.pi / N
        #     J12 = 0.25 * torch.sum(rdotn * (-x * y) * jac) * 2 * torch.pi / N
        #     J21 = 0.25 * torch.sum(rdotn * (-y * x) * jac) * 2 * torch.pi / N
        #     J22 = 0.25 * torch.sum(rdotn * (rho2 - y * y) * jac) * 2 * torch.pi / N

        #     J = torch.tensor([[J11, J12], [J21, J22]])
            
        #     D, V = torch.linalg.eig(J)
        #     ind = torch.argmin(torch.abs((D)))
        #     # % make sure that the first components of e-vectors have the same sign
        #     V = torch.real(V)
        #     if V[1, ind] < 0:
        #         V[:, ind] *= -1
        #     # % since V[1,ind] > 0, this will give angle between [0, pi]
        #     IA[k] = torch.arctan2(V[1, ind], V[0, ind])
        
        # Compute the centered coordinates
        Xcent = torch.vstack((X[:N] - center[0], X[N:] - center[1])).to(X.device)
        xCent = Xcent[:N]
        yCent = Xcent[N:]

        # Compute differential properties
        jacCent, tanCent, _ = self.diffProp(Xcent)
        # Normal vectors
        nxCent = tanCent[N:]
        nyCent = -tanCent[:N]
        # Dot product and rho^2
        rdotn = xCent * nxCent + yCent * nyCent
        rho2 = xCent**2 + yCent**2

        # Compute components of J
        J11 = 0.25 * torch.sum(rdotn * (rho2 - xCent**2) * jacCent, dim=0) * 2 * torch.pi / N
        J12 = 0.25 * torch.sum(rdotn * (-xCent * yCent) * jacCent, dim=0) * 2 * torch.pi / N
        J21 = 0.25 * torch.sum(rdotn * (-yCent * xCent) * jacCent, dim=0) * 2 * torch.pi / N
        J22 = 0.25 * torch.sum(rdotn * (rho2 - yCent**2) * jacCent, dim=0) * 2 * torch.pi / N
        
        # Assemble the Jacobian matrix, J shape: (batch_size, 2, 2)
        J_ = torch.concat((torch.stack((J11, J12)).T.unsqueeze(1), 
                          torch.stack((J21, J22)).T.unsqueeze(1)), dim=1)
        
        # Eigen decomposition
        eig_vals, eig_vecs = torch.linalg.eig(J_)
        
        # Select the eigenvector corresponding to the smallest eigenvalue
        min_index = torch.argmin(torch.abs(eig_vals), dim=1)
        
        V_ = torch.real(eig_vecs[torch.arange(nv), :, min_index]).T
        condition = V_[1, :] < 0
        # Apply -1 to the entire column where condition is True
        V_[:, condition] *= -1

        # % since V(2,ind) > 0, this will give angle between [0, pi]
        IA = torch.arctan2(V_[1], V_[0])

        return IA
    
    
    def getPrincAxesGivenCentroid(self, X, center):
        """
        Compute the principal axes given the centroid.
        
        Parameters:
        o       : Object with a method `diffProp` that returns jacCent, tanCent, and curvCent
        X       : 2D numpy array of shape (2N, nv)
        center  : 2D numpy array of shape (2, nv)
        
        Returns:
        V       : Principal axes as a 2D numpy array of shape (2, 1)
        """
        N = X.shape[0] // 2  # Number of points
        nv = X.shape[1]  # Number of variables
        # multiple_V = torch.zeros((2,nv), dtype=torch.float64)
        
        # for k in range(nv):
        #     # Compute the centered coordinates
        #     Xcent = torch.vstack((X[:N, k:k+1] - center[0, k], X[N:, k:k+1] - center[1, k])).to(X.device)
        #     xCent = Xcent[:N]
        #     yCent = Xcent[N:]
            
        #     # Compute differential properties
        #     jacCent, tanCent, curvCent = self.diffProp(Xcent)
            
        #     # Normal vectors
        #     nxCent = tanCent[N:]
        #     nyCent = -tanCent[:N]
            
        #     # Dot product and rho^2
        #     rdotn = xCent * nxCent + yCent * nyCent
        #     rho2 = xCent**2 + yCent**2
            
        #     # Compute components of J
        #     J11 = 0.25 * torch.sum(rdotn * (rho2 - xCent**2) * jacCent) * 2 * torch.pi / N
        #     J12 = 0.25 * torch.sum(rdotn * (-xCent * yCent) * jacCent) * 2 * torch.pi / N
        #     J21 = 0.25 * torch.sum(rdotn * (-yCent * xCent) * jacCent) * 2 * torch.pi / N
        #     J22 = 0.25 * torch.sum(rdotn * (rho2 - yCent**2) * jacCent) * 2 * torch.pi / N
            
        #     # Assemble the Jacobian matrix
        #     J = torch.tensor([[J11, J12], [J21, J22]])
            
        #     # Eigen decomposition
        #     eig_vals, eig_vecs = torch.linalg.eig(J)
            
        #     # Select the eigenvector corresponding to the smallest eigenvalue
        #     min_index = torch.argmin(torch.abs(eig_vals))
        #     V = eig_vecs[:, min_index]
            
        #     # Store the result for the current variable
        #     multiple_V[:,k] = torch.real(V)
        
        # Compute the centered coordinates
        Xcent = torch.vstack((X[:N] - center[0], X[N:] - center[1])).to(X.device)
        xCent = Xcent[:N]
        yCent = Xcent[N:]

        # Compute differential properties
        jacCent, tanCent, _ = self.diffProp(Xcent)
        # Normal vectors
        nxCent = tanCent[N:]
        nyCent = -tanCent[:N]
        # Dot product and rho^2
        rdotn = xCent * nxCent + yCent * nyCent
        rho2 = xCent**2 + yCent**2

        # Compute components of J
        J11 = 0.25 * torch.sum(rdotn * (rho2 - xCent**2) * jacCent, dim=0) * 2 * torch.pi / N
        J12 = 0.25 * torch.sum(rdotn * (-xCent * yCent) * jacCent, dim=0) * 2 * torch.pi / N
        J21 = 0.25 * torch.sum(rdotn * (-yCent * xCent) * jacCent, dim=0) * 2 * torch.pi / N
        J22 = 0.25 * torch.sum(rdotn * (rho2 - yCent**2) * jacCent, dim=0) * 2 * torch.pi / N
        
        # Assemble the Jacobian matrix, J shape: (batch_size, 2, 2)
        J = torch.concat((torch.stack((J11, J12)).T.unsqueeze(1), 
                          torch.stack((J21, J22)).T.unsqueeze(1)), dim=1)
        
        # Eigen decomposition
        eig_vals, eig_vecs = torch.linalg.eig(J)
        
        # Select the eigenvector corresponding to the smallest eigenvalue
        min_index = torch.argmin(torch.abs(eig_vals), dim=1)
        V = eig_vecs[torch.arange(nv), :, min_index]
        # Store the result for the current variable
        multiple_V = torch.real(V).T

        return multiple_V

    def getDXY(self, X):
        """Compute the derivatives of each component of X."""
        # % [Dx,Dy]=getDXY(X), compute the derivatives of each component of X 
        # % these are the derivatives with respect the parameterization 
        # % not arclength
        x, y = self.getXY(X)
        N = x.shape[0]
        nv = x.shape[1]
        f = fft1(N)
        IK = f.modes(N, nv).to(X.device)
        Dx = f.diffFT(x, IK)
        Dy = f.diffFT(y, IK)
        return Dx, Dy

    def diffProp(self, X):
        """Return differential properties of the curve."""
        # % [jacobian,tangent,curvature] = diffProp(X) returns differential
        # % properties of the curve each column of the matrix X. Each column of 
        # % X should be a closed curve defined in plane. The tangent is the 
        # % normalized tangent.
        N = X.shape[0] // 2
        nv = X.shape[1]

        Dx, Dy = self.getDXY(X)
        jacobian = torch.sqrt(Dx ** 2 + Dy ** 2)

        tangent = torch.vstack((Dx / jacobian, Dy / jacobian))

        f = fft1(N)
        IK = f.modes(N, nv)
        DDx = self.arcDeriv(Dx, 1, torch.ones((N, nv), device=X.device), IK.to(X.device))
        DDy = self.arcDeriv(Dy, 1, torch.ones((N, nv), device=X.device), IK.to(X.device))
        curvature = (Dx * DDy - Dy * DDx) / (jacobian ** 3)

        return jacobian, tangent, curvature

    def geomProp(self, X):
        """Calculate the length, area, and the reduced volume."""
        # % [reducedArea area length] = geomProp(X) calculate the length, area 
        # % and the reduced volume of domains inclose by columns of X. 
        # % Reduced volume is defined as 4*pi*A/L. 
        # % EXAMPLE:
        # %   X = boundary(64,'nv',3,'curly');
        # %   c = curve(X);
        # %   [rv A L] = c.geomProp(X);
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        
        x, y = self.getXY(X)
        N = x.shape[0]
        Dx, Dy = self.getDXY(X)
        length = torch.sum(torch.sqrt(Dx ** 2 + Dy ** 2),dim=0) * 2 * torch.pi / N
        area = torch.sum(x * Dy - y * Dx,dim=0) * torch.pi / N
        reducedArea = 4 * torch.pi * area / length ** 2
        return reducedArea, area, length


    def ellipse(self, N, ra):
        """
        Finds the ellipse (a*cos(theta), sin(theta)) so that the reduced area is ra.
        % X0 = o.ellipse(N,ra) finds the ellipse (a*cos(theta),sin(theta)) so
        % that the reduced area is ra.  Uses N points.  Parameter a is found 
        % by using bisection method
        """
        t = torch.arange(N) * 2 * torch.pi / N
        a = (1 - torch.sqrt(1 - ra**2)) / ra
        # Initial guess using approximation length = sqrt(2) * pi * sqrt(a^2 + 1)
        X0 = torch.concatenate((a * torch.cos(t), torch.sin(t)))[:,None]
        ra_new, _, _ = self.geomProp(X0)
        cond = torch.abs(ra_new - ra) / ra < 1e-4
        maxiter = 10
        iter = 0

        while (not cond[0] and iter < maxiter):
            iter += 1
            
            if ra_new > ra:
                a *= 0.9
            else:
                a *= 1.05

            # Update the major dim
            X0 = torch.concatenate((torch.cos(t), a*torch.sin(t)))[:,None]
            # Compute new possible configuration
            ra_new, _, _ = self.geomProp(X0)
            # Compute new reduced area
            cond = torch.abs(ra_new - ra) < 1e-2
            # % iteration quits if reduced area is achieved within 1% or 
            # % maxiter iterations have been performed

        return X0


    def correctAreaAndLength(self, X, area0, length0):
        """Change the shape of the vesicle by correcting the area and length."""
        
        # % Xnew = correctAreaAndLength(X,a0,l0) changes the shape of the vesicle
        # % by finding the shape Xnew that is closest to X in the L2 sense and
        # % has the same area and length as the original shape

        # % tolConstraint (which controls area and length) comes from the area-length
        # % tolerance for time adaptivity.
        tolConstraint = 1e-3
        tolFunctional = 1e-3

        # % Find the current area and length
        _, a, l = self.geomProp(X)
        eAt = torch.abs((a - area0) / area0)
        eLt = torch.abs((l - length0) / length0)
        if torch.max(eAt) < tolConstraint and torch.max(eLt) < tolConstraint:
            return X

        # N = X.shape[0] // 2
        
        print("entering a & l correction")
        options = {'maxiter': 300, 'disp': True}

        X = X.cpu().numpy()
        area0 = area0.cpu().numpy()
        length0 = length0.cpu().numpy()
        Xnew = np.zeros_like(X)

        # def mycallback(Xi):
        #     global num_iter
        #     print(f"scipy minimize iter {num_iter}")
        #     num_iter += 1

        for k in range(X.shape[1]):
            def minFun(z):
                return np.mean((z - X[:, k]) ** 2)

            cons = ({'type': 'eq', 'fun': lambda z: self.nonlcon(z, area0[k], length0[k])})
            res = minimize(minFun, X[:, k], constraints=cons, options=options) #, tol=1e-2 , callback=mycallback
            Xnew[:, k] = res.x
            # print(res.message)
            # print(f"function value{res.fun}") # , cons violation {res.maxcv}
            if not res.success:
                print('Correction scheme failed, do not correct at this step')
                Xnew[:, k] = X[:,k]

        return torch.from_numpy(Xnew)

    def nonlcon(self, X, a0, l0):
        """Non-linear constraints required by minimize."""
        _, a, l = self.geomProp(X[:,None])
        cEx = torch.hstack(((a - a0) / a0, (l - l0) / l0))
        return cEx

    def correctAreaAndLengthAugLag(self, X, area0, length0):
        """Change the shape of the vesicle by correcting the area and length."""
        
        # % Xnew = correctAreaAndLength(X,a0,l0) changes the shape of the vesicle
        # % by finding the shape Xnew that is closest to X in the L2 sense and
        # % has the same area and length as the original shape

        # % tolConstraint (which controls area and length) comes from the area-length
        # % tolerance for time adaptivity.
        
        # % Find the current area and length
        # _, a, l = self.geomProp(X)
        # eAt = torch.abs((a - area0) / area0)
        # eLt = torch.abs((l - length0) / length0)

        # N = X.shape[0] // 2
        tolConstraint = 1e-2
        tolFunctional = 1e-2

        # % Find the current area and length
        _, a, l = self.geomProp(X)
        area0 = area0.float()
        length0 = length0.float()
        eAt = torch.abs((a - area0) / area0)
        eLt = torch.abs((l - length0) / length0)
        if torch.max(eAt) < tolConstraint and torch.max(eLt) < tolConstraint:
            return X
        # print(f"initial rel err of a {torch.max(eAt)} and l {torch.max(eLt)}")

        maxiter = 100
        
        # # Xnew = torch.zeros_like(X)
        # def minFun(z, lamb, mu):
        #     _, a, l = self.geomProp(z)
        #     nv = z.shape[1]
        #     return torch.mean((z - X) ** 2) - lamb[:nv] * (a - area0) - lamb[nv:] * (l - length0) + 1/(2*mu) * ((a-area0)**2 + (l-length0)**2)
        
        def max_rel_err(z, x):
            return torch.max(torch.abs(z - x)/x)   
        def mean_rel_err(z, x): 
            return torch.mean(torch.norm(z - x, dim=0)/torch.norm(x, dim=0))
        class AugLag(nn.Module):
            def __init__(self, X):
                super().__init__()
                self.z = nn.Parameter(X)
                self.X = X.clone()
                self.c = Curve()

            def forward(self, lamb, mu):
                _, a, l = self.c.geomProp(self.z)
                nv = self.X.shape[1]
                a = a.float()
                l = l.float()
                return a, l, mean_rel_err(self.z, self.X), \
                      - 1.0/nv * torch.inner(lamb[:nv], (a - area0)) - 1.0/nv * torch.inner(lamb[nv:], (l - length0)) + 1/(2*mu) * torch.mean((a-area0)**2 + (l-length0)**2)
        
        def train_model(model, lamb, mu, n_iterations=20, lr=2e-5):
            # We initialize an optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            # We take n_iterations steps of gradient descent
            model.train()
            for iter in range(n_iterations):
                optimizer.zero_grad()
                a, l, loss_fun, loss_cons = model(lamb, mu)
                if iter % 5 == 0:
                    print(f"{iter} in ADAM, loss_fun is {loss_fun:.5e}, loss_cons is {loss_cons:.5e}, rel err of a {max_rel_err(a, area0):.5e}, of l {max_rel_err(l, length0):.5e}")
                if loss_fun < tolFunctional and \
                        max_rel_err(a, area0) < tolConstraint and \
                        max_rel_err(l, length0) < tolConstraint:
                    return a, l, True
                (loss_fun + loss_cons).backward()
                optimizer.step()

            return a.detach(), l.detach(), False
        
        it = 0
        lamb = torch.zeros(X.shape[1]*2, device=X.device, dtype=torch.float32)
        mu = 0.1
        aug_lag_model = AugLag(X.float()).to(X.device)
        
        while it < maxiter:
            if it % 5==0 :
                print(f"outside iter {it}")
            a, l, flag = train_model(aug_lag_model, lamb, mu)
            if flag:
                break
            
            lamb -= 1/mu * torch.concat((a-area0, l-length0))
            mu *= 0.8
            it += 1

        Xnew = aug_lag_model.z.detach().double()
            
        return Xnew

    
    def alignCenterAngle(self, Xorg, X):
        """Use translation and rotation to match X with Xorg."""
        # % Xnew = alignCenterAngle(o,Xorg,X) uses
        # % rigid body translation and rotation to match X having the corrected area 
        # % and length but wrong center and inclination angle with Xorg having the 
        # % right center and IA but wrong area and length. So that, Xnew has the
        # % correct area,length,center and inclination angle.

        # Xnew = torch.zeros_like(X)
        # for k in range(X.shape[1]):
        #     # initMean = torch.tensor([torch.mean(Xorg[:Xorg.shape[0] // 2, k]), torch.mean(Xorg[Xorg.shape[0] // 2:, k])])
        #     # newMean = torch.tensor([torch.mean(X[:X.shape[0] // 2, k]), torch.mean(X[X.shape[0] // 2:, k])])
        #     initCenter = self.getPhysicalCenter(Xorg[:, k:k+1])
        #     newCenter = self.getPhysicalCenter(X[:, k:k+1])

        #     initAngle = self.getIncAngle(Xorg[:, k:k+1])
        #     newAngle = self.getIncAngle(X[:, k:k+1])

        #     if newAngle > torch.pi:
        #         newAngle2 = newAngle - torch.pi
        #     else:
        #         newAngle2 = newAngle + torch.pi
        #     newAngles = torch.tensor([newAngle, newAngle2])
        #     diffAngles = torch.abs(initAngle - newAngles)
        #     id = torch.argmin(diffAngles)
        #     newAngle = newAngles[id]

        #     # % move to (0,0) new shape
        #     Xp = torch.concatenate((X[:X.shape[0] // 2, k] - newCenter[0], X[X.shape[0] // 2:, k] - newCenter[1]),dim=0)
        #     # % tilt it to the original angle
        #     thet = -newAngle+initAngle
        #     XpNew = torch.zeros_like(Xp)
        #     XpNew[:Xp.shape[0]//2] = Xp[:Xp.shape[0]//2] * torch.cos(thet) - Xp[Xp.shape[0]//2:] * torch.sin(thet)
        #     XpNew[Xp.shape[0]//2:] = Xp[:Xp.shape[0]//2] * torch.sin(thet) + Xp[Xp.shape[0]//2:] * torch.cos(thet)

        #     # % move to original center
        #     Xnew[:, k] = torch.concatenate((XpNew[:Xp.shape[0]//2] + initCenter[0], XpNew[Xp.shape[0]//2:] + initCenter[1]), dim=0)

        initCenter = self.getPhysicalCenter(Xorg)
        newCenter = self.getPhysicalCenter(X)
        initAngle = self.getIncAngle(Xorg)
        newAngle = self.getIncAngle(X)

        newAngle2 = torch.where(newAngle > torch.pi, newAngle - torch.pi, newAngle + torch.pi)
        newAngles = torch.stack([newAngle, newAngle2])
        diffAngles = torch.abs(initAngle - newAngles)
        ids = torch.argmin(diffAngles, axis=0) # ids indicates first row or second row
        newAngle = newAngles[ids, torch.arange(X.shape[1])]

        N = X.shape[0]//2
        # % move to (0,0) new shape
        Xp = torch.concatenate((X[:N] - newCenter[0], X[N:] - newCenter[1]),dim=0)
        # % tilt it to the original angle
        thet = -newAngle + initAngle
        XpNew = torch.zeros_like(Xp)
        XpNew[:N] = Xp[:N] * torch.cos(thet) - Xp[N:] * torch.sin(thet)
        XpNew[N:] = Xp[:N] * torch.sin(thet) + Xp[N:] * torch.cos(thet)

        # % move to original center
        Xnew = torch.concatenate((XpNew[:N] + initCenter[0], XpNew[N:] + initCenter[1]), dim=0)
        
        return Xnew

    def redistributeArcLength(self, X):
        """Redistribute the vesicle shape equispaced in arclength."""
        # % [X,u,sigma] = redistributeArcLength(o,X,u,sigma) redistributes
        # % the vesicle shape eqiuspaced in arclength and adjusts the tension and
        # % velocity according to the new parameterization

        N = X.shape[0] // 2
        nv = X.shape[1]
        # modes = torch.concatenate((torch.arange(0, N // 2), [0], torch.arange(-N // 2 + 1, 0)))
        modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(X.device).double()
        jac, _, _ = self.diffProp(X)
        tol = 1e-10

        # X_out = torch.zeros_like(X, device=X.device)
        # allGood = True

        # for k in range(nv):
        #     if torch.linalg.norm(jac[:, k] - torch.mean(jac[:, k]), ord=torch.inf) > tol * torch.mean(jac[:, k]):
        #         allGood = False
        #         theta, _ = self.arcLengthParameter(X[:N, k], X[N:, k])
        #         theta = torch.from_numpy(theta).to(X.device).squeeze()
        #         # print(theta)
        #         zX = X[:N, k] + 1j * X[N:, k]
        #         zXh = torch.fft.fft(zX) / N
        #         zX = torch.zeros(N, dtype=torch.complex128, device=X.device)
        #         for jj in range(N):
        #             zX += zXh[jj] * torch.exp(1j * modes[jj] * theta)
        #         X_out[:, k:k+1] = self.setXY(torch.real(zX)[:,None], torch.imag(zX)[:,None])
        #     else:
        #         X_out[:, k:k+1] = X[:, k:k+1]
        
        allGood = True
        X_out = X.clone()
        to_redistribute = torch.linalg.norm(jac - torch.mean(jac, dim=0), ord=torch.inf) > tol * torch.mean(jac, dim=0)
        if torch.any(to_redistribute):
            allGood = False
            ids = torch.arange(nv, device=X.device)[to_redistribute]
            tStart = time.time()
            theta, _ = self.arcLengthParameter(X[:N, ids], X[N:, ids])
            tEnd = time.time()
            print(f'arcLengthParameter {tEnd - tStart} sec.')
            theta = torch.from_numpy(theta).to(X.device)
            
            zX = X[:N, ids] + 1j * X[N:, ids]
            zXh = torch.fft.fft(zX, dim=0) / N
            # zX = torch.zeros((N, len(ids)), dtype=torch.complex128, device=X.device)
            # for jj in range(N): # use broadcasting to remove this loop
            #     zX += zXh[jj] * torch.exp(1j * modes[jj] * theta)
            zX_ = torch.einsum('mj,mnj->nj', zXh, torch.exp(1j * modes[:,None,None] * theta))
            X_out[:, ids] = self.setXY(torch.real(zX_), torch.imag(zX_))
        
        return X_out, allGood


    # def cubic_spline_interp(self, x, y, x_new):
    #     """
    #     Perform cubic spline interpolation with not-a-knot boundary conditions.
        
    #     Parameters:
    #         x (array-like): The known x-values (must be sorted in ascending order).
    #         y (array-like): The known y-values.
    #         x_new (array-like): The x-values to interpolate.

    #     Returns:
    #         np.array: Interpolated y-values at x_new.
    #     """
    #     n = len(x)
    #     h = np.diff(x)   # Calculate segment lengths
    #     b = np.diff(y) / h  # Calculate slopes

    #     # Build the coefficient matrix A and the right-hand side vector rhs for cubic spline
    #     A = np.zeros((n, n))
    #     rhs = np.zeros(n)

    #     # Not-a-knot boundary conditions at the start and end
    #     A[0, 0], A[0, 1] = h[1], -(h[0] + h[1])
    #     A[0, 2] = h[0]
    #     rhs[0] = 0

    #     A[-1, -3] = h[-1]
    #     A[-1, -2], A[-1, -1] = -(h[-2] + h[-1]), h[-2]
    #     rhs[-1] = 0

    #     # Fill the tridiagonal matrix for the interior points
    #     for i in range(1, n - 1):
    #         A[i, i - 1] = h[i - 1]
    #         A[i, i] = 2 * (h[i - 1] + h[i])
    #         A[i, i + 1] = h[i]
    #         rhs[i] = 3 * (b[i] - b[i - 1])

    #     # Solve the linear system for c (the second derivatives at the knots)
    #     c = np.linalg.solve(A, rhs)

    #     # Calculate the spline coefficients for each segment
    #     a = y[:-1]
    #     b = b - (h * (2 * c[:-1] + c[1:])) / 3
    #     d = (c[1:] - c[:-1]) / (3 * h)

    #     # Interpolating at the new points
    #     y_new = np.zeros_like(x_new)
    #     for j, xj in enumerate(x_new):
    #         # Find the segment that xj is in
    #         i = np.searchsorted(x, xj) - 1
    #         i = np.clip(i, 0, n - 2)  # Ensure i is within valid range
    #         dx = xj - x[i]
    #         y_new[j] = a[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3

    #     return y_new

    def arcLengthParameter(self, x, y):
        """
        % theta = arcLengthParamter(o,x,y) finds a discretization of parameter
        % space theta so that the resulting geometry will be equispaced in
        % arclength
        """
        N = len(x)
        t = torch.arange(N, dtype=torch.float64, device=x.device) * 2 * torch.pi / N
        X = torch.concatenate((x, y))
        if len(X.shape) < 2:
            X = X.unsqueeze(-1)
        _, _, length = self.geomProp(X)
        # Find total perimeter
        
        Dx, Dy = self.getDXY(X)
        # Find derivative
        arc = torch.sqrt(Dx**2 + Dy**2)
        arch = torch.fft.fft(arc, dim=0).T # (nv, N)
        modes = -1.0j / torch.hstack([torch.tensor([1e-10]).double(), (torch.arange(1,N // 2)), torch.tensor([1e-10]).double(), (torch.arange(-N//2+1,0))])  # FFT modes
        modes[0] = 0
        modes[N // 2] = 0
        modes = modes.to(x.device) #(N)
        
        arc_length = torch.real(torch.fft.ifft(modes * arch, dim=-1) - \
                                torch.sum(modes * arch, dim=-1).unsqueeze(-1) / N + arch[:,0:1] * t / N).T
        # arc_length shape: (N, nv)
        z1 = torch.concat((arc_length[-7:] - length, arc_length, arc_length[:7] + length), dim=0).cpu().numpy()
        z2 = torch.hstack([t[-7:] - 2 * torch.pi, t, t[:7] + 2 * torch.pi]).cpu().numpy()
        # % put in some overlap to account for periodicity

        # Interpolate to obtain equispaced points
        # dx = torch.diff(z1)
        # dx = abs(dx)
        # dump_z1 = torch.cumsum(torch.concat((z1[[0]], dx)), dim=0).cpu().numpy()
        # if torch.any(dx <= 0):
        #     print(dx)
        #     print("haha")
        
        theta = np.zeros((N, X.shape[1]))
        for i in range(X.shape[1]):
            theta[:,i] = CubicSpline(z1[:,i], z2)(torch.arange(N).cpu() * length[i].cpu() / N)

        # # Create interpolation function using cubic spline
        # interpolation_function = interp1d(z1, z2, kind='cubic')  # 'cubic' is equivalent to MATLAB's 'spline'
        # # Generate theta values with interpolation
        # theta = interpolation_function(torch.arange(N).cpu() * length.cpu() / N)

        # theta = self.cubic_spline_interp(z1, z2, torch.arange(N).cpu() * length.cpu() / N)

        # theta = eng.interp1(z1.numpy(),z2, np.arange(N)*length.cpu().numpy()/N,'spline')
        
        return theta, arc_length

    
    # def reparametrize(self, X, dX, maxIter=100):
    #     """Reparametrize to minimize the energy in the high frequencies."""
    #     # % [X,niter] = reparametrize applies the reparametrization with
    #     # % minimizing the energy in the high frequencies (Veerapaneni et al. 2011, 
    #     # % doi: 10.1016/j.jcp.2011.03.045, Section 6).

    #     pow = 4
    #     nv = X.shape[1]
    #     niter = torch.ones(nv, dtype=int)
    #     tolg = 1e-3
    #     if dX is None:
    #         _, _, length = self.geomProp(X)
    #         dX = length / X.shape[0]
    #         toly = 1e-5 * dX
    #     else:
    #         normDx = torch.sqrt(dX[:X.shape[0] // 2] ** 2 + dX[X.shape[0] // 2:] ** 2)
    #         toly = 1e-3 * torch.min(normDx)

    #     beta = 0.1
    #     dtauOld = 0.05

    #     for k in range(nv):
    #         # % Get initial coordinates of kth vesicle (upsample if necessary)
    #         x0 = X[:X.shape[0] // 2, [k]]
    #         y0 = X[X.shape[0] // 2:, [k]]
    #         # % Compute initial projected gradient energy
    #         g0 = self.computeProjectedGradEnergy(x0, y0, pow)
    #         x = x0
    #         y = y0
    #         g = g0
            
    #         # % Explicit reparametrization
    #         while niter[k] <= maxIter:
    #             dtau = dtauOld
    #             xn = x - g[:X.shape[0] // 2] * dtau
    #             yn = y - g[X.shape[0] // 2:] * dtau
    #             gn = self.computeProjectedGradEnergy(xn, yn, pow)
    #             while torch.linalg.norm(gn) > torch.linalg.norm(g):
    #                 dtau = dtau * beta
    #                 xn = x - g[:X.shape[0] // 2] * dtau
    #                 yn = y - g[X.shape[0] // 2:] * dtau
    #                 gn = self.computeProjectedGradEnergy(xn, yn, pow)
    #             dtauOld = dtau * 1 / beta
    #             # print(toly)
    #             if torch.linalg.norm(gn) < max(max(toly / dtau), tolg * torch.linalg.norm(g0)):
    #                 break
    #             x = xn
    #             y = yn
    #             g = gn
    #             niter[k] += 1
    #         X[:, [k]] = torch.vstack((xn, yn))

    #     return X

    def computeProjectedGradEnergy(self, x, y, pow):
        """Compute the projected gradient of the energy of the surface."""
        # % g = computeProjectedGradEnergy(o,x,y) computes the projected gradient of
        # % the energy of the surface. We use this in reparamEnergyMin(o,X). For the
        # % formulation see (Veerapaneni et al. 2011 doi: 10.1016/j.jcp.2011.03.045,
        # % Section 6)

        N = len(x)
        modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0)))[:,None]
        modes = modes.double()
        # % get tangent vector at each point (tang_x;tang_y) 
        _, tang, _ = self.diffProp(torch.concatenate((x, y)).reshape(-1,1))
        # % get x and y components of normal vector at each point
        nx = tang[N:]
        ny = -tang[:N]

        # % Compute gradE
        # % first, get Fourier coefficients
        zX = x + 1j * y
        zXh = torch.fft.fft(zX, dim=0) / N
        # % second, compute zX with a_k = k^pow
        zX = torch.fft.ifft(N * zXh * torch.abs(modes) ** pow, dim=0)
        # % Compute Energy
        gradE = torch.vstack((torch.real(zX), torch.imag(zX))) #[gradE_x;gradE_y]
        
        # % A dyadic product property (a (ban) a)b = a(a.b) can be used to avoid the
        # % for loop as follows
        normals = torch.vstack((nx, ny))
        # % do the dot product n.gradE
        prod = normals * gradE
        dotProd = prod[:N] + prod[N:]
        # % do (I-(n ban n))gradE = gradE - n(n.gradE) for each point

        g = gradE - normals * torch.vstack((dotProd, dotProd))
        
        return g
    
    def arcDeriv(self, f, m, isa, IK):
        """
        % f = arcDeriv(f,m,s,IK,col) is the arclength derivative of order m.
        % f is a matrix of scalar functions (each function is a column)
        % f is assumed to have an arbitrary parametrization
        % sa = d s/ d a, where a is the aribtrary parameterization
        % IK is the fourier modes which is saved and used to accelerate 
        % this routine
        """
        for _ in range(m):
            f = isa * torch.fft.ifft(IK * torch.fft.fft(f, dim=0), dim=0)
            
        return torch.real(f)
