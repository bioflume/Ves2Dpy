# %% 
import torch
import numpy as np
torch.set_default_dtype(torch.float32)
# from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
# from fft1 import fft1
# from scipy.interpolate import interp1d
import torch.nn as nn
import time
# import cupy as cp
from filter import interpft, upsample_fft
from typing import Tuple
# from scipy.io import loadmat
# import matlab.engine
# eng = matlab.engine.start_matlab()


class Curve():
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
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    # @torch.jit.script_method
    def getXY(self, X):
        """Get the [x,y] component of curves X."""
        N = X.shape[0] // 2
        return X[:N, :], X[N:, :]

    # @torch.jit.script_method
    def setXY(self, x, y):
        """Set the [x,y] component of vector V on the curve."""
        N = x.shape[0]
        V = torch.zeros(2 * N, x.shape[1], dtype=torch.float32, device=x.device)
        V[:N, :] = x
        V[N:, :] = y
        return V

    # def getCenter(self, X):
    #     """Find the center of each capsule."""
    #     center = torch.sqrt(torch.mean(X[:X.shape[0] // 2], dim=0) ** 2 +
    #                             torch.mean(X[X.shape[0] // 2:], dim=0) ** 2)
    #     return center
    
    @torch.jit.script_method
    def getPhysicalCenter_(self, X):
        """Fin the physical center of each capsule. Compatible with multi ves.
        returns center in shape (2, nv)
        """
        nv = X.shape[1]
        # Compute the differential properties of X
        jac, tan = self.diffProp_jac_tan(X)
        # Assign the normal as well
        nx, ny = tan[tan.shape[0] // 2:, :], -tan[:tan.shape[0] // 2, :] 
        x, y = X[:X.shape[0] // 2, :], X[X.shape[0] // 2:, :]
      
        center = torch.zeros((2, nv), dtype=torch.float32, device=X.device)
        xdotn = x * nx
        ydotn = y * ny
        xdotn_sum = torch.sum(xdotn * jac, dim=0)
        ydotn_sum = torch.sum(ydotn * jac, dim=0)
        # x-component of the center
        center[0] = 0.5 * torch.sum(x * xdotn * jac, dim=0) / xdotn_sum
        # y-component of the center
        center[1] = 0.5 * torch.sum(y * ydotn * jac, dim=0) / ydotn_sum

        return center 

    
    # @torch.jit.script_method
    # @torch.compile(backend='cudagraphs')
    def getPhysicalCenter(self, X):
        """Fin the physical center of each capsule. Compatible with multi ves.
        returns center in shape (2, nv)
        """
        N, nv = X.shape[0]//2, X.shape[1]
        
        x, y = X[:X.shape[0] // 2, :], X[X.shape[0] // 2:, :]
        center = torch.zeros((2, nv), dtype=torch.float32, device=X.device)
        # A = 0.5 * (torch.sum(x[:-1] * y[1:] - x[1:] * y[:-1], dim=0) + x[-1] * y[0] - x[0] * y[-1])
        Dx, Dy = self.getDXY(X)
        A = torch.sum(x * Dy - y * Dx,dim=0) #* torch.pi / N
        # x-component of the center
        center[0] = torch.sum(x**2 * Dy, dim=0) /A
        # y-component of the center
        center[1] = - torch.sum(y**2 * Dx, dim=0) /A

        return center 

    
    
    # @torch.compile(backend='cudagraphs')
    def getIncAngle(self, X):
        """Find the inclination angle of each capsule.
        % GK: THIS IS NEEDED IN STANDARDIZING VESICLE SHAPES 
        % WE NEED TO KNOW THE INCLINATION ANGLE AND ROTATE THE VESICLE TO pi/2
        % IA = getIncAngle(o,X) finds the inclination angle of each capsule
        % The inclination angle (IA) is the angle between the x-dim and the 
        % principal dim corresponding to the smallest principal moment of inertia
        """
        nv = X.shape[1]
        # IA = torch.zeros(nv, dtype=torch.float32)
        # % compute inclination angle on an upsampled grid
        N = X.shape[0] // 2
        # modes = torch.concatenate((torch.arange(0, N // 2), torch.tensor([0]), torch.arange(-N // 2 + 1, 0))).double()
        center = self.getPhysicalCenter(X)
        
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
        jacCent, tanCent = self.diffProp_jac_tan(Xcent)
        # Normal vectors
        nxCent = tanCent[N:]
        nyCent = -tanCent[:N]
        # Dot product and rho^2
        rdotn = xCent * nxCent + yCent * nyCent
        rho2 = xCent**2 + yCent**2

        # Compute components of J
        # J11 = 0.25 * torch.sum(rdotn * (rho2 - xCent**2) * jacCent, dim=0) * 2 * torch.pi / N
        # J12 = 0.25 * torch.sum(rdotn * (-xCent * yCent) * jacCent, dim=0) * 2 * torch.pi / N
        # J21 = 0.25 * torch.sum(rdotn * (-yCent * xCent) * jacCent, dim=0) * 2 * torch.pi / N
        # J22 = 0.25 * torch.sum(rdotn * (rho2 - yCent**2) * jacCent, dim=0) * 2 * torch.pi / N

        J11 = torch.sum(rdotn * (rho2 - xCent**2) * jacCent, dim=0) 
        J12 = torch.sum(rdotn * (-xCent * yCent) * jacCent, dim=0)
        J21 = torch.sum(rdotn * (-yCent * xCent) * jacCent, dim=0) 
        J22 = torch.sum(rdotn * (rho2 - yCent**2) * jacCent, dim=0)
        
        
        # Assemble the Jacobian matrix, J shape: (batch_size, 2, 2)
        J_ = torch.concat((torch.stack((J11, J12)).T.unsqueeze(1), 
                          torch.stack((J21, J22)).T.unsqueeze(1)), dim=1)
        
        # Eigen decomposition
        eig_vals, eig_vecs = torch.linalg.eig(J_)
        
        # Select the eigenvector corresponding to the smallest eigenvalue
        min_index = torch.argmin(torch.abs(eig_vals), dim=1)
        
        V_ = torch.real(eig_vecs[torch.arange(nv, device=X.device), :, min_index]).T
        condition = V_[1, :] < 0
        # Apply -1 to the entire column where condition is True
        V_[:, condition] *= -1

        # % since V(2,ind) > 0, this will give angle between [0, pi]
        IA = torch.arctan2(V_[1], V_[0])

        return IA
    
    def getIncAngle2(self, X):

        """
        Compute the inclination angle of each capsule. 
        The inclination angle (IA) is the angle between the x-axis and the 
        principal axis corresponding to the smallest principal moment of inertia.
        """
        nv = X.shape[1]
        IA = torch.zeros(nv, dtype=torch.float32, device=X.device)

        # Compute inclination angle on an upsampled grid
        N = X.shape[0] // 2
        modes = torch.cat((torch.arange(0, N // 2, dtype=torch.float32), 
                        torch.tensor([0.0]), 
                        torch.arange(-N // 2 + 1, 0, dtype=torch.float32)))

        # Center each capsule
        centX = self.getPhysicalCenter(X)
        # X[:N, :] -= centX[0]
        # X[N:, :] -= centX[1]

        for k in range(nv):
            x = X[:N, k] - centX[0, k]
            y = X[N:, k] - centX[1, k]
            
            Dx = torch.real(torch.fft.ifft(1.0j * modes * torch.fft.fft(x)))
            Dy = torch.real(torch.fft.ifft(1.0j * modes * torch.fft.fft(y)))
            jac = torch.sqrt(Dx ** 2 + Dy ** 2)
            tx = Dx / jac
            ty = Dy / jac
            nx, ny = ty, -tx  # n is the right-hand side of t
            rdotn = x * nx + y * ny
            rho2 = x ** 2 + y ** 2

            J11 = 0.25 * torch.sum(rdotn * (rho2 - x ** 2) * jac) * 2 * torch.pi / N
            J12 = 0.25 * torch.sum(rdotn * (-x * y) * jac) * 2 * torch.pi / N
            J21 = 0.25 * torch.sum(rdotn * (-y * x) * jac) * 2 * torch.pi / N
            J22 = 0.25 * torch.sum(rdotn * (rho2 - y ** 2) * jac) * 2 * torch.pi / N

            J = torch.tensor([[J11, J12], [J21, J22]])
            eigvals, eigvecs = torch.linalg.eig(J)
            
            # Get the eigenvector corresponding to the smallest eigenvalue
            ind = torch.argmin(torch.abs(eigvals))
            V = eigvecs[:, ind].real

            # Ensure first component of eigenvector has the same sign
            if V[1] < 0:
                V *= -1

            # Inclination angle in [0, pi]
            IA[k] = torch.atan2(V[1], V[0])

            # Rotate to pi/2 and compute rotated coordinates
            theta = -IA[k] + torch.pi / 2
            x0rot = x * torch.cos(theta) - y * torch.sin(theta)
            y0rot = x * torch.sin(theta) + y * torch.cos(theta)

            # Rotate derivatives
            Dx_rot = Dx * torch.cos(theta) - Dy * torch.sin(theta)
            Dy_rot = Dx * torch.sin(theta) + Dy * torch.cos(theta)

            # Compute areas for determining head-tail direction
            idcs_top = y0rot >= 0
            idcs_bot = ~idcs_top
            area_top = torch.sum(x0rot[idcs_top] * Dy_rot[idcs_top] - y0rot[idcs_top] * Dx_rot[idcs_top]) * torch.pi / N
            area_bot = torch.sum(x0rot[idcs_bot] * Dy_rot[idcs_bot] - y0rot[idcs_bot] * Dx_rot[idcs_bot]) * torch.pi / N

            if area_bot >= 1.1 * area_top:
                IA[k] += torch.pi
            elif area_top < 1.1 * area_bot:
                # Check areaRight and areaLeft
                idcs_left = x0rot < 0
                idcs_right = ~idcs_left
                area_right = torch.sum(x0rot[idcs_right] * Dy_rot[idcs_right] - y0rot[idcs_right] * Dx_rot[idcs_right]) * torch.pi / N
                area_left = torch.sum(x0rot[idcs_left] * Dy_rot[idcs_left] - y0rot[idcs_left] * Dx_rot[idcs_left]) * torch.pi / N
                if area_left >= 1.1 * area_right:
                    IA[k] += torch.pi

        return IA

    
    # @torch.jit.script_method
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
        # multiple_V = torch.zeros((2,nv), dtype=torch.float32)
        
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
        jacCent, tanCent = self.diffProp_jac_tan(Xcent)
        # Normal vectors
        nxCent = tanCent[N:]
        nyCent = -tanCent[:N]
        # Dot product and rho^2
        rdotn = xCent * nxCent + yCent * nyCent
        rho2 = xCent**2 + yCent**2

        # Compute components of J
        # J11 = 0.25 * torch.sum(rdotn * (rho2 - xCent**2) * jacCent, dim=0) * 2 * torch.pi / N
        # J12 = 0.25 * torch.sum(rdotn * (-xCent * yCent) * jacCent, dim=0) * 2 * torch.pi / N
        # J21 = 0.25 * torch.sum(rdotn * (-yCent * xCent) * jacCent, dim=0) * 2 * torch.pi / N
        # J22 = 0.25 * torch.sum(rdotn * (rho2 - yCent**2) * jacCent, dim=0) * 2 * torch.pi / N

        J11 = torch.sum(rdotn * (rho2 - xCent**2) * jacCent, dim=0)
        J12 = torch.sum(rdotn * (-xCent * yCent) * jacCent, dim=0)
        J21 = torch.sum(rdotn * (-yCent * xCent) * jacCent, dim=0)
        J22 = torch.sum(rdotn * (rho2 - yCent**2) * jacCent, dim=0)

        
        # J shape: (batch_size, 2, 2)
        J = torch.concat((torch.stack((J11, J12)).T.unsqueeze(1), 
                          torch.stack((J21, J22)).T.unsqueeze(1)), dim=1)
        
        if torch.any(torch.isnan(J)) or torch.any(torch.isinf(J)):
            np.save("problematic_X.npy", {'X': X.cpu().numpy(), 'center': center.cpu().numpy()})
            print("J matrix contains NaN or Inf values. problematic X saved.")
            # raise ValueError("J matrix contains NaN or Inf values.")
        
        # Eigen decomposition
        # import pdb; pdb.set_trace()
        eig_vals, eig_vecs = torch.linalg.eig(J)
        
        # Select the eigenvector corresponding to the smallest eigenvalue
        min_index = torch.argmin(torch.abs(eig_vals), dim=1)
        V = eig_vecs[torch.arange(nv, device=X.device), :, min_index]
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
        # f = fft1(N)
        # IK = f.modes(N, nv, device=X.device)
        IK = 1.0j * torch.concatenate((torch.arange(0, N / 2, device=X.device), torch.tensor([0.], device=X.device), torch.arange(-N / 2 + 1, 0, device=X.device))).to(X.device) #.double()
        IK = IK[:,None].expand(-1, nv)

        Dx = torch.real(torch.fft.ifft(IK * torch.fft.fft(x, dim=0), dim=0))
        Dy = torch.real(torch.fft.ifft(IK * torch.fft.fft(y, dim=0), dim=0))
        # Dx = f.diffFT(x, IK)
        # Dy = f.diffFT(y, IK)

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

        # f = fft1(N)
        # IK = f.modes(N, nv, X.device)
        IK = 1.0j * torch.concatenate((torch.arange(0, N / 2, device=X.device), torch.tensor([0], device=X.device), torch.arange(-N / 2 + 1, 0, device=X.device))).to(X.device) #.double()
        IK = IK[:,None].expand(-1, nv)

        DDx = self.arcDeriv(Dx, 1, torch.ones((N, nv), device=X.device), IK)
        DDy = self.arcDeriv(Dy, 1, torch.ones((N, nv), device=X.device), IK)
        curvature = (Dx * DDy - Dy * DDx) / (jacobian ** 3)

        return jacobian, tangent, curvature

    
    def diffProp_jac(self, X):
        """Return differential properties of the curve."""
        # % [jacobian,tangent,curvature] = diffProp(X) returns differential
        # % properties of the curve each column of the matrix X. Each column of 
        # % X should be a closed curve defined in plane. The tangent is the 
        # % normalized tangent.

        Dx, Dy = self.getDXY(X)
        jacobian = torch.sqrt(Dx ** 2 + Dy ** 2)

        return jacobian


    def diffProp_jac_tan(self, X):
        """Return differential properties of the curve."""
        # % [jacobian,tangent,curvature] = diffProp(X) returns differential
        # % properties of the curve each column of the matrix X. Each column of 
        # % X should be a closed curve defined in plane. The tangent is the 
        # % normalized tangent.

        Dx, Dy = self.getDXY(X)
        jacobian = torch.sqrt(Dx ** 2 + Dy ** 2)

        tangent = torch.vstack((Dx / jacobian, Dy / jacobian))

        return jacobian, tangent

    
    # @torch.jit.script_method
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


    
    def geomProp_length(self, X):
        """Calculate the length, area, and the reduced volume."""
        # % [reducedArea area length] = geomProp(X) calculate the length, area 
        # % and the reduced volume of domains inclose by columns of X. 
        # % Reduced volume is defined as 4*pi*A/L. 
        # % EXAMPLE:
        # %   X = boundary(64,'nv',3,'curly');
        # %   c = curve(X);
        # %   [rv A L] = c.geomProp(X);
        
        N = X.shape[0] // 2
        Dx, Dy = self.getDXY(X)
        length = torch.sum(torch.sqrt(Dx ** 2 + Dy ** 2),dim=0) * 2 * torch.pi / N

        return length


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
        tolConstraint = 1e-2
        tolFunctional = 1e-2

        # % Find the current area and length
        _, a, l = self.geomProp(X)
        eAt = torch.abs((a - area0) / area0)
        eLt = torch.abs((l - length0) / length0)
        if torch.max(eAt) < tolConstraint and torch.max(eLt) < tolConstraint:
            return X

        # N = X.shape[0] // 2
        # print(a)
        # print(l)
        
        # print("entering a & l correction with ", a, l)
        options = {'maxiter': 300, 'disp': False}

        X = X.cpu().double().numpy()
        area0 = area0.double().cpu().numpy()
        length0 = length0.double().cpu().numpy()
        Xnew = np.copy(X)

        # def mycallback(Xi):
        #     global num_iter
        #     print(f"scipy minimize iter {num_iter}")
        #     num_iter += 1

        for k in range(X.shape[1]):
            if eAt[k] < tolConstraint and eLt[k] < tolConstraint:
                continue
            def minFun(z):
                return np.mean((z - X[:, k]) ** 2)

            cons = ({'type': 'eq', 'fun': lambda z: self.nonlcon(z, area0[k], length0[k])})
            res = minimize(minFun, X[:, k], constraints=cons, options=options, tol=1e-2) # , callback=mycallback
            Xnew[:, k] = res.x
            # print(res.message)
            # print(f"function value{res.fun}") # , cons violation {res.maxcv}
            if not res.success:
                print('Correction scheme failed, do not correct at this step')
                Xnew[:, k] = X[:,k]

        return torch.from_numpy(Xnew).float()

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

        mask = (eAt > tolConstraint) | (eLt > tolConstraint)
        # num_should_correct = torch.sum(mask)
        # mask_skip = (eAt > 0.15) | (eLt > 0.15)
        # num_skip = torch.sum(mask_skip)
        # if num_skip > 0:
        #     print(f"****** should correct {num_should_correct} vesicles, but replacing {num_skip} vesicles")
        #     # print(f"****** skipped ves are {torch.arange(X.shape[1])[mask_skip]}, with eA {eAt[mask_skip]}, eL {eLt[mask_skip]}")
        #     print(f"****** replaced ves are {torch.arange(X.shape[1])[mask_skip]}, with eA {eAt[mask_skip]}, eL {eLt[mask_skip]}")
        # if num_should_correct - num_skip <= 0:
        #     return X
        
        # mask = mask & ~mask_skip    

        X_to_correct = X[:, mask]
        area0 = area0[mask]
        length0 = length0[mask]
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
                self.z = nn.Parameter(X, requires_grad=True)
                self.X = X.clone()
                self.c = Curve()
                # self.area0 = area0
                # self.length0 = length0

            def forward(self, lamb, mu):
                _, a, l = self.c.geomProp(self.z)
                nv = self.X.shape[1]
                a = a.float()
                l = l.float()
                return a, l, mean_rel_err(self.z, self.X), \
                      - 1.0/nv * torch.inner(lamb[:nv], (a -area0)/area0) - 1.0/nv * torch.inner(lamb[nv:], (l - length0)/length0) + \
                        1/(2*mu) * torch.mean(((a-area0)/area0)**2 + ((l-length0)/length0)**2)
        
        def train_model(model, lamb, mu, n_iterations=20, lr=3e-5):
            # We initialize an optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            # model = torch.compile(model, backend='cudagraphs')
            # We take n_iterations steps of gradient descent
            model.train()
            for iter in range(n_iterations):
                optimizer.zero_grad()
                a, l, loss_fun, loss_cons = model(lamb, mu)
                if iter % 15 == 0:
                    print(f"{iter} in ADAM, loss_fun is {loss_fun:.5e}, loss_cons is {loss_cons:.5e}, rel err of a {max_rel_err(a, area0):.5e}, of l {max_rel_err(l, length0):.5e}")
                if loss_fun < tolFunctional and \
                        max_rel_err(a, area0) < tolConstraint and \
                        max_rel_err(l, length0) < tolConstraint:
                    return a, l, True
                # import code; code.interact(local=locals())
                # import pdb; pdb.set_trace()
                # print(loss_fun.requires_grad, loss_cons.requires_grad, loss_fun.grad_fn, loss_cons.grad_fn)
                (loss_fun + loss_cons).backward()
                optimizer.step()

            return a.detach(), l.detach(), False
        
        it = 0
        lamb = torch.zeros(X_to_correct.shape[1]*2, device=X.device, dtype=torch.float32)
        mu = 0.1
        aug_lag_model = AugLag(X_to_correct.float()).to(X.device)
        
        while it < maxiter:
            if it % 5==0 :
                print(f"outside iter {it}")
            a, l, flag = train_model(aug_lag_model, lamb, mu)
            if flag:
                break
            
            lamb -= 1/mu * torch.concat(((a-area0)/area0, (l-length0)/length0))
            mu *= 0.8
            it += 1

        Xnew = aug_lag_model.z.detach() #.double()
        X_corrected = X.clone()
        X_corrected[:, mask] = Xnew.to(X.dtype)
            
        return X_corrected

    
    def correctAreaAndLengthAugLag_replace(self, X, area0, length0, oc):
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
            return X, torch.zeros(X.shape[1], dtype=torch.bool, device=X.device)
        # print(f"initial rel err of a {torch.max(eAt)} and l {torch.max(eLt)}")

        mask = (eAt > tolConstraint) | (eLt > tolConstraint)
        num_should_correct = torch.sum(mask)
        mask_skip = (eAt > 0.15) | (eLt > 0.15)
        num_skip = torch.sum(mask_skip)
        if num_skip > 0:
            self.logger.info(f"****** should correct {num_should_correct} vesicles, but replacing {num_skip} vesicles")
            # print(f"****** skipped ves are {torch.arange(X.shape[1])[mask_skip]}, with eA {eAt[mask_skip]}, eL {eLt[mask_skip]}")
            self.logger.info(f"****** replaced ves are {torch.arange(X.shape[1])[mask_skip]}, with eA {eAt[mask_skip]}, eL {eLt[mask_skip]}")
        if num_should_correct - num_skip <= 0:
            return X, mask_skip
        
        mask = mask & ~mask_skip    
        
        X_to_correct = X[:, mask]
        area0 = area0[mask]
        length0 = length0[mask]
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
            def __init__(self, X, oc):
                super().__init__()
                self.z = nn.Parameter(X, requires_grad=True)
                self.X = X.clone()
                self.c = oc
                # self.area0 = area0
                # self.length0 = length0

            def forward(self, lamb, mu):
                _, a, l = self.c.geomProp(self.z)
                nv = self.X.shape[1]
                a = a.float()
                l = l.float()
                # return a, l, mean_rel_err(self.z, self.X), \
                #       - 1.0/nv * torch.inner(lamb[:nv], (a -area0)) - 1.0/nv * torch.inner(lamb[nv:], (l - length0)) + \
                        # 1/(2*mu) * torch.mean((a-area0)**2 + (l-length0)**2)
                return a, l, mean_rel_err(self.z, self.X), \
                      - 1.0/nv * torch.inner(lamb[:nv], (a -area0)/area0) - 1.0/nv * torch.inner(lamb[nv:], (l - length0)/length0) + \
                        1/(2*mu) * torch.mean(((a-area0)/area0)**2 + ((l-length0)/length0)**2)

        
        def train_model(model, lamb, mu, n_iterations=25, lr=3e-5):
            # We initialize an optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            # model = torch.compile(model, backend='cudagraphs')
            # We take n_iterations steps of gradient descent
            model.train()
            for iter in range(n_iterations):
                optimizer.zero_grad()
                a, l, loss_fun, loss_cons = model(lamb, mu)
                if iter % 15 == 0:
                    self.logger.info(f"{iter} in ADAM, loss_fun is {loss_fun:.5e}, loss_cons is {loss_cons:.5e}, rel err of a {max_rel_err(a, area0):.5e}, of l {max_rel_err(l, length0):.5e}")
                if loss_fun < tolFunctional and \
                        max_rel_err(a, area0) < tolConstraint and \
                        max_rel_err(l, length0) < tolConstraint:
                    return a, l, True
                # import code; code.interact(local=locals())
                # import pdb; pdb.set_trace()
                (loss_fun + loss_cons).backward()
                optimizer.step()

            return a.detach(), l.detach(), False
        
        it = 0
        lamb = torch.zeros(X_to_correct.shape[1]*2, device=X.device, dtype=torch.float32)
        mu = 0.1
        aug_lag_model = AugLag(X_to_correct.float(), oc).to(X.device)
        
        while it < maxiter:
            if it % 5==0 :
                self.logger.info(f"outside iter {it}")
            a, l, flag = train_model(aug_lag_model, lamb, mu)
            if flag:
                break
            
            # lamb -= 1/mu * torch.concat((a-area0, l-length0))
            lamb -= 1/mu * torch.concat(((a-area0)/area0, (l-length0)/length0))
            mu *= 0.8
            it += 1

        Xnew = aug_lag_model.z.detach() #.double()
        X_corrected = X.clone()
        X_corrected[:, mask] = Xnew
            
        return X_corrected, mask_skip

    
    # @torch.jit.script_method
    # @torch.jit.export
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
        newAngle = newAngles[ids, torch.arange(X.shape[1], device=X.device)]

        N = X.shape[0]//2
        # % move to (0,0) new shape
        Xp = torch.concatenate((X[:N] - newCenter[0], X[N:] - newCenter[1]),dim=0)
        # % tilt it to the original angle
        thet = -newAngle + initAngle
        XpNew = torch.zeros_like(Xp)
        # % move to original center
        XpNew[:N] = Xp[:N] * torch.cos(thet) - Xp[N:] * torch.sin(thet) + initCenter[0]
        XpNew[N:] = Xp[:N] * torch.sin(thet) + Xp[N:] * torch.cos(thet) + initCenter[1]
        
        return XpNew

    # @torch.jit.script_method
    # @torch.compile(backend='cudagraphs')
    def redistributeArcLength(self, X, modes):
        """Redistribute the vesicle shape equispaced in arclength."""
        # % [X,u,sigma] = redistributeArcLength(o,X,u,sigma) redistributes
        # % the vesicle shape eqiuspaced in arclength and adjusts the tension and
        # % velocity according to the new parameterization

        N = X.shape[0] // 2
        # nv = X.shape[1]
        # modes = torch.concatenate((torch.arange(0, N // 2), [0], torch.arange(-N // 2 + 1, 0)))
        # modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).to(X.device) #.double()

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        jac = self.diffProp_jac(X)


        tol = 1e-5

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
        # X_out = X.clone()
        to_redistribute = torch.linalg.norm(jac - torch.mean(jac, dim=0), ord=torch.inf) > tol * torch.mean(jac, dim=0)

        if torch.max(to_redistribute):
            allGood = False
            ids = torch.where(to_redistribute)[0]
            # start.record()
            # theta = self.arcLengthParameter(X[:N, ids], X[N:, ids])
            theta = self.my_arcLengthParameter(X[:, ids])
            # end.record()
            # torch.cuda.synchronize()
            # print(f'----------- inside redistribute, arcLengthPara  {start.elapsed_time(end)/1000} sec.')

            # theta = torch.from_numpy(theta).to(X.device)
            
            zX = X[:N, ids] + 1.0j * X[N:, ids]
            zXh = torch.fft.fft(zX.cfloat(), dim=0) / N
            # zX = torch.zeros((N, len(ids)), dtype=torch.complex128, device=X.device)
            # for jj in range(N): # use broadcasting to remove this loop
            #     zX += zXh[jj] * torch.exp(1j * modes[jj] * theta)
            zX_ = torch.einsum('mj,mnj->nj', zXh, torch.exp(1.0j * modes[:,None,None] * theta).cfloat())

            # zX_ = torch.exp(1j * modes[:,None,None] * theta).permute(1,2,0) @ zXh.T
            # X_out[:N, ids] = torch.real(zX_)
            # X_out[N:, ids] = torch.imag(zX_)
            X[:N, ids] = torch.real(zX_)
            X[N:, ids] = torch.imag(zX_)
        
        return X, allGood

    
    
    
    # def arcLengthParameter(self, x, y):
    #     """
    #     % theta = arcLengthParamter(o,x,y) finds a discretization of parameter
    #     % space theta so that the resulting geometry will be equispaced in
    #     % arclength
    #     """
    #     N = len(x)
    #     t = torch.arange(N, dtype=torch.float32, device=x.device) * 2 * torch.pi / N
    #     X = torch.concatenate((x, y))
    #     if len(X.shape) < 2:
    #         X = X.unsqueeze(-1)
    #     _, _, length = self.geomProp(X)
    #     # Find total perimeter
        
    #     Dx, Dy = self.getDXY(X)
    #     # Find derivative
    #     arc = torch.sqrt(Dx**2 + Dy**2)
    #     arch = torch.fft.fft(arc, dim=0).T # (nv, N)
    #     # modes = -1.0j / torch.hstack([torch.tensor([1e-10]).double(), (torch.arange(1,N // 2)), torch.tensor([1e-10]).double(), (torch.arange(-N//2+1,0))])  # FFT modes
    #     modes = -1.0j / torch.hstack([torch.tensor([1e-10]), (torch.arange(1,N // 2)), torch.tensor([1e-10]), (torch.arange(-N//2+1,0))])  # FFT modes
    #     modes[0] = 0
    #     modes[N // 2] = 0
    #     modes = modes.to(x.device) #(N)
        
    #     arc_length = torch.real(torch.fft.ifft(modes * arch, dim=-1) - \
    #                             torch.sum(modes * arch, dim=-1).unsqueeze(-1) / N + arch[:,0:1] * t / N).T
    #     # arc_length shape: (N, nv)
    #     z1 = torch.concat((arc_length[-7:] - length, arc_length, arc_length[:7] + length), dim=0).cpu().numpy()
    #     z2 = torch.hstack([t[-7:] - 2 * torch.pi, t, t[:7] + 2 * torch.pi]).cpu().numpy()
    #     # % put in some overlap to account for periodicity

    #     # Interpolate to obtain equispaced points
    #     # dx = torch.diff(z1)
    #     # dx = abs(dx)
    #     # dump_z1 = torch.cumsum(torch.concat((z1[[0]], dx)), dim=0).cpu().numpy()
    #     # if torch.any(dx <= 0):
    #     #     print(dx)
    #     #     print("haha")
        
    #     theta = np.zeros((N, X.shape[1]))
    #     for i in range(X.shape[1]):
    #         try:
    #             theta[:,i] = CubicSpline(z1[:,i], z2)(torch.arange(N).cpu() * length[i].cpu() / N)
    #             # theta[:,i] = interp1d(z1[:,i], z2, 'linear')(np.arange(N) * length[i].cpu().numpy() / N)
    #         except:
    #             print("CubicSpline has error")
    #             print(f"we are at {i}-th vesicle, with shape {X[:, i]}")
                
    #     # # Create interpolation function using cubic spline
    #     # interpolation_function = interp1d(z1, z2, kind='cubic')  # 'cubic' is equivalent to MATLAB's 'spline'
    #     # # Generate theta values with interpolation
    #     # theta = interpolation_function(torch.arange(N).cpu() * length.cpu() / N)

    #     # theta = self.cubic_spline_interp(z1, z2, torch.arange(N).cpu() * length.cpu() / N)

    #     # theta = eng.interp1(z1.numpy(),z2, np.arange(N)*length.cpu().numpy()/N,'spline')
        
    #     return theta

    
    # @torch.jit.script_method
    # @torch.compile(backend='cudagraphs')
    def my_arcLengthParameter(self, x):
        """
        % theta = arcLengthParamter(o,x,y) finds a discretization of parameter
        % space theta so that the resulting geometry will be equispaced in
        % arclength
        """
        N = x.shape[0]//2
        Nup = N * 6
        t = torch.arange(Nup, dtype=torch.float32, device=x.device) * 2 * torch.pi / Nup

        # X = torch.concatenate((interpft(x, Nup), interpft(y, Nup)))
        X = upsample_fft(x, Nup)
        if len(X.shape) < 2:
            X = X.unsqueeze(-1)
        length = self.geomProp_length(x)
        # Find total perimeter
        
        Dx, Dy = self.getDXY(X)
        # Find derivative
        arc = torch.sqrt(Dx**2 + Dy**2)
        arch = torch.fft.fft(arc.T, dim=1) # (nv, N)
        # modes = -1.0j / torch.hstack([torch.tensor([1e-10]).double(), (torch.arange(1,Nup // 2)),
        #                                torch.tensor([1e-10]).double(), (torch.arange(-Nup//2+1,0))])  # FFT modes
        
        modes = -1.0j / torch.hstack([torch.tensor([torch.inf],device=X.device), (torch.arange(1,Nup // 2, device=X.device)),
                                       torch.tensor([torch.inf],device=X.device), (torch.arange(-Nup//2+1,0, device=X.device))])  # FFT modes
        
        # modes[0] = 0
        # modes[Nup // 2] = 0
        modes = modes.to(x.device) #(N)
        
        # arc_length shape: (nv, N)
        arc_length = torch.real(torch.fft.ifft(modes * arch, dim=-1) - \
                                torch.sum(modes * arch, dim=-1, keepdim=True) / Nup + arch[:,0:1] * t / Nup)
        
        # z1 = torch.concat((arc_length.T[-7:] - length, arc_length.T, arc_length.T[:7] + length), dim=0)
        # z2 = torch.hstack([t[-7:] - 2 * torch.pi, t, t[:7] + 2 * torch.pi])
        # % put in some overlap to account for periodicity

        # Interpolate to obtain equispaced points
        
        theta = torch.zeros((X.shape[1], N), device=X.device, dtype=X.dtype)

        
        target = torch.arange(1, N, device=X.device, dtype=X.dtype) * length[:, None] / N  # (nv, N-1)
        indices = torch.searchsorted(arc_length.contiguous(), target) # torch real makes arc_length not contiguous
        theta[:, 1:] = t[None, :].expand(X.shape[1], -1).gather(1, indices - 1) + \
            (target - arc_length.gather(1, indices - 1)) *  (2 * torch.pi)

        # theta_ = np.zeros((N, X.shape[1]))
        # for i in range(X.shape[1]):
        #      theta_[:,i] = interp1d(z1[:,i].cpu().numpy(), z2.cpu().numpy(), 'linear')(np.arange(N) * length[i].cpu().numpy() / N)
  
        return theta.T

    
    
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
        modes = modes #.double()
        # % get tangent vector at each point (tang_x;tang_y) 
        _, tang, _ = self.diffProp(torch.concatenate((x, y)).reshape(-1,1))
        # % get x and y components of normal vector at each point
        nx = tang[N:]
        ny = -tang[:N]

        # % Compute gradE
        # % first, get Fourier coefficients
        zX = x + 1.0j * y
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



# # import matplotlib.pyplot as plt
# c = Curve()
# Xics = loadmat("../1000vesShape8.mat").get('X')
# x = torch.from_numpy(Xics)
# # center = torch.ones((2, 4))
# # print(x)

# # %matplotlib inline
# # plt.plot(x[:32, :], x[32:, :])
# # plt.axis('scaled')
# # plt.show()

# def rotationOperator(X, theta, rotCent):
#     ''' Shan: compatible with multi ves
#     theta of shape (1,nv), rotCent of shape (2,nv)'''
#     Xrot = torch.zeros_like(X)
#     x = X[:len(X)//2] - rotCent[0]
#     y = X[len(X)//2:] - rotCent[1]

#     # Rotated shape
#     xrot = x * torch.cos(theta) - y * torch.sin(theta)
#     yrot = x * torch.sin(theta) + y * torch.cos(theta)

#     Xrot[:len(X)//2] = xrot + rotCent[0]
#     Xrot[len(X)//2:] = yrot + rotCent[1]
#     return Xrot


# center1 = c.getPhysicalCenter(x) # new one
# # center2 = c.getPhysicalCenter_(x)
# Xref = rotationOperator(x, torch.ones(x.shape[1])*0.48, center1)

# center2 = c.getPhysicalCenter(Xref)
# print(center1)
# print(center2)
# print(center1 - center2)

