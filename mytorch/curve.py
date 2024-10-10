import torch
import numpy as np
torch.set_default_dtype(torch.float64)
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from fft1 import fft1


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
        x = X[:N, :]
        y = X[N:, :]
        return x, y

    def setXY(self, x, y):
        """Set the [x,y] component of vector V on the curve."""
        N = x.shape[0]
        V = torch.zeros(2 * N, x.shape[1], dtype=torch.float64, device=x.device)
        V[:N, :] = x
        V[N:, :] = y
        return V

    def getCenter(self, X):
        """Find the center of each capsule."""
        center = torch.sqrt(torch.mean(X[:X.shape[0] // 2], dim=0) ** 2 +
                                torch.mean(X[X.shape[0] // 2:], dim=0) ** 2)
        return center
    
    def getPhysicalCenter(self, X):
        """Fin the physical center of each capsule. Compatible with multi ves.
        returns center in shape (2, nv)
        """
        nv = X.shape[1]
        # Compute the differential properties of X
        jac, tan, curv = self.diffProp(X)
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
        IA = torch.zeros(nv, dtype=torch.float64)
        # % compute inclination angle on an upsampled grid
        N = X.shape[0] // 2
        modes = torch.concatenate((torch.arange(0, N // 2), torch.tensor([0]), torch.arange(-N // 2 + 1, 0))).double()
        
        tempX = torch.zeros_like(X)
        tempX[:X.shape[0] // 2] = X[:X.shape[0] // 2] - torch.mean(X[:X.shape[0] // 2], dim=0)
        tempX[X.shape[0] // 2:] = X[X.shape[0] // 2:] - torch.mean(X[X.shape[0] // 2:], dim=0)

        for k in range(nv):
            x = tempX[:N, k]
            y = tempX[N:, k]
            
            Dx = torch.real(torch.fft.ifft(1j * modes * torch.fft.fft(x)))
            Dy = torch.real(torch.fft.ifft(1j * modes * torch.fft.fft(y)))
            jac = torch.sqrt(Dx ** 2 + Dy ** 2)
            tx = Dx / jac
            ty = Dy / jac
            nx = ty
            ny = -tx #Shan: n is the right hand side of t
            rdotn = x * nx + y * ny
            rho2 = x ** 2 + y ** 2

            J11 = 0.25 * torch.sum(rdotn * (rho2 - x * x) * jac) * 2 * torch.pi / N
            J12 = 0.25 * torch.sum(rdotn * (-x * y) * jac) * 2 * torch.pi / N
            J21 = 0.25 * torch.sum(rdotn * (-y * x) * jac) * 2 * torch.pi / N
            J22 = 0.25 * torch.sum(rdotn * (rho2 - y * y) * jac) * 2 * torch.pi / N

            J = torch.tensor([[J11, J12], [J21, J22]])
            
            D, V = torch.linalg.eig(J)
            ind = torch.argmin(torch.abs((D)))
            # % make sure that the first components of e-vectors have the same sign
            V = torch.real(V)
            if V[1, ind] < 0:
                V[:, ind] *= -1
            # % since V(2,ind) > 0, this will give angle between [0, pi]
            IA[k] = torch.arctan2(V[1, ind], V[0, ind])

        return IA
    
    def getPrincAxesGivenCentroid(self, X, center):
        """Find the principal axes given centroid.
        % GK: THIS IS NEEDED IN STANDARDIZING VESICLE SHAPES 
        """
        nv = X.shape[1]
        # % compute inclination angle on an upsampled grid
        N = X.shape[0] // 2
        modes = torch.concatenate((torch.arange(0, N // 2), torch.tensor([0]), torch.arange(-N // 2 + 1, 0))).double()
        
        tempX = torch.zeros_like(X)
        tempX[:X.shape[0] // 2] = X[:X.shape[0] // 2] - torch.mean(X[:X.shape[0] // 2], dim=0)
        tempX[X.shape[0] // 2:] = X[X.shape[0] // 2:] - torch.mean(X[X.shape[0] // 2:], dim=0)

        multiple_V = torch.zeros((2,nv), dtype=torch.float64)
        for k in range(nv):
            x = tempX[:N, k]
            y = tempX[N:, k]
            
            x -= center[0,k]
            y -= center[1,k]
            
            Dx = torch.real(torch.fft.ifft(1j * modes * torch.fft.fft(x)))
            Dy = torch.real(torch.fft.ifft(1j * modes * torch.fft.fft(y)))
            jac = torch.sqrt(Dx ** 2 + Dy ** 2)
            tx = Dx / jac
            ty = Dy / jac
            nx = ty
            ny = -tx #Shan: n is the right hand side of t
            rdotn = x * nx + y * ny
            rho2 = x ** 2 + y ** 2

            J11 = 0.25 * torch.sum(rdotn * (rho2 - x * x) * jac) * 2 * torch.pi / N
            J12 = 0.25 * torch.sum(rdotn * (-x * y) * jac) * 2 * torch.pi / N
            J21 = 0.25 * torch.sum(rdotn * (-y * x) * jac) * 2 * torch.pi / N
            J22 = 0.25 * torch.sum(rdotn * (rho2 - y * y) * jac) * 2 * torch.pi / N

            J = torch.tensor([[J11, J12], [J21, J22]])
            # Shan
            D, V = torch.linalg.eig(J)
            ind = torch.argmin(torch.abs((D)))
            # % make sure that the first components of e-vectors have the same sign
            multiple_V[:,k] = torch.real(V[:,ind])
            
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
        
        # % Find the current area and length
        # _, a, l = self.geomProp(X)
        # eAt = torch.abs((a - area0) / area0)
        # eLt = torch.abs((l - length0) / length0)

        N = X.shape[0] // 2
        # tolConstraint = 1e-2
        # tolFunctional = 1e-2

        options = {'maxiter': 3000, 'disp': False}

        X = X.cpu().numpy()
        area0 = area0.cpu().numpy()
        length0 = length0.cpu().numpy()
        Xnew = np.zeros_like(X)
        for k in range(X.shape[1]):
            def minFun(z):
                return np.mean((z - X[:, k]) ** 2)

            cons = ({'type': 'eq', 'fun': lambda z: self.nonlcon(z, area0[k], length0[k])})
            res = minimize(minFun, X[:, k], constraints=cons, options=options)
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

    def alignCenterAngle(self, Xorg, X):
        """Use translation and rotation to match X with Xorg."""
        # % Xnew = alignCenterAngle(o,Xorg,X) uses
        # % rigid body translation and rotation to match X having the corrected area 
        # % and length but wrong center and inclination angle with Xorg having the 
        # % right center and IA but wrong area and length. So that, Xnew has the
        # % correct area,length,center and inclination angle.

        Xnew = torch.zeros_like(X)
        for k in range(X.shape[1]):
            initMean = torch.tensor([torch.mean(Xorg[:Xorg.shape[0] // 2, k]), torch.mean(Xorg[Xorg.shape[0] // 2:, k])])
            newMean = torch.tensor([torch.mean(X[:X.shape[0] // 2, k]), torch.mean(X[X.shape[0] // 2:, k])])

            initAngle = self.getIncAngle(Xorg[:, [k]])
            newAngle = self.getIncAngle(X[:, [k]])

            if newAngle > torch.pi:
                newAngle2 = newAngle - torch.pi
            else:
                newAngle2 = newAngle + torch.pi
            newAngles = torch.tensor([newAngle, newAngle2])
            diffAngles = torch.abs(initAngle - newAngles)
            id = torch.argmin(diffAngles)
            newAngle = newAngles[id]

            # % move to (0,0) new shape
            Xp = torch.concatenate((X[:X.shape[0] // 2, k] - newMean[0], X[X.shape[0] // 2:, k] - newMean[1]),dim=0)
            # % tilt it to the original angle
            thet = -newAngle+initAngle
            XpNew = torch.zeros_like(Xp)
            XpNew[:Xp.shape[0]//2] = Xp[:Xp.shape[0]//2] * torch.cos(thet) - Xp[Xp.shape[0]//2:] * torch.sin(thet)
            XpNew[Xp.shape[0]//2:] = Xp[:Xp.shape[0]//2] * torch.sin(thet) + Xp[Xp.shape[0]//2:] * torch.cos(thet)

            # % move to original center
            Xnew[:, k] = torch.concatenate((XpNew[:Xp.shape[0]//2] + initMean[0], XpNew[Xp.shape[0]//2:] + initMean[1]), dim=0)

        return Xnew

    def redistributeArcLength(self, X):
        """Redistribute the vesicle shape equispaced in arclength."""
        # % [X,u,sigma] = redistributeArcLength(o,X,u,sigma) redistributes
        # % the vesicle shape eqiuspaced in arclength and adjusts the tension and
        # % velocity according to the new parameterization

        N = X.shape[0] // 2
        nv = X.shape[1]
        # modes = torch.concatenate((torch.arange(0, N // 2), [0], torch.arange(-N // 2 + 1, 0)))
        modes = torch.concatenate((torch.arange(0, N // 2), torch.arange(-N // 2, 0))).double()
        jac, _, _ = self.diffProp(X)
        tol = 1e-10
        # u = None
        # sigma = None
        X_out = torch.zeros_like(X, device=X.device)
        allGood = True

        for k in range(nv):
            if torch.linalg.norm(jac[:, k] - torch.mean(jac[:, k]), ord=torch.inf) > tol * torch.mean(jac[:, k]):
                allGood = False
                theta, _ = self.arcLengthParameter(X[:N, k], X[N:, k])
                theta = torch.from_numpy(theta).to(X.device)
                # print(theta)
                zX = X[:N, k] + 1j * X[N:, k]
                zXh = torch.fft.fft(zX) / N
                zX = torch.zeros(N, dtype=torch.complex128, device=X.device)
                for j in range(N):
                    zX += zXh[j] * torch.exp(1j * modes[j] * theta)
                X_out[:, [k]] = self.setXY(torch.real(zX)[:,None], torch.imag(zX)[:,None])
            else:
                X_out[:, [k]] = X[:, [k]]
                
                # if u is not None:
                #     zu = u[:N, k] + 1j * u[N:, k]
                #     zuh = torch.fft.fft(zu) / N
                #     sigmah = torch.fft.fft(sigma[:, k]) / N
                #     zu = torch.zeros(N, dtype=torch.complex64)
                #     sigma[:, k] = torch.zeros(N)
                #     for j in range(N):
                #         zu += zuh[j] * torch.exp(1j * modes[j] * theta)
                #         sigma[:, k] += sigmah[j] * torch.exp(1j * modes[j] * theta)
                #     sigma = torch.real(sigma)
                #     u[:, k] = self.setXY(torch.real(zu), torch.imag(zu))
        return X_out, allGood

    def arcLengthParameter(o, x, y):
        """
        % theta = arcLengthParamter(o,x,y) finds a discretization of parameter
        % space theta so that the resulting geometry will be equispaced in
        % arclength
        """
        N = len(x)
        t = torch.arange(N, dtype=torch.float64, device=x.device) * 2 * torch.pi / N
        _, _, length = o.geomProp(torch.concatenate((x, y))[:,None])
        # Find total perimeter
        
        Dx, Dy = o.getDXY(torch.concatenate((x, y))[:,None])
        # Find derivative
        arc = torch.sqrt(Dx**2 + Dy**2)
        arch = torch.fft.fft(arc.reshape(-1))
        modes = -1.0j / torch.hstack([torch.tensor([1e-10]).double(), (torch.arange(1,N // 2)), torch.tensor([1e-10]).double(), (torch.arange(-N//2+1,0))])  # FFT modes
        modes[0] = 0
        modes[N // 2] = 0
        modes = modes.to(x.device)
        
        arc_length = torch.real(torch.fft.ifft(modes * arch) - torch.sum(modes * arch) / N + arch[0] * t / N)
        # print(arc_length)
        z1 = torch.hstack([arc_length[-7:] - length, arc_length, arc_length[:7] + length])
        z2 = torch.hstack([t[-7:] - 2 * torch.pi, t, t[:7] + 2 * torch.pi]).cpu().numpy()
        # % put in some overlap to account for periodicity

        # Interpolate to obtain equispaced points
        dx = torch.diff(z1)
        dx = abs(dx)
        z1 = torch.cumsum(torch.concat((z1[[0]], dx)), dim=0).cpu().numpy()
        if torch.any(dx <= 0):
            print(dx)
            print("haha")
        theta = CubicSpline(z1, z2)(torch.arange(N).cpu() * length.cpu() / N)

        return theta, arc_length

    def reparametrize(self, X, dX, maxIter=100):
        """Reparametrize to minimize the energy in the high frequencies."""
        # % [X,niter] = reparametrize applies the reparametrization with
        # % minimizing the energy in the high frequencies (Veerapaneni et al. 2011, 
        # % doi: 10.1016/j.jcp.2011.03.045, Section 6).

        pow = 4
        nv = X.shape[1]
        niter = torch.ones(nv, dtype=int)
        tolg = 1e-3
        if dX is None:
            _, _, length = self.geomProp(X)
            dX = length / X.shape[0]
            toly = 1e-5 * dX
        else:
            normDx = torch.sqrt(dX[:X.shape[0] // 2] ** 2 + dX[X.shape[0] // 2:] ** 2)
            toly = 1e-3 * torch.min(normDx)

        beta = 0.1
        dtauOld = 0.05

        for k in range(nv):
            # % Get initial coordinates of kth vesicle (upsample if necessary)
            x0 = X[:X.shape[0] // 2, [k]]
            y0 = X[X.shape[0] // 2:, [k]]
            # % Compute initial projected gradient energy
            g0 = self.computeProjectedGradEnergy(x0, y0, pow)
            x = x0
            y = y0
            g = g0
            
            # % Explicit reparametrization
            while niter[k] <= maxIter:
                dtau = dtauOld
                xn = x - g[:X.shape[0] // 2] * dtau
                yn = y - g[X.shape[0] // 2:] * dtau
                gn = self.computeProjectedGradEnergy(xn, yn, pow)
                while torch.linalg.norm(gn) > torch.linalg.norm(g):
                    dtau = dtau * beta
                    xn = x - g[:X.shape[0] // 2] * dtau
                    yn = y - g[X.shape[0] // 2:] * dtau
                    gn = self.computeProjectedGradEnergy(xn, yn, pow)
                dtauOld = dtau * 1 / beta
                # print(toly)
                if torch.linalg.norm(gn) < max(max(toly / dtau), tolg * torch.linalg.norm(g0)):
                    break
                x = xn
                y = yn
                g = gn
                niter[k] += 1
            X[:, [k]] = torch.vstack((xn, yn))

        return X

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
