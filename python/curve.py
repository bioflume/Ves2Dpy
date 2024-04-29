import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from python.fft1 import fft1


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
        V = np.zeros((2 * N, x.shape[1]))
        V[:N, :] = x
        V[N:, :] = y
        return V

    def getCenter(self, X):
        """Find the center of each capsule."""
        # nv = X.shape[1]
        # center = np.zeros(nv)
        # for k in range(nv):
        #     center[k] = np.sqrt(np.mean(X[:X.shape[0] // 2, k]) ** 2 +
        #                         np.mean(X[X.shape[0] // 2:, k]) ** 2)
        center = np.sqrt(np.mean(X[:X.shape[0] // 2], axis=0) ** 2 +
                                np.mean(X[X.shape[0] // 2:], axis=0) ** 2)
        return center

    def getIncAngle2(self, X):
        """Find the inclination angle of each capsule.
        % GK: THIS IS NEEDED IN STANDARDIZING VESICLE SHAPES 
        % WE NEED TO KNOW THE INCLINATION ANGLE AND ROTATE THE VESICLE TO pi/2
        % IA = getIncAngle(o,X) finds the inclination angle of each capsule
        % The inclination angle (IA) is the angle between the x-axis and the 
        % principal axis corresponding to the smallest principal moment of inertia
        """
        nv = X.shape[1]
        IA = np.zeros(nv)
        # % compute inclination angle on an upsampled grid
        N = X.shape[0] // 2
        modes = np.concatenate((np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0)))
        
        
        X[:X.shape[0] // 2] -= np.mean(X[:X.shape[0] // 2], axis=0)
        X[X.shape[0] // 2:] -= np.mean(X[X.shape[0] // 2:], axis=0)

        for k in range(nv):
            x = X[:N, k]
            y = X[N:, k]
            
            Dx = np.real(np.fft.ifft(1j * modes * np.fft.fft(x)))
            Dy = np.real(np.fft.ifft(1j * modes * np.fft.fft(y)))
            jac = np.sqrt(Dx ** 2 + Dy ** 2)
            tx = Dx / jac
            ty = Dy / jac
            nx = ty
            ny = -tx #Shan: n is the right hand side of t
            rdotn = x * nx + y * ny
            rho2 = x ** 2 + y ** 2

            J11 = 0.25 * np.sum(rdotn * (rho2 - x * x) * jac) * 2 * np.pi / N
            J12 = 0.25 * np.sum(rdotn * (-x * y) * jac) * 2 * np.pi / N
            J21 = 0.25 * np.sum(rdotn * (-y * x) * jac) * 2 * np.pi / N
            J22 = 0.25 * np.sum(rdotn * (rho2 - y * y) * jac) * 2 * np.pi / N

            J = np.array([[J11, J12], [J21, J22]])
            # Shan
            D, V = np.linalg.eig(J)
            ind = np.argmin(np.abs((D)))
            # % make sure that the first components of e-vectors have the same sign
            if V[1, ind] < 0:
                V[:, ind] *= -1
            # % since V(2,ind) > 0, this will give angle between [0, pi]
            IA[k] = np.arctan2(V[1, ind], V[0, ind])

            # % FIND DIRECTION OF PRIN. AXIS DEPENDING ON HEAD-TAIL
            # % 1) already translated to (0,0), so rotate to pi/2
            x0rot = x * np.cos(-IA[k] + np.pi / 2) - y * np.sin(-IA[k] + np.pi / 2)
            y0rot = x * np.sin(-IA[k] + np.pi / 2) + y * np.cos(-IA[k] + np.pi / 2)
            
            # % 2) find areas (top, bottom)
            # % need derivatives, so rotate the computed ones
            Dx = Dx * np.cos(-IA[k] + np.pi / 2) - Dy * np.sin(-IA[k] + np.pi / 2)
            Dy = Dx * np.sin(-IA[k] + np.pi / 2) + Dy * np.cos(-IA[k] + np.pi / 2)

            idcsTop = np.where(y0rot >= 0)[0]
            idcsBot = np.where(y0rot < 0)[0]
            areaTop = np.sum(x0rot[idcsTop] * Dy[idcsTop] - y0rot[idcsTop] * Dx[idcsTop]) / N * np.pi
            areaBot = np.sum(x0rot[idcsBot] * Dy[idcsBot] - y0rot[idcsBot] * Dx[idcsBot]) / N * np.pi

            if areaBot >= 1.1 * areaTop:
                IA[k] += np.pi
            elif areaTop < 1.1 * areaBot:
                # % if areaTop ~ areaBot, then check areaRight, areaLeft  
                idcsLeft = np.where(x0rot < 0)[0]
                idcsRight = np.where(x0rot >= 0)[0]
                areaRight = np.sum(x0rot[idcsRight] * Dy[idcsRight] - y0rot[idcsRight] * Dx[idcsRight]) / N * np.pi
                areaLeft = np.sum(x0rot[idcsLeft] * Dy[idcsLeft] - y0rot[idcsLeft] * Dx[idcsLeft]) / N * np.pi
                if areaLeft >= 1.1 * areaRight:
                    IA[k] += np.pi
        return IA

    def getDXY(self, X):
        """Compute the derivatives of each component of X."""
        # % [Dx,Dy]=getDXY(X), compute the derivatives of each component of X 
        # % these are the derivatives with respect the parameterization 
        # % not arclength
        x, y = self.getXY(X)
        N = x.shape[0]
        nv = x.shape[1]
        f = fft1(N)
        IK = f.modes(N, nv)
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
        jacobian = np.sqrt(Dx ** 2 + Dy ** 2)

        tangent = np.vstack((Dx / jacobian, Dy / jacobian))

        f = fft1(N)
        IK = f.modes(N, nv)
        DDx = self.arcDeriv(Dx, 1, np.ones((N, nv)), IK)
        
        DDy = self.arcDeriv(Dy, 1, np.ones((N, nv)), IK)
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
        
        x, y = self.getXY(X)
        N = x.shape[0]
        Dx, Dy = self.getDXY(X)
        length = np.sum(np.sqrt(Dx ** 2 + Dy ** 2),axis=0) * 2 * np.pi / N
        area = np.sum(x * Dy - y * Dx,axis=0) * np.pi / N
        reducedArea = 4 * np.pi * area / length ** 2
        return reducedArea, area, length


    def ellipse(self, N, ra):
        """
        Finds the ellipse (a*cos(theta), sin(theta)) so that the reduced area is ra.
        % X0 = o.ellipse(N,ra) finds the ellipse (a*cos(theta),sin(theta)) so
        % that the reduced area is ra.  Uses N points.  Parameter a is found 
        % by using bisection method
        """
        t = np.arange(N) * 2 * np.pi / N
        a = (1 - np.sqrt(1 - ra**2)) / ra
        # Initial guess using approximation length = sqrt(2) * pi * sqrt(a^2 + 1)
        X0 = np.concatenate((a * np.cos(t), np.sin(t)))[:,None]
        ra_new, _, _ = self.geomProp(X0)
        cond = np.abs(ra_new - ra) / ra < 1e-4
        maxiter = 10
        iter = 0

        while (not cond[0] and iter < maxiter):
            iter += 1
            
            if ra_new > ra:
                a *= 0.9
            else:
                a *= 1.05

            # Update the major axis
            X0 = np.concatenate((np.cos(t), a*np.sin(t)))[:,None]
            # Compute new possible configuration
            ra_new, _, _ = self.geomProp(X0)
            # Compute new reduced area
            cond = np.abs(ra_new - ra) < 1e-2
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
        # eAt = np.abs((a - area0) / area0)
        # eLt = np.abs((l - length0) / length0)

        N = X.shape[0] // 2
        # tolConstraint = 1e-2
        # tolFunctional = 1e-2

        options = {'maxiter': 3000, 'disp': False}

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

        return Xnew

    def nonlcon(self, X, a0, l0):
        """Non-linear constraints required by minimize."""
        _, a, l = self.geomProp(X[:,None])
        cEx = np.hstack(((a - a0) / a0, (l - l0) / l0))
        return cEx

    def alignCenterAngle(self, Xorg, X):
        """Use translation and rotation to match X with Xorg."""
        # % Xnew = alignCenterAngle(o,Xorg,X) uses
        # % rigid body translation and rotation to match X having the corrected area 
        # % and length but wrong center and inclination angle with Xorg having the 
        # % right center and IA but wrong area and length. So that, Xnew has the
        # % correct area,length,center and inclination angle.

        Xnew = np.zeros_like(X)
        for k in range(X.shape[1]):
            initMean = np.array([np.mean(Xorg[:Xorg.shape[0] // 2, k]), np.mean(Xorg[Xorg.shape[0] // 2:, k])])
            newMean = np.array([np.mean(X[:X.shape[0] // 2, k]), np.mean(X[X.shape[0] // 2:, k])])

            initAngle = self.getIncAngle2(Xorg[:, [k]])
            newAngle = self.getIncAngle2(X[:, [k]])

            if newAngle > np.pi:
                newAngle2 = newAngle - np.pi
            else:
                newAngle2 = newAngle + np.pi
            newAngles = np.array([newAngle, newAngle2])
            diffAngles = np.abs(initAngle - newAngles)
            id = np.argmin(diffAngles)
            newAngle = newAngles[id]

            # % move to (0,0) new shape
            Xp = np.concatenate((X[:X.shape[0] // 2, k] - newMean[0], X[X.shape[0] // 2:, k] - newMean[1]),axis=0)
            # % tilt it to the original angle
            thet = -newAngle+initAngle
            XpNew = np.zeros_like(Xp)
            XpNew[:Xp.shape[0]//2] = Xp[:Xp.shape[0]//2] * np.cos(thet) - Xp[Xp.shape[0]//2:] * np.sin(thet)
            XpNew[Xp.shape[0]//2:] = Xp[:Xp.shape[0]//2] * np.sin(thet) + Xp[Xp.shape[0]//2:] * np.cos(thet)

            # % move to original center
            Xnew[:, k] = np.concatenate((XpNew[:Xp.shape[0]//2] + initMean[0], XpNew[Xp.shape[0]//2:] + initMean[1]), axis=0)

        return Xnew

    def redistributeArcLength(self, X, u=None, sigma=None):
        """Redistribute the vesicle shape equispaced in arclength."""
        # % [X,u,sigma] = redistributeArcLength(o,X,u,sigma) redistributes
        # % the vesicle shape eqiuspaced in arclength and adjusts the tension and
        # % velocity according to the new parameterization

        N = X.shape[0] // 2
        nv = X.shape[1]
        modes = np.concatenate((np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0)))
        jac, _, _ = self.diffProp(X)
        tol = 1e-10
        # u = None
        # sigma = None
        X_out = np.zeros_like(X)

        for k in range(nv):
            if np.linalg.norm(jac[:, k] - np.mean(jac[:, k]), ord=np.inf) > tol * np.mean(jac[:, k]):
                theta, _ = self.arcLengthParameter(X[:N, k], X[N:, k])
                zX = X[:N, k] + 1j * X[N:, k]
                zXh = np.fft.fft(zX) / N
                zX = np.zeros(N, dtype=np.complex64)
                for j in range(N):
                    zX += zXh[j] * np.exp(1j * modes[j] * theta)
                X_out[:, [k]] = self.setXY(np.real(zX)[:,None], np.imag(zX)[:,None])
                # if u is not None:
                #     zu = u[:N, k] + 1j * u[N:, k]
                #     zuh = np.fft.fft(zu) / N
                #     sigmah = np.fft.fft(sigma[:, k]) / N
                #     zu = np.zeros(N, dtype=np.complex64)
                #     sigma[:, k] = np.zeros(N)
                #     for j in range(N):
                #         zu += zuh[j] * np.exp(1j * modes[j] * theta)
                #         sigma[:, k] += sigmah[j] * np.exp(1j * modes[j] * theta)
                #     sigma = np.real(sigma)
                #     u[:, k] = self.setXY(np.real(zu), np.imag(zu))
        return X_out, u, sigma

    def arcLengthParameter(o, x, y):
        """
        % theta = arcLengthParamter(o,x,y) finds a discretization of parameter
        % space theta so that the resulting geometry will be equispaced in
        % arclength
        """
        N = len(x)
        t = np.arange(N) * 2 * np.pi / N
        _, _, length = o.geomProp(np.concatenate((x, y))[:,None])

        # Find total perimeter
        Dx, Dy = o.getDXY(np.concatenate((x, y))[:,None])
        # Find derivative
        arc = np.sqrt(Dx**2 + Dy**2)
        arch = np.fft.fft(arc.reshape(-1))
        modes = -1j / np.hstack([1e-10, (np.arange(1,N // 2)), 1e-10, (np.arange(-N//2+1,0))])  # FFT modes
        modes[0] = 0
        modes[N // 2] = 0
        
        arc_length = np.real(np.fft.ifft(modes * arch) - np.sum(modes * arch) / N + arch[0] * t / N)
        # print(arc_length)
        z1 = np.hstack([arc_length[-7:] - length, arc_length, arc_length[:7] + length])
        z2 = np.hstack([t[-7:] - 2 * np.pi, t, t[:7] + 2 * np.pi])
        # % put in some overlap to account for periodicity

        # Interpolate to obtain equispaced points
        # print(z1)
        # print(z2)
        theta = CubicSpline(z1, z2)(np.arange(N) * length / N)

        return theta, arc_length

    def reparametrize(self, X, dX, maxIter=100):
        """Reparametrize to minimize the energy in the high frequencies."""
        # % [X,niter] = reparametrize applies the reparametrization with
        # % minimizing the energy in the high frequencies (Veerapaneni et al. 2011, 
        # % doi: 10.1016/j.jcp.2011.03.045, Section 6).

        pow = 4
        nv = X.shape[1]
        niter = np.ones(nv, dtype=int)
        tolg = 1e-3
        if dX is None:
            _, _, length = self.geomProp(X)
            dX = length / X.shape[0]
            toly = 1e-5 * dX
        else:
            normDx = np.sqrt(dX[:X.shape[0] // 2] ** 2 + dX[X.shape[0] // 2:] ** 2)
            toly = 1e-3 * np.min(normDx)

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
                while np.linalg.norm(gn) > np.linalg.norm(g):
                    dtau = dtau * beta
                    xn = x - g[:X.shape[0] // 2] * dtau
                    yn = y - g[X.shape[0] // 2:] * dtau
                    gn = self.computeProjectedGradEnergy(xn, yn, pow)
                dtauOld = dtau * 1 / beta
                # print(toly)
                if np.linalg.norm(gn) < max(max(toly / dtau), tolg * np.linalg.norm(g0)):
                    break
                x = xn
                y = yn
                g = gn
                niter[k] += 1
            X[:, [k]] = np.vstack((xn, yn))

        return X

    def computeProjectedGradEnergy(self, x, y, pow):
        """Compute the projected gradient of the energy of the surface."""
        # % g = computeProjectedGradEnergy(o,x,y) computes the projected gradient of
        # % the energy of the surface. We use this in reparamEnergyMin(o,X). For the
        # % formulation see (Veerapaneni et al. 2011 doi: 10.1016/j.jcp.2011.03.045,
        # % Section 6)

        N = len(x)
        modes = np.concatenate((np.arange(0, N // 2), np.arange(-N // 2, 0)))[:,None]
        # % get tangent vector at each point (tang_x;tang_y) 
        _, tang, _ = self.diffProp(np.concatenate((x, y)).reshape(-1,1))
        # % get x and y components of normal vector at each point
        nx = tang[N:]
        ny = -tang[:N]

        # % Compute gradE
        # % first, get Fourier coefficients
        zX = x + 1j * y
        zXh = np.fft.fft(zX, axis=0) / N
        # % second, compute zX with a_k = k^pow
        zX = np.fft.ifft(N * zXh * np.abs(modes) ** pow, axis=0)
        # % Compute Energy
        gradE = np.vstack((np.real(zX), np.imag(zX))) #[gradE_x;gradE_y]
        
        # % A dyadic product property (a (ban) a)b = a(a.b) can be used to avoid the
        # % for loop as follows
        normals = np.vstack((nx, ny))
        # % do the dot product n.gradE
        prod = normals * gradE
        dotProd = prod[:N] + prod[N:]
        # % do (I-(n ban n))gradE = gradE - n(n.gradE) for each point

        g = gradE - normals * np.vstack((dotProd, dotProd))
        
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
            f = isa * np.fft.ifft(IK * np.fft.fft(f, axis=0), axis=0)
            
        return np.real(f)
