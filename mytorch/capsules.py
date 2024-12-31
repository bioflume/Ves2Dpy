import torch
from scipy.interpolate import interp1d
from curve_batch import Curve
from fft1 import fft1

class capsules:
    """
    This class implements standard calculations that need to
    be done to a vesicle, solid wall, or a collection of arbitrary
    target points (such as tracers or pressure/stress targets)
    % Given a vesicle, the main tasks that can be performed are
    % computing the required derivatives (bending, tension, surface
    % divergence), the traction jump, the pressure and stress, 
    % and constructing structures required for near-singluar
    % integration
    """

    def __init__(self, X, sigma, u, kappa, viscCont):
        """
        Initialize the capsules class with parameters.
        % capsules(X,sigma,u,kappa,viscCont) sets parameters and options for
        % the class; no computation takes place here.  
        %
        % sigma and u are not needed and typically unknown, so just set them to
        % an empty array.

        """
        self.N = X.shape[0] // 2  # points per vesicle
        self.nv = X.shape[1]  # number of vesicles
        self.X = X  # position of vesicle
        oc = Curve()
        # Jacobian, tangent, and curvature
        self.sa, self.xt, self.cur = oc.diffProp(self.X)
        self.isa = 1. / self.sa
        self.sig = sigma  # Tension of vesicle
        self.u = u  # Velocity of vesicle
        self.kappa = kappa  # Bending modulus
        self.viscCont = viscCont  # Viscosity contrast

        # center of vesicle.  Required for center of rotlets and
        # stokeslets in confined flows
        # self.center = torch.concat((torch.mean(X[:self.N, :], dim=0), torch.mean(X[self.N:, :], dim=0)))

        # minimum arclength needed for near-singular integration
        _, _, length = oc.geomProp(X)
        self.length = torch.min(length)

        # ordering of the fourier modes.  It is faster to compute once here and
        # pass it around to the fft differentitation routine
        f = fft1(self.N)
        self.IK = f.modes(self.N, self.nv).to(self.X.device)

    def tracJump(self, f, sigma):
        """
        % tracJump(f,sigma) computes the traction jump where the derivatives
        % are taken with respect to a linear combiation of previous time steps
        % which is stored in object o Xm is 2*N x nv and sigma is N x nv
        """
        return self.bendingTerm(f) + self.tensionTerm(sigma)

    def bendingTerm(self, f):
        """
        Compute the term due to bending.
        """
        c = Curve()
        return torch.vstack([-self.kappa * c.arcDeriv(f[:self.N, :], 4, self.isa, self.IK),
                          -self.kappa * c.arcDeriv(f[self.N:, :], 4, self.isa, self.IK)])

    def tensionTerm(self, sig):
        """
        % ten = tensionTerm(o,sig) computes the term due to tension (\sigma *
        % x_{s})_{s}
        """
        c = Curve()
        # print(sig.device)
        # print(self.IK.device)
        # print(self.xt.device)
        return torch.vstack([c.arcDeriv(sig * self.xt[:self.N, :], 1, self.isa, self.IK),
                          c.arcDeriv(sig * self.xt[self.N:, :], 1, self.isa, self.IK)])

    def surfaceDiv(self, f):
        """
        Compute the surface divergence of f with respect to the vesicle.
        f has size N x nv
        """
        oc = Curve()
        fx, fy = oc.getXY(f)
        tangx, tangy = oc.getXY(self.xt)
        return oc.arcDeriv(fx, 1, self.isa, self.IK) * tangx + oc.arcDeriv(fy, 1, self.isa, self.IK) * tangy

    def computeDerivs(self):
        """
        % [Ben,Ten,Div] = computeDerivs computes the matricies that takes a
        % periodic function and maps it to the fourth derivative, tension, and
        % surface divergence all with respect to arclength.  Everything in this
        % routine is matrix free at the expense of having repmat calls
        """
        Ben = torch.zeros((2 * self.N, 2 * self.N, self.nv))
        Ten = torch.zeros((2 * self.N, self.N, self.nv))
        Div = torch.zeros((self.N, 2 * self.N, self.nv))

        f = fft1(self.N)
        D1 = f.fourierDiff(self.N)

        for k in range(self.nv):
            # compute single arclength derivative matrix
            
            isa = self.isa[:, k]
            arcDeriv = isa[:, torch.newdim] * D1
            # This line is equivalent to repmat(o.isa(:,k),1,o.N).*D1 but much
            # faster.

            D4 = torch.dot(arcDeriv, arcDeriv)
            D4 = torch.dot(D4, D4)
            Ben[:, :, k] = torch.vstack([torch.hstack([torch.real(D4), torch.zeros((self.N, self.N))]),
                                       torch.hstack([torch.zeros((self.N, self.N)), torch.real(D4)])])
            
            Ten[:, :, k] = torch.vstack([torch.dot(torch.real(arcDeriv), torch.diag(self.xt[:self.N, k])),
                                       torch.dot(torch.real(arcDeriv), torch.diag(self.xt[self.N:, k]))])
            
            Div[:, :, k] = torch.hstack([torch.dot(torch.diag(self.xt[:self.N, k]), torch.real(arcDeriv)),
                                      torch.dot(torch.diag(self.xt[self.N:, k]), torch.real(arcDeriv))])
            
        Ben = torch.real(Ben)
        Ten = torch.real(Ten)
        Div = torch.real(Div)
        # Imaginary part should be 0 since we are preforming a real operation

        return Ben, Ten, Div

    
    def repulsionForce(self, X, W):
        """
        repulsion_force(o, X, W) computes the artificial repulsion between vesicles.
        W is the repulsion strength -- depends on the length and velocity scale
        of the flow.

        Repulsion is computed using the discrete penalty layers given in Grinspun
        et al. (2009), Asynchronous Contact Mechanics.
        """
        # Number of vesicles and points
        nv = X.shape[1]
        N = X.shape[0] // 2

        # Initialize net repulsive force on each point of each vesicle
        rep_force = torch.zeros((2 * N, nv), dtype=torch.float64)

        # Pairwise differences for x and y across all points and vesicles
        dist_x = X[:N, :].unsqueeze(1).unsqueeze(3) - X[:N, :].unsqueeze(0).unsqueeze(2)  # Shape: (N, N, nv, nv)
        dist_y = X[N:, :].unsqueeze(1).unsqueeze(3) - X[N:, :].unsqueeze(0).unsqueeze(2)  # Shape: (N, N, nv, nv)

        # Compute the pairwise distances
        dist = torch.sqrt(dist_x**2 + dist_y**2 + 1e-12)

        # Mask out self-interactions
        mask = torch.eye(nv, dtype=torch.bool, device=dist.device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, nv, nv)
        dist.masked_fill_(mask, float('inf'))  # Set self-distances to infinity

        # Compute the maximum number of layers (L) for each distance
        # eta = 0.5 * VesicleLength/Number of Points
        eta = 0.5 / N
        L = torch.floor(eta / dist)

        # Compute stiffness values (dF)
        dF = -L * (2 * L + 1) * (L + 1) / 3 + L * (L + 1) * eta / dist  # Shape: (N, N, nv, nv)

        # Compute repulsion forces for all points in parallel
        repx = torch.sum(dF * dist_x, dim=(1, 3)) # Shape: (N, nv)
        repy = torch.sum(dF * dist_y, dim=(1, 3))  

        # Concatenate x and y components and multiply by strength W
        rep_force[:N, :] = W * repx
        rep_force[N:, :] = W * repy

        return rep_force