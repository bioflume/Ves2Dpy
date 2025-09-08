import torch
torch.set_default_dtype(torch.float32)
# from scipy.interpolate import interp1d
from curve_batch_compile import Curve
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
        self.IK = f.modes(self.N, self.nv, X.device)

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
        N, nv = self.N, self.nv
        # Ben = torch.zeros((2 * self.N, 2 * self.N, self.nv))
        # Ten = torch.zeros((2 * self.N, self.N, self.nv))
        # Div = torch.zeros((self.N, 2 * self.N, self.nv))

        f = fft1(self.N)
        D1 = f.fourierDiff(self.N)
        D1 = D1.to(self.X.device)

        # for k in range(self.nv):
        #     # compute single arclength derivative matrix
            
        #     isa = self.isa[:, k]
        #     arcDeriv = isa[:, None] * D1

        #     D4 = (arcDeriv @ arcDeriv)
        #     D4 = D4 @ D4
        #     Ben[:, :, k] = torch.vstack([torch.hstack([torch.real(D4), torch.zeros((self.N, self.N), device=self.X.device)]),
        #                                torch.hstack([torch.zeros((self.N, self.N), device=self.X.device), torch.real(D4)])])
            
        #     Ten[:, :, k] = torch.vstack([torch.matmul(torch.real(arcDeriv), torch.diag(self.xt[:self.N, k])),
        #                                torch.matmul(torch.real(arcDeriv), torch.diag(self.xt[self.N:, k]))])
            
        #     Div[:, :, k] = torch.hstack([torch.matmul(torch.diag(self.xt[:self.N, k]), torch.real(arcDeriv)),
        #                               torch.matmul(torch.diag(self.xt[self.N:, k]), torch.real(arcDeriv))])
            

        device = self.X.device

        isa_ = self.isa  # shape: (N, nv)
        arcDeriv_ = isa_.unsqueeze(1) * D1.unsqueeze(-1) # shape: (N, N, nv)
        arcDeriv_ = arcDeriv_.permute(2, 0, 1)   # shape: (nv, N, N)
        arcDeriv_real = torch.real(arcDeriv_)  # shape: (nv, N, N)

        # Compute D4 = (arcDeriv @ arcDeriv)^2
        D2 = arcDeriv_ @ arcDeriv_  # shape: (nv, N, N)
        D4_real = torch.real(D2 @ D2)  # still (nv, N, N)

        # Create Ben (2N x 2N x nv)
        zero_NN = torch.zeros((nv, N, N), device=device)
        top = torch.cat([D4_real, zero_NN], dim=2)  # (nv, N, 2N)
        bot = torch.cat([zero_NN, D4_real], dim=2)  # (nv, N, 2N)
        Ben_ = torch.cat([top, bot], dim=1).permute(1, 2, 0)  # (2N, 2N, nv)

        # Create Ten (2N x N x nv)
        xt_top = self.xt[:N, :]  # (N, nv)
        xt_bot = self.xt[N:, :]  # (N, nv)

        Ten_top = torch.matmul(arcDeriv_real, torch.diag_embed(xt_top.T))  # (nv, N, N)
        Ten_bot = torch.matmul(arcDeriv_real, torch.diag_embed(xt_bot.T))  # (nv, N, N)
        Ten_ = torch.cat([Ten_top, Ten_bot], dim=1).permute(1, 2, 0)  # (2N, N, nv)

        # Create Div (N x 2N x nv)
        Div_left = torch.matmul(torch.diag_embed(xt_top.T), arcDeriv_real)  # (nv, N, N)
        Div_right = torch.matmul(torch.diag_embed(xt_bot.T), arcDeriv_real)  # (nv, N, N)
        Div_ = torch.cat([Div_left, Div_right], dim=2).permute(1, 2, 0)  # (N, 2N, nv)

        # print("computeDerivs checking...")
        # if not torch.allclose(Ben, Ben_, rtol=5e-5):
        #     print(torch.max((Ben - Ben_)/Ben))
        #     print(torch.min((Ben - Ben_)/Ben))
        #     raise "Ben mismatch!"
             
        # if not torch.allclose(Ten, Ten_, rtol=4e-5):
        #     raise "Ten mismatch!"
              
        # if not torch.allclose(Div, Div_, rtol=4e-5):
        #     raise "Div mismatch!"
    
        # # Assume self.isa is of shape (N, nv), D1 is (N, N)
        # isa_expanded = self.isa.unsqueeze(1)  # Shape: (N, 1, nv)
        # arcDeriv = isa_expanded * D1.unsqueeze(-1)  # Shape: (N, N, nv)

        # # Compute D4 efficiently (avoiding explicit loops)
        # D4 = arcDeriv @ arcDeriv  # Shape: (N, N, nv)
        # D4 = D4 @ D4  # Shape: (N, N, nv)

        # # Construct Ben (batching operations)
        # zero_block = torch.zeros((self.N, self.N, self.nv), device=self.X.device)
        # Ben_ = torch.cat([
        #     torch.cat([D4.real, zero_block], dim=1),  
        #     torch.cat([zero_block, D4.real], dim=1)
        # ], dim=0)  # Shape: (2N, 2N, nv)

        # # Construct Ten
        # arcDeriv_real = arcDeriv.real  # Shape: (N, N, nv)
        # xt_top = self.xt[:self.N, :].unsqueeze(1)  # Shape: (N, 1, nv)
        # xt_bottom = self.xt[self.N:, :].unsqueeze(1)  # Shape: (N, 1, nv)

        # Ten_ = torch.cat([
        #     torch.matmul(arcDeriv_real, xt_top.diagonal(dim1=0, dim2=1)),  
        #     torch.matmul(arcDeriv_real, xt_bottom.diagonal(dim1=0, dim2=1))
        # ], dim=0)  # Shape: (2N, N, nv)

        # # Construct Div
        # Div_ = torch.cat([
        #     torch.matmul(xt_top.diagonal(dim1=0, dim2=1), arcDeriv_real),  
        #     torch.matmul(xt_bottom.diagonal(dim1=0, dim2=1), arcDeriv_real)
        # ], dim=1)  # Shape: (N, 2N, nv)

        
        # print(torch.allclose(Ben, Ben_))
            
        Ben = torch.real(Ben_)
        Ten = torch.real(Ten_)
        Div = torch.real(Div_)
        # Imaginary part should be 0 since we are preforming a real operation

        return Ben, Ten, Div

    
    def repulsionForce(self, X, W, eta):
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
        rep_force = torch.zeros((2 * N, nv), dtype=torch.float32)

        # Pairwise differences for x and y across all points and vesicles
        dist_x = X[:N, :].unsqueeze(1).unsqueeze(3) - X[:N, :].unsqueeze(0).unsqueeze(2)  # Shape: (N, N, nv, nv)
        dist_y = X[N:, :].unsqueeze(1).unsqueeze(3) - X[N:, :].unsqueeze(0).unsqueeze(2)  # Shape: (N, N, nv, nv)

        # Compute the pairwise distances
        dist = torch.sqrt(dist_x**2 + dist_y**2 + 1e-10)

        # Mask out self-interactions
        mask = torch.eye(nv, dtype=torch.bool, device=dist.device).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, nv, nv)
        dist.masked_fill_(mask, float('inf'))  # Set self-distances to infinity

        # Compute the maximum number of layers (L) for each distance
        # eta = 0.5 * VesicleLength/Number of Points
        # eta = 1 / N
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