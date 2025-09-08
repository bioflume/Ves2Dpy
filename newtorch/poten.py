import torch
import numpy as np
from fft1 import fft1
from capsules import capsules
from curve_batch_compile import Curve
from filter import interpft_vec
import torch
import torch.fft
import math
torch.set_default_dtype(torch.float32)
from biem_support import  exactStokesSL_, exactStokesSL_onlyself, exactStokesSL_onlyself_old
torch.set_default_device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Poten:
    def __init__(self, N):
        """
        Constructor for the Poten class.
        
        Parameters:
        N : int
            Number of points per curve.
        """
        self.N = N
        self.Nup = N * math.ceil(math.sqrt(N))

        # Compute singular quadrature weights and rotation matrices
        self.qw, self.qp, self.Rbac, self.Rfor = self.singQuadStokesSLmatrix(self.Nup)

        # Compute quadrature size
        self.Nquad = self.qw.numel()    

        # Expand qw to match the upsampled grid
        self.qw = self.qw.unsqueeze(1).repeat(1, self.Nup)  # Equivalent to MATLAB: o.qw(:, ones(o.Nup,1));

        self.interpMat = self.lagrange_interp()
        # Compute restriction and prolongation matrices
        self.Restrict_LP, self.Prolong_LP = fft1.fourierRandP(self.N, self.Nup)


    def stokesSLmatrix(self, vesicle):
        # Creating vesicleUp (similar to MATLAB's capsules_py function)
        # print("start of Galpert")
        vesicleUp = capsules(interpft_vec(vesicle.X, self.Nup), None, None, 1, 1)

        # Extract Jacobian
        sa = vesicleUp.sa[None, :].expand(vesicleUp.N, -1, -1) # (vesicleUp.N, vesicleUp.N, vesicle.nv)
        sa = sa.float()

        x, y = vesicleUp.X[:self.Nup, :], vesicleUp.X[self.Nup:, :]

        # Compute target points
        xtar = x[None, :] #.repeat(self.Nquad, 1, 1) # (self.Nquad, self.Nup, vesicle.nv)
        ytar = y[None, :] #.repeat(self.Nquad, 1, 1)

        # Compute source points
        xsou = x[None, :] #.repeat(vesicleUp.N, 1, 1) # (self.Nup, self.Nup, vesicle.nv)
        ysou = y[None, :] #.repeat(vesicleUp.N, 1, 1)

        # Apply circular shift using self.Rfor
        xsou = xsou.reshape(-1, vesicle.nv)[self.Rfor % self.Nup].reshape(self.Nup, self.Nup, vesicle.nv)
        diffx = xtar.permute(2,0,1) - torch.matmul(self.qp[None, :], xsou.permute(2, 0, 1))
        del xsou
        torch.cuda.empty_cache()

        ysou = ysou.reshape(-1, vesicle.nv)[self.Rfor % self.Nup].reshape(self.Nup, self.Nup, vesicle.nv)
        diffy = ytar.permute(2,0,1) - torch.matmul(self.qp[None, :], ysou.permute(2, 0, 1))
        del ysou
        torch.cuda.empty_cache()

        # print(diffx.shape)
        rho2 = 1./(diffx ** 2 + diffy ** 2)

        # print(f" max element in rho2 in Galpert :{torch.max(rho2)}")
        # Compute log-part contribution
        logpart = 0.5 * torch.matmul(self.qp.T[None, :], (self.qw.unsqueeze(0) * torch.log(rho2)))

        if torch.any(torch.isnan(logpart)) or torch.any(torch.isinf(logpart)):
            raise ValueError("NaN or Inf in the logpart")
        

        # Compute G-matrix components
        G11 = (logpart + torch.matmul(self.qp.T[None, :], self.qw[None, :] * diffx ** 2 * rho2)).permute(0, 2, 1).reshape(vesicle.nv, -1)[:, self.Rbac]
        G11 = G11.permute(0, 2, 1) * sa.permute(2,0,1)
        G22 = (logpart + torch.matmul(self.qp.T[None, :], self.qw[None, :] * diffy ** 2 * rho2)).permute(0, 2, 1).reshape(vesicle.nv, -1)[:, self.Rbac]
        G22 = G22.permute(0, 2, 1) * sa.permute(2,0,1)
        G12 = (torch.matmul(self.qp.T[None, :], self.qw[None, :] * diffy * diffx * rho2)).permute(0, 2, 1).reshape(vesicle.nv, -1)[:, self.Rbac]
        G12 = G12.permute(0, 2, 1) * sa.permute(2,0,1)
        # G12 = (torch.matmul(self.qp.T, self.qw * diffx * diffy * rho2))[self.Rbac].T * sa

        # Initializing G tensor
        G = torch.zeros(vesicle.nv, 2 * vesicle.N, 2 * vesicle.N)
        # Populate G tensor
        G[:, :self.N, :self.N] = torch.matmul(self.Restrict_LP[None, :], torch.matmul(G11, self.Prolong_LP[None, :]))
        G[:, :self.N, self.N:] = torch.matmul(self.Restrict_LP[None, :], torch.matmul(G12, self.Prolong_LP[None, :]))
        G[:, self.N:, :self.N] = G[:, :self.N, self.N:]
        G[:, self.N:, self.N:] = torch.matmul(self.Restrict_LP[None, :], torch.matmul(G22, self.Prolong_LP[None, :]))

        # print("end of Galpert")
        return G.permute(1,2,0)


    def stokesDLmatrix(self, vesicle):
        """
        Computes the Stokes double-layer potential matrix (D).
        Parameters:
            o: An object containing interpolation and quadrature operators.
            vesicle: An object representing the vesicle structure.
        Returns:
            D: A (2N, 2N, nv) tensor representing the double-layer potential.
        """
        # Upsample vesicle positions
        Xup = interpft_vec(vesicle.X, self.Nup)
        
        # Create an upsampled vesicle object
        vesicleUp = capsules(Xup, None, None, 1, torch.ones(vesicle.nv, device=Xup.device))

        # Initialize D matrix
        D = torch.zeros((2 * vesicle.N, 2 * vesicle.N, vesicle.nv), device=Xup.device)

        # Mask for non-trivial viscosity contrast
        valid_idx = vesicle.viscCont != 1
        if torch.max(valid_idx):
            const_coeff = -(1 - vesicle.viscCont[valid_idx]).unsqueeze(0).unsqueeze(0)

            # Extract locations and tangent vectors
            xx, yy = Xup[:self.Nup, valid_idx], Xup[self.Nup:, valid_idx]
            tx, ty = vesicleUp.xt[:self.Nup, valid_idx], vesicleUp.xt[self.Nup:, valid_idx]
            sa = vesicleUp.sa[:, valid_idx]
            cur = vesicleUp.cur[:, valid_idx]

            # Create target and source point grids
            xtar, ytar = xx.unsqueeze(0), yy.unsqueeze(0)
            xsou, ysou = xx.unsqueeze(1), yy.unsqueeze(1)
            txsou, tysou = tx.unsqueeze(0), ty.unsqueeze(0)

            ids = torch.arange(self.Nup, device=Xup.device)
            # Compute differences
            diffx, diffy = xtar - xsou, ytar - ysou
            rho4 = (diffx**2 + diffy**2).pow(-2)
            rho4[ids, ids] = 0.  # Set diagonal terms to 0

            # Compute kernel
            kernel = (diffx * tysou - diffy * txsou) * rho4 * sa.unsqueeze(0)
            kernel = const_coeff * kernel

            # Compute blocks of D matrix
            D11 = kernel * diffx**2
            D12 = kernel * diffx * diffy
            D22 = kernel * diffy**2

            # Set diagonal limiting terms
            factor = 0.5 * const_coeff.squeeze(0) * cur * sa
            # diag_terms = 0.5 * const_coeff * cur.unsqueeze(-1) * sa * torch.stack([txsou**2, txsou * tysou, tysou**2], dim=0)
            # D11.diagonal(dim1=-2, dim2=-1).copy_(diag_terms[0])
            # D12.diagonal(dim1=-2, dim2=-1).copy_(diag_terms[1])
            # D22.diagonal(dim1=-2, dim2=-1).copy_(diag_terms[2])
            D11[ids, ids] = factor * txsou.squeeze(0)**2
            D12[ids, ids] = factor * (txsou * tysou).squeeze(0)
            D22[ids, ids] = factor * tysou.squeeze(0)**2

            # Apply restriction and prolongation operators
            D11 = self.Restrict_LP @ D11.permute(2,0,1) @ self.Prolong_LP # after this, batch dim is 0-th dim
            D12 = self.Restrict_LP @ D12.permute(2,0,1) @ self.Prolong_LP
            D22 = self.Restrict_LP @ D22.permute(2,0,1) @ self.Prolong_LP

            # Assemble full D matrix
            D_full = torch.cat([
                torch.cat([D11, D12], dim=-1),
                torch.cat([D12, D22], dim=-1)
            ], dim=-2)

            # Scale with arc-length spacing and divide by pi
            D[:, :, valid_idx] = (1 / torch.pi) * D_full.permute(1,2,0) * (2 * torch.pi / vesicleUp.N)

        return D

    def StokesN0Matrix(self, vesicle):
        """
        Generates the integral operator with kernel normal(x) ⊗ normal(y),
        which removes the rank-one deficiency of the double-layer potential.
        This operator is needed for solid walls.

        Parameters:
        vesicle - Object containing vesicle properties including positions,
                  normal vectors, arclength elements, and count.

        Returns:
        N0 - Tensor representing the integral operator matrix.
        """

        # Compute normal vectors
        normal = torch.cat((vesicle.xt[vesicle.N:2 * vesicle.N, 0],
                            -vesicle.xt[:vesicle.N, 0]), dim=0).to(vesicle.X.device)

        # Expand normal vectors along the second axis
        normal = normal[:, None].expand(-1, 2*vesicle.N)

        # Arclength elements
        # sa = vesicle.sa[:, 0].repeat(2)  # Duplicate first column for both components
        # sa = sa[:, None].repeat(1, 2 * vesicle.N)  # Expand along second axis
        sa = torch.cat((vesicle.sa[:, 0], vesicle.sa[:, 0]), dim=0).to(vesicle.X.device)
        sa = sa[:, None].expand(-1, 2 *vesicle.N)

        # Compute the N0 matrix using element-wise operations
        N0 = torch.zeros((2*vesicle.N, 2*vesicle.N, vesicle.nv), device=vesicle.X.device)
        N0[:, :, 0] = normal * normal.T * sa.T * (2 * torch.pi / vesicle.N)

        # # If multiple vesicles are present, expand dimensions
        # N0 = N0.unsqueeze(-1).repeat(1, 1, vesicle.nv)

        return N0

    def exactStokesSLdiag(self, vesicle, G, f):
        """
        Computes the diagonal term of the single-layer potential due to `f` around `vesicle`.
        The source and target points are the same. This uses Alpert's quadrature formula.

        Returns:
        SLP - Tensor of shape (2*N, nv), representing the computed single-layer potential.
        """
        # Compute SLP using batch matrix-vector multiplication
        SLP = torch.einsum('ijk,jk->ik', G, f)  # Equivalent to multiplying G(:,:,k) by f(:,k) for each k

        return SLP
    

    def exactStokesDLdiag(self, vesicle, D, f):
        """
        Computes the diagonal term of the double-layer potential due to `f` around all vesicles.
        The source and target points are the same. This uses the trapezoid rule with the 
        curvature at the diagonal to guarantee spectral accuracy.

        Returns:
        DLP - Tensor of shape (2*N, nv), representing the computed double-layer potential.
        """
        # Compute DLP using batch matrix-vector multiplication
        DLP = torch.einsum('ijk,jk->ik', D, f)  # Equivalent to multiplying D(:,:,k) by f(:,k) for each k

        return DLP


    def exactStokesN0diag(self, vesicle, N0, f):
        """
        Computes the diagonal term of the modification of the double-layer potential
        due to `f` around the outermost vesicle. The source and target points are the same.
        This uses the trapezoid rule.

        """
        if N0 is None:  # Check if N0 is empty
            N = f.shape[0] // 2

            # Compute force components
            fx, fy = f[:N, 0], f[N:, 0]
            fx *= vesicle.sa[:, 0]
            fy *= vesicle.sa[:, 0]

            # Compute tangent vectors
            tx, ty = vesicle.xt[:N, 0], vesicle.xt[N:, 0]

            # Compute constant term
            const = torch.sum(ty * fx - tx * fy) * (2 * torch.pi / N)

            # Compute N0 as a function of tangents
            N0 = const * torch.cat((ty, -tx))

        else:
            N0 = torch.matmul(N0[:, :, 0], f[:, 0])  # Compute modified potential

        return N0
    
    # % END OF ROUTINES THAT EVALUATE LAYER-POTENTIALS
    # % WHEN SOURCES == TARGETS

    # % START OF ROUTINES THAT EVALUATE LAYER-POTENTIALS
    # % WHEN SOURCES ~= TARGETS.  CAN COMPUTE LAYER POTENTIAL ON EACH
    # % VESICLE DUE TO ALL OTHER VESICLES (ex. stokesSLP) AND CAN
    # % COMPUTE LAYER POTENTIAL DUE TO VESICLES INDEXED IN K1 AT 
    # % TARGET POINTS Xtar

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
        
        
        Ntar = Xtar.shape[0] // 2
        ncol = Xtar.shape[1]
        stokesSLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float32, device=vesicle.X.device)
        

        den = f * torch.tile(vesicle.sa, (2, 1)) * 2 * torch.pi / vesicle.N

        xsou = vesicle.X[:vesicle.N, K1].flatten()
        ysou = vesicle.X[vesicle.N:, K1].flatten()
        xsou = torch.tile(xsou, (Ntar, 1)).T    # (N*(nv-1), Ntar)
        ysou = torch.tile(ysou, (Ntar, 1)).T

        denx = den[:vesicle.N, K1].flatten()
        deny = den[vesicle.N:, K1].flatten()
        denx = torch.tile(denx, (Ntar, 1)).T    # (N*(nv-1), Ntar)
        deny = torch.tile(deny, (Ntar, 1)).T

        for k in range(ncol):  # Loop over columns of target points
            if ncol != 1:
                raise "ncol != 1"
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


        return stokesSLPtar / (4 * torch.pi)


    def exactLaplaceDL(self, vesicle, f, Xtar, K1):
        """
        Computes the double-layer Laplace potential due to `f` around all vesicles except itself.
        Can also evaluate the potential at target points `Xtar` using vesicles indexed by `K1`.

        """

        # Normal vectors
        nx = vesicle.xt[vesicle.N:2 * vesicle.N, :]
        ny = -vesicle.xt[:vesicle.N, :]

        # Target dimensions
        N = vesicle.N
        Ntar = Xtar.shape[0] // 2
        ncol = Xtar.shape[1]

        # Initialize output
        laplaceDLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float32)

        # Multiply by arclength term
        den = f * torch.cat((vesicle.sa, vesicle.sa), dim=0) * (2 * torch.pi / vesicle.N)

        # Source coordinates
        xsou, ysou = vesicle.X[:N, K1], vesicle.X[N:, K1]
        xsou = xsou.T.flatten().expand(Ntar, -1)
        ysou = ysou.T.flatten().expand(Ntar, -1)

        # Density function
        denx, deny = den[:N, K1], den[N:, K1]
        denx = denx.T.flatten().expand(Ntar, -1)
        deny = deny.T.flatten().expand(Ntar, -1)

        # Normal vectors
        nxK1, nyK1 = nx[:, K1], ny[:, K1]
        nxK1 = nxK1.T.flatten().expand(Ntar, -1)
        nyK1 = nyK1.T.flatten().expand(Ntar, -1)

        # Loop over target point columns
        for k2 in range(ncol):
            xtar, ytar = Xtar[:Ntar, k2], Xtar[Ntar:, k2]
            xtar = xtar[:, None].expand(-1, vesicle.N * len(K1))
            ytar = ytar[:, None].expand(-1, vesicle.N * len(K1))

            # Compute differences and squared distance
            diffx = xsou - xtar
            diffy = ysou - ytar
            dis2 = diffx**2 + diffy**2

            # Compute coefficients
            coeff = (diffx * nxK1 + diffy * nyK1) / dis2

            # Compute the potential
            laplaceDLPtar[:Ntar, k2] = torch.sum(coeff * denx, dim=1)
            laplaceDLPtar[Ntar:, k2] = torch.sum(coeff * deny, dim=1)

        # Multiply by coefficient in front of the double-layer potential
        laplaceDLPtar /= (2 * torch.pi)

        return laplaceDLPtar

    def exactStokesDL(self, vesicle, f, Xtar=None, K1=None):
        """
        Computes the double-layer potential due to `f` around all vesicles except itself.
        Can also evaluate the potential at target points `Xtar` using vesicles indexed by `K1`.

        """
        # Compute normal vectors
        normal = torch.cat((vesicle.xt[vesicle.N:2 * vesicle.N, :], 
                            -vesicle.xt[:vesicle.N, :]), dim=0)
        
        Ntar = Xtar.shape[0] // 2
        ncol = Xtar.shape[1]
        stokesDLPtar = torch.zeros((2 * Ntar, ncol), dtype=torch.float32, device=vesicle.X.device)

        # Compute density function with jacobian term and viscosity contrast scaling
        # 可能需要改成对角矩阵
        den = (f * torch.cat((vesicle.sa, vesicle.sa), dim=0) * (2 * torch.pi / vesicle.N)) @ torch.eye(vesicle.nv) * (1 - vesicle.viscCont)

        oc = Curve()

        # Source coordinates
        xsou, ysou = oc.getXY(vesicle.X[:, K1])
        xsou = xsou.T.flatten().expand(Ntar, -1)
        ysou = ysou.T.flatten().expand(Ntar, -1)

        # Density function components
        denx, deny = oc.getXY(den[:, K1])
        denx = denx.T.flatten().expand(Ntar, -1)
        deny = deny.T.flatten().expand(Ntar, -1)

        # Normal vectors
        normalx, normaly = oc.getXY(normal[:, K1])
        normalx = normalx.T.flatten().expand(Ntar, -1)
        normaly = normaly.T.flatten().expand(Ntar, -1)

        # Loop over target columns
        for k in range(ncol):
            xtar, ytar = Xtar[:Ntar, k], Xtar[Ntar:, k]
            xtar = xtar[:, None].expand(-1, vesicle.N * len(K1))
            ytar = ytar[:, None].expand(-1, vesicle.N * len(K1))

            # Compute differences and squared distances
            diffx = xtar - xsou
            diffy = ytar - ysou
            dis2 = diffx**2 + diffy**2

            # Compute rdotnTIMESrdotf term
            rdotnTIMESrdotf = ((diffx * normalx + diffy * normaly) / dis2**2) * (diffx * denx + diffy * deny)

            # Compute the potential
            stokesDLPtar[:Ntar, k] += torch.sum(rdotnTIMESrdotf * diffx, dim=1)
            stokesDLPtar[Ntar:, k] += torch.sum(rdotnTIMESrdotf * diffy, dim=1)

        # Apply scaling factor
        stokesDLPtar /= torch.pi

        return stokesDLPtar

    # % END OF ROUTINES THAT EVALUATE LAYER-POTENTIALS
    # % WHEN SOURCES ~= TARGETS

    def filter_to_be_implemented(self, tensor):
        """
        a filter to be implemented, should be equivalent to MATLAB's filter
        """
        return tensor

    def nearSingInt_hh(self, vesicleSou, f, selfMat, NearV2V, kernelDirect, vesicleTar, tEqualS):
        """
        Computes a layer potential due to `f` at all points in `vesicleTar.X`.

        Parameters:
        vesicleSou - Source vesicles object
        f - Density function
        selfMat - Function computing self-interactions
        NearStruct - Structure with near-zone data
        kernelDirect - Function for direct kernel evaluations
        vesicleTar - Target vesicles object
        tEqualS - Boolean, true if sources == targets
        o - Object containing interpolation and upsampling parameters

        Returns:
        LP - Computed layer potential
        """

        # If only a single vesicle exists, return zeros
        if tEqualS and vesicleSou.X.shape[1] == 1:
            return torch.zeros_like(vesicleSou.X)

        device = f.device
        # Extract data from NearStruct
        # dist, zone, nearest, icp, argnear = NearStruct.dist, NearStruct.zone, NearStruct.nearest, NearStruct.icp, NearStruct.argnear

        Xsou, Nsou, nvSou = vesicleSou.X, vesicleSou.X.shape[0] // 2, vesicleSou.X.shape[1]
        Xtar, Ntar, nvTar = vesicleTar.X, vesicleTar.X.shape[0] // 2, vesicleTar.X.shape[1]
        h = vesicleSou.length / Nsou  # Arc length

        # Upsample sources
        Nup = self.Nup
        Xup = interpft_vec(Xsou, Nup)  # Upsample source points
        fup = interpft_vec(f, Nup)  # Upsample density function

        # Compute self-interaction
        vself = selfMat(f)

        # Upsampled vesicle object
        vesicleUp = capsules(Xup, None, None, vesicleSou.kappa, vesicleSou.viscCont)

        interpOrder = self.interpMat.shape[0]
        p = (interpOrder + 1) // 2

        if nvSou >= 1:
            # Compute far-field ignoring self-interactions
            # idx = torch.arange(nvSou)
            # farField = torch.stack([kernelDirect(vesicleUp, fup, Xtar[:, k], idx[idx != k]) for k in range(nvSou)], dim=1)
            # farField = kernelDirect(vesicleUp.X, vesicleUp.sa, fup, Xtar, NearV2V[-1][0], NearV2V[-1][1], NearV2V[-1][2])
            farField = kernelDirect(vesicleUp.X, vesicleUp.sa, fup, Xtar, NearV2V[-1])
        # elif not tEqualS:
        #     farField = kernelDirect(vesicleUp.X, vesicleUp.sa, fup, Xtar, torch.arange(nvSou))
        else:
            farField = torch.zeros((2 * Ntar, nvTar), dtype=torch.float32, device=device)

        # Initialize nearField
        nearField = torch.zeros((2 * Ntar, nvTar), dtype=torch.float32, device=device)

        beta = 1.1  # Buffer factor for interpolation points

        # Vectorized computation for near-field interactions
        for k1 in range(nvSou):
            if tEqualS:
                K = [k for k in range(nvTar) if k != k1]
                # K = torch.arange(nvTar, device=device)
                # K = K[K != k1]
            else:
                K = range(nvTar)
                

            for k2 in K: # in nvTar
                # J = torch.where(zone[k1][:, k2] == 1)[0]  # set of points on vesicle k2 close to vesicle k1
                id1, id2 = NearV2V[3]
                J = id1[(Ntar * k2 <= id1) & (id1 < Ntar * (k2+1)) & (id2 == k1)] % Ntar
                if len(J) == 0:
                    continue  # Skip if no near-zone points

                # indcp = icp[k1][J, k2] # closest point on vesicle k1 to each point on vesicle k2 that is close to vesicle k1
                dist_closest, idx_closest = NearV2V[0][k1, k2, J], NearV2V[1][k1, k2, J]
                # index of points to the left and right of the closest point
                pn = ((idx_closest[:, None] - p + 1 + torch.arange(interpOrder)) % Nsou).long() 

                # # vel = torch.zeros((2 * len(J), nvTar, nvSou), dtype=torch.float32, device=device)
                vel = torch.zeros((2 * Ntar, nvTar, nvSou), dtype=torch.float32, device=device)
                v = self.filter_to_be_implemented(self.interpMat @ vself[pn, k1].T)
                vel[J, k2, k1]  = v[-1, :]
                v = self.filter_to_be_implemented(self.interpMat @ vself[pn + Nsou, k1].T)
                vel[J + Ntar, k2, k1]  = v[-1, :]


                # Compute Lagrange interpolation points
                nx = (Xtar[J, k2] - Xsou[idx_closest, k1]) / dist_closest
                # ny = (Xtar[J + Ntar, k2] - nearest[k1][J + Ntar, k2]) / dist[k1][J + Ntar, k2]
                ny = (Xtar[J + Ntar, k2] - Xsou[idx_closest + Nsou, k1]) / dist_closest

                # XLag_x = nearest[k1][J, k2].unsqueeze(1) + beta * h * nx.unsqueeze(1) * torch.arange(1, interpOrder)
                # XLag_y = nearest[k1][J + Ntar, k2].unsqueeze(1) + beta * h * ny.unsqueeze(1) * torch.arange(1, interpOrder)
                XLag_x = Xsou[idx_closest, k1].unsqueeze(1) + beta * h * nx.unsqueeze(1) * torch.arange(1, interpOrder)
                XLag_y = Xsou[idx_closest + Nsou, k1].unsqueeze(1) + beta * h * ny.unsqueeze(1) * torch.arange(1, interpOrder)

                XLag = torch.cat((XLag_x, XLag_y), dim=0)

                # lagrangePts = kernelDirect(vesicleUp, fup, XLag, k1)
                # lagrangePts, _ = allExactStokesSLTarget_compare1(vesicleUp.X[:, k1], vesicleUp.sa[:, k1], fup, XLag)
                # print(f"Xlag shape {XLag.shape}")
                lagrangePts = exactStokesSL_(vesicleUp, fup, XLag, [k1]) # may need optimization

                num_J = len(J)

                # Get Px and Py all at once
                Px = torch.bmm(
                    self.interpMat.unsqueeze(0).expand(num_J, -1, -1),  # [num_J, 7, 7]
                    torch.concat([
                        vel[J, k2, k1].unsqueeze(-1),                    # [num_J, 1]
                        lagrangePts[:num_J, :]            # [num_J, interpOrder-1]
                    ], dim=1).unsqueeze(-1)                              # [num_J, interpOrder]
                ).squeeze()  # [num_J, 7, 1] → [num_J, 7]

                Py = torch.bmm(
                    self.interpMat.unsqueeze(0).expand(num_J, -1, -1),
                    torch.concat([
                        vel[J + Ntar, k2, k1].unsqueeze(-1), 
                        lagrangePts[num_J:, :]
                    ], dim=1).unsqueeze(-1)
                ).squeeze()

                # Get dscaled values
                # dscaled = (dist_closest / (beta * h * (interpOrder - 1))).unsqueeze(1)
                Px = self.filter_to_be_implemented(Px)
                nearField[J, k2] += Px[:, -1]
                Py = self.filter_to_be_implemented(Py)
                nearField[J + Ntar, k2] += Py[:, -1]
        
        # print(nearField)
        return farField + nearField, nearField



    # def hh()
    #     for k1 in range(nvSou):
    #         # if tEqualS:
    #         K = [k for k in range(nvTar) if k != k1]
    #         # else:
    #         #     K = range(nvTar)
                

    #         for k2 in K: # in nvTar
    #             # set of points on vesicle k2 close to vesicle k1
    #             id1, id2 = NearV2V[3]
    #             J = id1[(Ntar * k2 <= id1) & (id1 < Ntar * (k2+1)) & (id2 == k1)] % Ntar
    #             if len(J) == 0:
    #                 continue  # Skip if no near-zone points

    #             # indcp = icp[k1][J, k2] # closest point on vesicle k1 to each point on vesicle k2 that is close to vesicle k1
    #             dist_closest, idx_closest = NearV2V[0][k1, k2, J], NearV2V[1][k1, k2, J]
    #             # index of points to the left and right of the closest point
    #             pn = ((idx_closest[:, None] - p + torch.arange(interpOrder)) % Nsou).long() 

    #             vel = torch.zeros((2 * Ntar, nvTar, nvSou), dtype=torch.float32, device=device)
    #             v = self.interpMat @ vself[pn, k1].T
    #             vel[J, k2, k1]  = v[-1, :]
    #             v = self.interpMat @ vself[pn + Nsou, k1].T
    #             vel[J + Ntar, k2, k1]  = v[-1, :]

                
    #             # Compute Lagrange interpolation points
    #             nx = (Xtar[J, k2] - Xsou[idx_closest, k1]) / dist_closest
    #             ny = (Xtar[J + Ntar, k2] - Xsou[idx_closest + Nsou, k1]) / dist_closest

    #              XLag_x = Xsou[idx_closest, k1].unsqueeze(1) + beta * h * nx.unsqueeze(1) * torch.arange(1, interpOrder)
    #             XLag_y = Xsou[idx_closest + Nsou, k1].unsqueeze(1) + beta * h * ny.unsqueeze(1) * torch.arange(1, interpOrder)

    #             XLag = torch.cat((XLag_x, XLag_y), dim=0)

    #             lagrangePts = exactStokesSL(vesicleUp, fup, XLag, [k1]) # may need optimization


    def nearSingInt_rbf(self, vesicleSou, f, selfMat, NearV2V, kernelDirect, vesicleTar, tEqualS):
        """
        Computes a layer potential due to `f` at all points in `vesicleTar.X`.

        Parameters:
        vesicleSou - Source vesicles object
        f - Density function
        selfMat - Function computing self-interactions
        NearStruct - Structure with near-zone data
        kernelDirect - Function for direct kernel evaluations
        vesicleTar - Target vesicles object
        tEqualS - Boolean, true if sources == targets
        o - Object containing interpolation and upsampling parameters

        Returns:
        LP - Computed layer potential
        """

        # If only a single vesicle exists, return zeros
        if tEqualS and vesicleSou.X.shape[1] == 1:
            return torch.zeros_like(vesicleSou.X)

        device = f.device
        # Extract data from NearStruct
        # dist, zone, nearest, icp, argnear = NearStruct.dist, NearStruct.zone, NearStruct.nearest, NearStruct.icp, NearStruct.argnear

        Xsou, Nsou, nvSou = vesicleSou.X, vesicleSou.X.shape[0] // 2, vesicleSou.X.shape[1]
        Xtar, Ntar, nvTar = vesicleTar.X, vesicleTar.X.shape[0] // 2, vesicleTar.X.shape[1]
        # h = vesicleSou.length / Nsou  # Arc length

        # Upsample sources
        Nup = self.Nup 
        Xup = interpft_vec(Xsou, Nup)  # Upsample source points
        fup = interpft_vec(f, Nup)  # Upsample density function

        # Compute self-interaction
        vself = selfMat(f)

        # Upsampled vesicle object
        vesicleUp = capsules(Xup, None, None, vesicleSou.kappa, vesicleSou.viscCont)

        # interpOrder = self.interpMat.shape[0]
        # p = (interpOrder + 1) // 2

        if tEqualS and nvSou > 1:
            # Compute far-field ignoring self-interactions
            # idx = torch.arange(nvSou)
            # farField = torch.stack([kernelDirect(vesicleUp, fup, Xtar[:, k], idx[idx != k]) for k in range(nvSou)], dim=1)
            farField = kernelDirect(vesicleUp.X, vesicleUp.sa, fup, Xtar, NearV2V[-1])
            # farField = kernelDirect(vesicleUp.X, vesicleUp.sa, fup, Xtar, (torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))) 
        elif not tEqualS:
            farField = kernelDirect(vesicleUp, fup, Xtar, torch.arange(nvSou))
        else:
            farField = torch.zeros((2 * Ntar, nvTar), dtype=torch.float32, device=device)

        # if len(NearV2V[2][0]) == 0 and len(NearV2V[2][1]) == 0:
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        upsample = -1
        if Nsou == 32:
            if upsample <=1 :
                const = 0.672 #* self.len[0].item()
            elif upsample == 2:
                const = 0.566
                vself = interpft_vec(vself, upsample * Nsou)
            elif upsample == 4:
                # const = 0.495
                const = 0.305
                vself = interpft_vec(vself, upsample * Nsou)
        else:
            upsample = -1
            const = 0.0132
        
        nlayers=3
        Nup_for_layers = Nsou * (nlayers-1) * math.ceil(math.sqrt((nlayers-1)*Nsou))
        # Nup_for_layers = Nsou * 2 * math.ceil(math.sqrt(2*Nsou))
        Xup = interpft_vec(Xsou, Nup_for_layers)  # Upsample source points
        fup = interpft_vec(f, Nup_for_layers)  # Upsample density function
        vesicleUp = capsules(Xup, None, None, vesicleSou.kappa, vesicleSou.viscCont)

        # print(f"using nlayers {nlayers} and upsample {upsample}")
        N = Xsou.shape[0]//2
        if upsample > 0:
            Nup = N * upsample # is different from vesicleUp.N !!
            Xup = interpft_vec(Xsou, Nup)  # Upsample source points
        else:
            Nup = N
            Xup = Xsou

        # vesicleUp = capsules(Xup, None, None, vesicle.kappa, vesicle.viscCont)
        oc = Curve()
        dlayer = torch.linspace(0, 1/N, nlayers, dtype=torch.float32, device=Xsou.device)
        _, tang = oc.diffProp_jac_tan(Xup)
        rep_nx = tang[Nup:, :, None].expand(-1,-1,nlayers-1)
        rep_ny = -tang[:Nup, :, None].expand(-1,-1,nlayers-1)
        dx =  rep_nx * dlayer[[1,2,3]] if nlayers == 4 else rep_nx * dlayer[[1,2]] # (N, nv, nlayers-1)
        dy =  rep_ny * dlayer[[1,2,3]] if nlayers == 4 else rep_ny * dlayer[[1,2]]
        tracers = torch.permute(
            torch.vstack([torch.repeat_interleave(Xup[:Nup, :, None], nlayers-1, dim=-1) + dx,
                        torch.repeat_interleave(Xup[Nup:, :, None], nlayers-1, dim=-1) + dy]), (0,2,1)) # (2*N, nlayers-1, nv)

        velx, vely, xlayers, ylayers = exactStokesSL_onlyself(vesicleUp.X, vesicleUp.sa, fup, Nup, Xup, vself, tracers)
        # velx_, vely_, xlayers_, ylayers_ = exactStokesSL_onlyself_old(vesicleUp, fup, Xsou, vself, nlayers, upsample)
        # np.save("vel_layers.npy", {"velx": velx.cpu().numpy(), "vely": vely.cpu().numpy(), "xlayers": xlayers.cpu().numpy(), "ylayers": ylayers.cpu().numpy()})


        # if not torch.allclose(velx, velx_):
        #     print("velx not equal")
        #     print(torch.max(torch.abs(velx - velx_)))

        # if not torch.allclose(vely, vely_):
        #     print("vely not equal")
        #     print(torch.max(torch.abs(vely - vely_)))
        
        # if not torch.allclose(xlayers, xlayers_):
        #     print("xlayers not equal")
        #     print(torch.max(torch.abs(xlayers - xlayers_)))
        
        # if not torch.allclose(ylayers, ylayers_):
        #     print("ylayers not equal")
        #     print(torch.max(torch.abs(ylayers - ylayers_)))


        all_X = torch.concat((xlayers.reshape(-1,1,nvSou), ylayers.reshape(-1,1,nvSou)), dim=1) # (nlayers * N, 2, nv), 2 for x and y
        all_X = all_X /const
        matrices = torch.exp(- torch.sum((all_X[:, None] - all_X[None, ...])**2, dim=-2)) 
        matrices += (torch.eye(all_X.shape[0]).unsqueeze(-1) * 1e-6).expand(-1,-1,nvSou) # (nlayers*N, nlayers*N, nv)
        L = torch.linalg.cholesky(matrices.permute(2, 0, 1))
        
        # old_farField = farField.clone()
        self.nearFieldCorrectionUP_SOLVE(vesicleTar, upsample, NearV2V[2], L, farField, velx, vely, xlayers, ylayers, nlayers=nlayers)

        # end.record()
        # torch.cuda.synchronize()
        # print(f'near correction takes {start.elapsed_time(end)/1000} sec.')


        # print(old_farField - farField)
        # return farField, farField - old_farField
        return farField


    def nearFieldCorrectionUP_SOLVE(self, vesicle, upsample, info, L, far_field, velx, vely, xlayers, ylayers, nlayers):
        if  len(info[0])==0 or len(info[1])==0:
            return
        
        N = vesicle.N
        nv = vesicle.nv

        all_points = torch.concat((vesicle.X[:N, :].T.reshape(-1,1), vesicle.X[N:, :].T.reshape(-1,1)), dim=1)
        # correction = torch.zeros((N*nv, 2), dtype=torch.float32, device=trac_jump.device)
        
        if N == 32:
            if upsample <=0 :
                const = 0.672  #* self.len0[0].item()
            elif upsample == 2:
                const = 0.566
            elif upsample == 4:
                # const = 0.495 
                const = 0.305 
        else:
            const = 0.0132


        all_X = torch.concat((xlayers.reshape(-1,1,nv), ylayers.reshape(-1,1,nv)), dim=1) # (3 * N, 2, nv), 2 for x and y
        all_X = all_X /const   

        rhs = torch.concat((velx.reshape(-1,1,nv), vely.reshape(-1,1,nv)), dim=1) # (3 * N), 2, nv), 2 for x and y
        
        y = torch.linalg.solve_triangular(L, rhs.permute(2, 0, 1), upper=False)
        coeffs = torch.linalg.solve_triangular(L.permute(0, 2, 1), y, upper=True)
            

        # tStart = time.time()
        # for k in range(nv):
        #     rows_with_true = torch.any(nbrs_mask[:, k*Nup : (k+1)*Nup], dim=1)
        #     if not torch.any(rows_with_true):
        #         continue
        #     ids_in = torch.arange(N*nv)[rows_with_true] # which ves points are in k's near zone
            
        #     points_query = all_points[ids_in] # and their coords
        #     ves_id = torch.IntTensor([k])

        #     r2 = torch.sum((points_query[:, None]/const * N - all_X[None, ..., ves_id].squeeze(-1))**2, dim=-1)
        #     matrix = torch.exp(- 1 * r2) 
        
        #     rbf_vel = matrix @ coeffs[k]

        #     correction[ids_in, 0] += rbf_vel[:, 0]
        #     correction[ids_in, 1] += rbf_vel[:, 1]

        # tEnd = time.time()
        # print(f'x1 nearFieldCorrection for loop {tEnd - tStart} sec.')


        # nbrs_mask_reshaped = nbrs_mask.reshape(nv*N, nv, Nup)
        # rows_with_true = torch.any(nbrs_mask_reshaped, dim=-1).unsqueeze(1) # (N*nv, 1, nv)
 
        # r = torch.norm(all_points.unsqueeze(1).unsqueeze(-1)/const * N - all_X[None, ...], dim=-2) # (N*nv, N*nlayers, nv)
        # r.masked_fill_(~rows_with_true, torch.inf)
        # matrix = torch.exp(- 1 * r**2) 

        # rr = torch.full((N*nv, N*5, nv), float('inf'))
        # id1, id2, id3 = torch.where(rows_with_true.expand(-1, N*5, -1))
        # rr[id1, id2, id3] = torch.norm(all_points[id1]/const*N - all_X[id2, :, id3], dim=-1)
        # matrix = torch.exp(- 1 * rr**2) 

        # matrix = torch.zeros((N*nv, N*5, nv), dtype=torch.float32, device=far_field.device)
        # id1, id2, id3 = torch.where(rows_with_true.expand(-1, N*5, -1))
        # matrix[id1, id2, id3] = torch.exp(- torch.norm(all_points[id1]/const*N - all_X[id2, :, id3], dim=-1)**2)
        
        # if not torch.allclose(r, rr):
        #     raise "r and rr are not the same"

        # sp_matrix = matrix.reshape(N*nv, N*5 * nv).to_sparse()
        # c = torch.sparse.mm(sp_matrix, coeffs.permute(1,0,2).reshape(nv * N*5, 2))
        # correction = torch.einsum("Nnv, vnc -> Nc", matrix, coeffs)  #matrix @ coeffs
        # correction_ = matrix.reshape(N*nv, N*5 * nv) @ coeffs.permute(1,0,2).reshape(nv * N*5, 2)

        # id1, id2 = torch.where(rows_with_true.expand(-1, N*5, -1).reshape(N*nv, -1))
        # sp_matrix = torch.sparse_coo_tensor(torch.vstack((id1, id2)), 
        #                 torch.exp(-torch.norm(all_points[id1]/const*N - all_X.permute(0,2,1).reshape(-1, 2)[id2, :], dim=-1)**2),
        #                 size=(N*nv, N*5 * nv))  
        # correction = torch.sparse.mm(sp_matrix, coeffs.permute(1,0,2).reshape(nv * N*5, 2))


        # id1_, id2_ = torch.where(rows_with_true.squeeze())
        id1_, id2_ = info
        if upsample <= 1:
            id2_ = id2_[:, None] + torch.arange(0, N*nlayers*nv, nv).to(id2_.device)
            id2_ = id2_.reshape(-1)
            id1_ = id1_[:, None].expand(-1, N*nlayers).reshape(-1)
            sp_matrix_ = torch.sparse_coo_tensor(torch.vstack((id1_, id2_)), 
                            torch.exp(-torch.norm(all_points[id1_]/const - all_X.permute(0,2,1).reshape(-1, 2)[id2_, :], dim=-1)**2),
                            size=(N*nv, N * nlayers * nv))
            correction = torch.sparse.mm(sp_matrix_, coeffs.permute(1,0,2).reshape(nv * N* nlayers, 2))
        else:
            id2_ = id2_[:, None] + torch.arange(0, upsample * N*nlayers*nv, nv).to(id2_.device)
            id2_ = id2_.reshape(-1)
            id1_ = id1_[:, None].expand(-1, upsample * N*nlayers).reshape(-1)
            sp_matrix_ = torch.sparse_coo_tensor(torch.vstack((id1_, id2_)), 
                            torch.exp(-torch.norm(all_points[id1_]/const - all_X.permute(0,2,1).reshape(-1, 2)[id2_, :], dim=-1)**2),
                            size=(N*nv, upsample  * N * nlayers * nv))
            correction = torch.sparse.mm(sp_matrix_, coeffs.permute(1,0,2).reshape(nv * upsample  * N* nlayers, 2))
        

        correction = correction.view(nv, N, 2).permute(2, 1, 0).reshape(2 * N, nv)
        far_field += correction
        return 




    def singQuadStokesSLmatrix(self, N):
        # Define the weights
        v = torch.tensor([
            6.531815708567918e-3,
            9.086744584657729e-2,
            3.967966533375878e-1,
            1.027856640525646e+0,
            1.945288592909266e+0,
            2.980147933889640e+0,
            3.998861349951123e+0
        ], dtype=torch.float32)
        
        u = torch.tensor([
            2.462194198995203e-2,
            1.701315866854178e-1,
            4.609256358650077e-1,
            7.947291148621895e-1,
            1.008710414337933e+0,
            1.036093649726216e+0,
            1.004787656533285e+0
        ], dtype=torch.float32)
        
        a = 5
        h = 2 * np.pi / N
        n = N - 2 * a + 1
        
        # Generate quadrature weights and points
        yt = h * torch.arange(a, n + a, dtype=torch.float32)
        wt = torch.cat([h * u, h * torch.ones(len(yt)), h * torch.flip(u, [0])]) / (4 * np.pi)
        
        # Construct matrix B
        B = torch.zeros(len(yt), N)
        pos = torch.arange(a, n + a, dtype=torch.long)
        B[torch.arange(len(yt)), pos] = 1
        
        # Compute interpolation matrices
        # Placeholder for sinc interpolation function (replace with actual implementation)
        of = fft1(N)
        
        A1 = of.sinterpS_(N, v*h)
        A2 = of.sinterpS_(N, 2*torch.pi-torch.flip(v*h, [0]))
            
        A = torch.cat([A1, B, A2])
        
        qw = torch.cat([wt.unsqueeze(1), A], dim=1).float()
        qp = qw[:, 1:]
        qw = qw[:, 0]
        
        # Compute index shifts for circulant matrix operations
        ind = torch.arange(N)
        Rfor = torch.zeros((N, N), dtype=torch.long)
        Rbac = torch.zeros((N, N), dtype=torch.long)
        
        Rfor[:, 0] = ind
        Rbac[:, 0] = ind
        
        # for k in range(1, N):
        #     Rfor[:, k] = (k) * N + torch.cat([ind[k:], ind[:k]])
        #     Rbac[:, k] = (k) * N + torch.cat([ind[-k:], ind[:-k]])
        # Vectorized computation of Rfor and Rbac

        indices = torch.arange(N)
        shift = torch.arange(N).unsqueeze(0)  # Shape (1, N)

        # Generate forward indices
        Rfor = (indices.unsqueeze(1) + shift) % N  # Circular shift forward
        # Rfor = Rfor * N + indices.unsqueeze(0)
        Rfor += shift * N

        # Generate backward indices
        Rbac = (indices.unsqueeze(1) - shift) % N  # Circular shift backward
        # Rbac = Rbac * N + indices.unsqueeze(0)
        Rbac += shift * N

        
        return qw, qp, Rbac, Rfor


    @staticmethod
    def lagrange_interp():
        # Define the interpolation matrix LP directly
        LP = torch.tensor([
            [  64.80,  -388.80,   972.00, -1296.00,   972.00,  -388.80,   64.80 ],
            [-226.80,  1296.00, -3078.00,  3888.00, -2754.00,  1036.80, -162.00 ],
            [ 315.00, -1674.00,  3699.00, -4356.00,  2889.00, -1026.00,  153.00 ],
            [-220.50,  1044.00, -2074.50,  2232.00, -1381.50,   468.00,  -67.50 ],
            [  81.20,  -313.20,   526.50,  -508.00,   297.00,   -97.20,   13.70 ],
            [ -14.70,    36.00,   -45.00,    40.00,   -22.50,     7.20,   -1.00 ],
            [   1.00,     0.00,     0.00,     0.00,     0.00,     0.00,    0.00 ]
        ], dtype=torch.float32)

        return LP

# Example usage
# N = 16  # Set an appropriate N value
# qw, qp, Rbac, Rfor = singQuadStokesSLmatrix(N)
# # print(qw)
# # print(qp)
# print(Rbac)
# print(Rfor)

# N = 16
# X = torch.tensor(
#     [[0.8599, 0.0863, 0.4132],
#         [0.3454, 0.4931, 0.9693],
#         [0.8920, 0.1510, 0.9593],
#         [0.8617, 0.0423, 0.8422],
#         [0.5763, 0.4830, 0.0940],
#         [0.1764, 0.8653, 0.7828],
#         [0.8798, 0.8578, 0.0333],
#         [0.4542, 0.5439, 0.1797],
#         [0.5168, 0.3419, 0.0768],
#         [0.0115, 0.6494, 0.1903],
#         [0.0993, 0.7038, 0.8806],
#         [0.2223, 0.0909, 0.2299],
#         [0.8176, 0.8592, 0.5057],
#         [0.6101, 0.3934, 0.7653],
#         [0.8687, 0.5387, 0.0578],
#         [0.2790, 0.6853, 0.8112]]
# )

# X = torch.tensor(
#     [[0.1818, 0.1459, 0.1868, 0.0616],
#         [0.0265, 0.3467, 0.8279, 0.0041],
#         [0.7317, 0.4133, 0.6540, 0.7822],
#         [0.9248, 0.4949, 0.9423, 0.3238],
#         [0.8524, 0.9184, 0.5882, 0.7911],
#         [0.1531, 0.1649, 0.0686, 0.0024],
#         [0.8971, 0.7035, 0.2109, 0.0119],
#         [0.7806, 0.2802, 0.8062, 0.8937],
#         [0.4480, 0.4619, 0.1555, 0.6016],
#         [0.4711, 0.6571, 0.3333, 0.1418],
#         [0.2738, 0.2125, 0.3279, 0.3178],
#         [0.5723, 0.1634, 0.6184, 0.7612],
#         [0.9286, 0.9554, 0.6151, 0.0986],
#         [0.9285, 0.0014, 0.3947, 0.3728],
#         [0.6724, 0.1632, 0.1417, 0.1725],
#         [0.3674, 0.8188, 0.0189, 0.0784]]
# )
# f = torch.tensor(
#     [[0.2150, 0.4940, 0.8152, 0.3232],
#         [0.8916, 0.8903, 0.3387, 0.1970],
#         [0.7910, 0.3858, 0.9156, 0.3808],
#         [0.3705, 0.7803, 0.8281, 0.1586],
#         [0.1098, 0.2964, 0.8263, 0.9500],
#         [0.6343, 0.5572, 0.4944, 0.0960],
#         [0.5035, 0.3945, 0.5819, 0.3653],
#         [0.3040, 0.6726, 0.3060, 0.0929],
#         [0.4984, 0.5615, 0.6630, 0.7215],
#         [0.1431, 0.1372, 0.7046, 0.5475],
#         [0.9208, 0.7720, 0.8337, 0.2339],
#         [0.4214, 0.5023, 0.4246, 0.4220],
#         [0.5682, 0.5950, 0.4241, 0.6812],
#         [0.9907, 0.5678, 0.4912, 0.5115],
#         [0.4176, 0.5109, 0.2031, 0.4541],
#         [0.0429, 0.1609, 0.2672, 0.1612]]
# )
# ves = capsules(X, None, None, torch.tensor([1.]), torch.tensor([1., 0.5, 0.3, 0.2]))
# op = Poten(N//2)
# D = op.stokesDLmatrix(ves)
# print(D.shape)