import torch
torch.set_default_dtype(torch.float64)
import matplotlib.pyplot as plt

class fft1:
    # class implements fft transformations. This includes computing
    # the fourier differentiation matrix, doing interpolation required
    # by Alpert's quadrature rules, and defining the Fourier frequencies

    def __init__(self, N):
        self.N = N  # Number of points in the incoming periodic functions

    def arbInterp(self, f, y):
        # fo = arbInterp(f) interpolates the function f given at
        # regular points, at arbitrary points y. The points y are assumed
        # to be in the 0-2*pi range. Matrix is built. This routine is
        # only for testing and is never called in the vesicle code

        N = f.shape[0]

        # build interpolation matrix
        A = torch.zeros((len(y), N), dtype=torch.float64)
        for j in range(N):
            g = torch.zeros(N)
            g[j] = 1
            fhat = torch.fft.fftshift(torch.fft.fft(g) / N)
            for k in range(N):
                A[:, j] = A[:, j] + fhat[k] * torch.exp(1j * (-N / 2 + k) * y)
        A = torch.real(A)
        print(A)
        # interpolate
        fo = A@f
        return fo, A

    def sinterpS(self, N, y):
        # A = sinterpS(N,y) constructs the interpolation matrix A that maps
        # a function defined periodically at N equispaced points to the
        # function value at the points y
        # The points y are assumed to be in the 0-2*pi range.

        A = torch.zeros((len(y), N), dtype=torch.float64)
        modes = torch.concatenate((torch.arange(0, N / 2), [0], torch.arange(-N / 2 + 1, 0)))
        f = torch.zeros(N, dtype=torch.float64)

        for j in range(N):
            f[j] = 1
            fhat = torch.fft.fft(f) / N
            fhat = torch.tile(fhat, (len(y), 1))
            A[:, j] = A[:, j] + torch.sum(fhat * torch.exp(1j * torch.outer(y, modes)), dim=1)
            f[j] = 0
        A = torch.real(A)
        return A

    def test(self):
        print('testing differentiation test:')
        # Differentiation of a hat function
        N = 24
        h = 2 * torch.pi / N
        x = h * torch.arange(1, N + 1)
        v1 = torch.maximum(0, 1 - torch.abs(x - torch.pi) / 2)
        w = self.diffFT(v1)
        plt.subplot(3, 2, 1)
        plt.plot(x, v1, '.-', markersize=13)
        plt.dim([0, 2 * torch.pi, -0.5, 1.5])
        plt.grid(True)
        plt.title('function')
        plt.subplot(3, 2, 2)
        plt.plot(x, w, '.-', markersize=13)
        plt.dim([0, 2 * torch.pi, -1, 1])
        plt.grid(True)
        plt.title('spectral derivative')

        # Differentiation of exp(sin(x))
        v2 = torch.exp(torch.sin(x))
        vprime = torch.cos(x) * v2
        w = self.diffFT(v2)
        error = torch.linalg.norm(w - vprime, torch.inf)
        plt.subplot(3, 2, 3)
        plt.plot(x, v2, '.-', markersize=13)
        plt.dim([0, 2 * torch.pi, 0, 3])
        plt.grid(True)
        plt.subplot(3, 2, 4)
        plt.plot(x, w, '.-', markersize=13)
        plt.dim([0, 2 * torch.pi, -2, 2])
        plt.grid(True)
        plt.text(2.2, 1.4, f'max error = {error}')

        print('-----------------------')
        print('exp(sin(x)) derivative:')
        print('-----------------------')
        for N in [8, 16, 32, 64, 128]:
            h = 2 * torch.pi / N
            x = h * torch.arange(1, N + 1)
            v = torch.exp(torch.sin(x))
            vprime = torch.cos(x) * v
            w = self.diffFT(v)
            error = torch.linalg.norm(w - vprime, torch.inf)
            print(f'  {N} points: |e|_inf = {error}')

        # check interpolation
        y = torch.random.rand(20) * 2 * torch.pi
        v2y = torch.exp(torch.sin(y))
        fy = self.arb_interp(v, y)

        print('--------------------------------------------------------')
        print('cos(x)sin(x)+sin(x)cos(10x)+sin(20x)cos(13x) first derivative:')
        print('--------------------------------------------------------')
        for J in range(4, 11):
            N = 2 ** J
            h = 2 * torch.pi / N
            x = h * torch.arange(1, N + 1)
            v = torch.cos(x) * torch.sin(x) + torch.sin(x) * torch.cos(10 * x) + torch.sin(20 * x) * torch.cos(13 * x)
            vprime = torch.cos(2 * x) - 9 / 2 * torch.cos(9 * x) + 11 / 2 * torch.cos(11 * x) + 7 / 2 * torch.cos(7 * x) + 33 / 2 * torch.cos(33 * x)
            w = self.diffFT(v)
            error = torch.linalg.norm(w - vprime, torch.inf)
            print(f'  {N} points: |e|_inf = {error}')

    
    # @staticmethod
    # def D1(N):
    #     # Deriv = D1(N) constructs a N by N fourier differentiation matrix
    #     FF, FFI = fft1.fourierInt(N)
    #     Deriv = torch.dot(FFI, torch.dot(torch.diag(1j * torch.concatenate(([0], torch.arange(-N / 2 + 1, N / 2)))), FF))
    #     Deriv = torch.real(Deriv)
    #     return Deriv

    @staticmethod
    def diffFT(f, IK):
        """
        Computes the first derivative of an array of periodic functions f using Fourier transform.
        
        Parameters:
            f (ndarray): Array of periodic functions. Each column represents a function.
            IK (ndarray): Index of the Fourier modes.
        
        Returns:
            ndarray: First derivative of the itorchut functions.
        """
        # N = f.shape[0]
        # IK = torch.concatenate((torch.arange(0, N // 2), [0], torch.arange(-N // 2 + 1, 0))) * 1j
        df = torch.real(torch.fft.ifft(IK * torch.fft.fft(f, dim=0), dim=0))
        return df


    @staticmethod
    def modes(N, nv):
        # IK = modes(N) builds the order of the Fourier modes required for using
        # fft and ifft to do spectral differentiation

        IK = 1j * torch.concatenate((torch.arange(0, N / 2), torch.tensor([0]), torch.arange(-N / 2 + 1, 0))).double()
        IK = IK[:,None]
        if nv == 2:
            IK = torch.column_stack((IK, IK))
        elif nv == 3:
            IK = torch.column_stack((IK, IK, IK))
        elif nv > 3:
            IK = torch.tile(IK, (1, nv))
        return IK

    @staticmethod
    def fourierDiff(N):
        """
        Creates the Fourier differentiation matrix.
        """
        D1 = fft1.fourierInt(N)[0]
        modes = torch.arange(-N//2, N//2).double()
        D1 = torch.conj(D1.T) @ torch.diag(1j * N * modes) @ D1

        return D1

    @staticmethod
    def fourierRandP(N, Nup):
        # [R,P] = fourierRandP(N,Nup) computes the Fourier restriction and
        # prolongation operators so that functions can be interpolated from N
        # points to Nup points (prolongation) and from Nup points to N points
        # (restriction)

        R = torch.zeros((N, Nup), dtype=torch.float64)
        P = torch.zeros((Nup, N), dtype=torch.float64)

        FF1, FFI1 = fft1.fourierInt(N)
        FF2, FFI2 = fft1.fourierInt(Nup)

        R = torch.dot(FFI1, torch.hstack((torch.zeros((N, (Nup - N) // 2)), torch.eye(N), torch.zeros((N, (Nup - N) // 2))))) \
            .dot(FF2)
        R = torch.real(R)
        P = R.T * Nup / N
        return R, P

    @staticmethod
    def fourierInt(N):
        # [FF,FFI] = fourierInt(N) returns a matrix that takes in the point
        # values of a function in [0,2*pi) and returns the Fourier coefficients
        # (FF) and a matrix that takes in the Fourier coefficients and returns
        # the function values (FFI)

        theta = torch.arange(N).reshape(-1, 1) * 2 * torch.pi / N
        theta = theta.double()
        modes = torch.concatenate(([-N / 2], torch.arange(-N / 2 + 1, N / 2))).double()
        FF = torch.exp(-1j * torch.outer(modes, theta)) / N

        if True:  # nargout > 1
            FFI = torch.exp(1j * torch.outer(theta, modes))
        else:
            FFI = None
        return FF, FFI
