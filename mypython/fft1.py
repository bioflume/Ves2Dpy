import numpy as np
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
        A = np.zeros((len(y), N))
        for j in range(N):
            g = np.zeros(N)
            g[j] = 1
            fhat = np.fft.fftshift(np.fft.fft(g) / N)
            for k in range(N):
                A[:, j] = A[:, j] + fhat[k] * np.exp(1j * (-N / 2 + k) * y)
        A = np.real(A)
        print(A)
        # interpolate
        fo = A@f
        return fo, A

    def sinterpS(self, N, y):
        # A = sinterpS(N,y) constructs the interpolation matrix A that maps
        # a function defined periodically at N equispaced points to the
        # function value at the points y
        # The points y are assumed to be in the 0-2*pi range.

        A = np.zeros((len(y), N))
        modes = np.concatenate((np.arange(0, N / 2), [0], np.arange(-N / 2 + 1, 0)))
        f = np.zeros(N)

        for j in range(N):
            f[j] = 1
            fhat = np.fft.fft(f) / N
            fhat = np.tile(fhat, (len(y), 1))
            A[:, j] = A[:, j] + np.sum(fhat * np.exp(1j * np.outer(y, modes)), axis=1)
            f[j] = 0
        A = np.real(A)
        return A

    def test(self):
        print('testing differentiation test:')
        # Differentiation of a hat function
        N = 24
        h = 2 * np.pi / N
        x = h * np.arange(1, N + 1)
        v1 = np.maximum(0, 1 - np.abs(x - np.pi) / 2)
        w = self.diffFT(v1)
        plt.subplot(3, 2, 1)
        plt.plot(x, v1, '.-', markersize=13)
        plt.axis([0, 2 * np.pi, -0.5, 1.5])
        plt.grid(True)
        plt.title('function')
        plt.subplot(3, 2, 2)
        plt.plot(x, w, '.-', markersize=13)
        plt.axis([0, 2 * np.pi, -1, 1])
        plt.grid(True)
        plt.title('spectral derivative')

        # Differentiation of exp(sin(x))
        v2 = np.exp(np.sin(x))
        vprime = np.cos(x) * v2
        w = self.diffFT(v2)
        error = np.linalg.norm(w - vprime, np.inf)
        plt.subplot(3, 2, 3)
        plt.plot(x, v2, '.-', markersize=13)
        plt.axis([0, 2 * np.pi, 0, 3])
        plt.grid(True)
        plt.subplot(3, 2, 4)
        plt.plot(x, w, '.-', markersize=13)
        plt.axis([0, 2 * np.pi, -2, 2])
        plt.grid(True)
        plt.text(2.2, 1.4, f'max error = {error}')

        print('-----------------------')
        print('exp(sin(x)) derivative:')
        print('-----------------------')
        for N in [8, 16, 32, 64, 128]:
            h = 2 * np.pi / N
            x = h * np.arange(1, N + 1)
            v = np.exp(np.sin(x))
            vprime = np.cos(x) * v
            w = self.diffFT(v)
            error = np.linalg.norm(w - vprime, np.inf)
            print(f'  {N} points: |e|_inf = {error}')

        # check interpolation
        y = np.random.rand(20) * 2 * np.pi
        v2y = np.exp(np.sin(y))
        fy = self.arb_interp(v, y)

        print('--------------------------------------------------------')
        print('cos(x)sin(x)+sin(x)cos(10x)+sin(20x)cos(13x) first derivative:')
        print('--------------------------------------------------------')
        for J in range(4, 11):
            N = 2 ** J
            h = 2 * np.pi / N
            x = h * np.arange(1, N + 1)
            v = np.cos(x) * np.sin(x) + np.sin(x) * np.cos(10 * x) + np.sin(20 * x) * np.cos(13 * x)
            vprime = np.cos(2 * x) - 9 / 2 * np.cos(9 * x) + 11 / 2 * np.cos(11 * x) + 7 / 2 * np.cos(7 * x) + 33 / 2 * np.cos(33 * x)
            w = self.diffFT(v)
            error = np.linalg.norm(w - vprime, np.inf)
            print(f'  {N} points: |e|_inf = {error}')

    
    # @staticmethod
    # def D1(N):
    #     # Deriv = D1(N) constructs a N by N fourier differentiation matrix
    #     FF, FFI = fft1.fourierInt(N)
    #     Deriv = np.dot(FFI, np.dot(np.diag(1j * np.concatenate(([0], np.arange(-N / 2 + 1, N / 2)))), FF))
    #     Deriv = np.real(Deriv)
    #     return Deriv

    @staticmethod
    def diffFT(f, IK):
        """
        Computes the first derivative of an array of periodic functions f using Fourier transform.
        
        Parameters:
            f (ndarray): Array of periodic functions. Each column represents a function.
            IK (ndarray): Index of the Fourier modes.
        
        Returns:
            ndarray: First derivative of the input functions.
        """
        # N = f.shape[0]
        # IK = np.concatenate((np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0))) * 1j
        df = np.real(np.fft.ifft(IK * np.fft.fft(f, axis=0), axis=0))
        return df


    @staticmethod
    def modes(N, nv):
        # IK = modes(N) builds the order of the Fourier modes required for using
        # fft and ifft to do spectral differentiation

        IK = 1j * np.concatenate((np.arange(0, N / 2), [0], np.arange(-N / 2 + 1, 0)))
        IK = IK[:,None]
        if nv == 2:
            IK = np.column_stack((IK, IK))
        elif nv == 3:
            IK = np.column_stack((IK, IK, IK))
        elif nv > 3:
            IK = np.tile(IK, (1, nv))
        return IK

    @staticmethod
    def fourierDiff(N):
        """
        Creates the Fourier differentiation matrix.
        """
        D1 = fft1.fourierInt(N)[0]
        modes = np.arange(-N//2, N//2)
        D1 = np.conj(D1.T) @ np.diag(1j * N * modes) @ D1

        return D1

    @staticmethod
    def fourierRandP(N, Nup):
        # [R,P] = fourierRandP(N,Nup) computes the Fourier restriction and
        # prolongation operators so that functions can be interpolated from N
        # points to Nup points (prolongation) and from Nup points to N points
        # (restriction)

        R = np.zeros((N, Nup))
        P = np.zeros((Nup, N))

        FF1, FFI1 = fft1.fourierInt(N)
        FF2, FFI2 = fft1.fourierInt(Nup)

        R = np.dot(FFI1, np.hstack((np.zeros((N, (Nup - N) // 2)), np.eye(N), np.zeros((N, (Nup - N) // 2))))) \
            .dot(FF2)
        R = np.real(R)
        P = R.T * Nup / N
        return R, P

    @staticmethod
    def fourierInt(N):
        # [FF,FFI] = fourierInt(N) returns a matrix that takes in the point
        # values of a function in [0,2*pi) and returns the Fourier coefficients
        # (FF) and a matrix that takes in the Fourier coefficients and returns
        # the function values (FFI)

        theta = np.arange(N).reshape(-1, 1) * 2 * np.pi / N
        modes = np.concatenate(([-N / 2], np.arange(-N / 2 + 1, N / 2)))
        FF = np.exp(-1j * np.outer(modes, theta)) / N

        if True:  # nargout > 1
            FFI = np.exp(1j * np.outer(theta, modes))
        else:
            FFI = None
        return FF, FFI
