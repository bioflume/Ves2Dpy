import torch
import torch.fft
torch.set_default_dtype(torch.float32)
# import matplotlib.pyplot as plt

def interpft(x, N_new):
    """
    Interpolates data x using Fourier transform method to a new length N_new.
    Handles both upsampling and downsampling.
    
    Parameters:
        x (Tensor): Input data of shape (N, ...), where N is the length along the interpolation axis.
        N_new (int): The desired new length after interpolation.
    
    Returns:
        Tensor: Interpolated data of length N_new.
    """
    N = x.size(0)
    nv = x.size(1)
    if N == N_new:
        return x
    X = torch.fft.fft(x, dim=0)
    
    # If N_new > N, upsample (add zeros in the middle)
    if N_new > N:
        X_new = torch.zeros((N_new, nv), dtype=X.dtype, device=x.device)
        X_new[:N//2] = X[:N//2]
        X_new[-(N//2):] = X[-(N//2):]
    # If N_new < N, downsample (truncate the Fourier modes)
    else:
        X_new = torch.cat([X[:N_new//2], X[-(N_new//2):]])
    
    # Inverse FFT and scale to maintain the correct amplitude
    return torch.fft.ifft(X_new, dim=0).real * (N_new / N)

# def upsThenFilterShape(X, Nup, modeCut):
#     """
#     Delete high frequencies from the vesicle shape by upsampling and applying a filter.
    
#     Parameters:
#         X (Tensor): Shape of the vesicle, with 2*N rows (x and y components) and nv columns.
#         Nup (int): Number of points to upsample.
#         modeCut (int): Cutoff mode to filter high frequencies.
    
#     Returns:
#         Xfinal (Tensor): The filtered shape.
#     """
#     N = X.size(0) // 2  # Get the number of points (half of the length of X)
#     nv = X.size(1)  # Get the number of columns (number of vesicles)

#     # Frequency modes
#     modes = torch.cat([torch.arange(0, Nup//2, device=X.device), torch.arange(-Nup//2, 0, device=X.device)])

#     # Upsample x and y components
#     xup = torch.stack([interpft(X[:N, k], Nup) for k in range(nv)], dim=1)
#     yup = torch.stack([interpft(X[N:, k], Nup) for k in range(nv)], dim=1)

#     Xfinal = torch.zeros_like(X)  # Initialize the result tensor

#     for k in range(nv):
#         z = xup[:, k] + 1j * yup[:, k]  # Complex form of the shape (z = x + iy)
#         z_fft = torch.fft.fft(z, dim=0)  # FFT of z
#         z_fft[torch.abs(modes) > modeCut] = 0  # Apply frequency cutoff
#         z_ifft = torch.fft.ifft(z_fft, dim=0)  # Inverse FFT

#         # Downsample back to original length and assign to result
#         Xfinal[:N, k] = interpft(z_ifft.real, N)
#         Xfinal[N:, k] = interpft(z_ifft.imag, N)

#     return Xfinal

def filterShape(X, modeCut):
    """
    Delete high frequencies from the vesicle shape by applying a filter.
    
    Parameters:
        X (Tensor): Shape of the vesicle, with 2*N rows (x and y components) and nv columns.
        modeCut (int): Cutoff mode to filter high frequencies.
    
    Returns:
        Xfinal (Tensor): The filtered shape.
    """
    N = X.size(0) // 2  # Get the number of points (half of the length of X)
    # nv = X.size(1)  # Get the number of columns (number of vesicles)

    # Frequency modes
    modes = torch.cat([torch.arange(0, N//2, device=X.device), torch.arange(-N//2, 0, device=X.device)])

    z = X[:N] + 1j * X[N:]  # Complex form of the shape (z = x + iy)
    z_fft = torch.fft.fft(z, dim=0)  # FFT of z
    z_fft[torch.abs(modes) > modeCut] = 0  # Apply frequency cutoff
    z_ifft = torch.fft.ifft(z_fft, dim=0)  # Inverse FFT

    # Xfinal[:N] = z_ifft.real
    # Xfinal[N:] = z_ifft.imag
    return torch.vstack((z_ifft.real, z_ifft.imag))


# def upsThenFilterTension(X, Nup, modeCut):
#     """
#     Delete high frequencies from the vesicle shape by upsampling and applying a filter.
    
#     Parameters:
#         X (Tensor): Shape of the vesicle, with 2*N rows (x and y components) and nv columns.
#         Nup (int): Number of points to upsample.
#         modeCut (int): Cutoff mode to filter high frequencies.
    
#     Returns:
#         Xfinal (Tensor): The filtered shape.
#     """
#     N = X.size(0)  # Get the number of points (half of the length of X)
#     nv = X.size(1)  # Get the number of columns (number of vesicles)

#     # Frequency modes
#     modes = torch.cat([torch.arange(0, Nup//2, device=X.device), torch.arange(-Nup//2, 0, device=X.device)])

#     xup = torch.stack([interpft(X[:, k], Nup) for k in range(nv)], dim=1)    

#     Xfinal = torch.zeros_like(X)  # Initialize the result tensor

#     for k in range(nv):
#         z = xup[:, k] 
#         z_fft = torch.fft.fft(z, dim=0)  # FFT of z
#         z_fft[torch.abs(modes) > modeCut] = 0  # Apply frequency cutoff
#         z_ifft = torch.fft.ifft(z_fft, dim=0)  # Inverse FFT

#         # Downsample back to original length and assign to result
#         Xfinal[:, k] = interpft(z_ifft.real, N)

#     return Xfinal


def filterTension(X, modeCut):
    """
    Delete high frequencies from tension by applying a filter.
    
    Parameters:
        X (Tensor): Shape of the vesicle, with 2*N rows (x and y components) and nv columns.
        modeCut (int): Cutoff mode to filter high frequencies.
    
    Returns:
        Xfinal (Tensor): The filtered shape.
    """
    N = X.size(0)  # Get the number of points (half of the length of X)
    # nv = X.size(1)  # Get the number of columns (number of vesicles)

    # Frequency modes
    modes = torch.cat([torch.arange(0, N//2), torch.arange(-N//2, 0)])  

    z_fft = torch.fft.fft(X, dim=0) 
    z_fft[torch.abs(modes) > modeCut] = 0  # Apply frequency cutoff
    z_ifft = torch.fft.ifft(z_fft, dim=0)  # Inverse FFT

    return z_ifft.real


# device = torch.device("cuda:0")
# for _ in range(10):
#     x = torch.rand(128, 2)
#     x1 = upsThenFilterTension(x, 128*2, 16)
#     x2 = filterTension(x, 16)

#     print(torch.allclose(x1, x2))

# x21 = upsThenFilterTension(x[:8,:], 4*8, 4)
# x22 = upsThenFilterTension(x[8:,:], 4*8, 4)
# print(x1)

# x = torch.sin(torch.arange(32)*0.2) + torch.sin(torch.arange(32)*(-0.4)+5)
# %matplotlib inline

# plt.figure()


# plt.plot(torch.arange(32), upsThenFilterTension(x.unsqueeze(-1), 4*32, 2), color='b')
# plt.plot(torch.arange(32), x, color='r')
# upsThenFilterShape(torch.rand(256, 2), 512, 16)
