import torch
import torch.fft
torch.set_default_dtype(torch.float32)
# import matplotlib.pyplot as plt


@torch.jit.script
def upsample_fft(x, N_new: int):
    """
    Upsamples data x using Fourier transform method to a new length N_new.
    
    Parameters:
        x (Tensor): Input data of shape (2N, nv)
        N_new (int): The desired new length after upsampling.
    
    Returns:
        Tensor: Upsampled data of length N_new.
    """
    N = x.size(0)//2 
    nv = x.size(1)
    # if N == N_new:
    #     return x
    X = torch.fft.fft(x.reshape(2, N, nv), dim=1)
    X_new = torch.zeros((2, N_new, nv), dtype=X.dtype, device=x.device)
    X_new[:, :N//2] = X[:, :N//2]
    X_new[:, -N//2:] = X[:, -N//2:]
    return torch.fft.ifft(X_new, dim=1).reshape(2*N_new, nv).real * (N_new / N)


def gaussian_filter_shape(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Applies a 1D Gaussian filter in the frequency domain.
    
    Args:
        signal (torch.Tensor): Input 1D signal (shape: [N]).
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        torch.Tensor: Filtered signal in the time domain.
    """
    N = signal.shape[0] // 2 # Signal length
    device = signal.device
    
    # Compute frequency components
    freqs = torch.fft.fftfreq(N, device=device)  # Normalized frequency range (-0.5 to 0.5)
    
    # Create Gaussian filter in frequency space
    gaussian_filter = torch.exp(-0.5 * (freqs / sigma) ** 2)
    
    # FFT of the signal
    signal_fft = torch.fft.fft(signal[:N] + 1j * signal[N:], dim=0)
    
    # Apply the Gaussian filter
    filtered_fft = signal_fft * gaussian_filter[:, None]
    
    # Inverse FFT to return to time domain
    filtered_signal = torch.fft.ifft(filtered_fft, dim=0)
    filtered_signal = torch.vstack((filtered_signal.real, filtered_signal.imag))
    
    return filtered_signal



def gaussian_filter_1d(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Applies a 1D Gaussian filter in the frequency domain.
    
    Args:
        signal (torch.Tensor): Input 1D signal (shape: [N]).
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        torch.Tensor: Filtered signal in the time domain.
    """
    N = signal.shape[0] # Signal length
    device = signal.device
    
    # Compute frequency components
    freqs = torch.fft.fftfreq(N, device=device)  # Normalized frequency range (-0.5 to 0.5)
    
    # Create Gaussian filter in frequency space
    gaussian_filter = torch.exp(-0.5 * (freqs / sigma) ** 2)
    
    # FFT of the signal
    signal_fft = torch.fft.fft(signal, dim=0)
    
    # Apply the Gaussian filter
    filtered_fft = signal_fft * gaussian_filter[:, None]
    
    # Inverse FFT to return to time domain
    filtered_signal = torch.fft.ifft(filtered_fft, dim=0).real
    # filtered_signal = torch.vstack((filtered_fft.real, filtered_fft.imag))
    
    return filtered_signal


def gaussian_filter_1d_energy_preserve(signal: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Applies a 1D Gaussian filter in the frequency domain.
    
    Args:
        signal (torch.Tensor): Input 1D signal (shape: [N]).
        sigma (float): Standard deviation of the Gaussian filter.

    Returns:
        torch.Tensor: Filtered signal in the time domain.
    """
    N = signal.shape[0] # Signal length
    device = signal.device
    
    # Compute frequency components
    freqs = torch.fft.fftfreq(N, device=device)  # Normalized frequency range (-0.5 to 0.5)
    
    # Create Gaussian filter in frequency space
    gaussian_filter = torch.exp(-0.5 * (freqs / sigma) ** 2)
    
    # FFT of the signal
    signal_fft = torch.fft.fft(signal, dim=0)

    num_modes = N//2 + 1
    mag_fft = torch.abs(torch.fft.fft(torch.norm(signal, dim=1), dim=0))
    mask = mag_fft[0, ...] / torch.sum(mag_fft[:num_modes], dim=0) < 0.75
    
    # Apply the Gaussian filter
    filtered_fft = signal_fft.clone()
    filtered_fft[:, :, mask] = signal_fft[:, :, mask] * gaussian_filter[:, None, None]

    original_energy = torch.sum(torch.abs(signal_fft)**2, dim=0)
    filtered_energy = torch.sum(torch.abs(filtered_fft)**2, dim=0)

    filtered_fft *= torch.sqrt(original_energy / filtered_energy).unsqueeze(0)
    
    
    # Inverse FFT to return to time domain
    filtered_signal = torch.fft.ifft(filtered_fft, dim=0).real
    # filtered_signal = torch.vstack((filtered_fft.real, filtered_fft.imag))

    return filtered_signal




def rescale_outlier_vel(vel, alpha=1.8):

    mag_vel = torch.norm(torch.concat((vel[:32, None], vel[32:, None]), dim=1), dim=1) # (N, nv)
    mag_vel = torch.max(mag_vel, dim=0)[0] # (nv)

    thres = torch.mean(mag_vel) + alpha * torch.std(mag_vel)
    mask = mag_vel > thres
    num_mask = torch.sum(mask)
    if num_mask > 0:
        print(f"rel vel: rescaling {num_mask} of {mag_vel.shape[0]} vesicles" )

        vel[:, mask] = vel[:, mask] / mag_vel[mask].unsqueeze(0) * thres

    return vel

def rescale_outlier_vel_abs(vel, c = 0.3, logger=None):
    '''***'''
    N = vel.shape[0] // 2
    mag_vel = torch.norm(torch.concat((vel[:N, None], vel[N:, None]), dim=1), dim=1) # (N, nv)
    mag_vel = torch.max(mag_vel, dim=0)[0] # (nv)

    thres = c / 32 * 1e5
    mask = mag_vel > thres
    num_mask = torch.sum(mask)
    if num_mask > 0:
        # logger.info(f"abs vel: rescaling {num_mask} of {mag_vel.shape[0]} vesicles" )
        print(f"abs vel: rescaling {num_mask} of {mag_vel.shape[0]} vesicles")

        vel[:, mask] = vel[:, mask] / mag_vel[mask].unsqueeze(0) * thres

    return vel


def rescale_outlier_trans(Xadv, Xold, c = 0.28):

    disp = Xadv - Xold
    disp_2ch = torch.concat((disp[:32, None], disp[32:, None]), dim=1) # (N, 2, nv)
    norm_disp = torch.norm(disp_2ch, dim=1) # (N, nv)

    max_norm_disp = torch.max(norm_disp, dim=0)[0] # (nv)
    mask = max_norm_disp > c / 32
    num_mask = torch.sum(mask)
    if num_mask > 0:
        print(f"rescaling vinf translation of {num_mask} of {Xadv.shape[1]} vesicles" )

        Xadv[:, mask] = Xold[:, mask] + disp[:, mask] / max_norm_disp[mask].unsqueeze(0) * c / 32

    return Xadv

def interpft_vec(X, N_new: int):
    N = X.shape[0] // 2
    return torch.concat((interpft(X[:N], N_new),  interpft(X[N:], N_new)), dim=0).to(X.device)
    
@torch.jit.script
def downsample_fft(x, N_new: int):
    """
    Upsamples data x using Fourier transform method to a new length N_new.
    
    Parameters:
        x (Tensor): Input data of shape (2N, nv)
        N_new (int): The desired new length after upsampling.
    
    Returns:
        Tensor: Upsampled data of length N_new.
    """
    N = x.size(0)//2 
    nv = x.size(1)
    # if N == N_new:
    #     return x
    X = torch.fft.fft(x.reshape(2, N, nv), dim=1)
    X_new = torch.zeros((2, N_new, nv), dtype=X.dtype, device=x.device)
    X_new[:, :N_new//2] = X[:, :N_new//2]
    X_new[:, -N_new//2:] = X[:, -N_new//2:]
    return torch.fft.ifft(X_new, dim=1).reshape(2*N_new, nv).real * (N_new / N)




def interpft(x, N_new: int):
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


# for _ in range(10):
#     N = 32
#     y = torch.rand(N*2, 4)
#     y1 = gaussian_filter_shape(y, sigma=0.25)

#     y2 = torch.concat((gaussian_filter_1d(y[:N], sigma=0.25), gaussian_filter_1d(y[N:], sigma=0.25)), dim=0)
#     print(torch.allclose(y1, y2))

# for _ in range(10):
#     X = torch.rand(64 * 6, 4)
#     X1 = interpft_vec(X, 32)
#     X2 = downsample_fft(X, 32)
    
#     print(torch.allclose(X1, X2))