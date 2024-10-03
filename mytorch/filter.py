import torch
import torch.fft

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
    X = torch.fft.fft(x, dim=0)
    
    # If N_new > N, upsample (add zeros in the middle)
    if N_new > N:
        X_new = torch.zeros(N_new, dtype=X.dtype, device=x.device)
        X_new[:N//2] = X[:N//2]
        X_new[-(N//2):] = X[-(N//2):]
    # If N_new < N, downsample (truncate the Fourier modes)
    else:
        X_new = torch.cat([X[:N_new//2], X[-(N_new//2):]])
    
    # Inverse FFT and scale to maintain the correct amplitude
    return torch.fft.ifft(X_new, dim=0).real * (N_new / N)

def upsThenFilterShape(X, Nup, modeCut):
    """
    Delete high frequencies from the vesicle shape by upsampling and applying a filter.
    
    Parameters:
        X (Tensor): Shape of the vesicle, with 2*N rows (x and y components) and nv columns.
        Nup (int): Number of points to upsample.
        modeCut (int): Cutoff mode to filter high frequencies.
    
    Returns:
        Xfinal (Tensor): The filtered shape.
    """
    N = X.size(0) // 2  # Get the number of points (half of the length of X)
    nv = X.size(1)  # Get the number of columns (number of vesicles)

    # Frequency modes
    modes = torch.cat([torch.arange(0, Nup//2, device=X.device), torch.arange(-Nup//2, 0, device=X.device)])

    # Upsample x and y components
    xup = torch.stack([interpft(X[:N, k], Nup) for k in range(nv)], dim=1)
    yup = torch.stack([interpft(X[N:, k], Nup) for k in range(nv)], dim=1)

    Xfinal = torch.zeros_like(X)  # Initialize the result tensor

    for k in range(nv):
        z = xup[:, k] + 1j * yup[:, k]  # Complex form of the shape (z = x + iy)
        z_fft = torch.fft.fft(z, dim=0)  # FFT of z
        z_fft[torch.abs(modes) > modeCut] = 0  # Apply frequency cutoff
        z_ifft = torch.fft.ifft(z_fft, dim=0)  # Inverse FFT

        # Downsample back to original length and assign to result
        Xfinal[:N, k] = interpft(z_ifft.real, N)
        Xfinal[N:, k] = interpft(z_ifft.imag, N)

    return Xfinal

# upsThenFilterShape(torch.rand(256, 2), 512, 16)