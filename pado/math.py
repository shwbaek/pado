########################################################
# The MIT License (MIT)
#
# PADO (Pytorch Automatic Differentiable Optics)
# Copyright (c) 2023 by POSTECH Computer Graphics Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Contact:
# Lead Developer: Dong-Ha Shin (0218sdh@gmail.com)
# Corresponding Author: Seung-Hwan Baek (shwbaek@postech.ac.kr)
#
########################################################

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

"""
At Pado, all measurements adhere to the International System of Units (SI).
"""
nm: float = 1e-9
um: float = 1e-6
mm: float = 1e-3
cm: float = 1e-2
m: float = 1

s: float = 1
ms: float = 1e-3
us: float = 1e-6
ns: float = 1e-9

def wrap_phase(phase_u: torch.Tensor, stay_positive: bool = False) -> torch.Tensor:
    """Wrap phase values to [-π, π] or [0, 2π] range.

    Args:
        phase_u (torch.Tensor): Unwrapped phase values tensor
        stay_positive (bool): If True, output range is [0, 2π]. If False, [-π, π]

    Returns:
        torch.Tensor: Wrapped phase values tensor

    Examples:
        >>> phase = torch.tensor([3.5 * np.pi, -2.5 * np.pi])
        >>> wrapped = wrap_phase(phase)  # tensor([0.5000 * π, -0.5000 * π])
    """
    phase = phase_u % (2 * np.pi)
    if not stay_positive:
        phase[phase > torch.pi] -= 2 * np.pi
    return phase


def fft(arr_c: torch.Tensor, normalized: str = "backward", 
        pad_width: Optional[Tuple[int, int, int, int]] = None, 
        padval: int = 0, shift: bool = True) -> torch.Tensor:
    """Compute 2D FFT of a complex tensor with optional padding and frequency shifting.

    Args:
        arr_c (torch.Tensor): Complex tensor [B, Ch, H, W]
        normalized (str): FFT normalization mode: "backward", "forward", or "ortho"
        pad_width (tuple): Padding as (left, right, top, bottom)
        padval (int): Padding value (only 0 supported)
        shift (bool): If True, center zero-frequency component

    Returns:
        torch.Tensor: FFT result tensor

    Examples:
        >>> light = Light(dim=(1, 1, 100, 100), pitch=2e-6, wvl=500e-9)
        >>> field_fft = fft(light.field)
    """
    if pad_width is not None:
        if padval == 0:
            arr_c = F.pad(arr_c, pad_width)
        else:
            raise NotImplementedError("Only zero padding is implemented.")
    
    arr_c_shifted = torch.fft.ifftshift(arr_c, dim=(-2, -1)) if shift else arr_c
    arr_c_fft = torch.fft.fft2(arr_c_shifted, norm=normalized)
    return torch.fft.fftshift(arr_c_fft, dim=(-2, -1)) if shift else arr_c_fft


def ifft(arr_c: torch.Tensor, normalized: str = "backward", 
         pad_width: Optional[Tuple[int, int, int, int]] = None, 
         shift: bool = True) -> torch.Tensor:
    """Compute 2D inverse FFT of a complex tensor with optional padding and shifting.

    Args:
        arr_c (torch.Tensor): Complex tensor [B, Ch, H, W]
        normalized (str): IFFT normalization mode: "backward", "forward", or "ortho"
        pad_width (tuple): Padding as (left, right, top, bottom)
        shift (bool): If True, center zero-frequency component

    Returns:
        torch.Tensor: IFFT result tensor

    Examples:
        >>> field = torch.ones((1, 1, 64, 64), dtype=torch.complex64)
        >>> field_freq = fft(field)
        >>> field_restored = ifft(field_freq)
    """
    arr_c_shifted = torch.fft.ifftshift(arr_c, dim=(-2, -1)) if shift else arr_c
    arr_c_fft = torch.fft.ifft2(arr_c_shifted, norm=normalized)
    arr_c_result = torch.fft.fftshift(arr_c_fft, dim=(-2, -1)) if shift else arr_c_fft
    
    if pad_width is not None:
        if pad_width[2] != 0 and pad_width[3] != 0:
            arr_c_result = arr_c_result[..., pad_width[2]:-pad_width[3], :]
        if pad_width[0] != 0 and pad_width[1] != 0:
            arr_c_result = arr_c_result[..., :, pad_width[0]:-pad_width[1]]
    
    return arr_c_result

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: Optional[float] = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio between multi-channel tensors.

    Args:
        img1 (torch.Tensor): First tensor [B, Channel, R, C]
        img2 (torch.Tensor): Second tensor [B, Channel, R, C]
        data_range (float, optional): The data range of the input image (e.g., 1.0 for normalized images, 
                            255 for uint8 images). If None, uses the maximum value from images.

    Returns:
        float: PSNR value in dB, infinity if images are identical

    Examples:
        >>> intensity1 = light1.get_intensity()  # [B, Channel, R, C]
        >>> intensity2 = light2.get_intensity()  # [B, Channel, R, C]
        >>> psnr = calculate_psnr(intensity1, intensity2)
    """
    if img1.shape != img2.shape:
        raise ValueError("Input tensors must have the same shape")
        
    img2 = img2.to(img1.device)
    
    # If data_range is None, determine it from the input images
    if data_range is None:
        data_range = max(torch.max(img1).item(), torch.max(img2).item())
    
    # If tensor is 4D [B, C, H, W], compute MSE per batch and channel, then average
    if len(img1.shape) == 4:
        mse = torch.mean((img1 - img2) ** 2, dim=(-1, -2))  # MSE per batch and channel
        mse = torch.mean(mse)  # Average over batches and channels
    else:
        mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    epsilon = 1e-10

    return 20 * torch.log10(data_range / torch.sqrt(mse + epsilon))

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, 
                  window_size: int = 21, 
                  sigma: Optional[float] = None, 
                  data_range: float = 1.0) -> float:
    """Calculate Structural Similarity Index between multi-channel tensors.

    Args:
        img1 (torch.Tensor): First tensor [B, Channel, H, W]
        img2 (torch.Tensor): Second tensor [B, Channel, H, W]
        window_size (int): Size of Gaussian window (odd number)
        sigma (float, optional): Standard deviation of Gaussian window. 
                               If None, defaults to window_size/6
        data_range (float): Dynamic range of images

    Returns:
        float: SSIM score (-1 to 1, where 1 indicates identical images)

    Examples:
        >>> intensity1 = light1.get_intensity()  # [B, Channel, R, C]
        >>> intensity2 = light2.get_intensity()  # [B, Channel, R, C]
        >>> similarity = calculate_ssim(intensity1, intensity2)
    """
    if sigma is None:
        sigma = window_size / 6

    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    img2 = img2.to(img1.device)
    window = gaussian_window(window_size, sigma).to(img1.device)
    window = window.unsqueeze(0).unsqueeze(0)
    
    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Compute means (window will be automatically broadcasted across batch and channels)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img2.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def gaussian_window(size: int, sigma: float) -> torch.Tensor:
    """Create normalized 2D Gaussian window.

    Args:
        size (int): Width and height of square window
        sigma (float): Standard deviation of Gaussian

    Returns:
        torch.Tensor: Normalized 2D Gaussian window [size, size]

    Examples:
        >>> window = gaussian_window(11, 1.5)
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    grid = torch.meshgrid(coords, coords, indexing='ij')
    window = torch.exp(-(grid[0] ** 2 + grid[1] ** 2) / (2 * sigma ** 2))
    return window / window.sum()

##########################
# Additional Helper Functions for Sc-ASM(Scaled Angular Spectrum Method)
##########################
def sc_dft_1d(g: torch.Tensor, M: int, delta_x: float, delta_fx: float) -> torch.Tensor:
    """Compute 1D scaled DFT for optical field propagation.

    Args:
        g (torch.Tensor): Input complex field [M]
        M (int): Number of sample points
        delta_x (float): Spatial sampling interval (m)
        delta_fx (float): Frequency sampling interval (1/m)

    Returns:
        torch.Tensor: Transformed complex field [M]

    Examples:
        >>> M = 1000
        >>> pitch = 2e-6
        >>> g = torch.exp(-x**2 / (2 * (100*um)**2)).to(torch.complex64)
        >>> G = sc_dft_1d(g, M, pitch, 1/(M*pitch))
    """
    device = g.device
    beta = np.pi * delta_fx * delta_x

    M2 = 2*M

    g_padded = torch.zeros(M2, dtype=torch.complex64, device=device)
    g_padded[M//2:M//2+M] = g

    m_big = torch.arange(-M, M, dtype=torch.float32, device=device)  # length = M2

    q1 = g_padded * torch.exp(-1j * beta * (m_big**2))
    q2 = torch.exp(1j * beta * (m_big**2))

    Q1 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q1)))
    Q2 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q2)))
    conv = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(Q1*Q2)))
    conv = conv[M//2:M//2+M]

    p = torch.arange(-M//2, M//2, dtype=torch.float32, device=device)
    G = delta_x * torch.exp(-1j * beta * (p**2)) * conv

    return G

def sc_idft_1d(G: torch.Tensor, M: int, delta_fx: float, delta_x: float) -> torch.Tensor:
    """Compute 1D scaled inverse DFT for optical field reconstruction.

    Args:
        G (torch.Tensor): Frequency domain input [M]
        M (int): Number of samples
        delta_fx (float): Frequency sampling interval (1/m)
        delta_x (float): Spatial sampling interval (m)

    Returns:
        torch.Tensor: Spatial domain output [M]

    Examples:
        >>> M = 1000
        >>> G = torch.ones(M, dtype=torch.complex64)
        >>> field = sc_idft_1d(G, M, 1/(M*2e-6), 4e-6)
    """
    device = G.device
    beta = np.pi * delta_fx * delta_x
    M2 = 2*M

    G_padded = torch.zeros(M2, dtype=torch.complex64, device=device)
    G_padded[M//2:M//2+M] = G

    m_big = torch.arange(-M, M, dtype=torch.float32, device=device)

    q1_inv = G_padded * torch.exp(-1j * beta * (m_big**2))
    q2_inv = torch.exp(1j * beta * (m_big**2))

    Q1 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q1_inv)))
    Q2 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q2_inv)))
    conv = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(Q1*Q2)))
    conv = conv[M//2:M//2+M]

    p = torch.arange(-M//2, M//2, dtype=torch.float32, device=device)

    xdomain = delta_fx * torch.exp(-1j*beta*(p**2)) * conv

    return xdomain

def sc_dft_2d(u: torch.Tensor, Mx: int, My: int, 
             delta_x: float, delta_y: float, 
             delta_fx: float, delta_fy: float) -> torch.Tensor:
    """Perform 2D scaled DFT using separable 1D transforms.

    Args:
        u (torch.Tensor): Input field [My, Mx]
        Mx, My (int): Number of samples in x,y directions
        delta_x, delta_y (float): Spatial sampling intervals (m)
        delta_fx, delta_fy (float): Frequency sampling intervals (1/m)

    Returns:
        torch.Tensor: Transformed field [My, Mx]

    Examples:
        >>> field = light.get_field().squeeze()
        >>> U = sc_dft_2d(field, 1024, 1024, pitch, pitch, 1/(pitch*1024), 1/(pitch*1024))
    """
    U_intermediate = torch.zeros_like(u, dtype=torch.complex64)
    for iy in range(My):
        U_intermediate[iy, :] = sc_dft_1d(u[iy, :], Mx, delta_x, delta_fx)

    U_final = torch.zeros_like(U_intermediate, dtype=torch.complex64)
    for ix in range(Mx):
        U_final[:, ix] = sc_dft_1d(U_intermediate[:, ix], My, delta_y, delta_fy)

    return U_final

def sc_idft_2d(U: torch.Tensor, Mx: int, My: int, 
              delta_x: float, delta_y: float, 
              delta_fx: float, delta_fy: float) -> torch.Tensor:
    """Perform 2D scaled inverse DFT using separable 1D transforms.

    Args:
        U (torch.Tensor): Frequency domain input [My, Mx]
        Mx, My (int): Number of samples in x,y directions
        delta_x, delta_y (float): Target spatial sampling intervals (m)
        delta_fx, delta_fy (float): Frequency sampling intervals (1/m)

    Returns:
        torch.Tensor: Spatial domain output [My, Mx]

    Examples:
        >>> U = sc_dft_2d(field, Mx, My, dx, dy, dfx, dfy)
        >>> field_recovered = sc_idft_2d(U, Mx, My, dx, dy, dfx, dfy)
    """
    u_intermediate = torch.zeros_like(U, dtype=torch.complex64)
    for ix in range(Mx):
        u_intermediate[:, ix] = sc_idft_1d(U[:, ix], My, delta_fy, delta_y)

    u_final = torch.zeros_like(u_intermediate, dtype=torch.complex64)
    for iy in range(My):
        u_final[iy, :] = sc_idft_1d(u_intermediate[iy, :], Mx, delta_fx, delta_x)

    return u_final

def compute_scasm_transfer_function(Mx: int, My: int, 
                                   delta_fx: float, delta_fy: float, 
                                   λ: float, z: float) -> torch.Tensor:
    """Compute transfer function for Scaled Angular Spectrum Method propagation.

    Args:
        Mx, My (int): Number of sampling points in x,y directions
        delta_fx, delta_fy (float): Frequency sampling intervals (1/m)
        λ (float): Wavelength (m)
        z (float): Propagation distance (m)

    Returns:
        torch.Tensor: Transfer function H(fx,fy) [My, Mx]

    Examples:
        >>> H = compute_scasm_transfer_function(1024, 1024, 1/(1024*2e-6), 1/(1024*2e-6), 633e-9, 0.1)
        >>> U_prop = torch.fft.fft2(light.get_field()) * H
    """
    fx = (torch.arange(-Mx//2, Mx//2, dtype=torch.float32)*delta_fx)
    fy = (torch.arange(-My//2, My//2, dtype=torch.float32)*delta_fy)
    fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')
    k = 2*np.pi/λ

    gamma = torch.sqrt(1 - (λ*fxx)**2 - (λ*fyy)**2 + 0j)
    H = torch.exp(1j*k*z*gamma)
    return H