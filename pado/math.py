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

import math as _math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ._propagation import unpad as _unpad

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
        >>> phase = torch.tensor([3.5 * torch.pi, -2.5 * torch.pi])
        >>> wrapped = wrap_phase(phase)  # tensor([-0.5000 * π, -0.5000 * π])
    """
    # Remainder by a rounded 2*pi accumulates a large argument-reduction
    # error. Trigonometric kernels reduce the represented input accurately,
    # including float32 phases much larger than 2*pi.
    if not isinstance(phase_u, torch.Tensor):
        phase_u = torch.as_tensor(phase_u, dtype=torch.float64)
    phase = torch.atan2(torch.sin(phase_u), torch.cos(phase_u))
    if stay_positive:
        phase = torch.where(phase < 0, phase + 2 * torch.pi, phase)
    else:
        phase = torch.where(phase == -torch.pi, phase + 2 * torch.pi, phase)
    return phase


def _fft_nd(value: torch.Tensor, dim: Tuple[int, ...], *, inverse: bool = False,
            norm: Optional[str] = "backward") -> torch.Tensor:
    """Apply FFT normalization outside the CPU backend.

    Some CPU FFT backends mis-scale normalized complex64 transforms at large
    sizes. Unnormalized transforms followed by the mathematical scale avoid
    that backend path without changing the process-wide Torch environment.
    CUDA retains its native normalization.
    """
    operation = torch.fft.ifftn if inverse else torch.fft.fftn
    if value.device.type != "cpu":
        return operation(value, dim=dim, norm=norm)
    if norm not in (None, "backward", "forward", "ortho"):
        raise ValueError(f"Invalid FFT normalization mode: {norm!r}")
    result = operation(value, dim=dim, norm="forward" if inverse else "backward")
    size = _math.prod(value.shape[axis] for axis in dim)
    if norm == "ortho":
        return result / _math.sqrt(size)
    if (inverse and norm in (None, "backward")) or (not inverse and norm == "forward"):
        return result / size
    return result


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
        if padval != 0:
            raise NotImplementedError("Only zero padding is implemented.")
        if len(pad_width) != 4 or any(pad_width):
            arr_c = F.pad(arr_c, pad_width)
    
    arr_c_shifted = torch.fft.ifftshift(arr_c, dim=(-2, -1)) if shift else arr_c
    arr_c_fft = _fft_nd(arr_c_shifted, (-2, -1), norm=normalized)
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
    arr_c_fft = _fft_nd(arr_c_shifted, (-2, -1), inverse=True, norm=normalized)
    arr_c_result = torch.fft.fftshift(arr_c_fft, dim=(-2, -1)) if shift else arr_c_fft
    
    if pad_width is not None and any(pad_width):
        arr_c_result = _unpad(arr_c_result, pad_width)
    
    return arr_c_result

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: Optional[float] = 1.0) -> torch.Tensor:
    """Calculate Peak Signal-to-Noise Ratio between multi-channel tensors.

    Args:
        img1 (torch.Tensor): First tensor [B, Channel, R, C]
        img2 (torch.Tensor): Second tensor [B, Channel, R, C]
        data_range (float, optional): The data range of the input image (e.g., 1.0 for normalized images, 
                            255 for uint8 images). If None, uses the detached maximum value
                            from both images, preserving the original normalization rule.

    Returns:
        torch.Tensor: Scalar PSNR in dB on the input device, +inf if images are identical

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
        data_range = torch.maximum(torch.max(img1), torch.max(img2)).detach()
    
    # If tensor is 4D [B, C, H, W], compute MSE per batch and channel, then average
    if len(img1.shape) == 4:
        mse = torch.mean((img1 - img2) ** 2, dim=(-1, -2))  # MSE per batch and channel
        mse = torch.mean(mse)  # Average over batches and channels
    else:
        mse = torch.mean((img1 - img2) ** 2)
    
    epsilon = 1e-10
    identical = mse == 0
    peak = torch.as_tensor(data_range, dtype=mse.dtype, device=mse.device)
    # Keep the unused branch finite even for two all-zero images, so backward
    # does not encounter the undefined derivative of log(0).
    peak = torch.where(identical, torch.ones_like(mse), peak)
    score = 20 * torch.log10(peak / torch.sqrt(mse + epsilon))
    return torch.where(identical, torch.full_like(score, float('inf')), score)

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, 
                  window_size: int = 21, 
                  sigma: Optional[float] = None, 
                  data_range: float = 1.0) -> torch.Tensor:
    """Calculate Structural Similarity Index between multi-channel tensors.

    Args:
        img1 (torch.Tensor): First tensor [B, Channel, H, W]
        img2 (torch.Tensor): Second tensor [B, Channel, H, W]
        window_size (int): Size of Gaussian window (odd number)
        sigma (float, optional): Standard deviation of Gaussian window. 
                               If None, defaults to window_size/6
        data_range (float): Dynamic range of images

    Returns:
        torch.Tensor: Scalar SSIM score (-1 to 1, where 1 indicates identical images)

    Examples:
        >>> intensity1 = light1.get_intensity()  # [B, Channel, R, C]
        >>> intensity2 = light2.get_intensity()  # [B, Channel, R, C]
        >>> similarity = calculate_ssim(intensity1, intensity2)
    """
    if sigma is None:
        sigma = window_size / 6

    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    img2 = img2.to(device=img1.device, dtype=img1.dtype)
    window = gaussian_window(window_size, sigma, device=img1.device, dtype=img1.dtype)
    window = window[None, None].expand(img1.size(1), 1, -1, -1).contiguous()
    
    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Each channel uses the same Gaussian kernel without mixing channels.
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

def gaussian_window(size: int, sigma: float, *, device: Optional[torch.device] = None,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create normalized 2D Gaussian window.

    Args:
        size (int): Width and height of square window
        sigma (float): Standard deviation of Gaussian
        device (torch.device, optional): Device for direct kernel construction
        dtype (torch.dtype): Real floating-point kernel dtype (default float32)

    Returns:
        torch.Tensor: Normalized 2D Gaussian window [size, size]

    Examples:
        >>> window = gaussian_window(11, 1.5)
    """
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    grid = torch.meshgrid(coords, coords, indexing='ij')
    window = torch.exp(-(grid[0] ** 2 + grid[1] ** 2) / (2 * sigma ** 2))
    return window / window.sum()

##########################
# Additional Helper Functions for Sc-ASM(Scaled Angular Spectrum Method)
##########################
def _scaled_transform_last_axis(g: torch.Tensor, M: int, delta_x: float,
                                delta_fx: float, *, inverse: bool = False) -> torch.Tensor:
    """Evaluate a centered Fourier sum with a batched chirp convolution."""
    if g.shape[-1] != M:
        raise ValueError(f"Expected a final axis of length {M}, got {g.shape[-1]}")
    if g.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError("Scaled transforms require float32, float64, complex64 or complex128 input")
    complex_dtype = torch.complex128 if g.dtype in (torch.float64, torch.complex128) else torch.complex64
    # Double precision argument construction prevents the chirp phase error
    # from growing quadratically with the grid index, even for complex64 FFTs.
    dx = torch.as_tensor(delta_x, dtype=torch.float64, device=g.device)
    dfx = torch.as_tensor(delta_fx, dtype=torch.float64, device=g.device)
    if dx.numel() != 1 or dfx.numel() != 1:
        raise ValueError("Scaled-transform sampling intervals must be scalars")
    dx, dfx = dx.reshape(()), dfx.reshape(())
    start = M - M // 2
    m_big = torch.arange(-M, M, dtype=torch.float64, device=g.device)
    # Reduce in units of pi with the exactly representable period 2 before
    # multiplying by pi. This retains discrete Fourier periodicity when the
    # sampling product or grid is large; reducing a rounded 2*pi does not.
    cycles = torch.remainder((dx * dfx) * m_big.square(), 2)
    phase = (1 if inverse else -1) * torch.pi * cycles
    chirp = torch.exp(1j * phase).to(complex_dtype)
    kernel = chirp.conj()
    # The product and inverse use native FFT order; paired frequency shifts
    # cancel algebraically and would otherwise copy the batched spectrum.
    # Multiply only nonzero input samples and build the padded native order
    # directly, avoiding a padded product and a full batched roll.
    weighted = g.to(complex_dtype) * chirp[start:start + M]
    zeros = weighted.new_zeros((*weighted.shape[:-1], M))
    native = torch.cat((weighted[..., M//2:], zeros, weighted[..., :M//2]), dim=-1)
    Q1 = _fft_nd(native, (-1,))
    del weighted, zeros, native
    Q2 = _fft_nd(torch.fft.ifftshift(kernel, dim=-1), (-1,))
    conv = _fft_nd(Q1 * Q2, (-1,), inverse=True)
    # Select only the requested centered window rather than rolling all 2*M samples.
    conv = torch.cat((conv[..., 2*M-M//2:], conv[..., :M-M//2]), dim=-1)
    scale = (dfx if inverse else dx).to(g.real.dtype)
    return scale * chirp[start:start + M] * conv


def sc_dft_1d(g: torch.Tensor, M: int, delta_x: float, delta_fx: float) -> torch.Tensor:
    """Compute 1D scaled DFT for optical field propagation.

    The sum is ``delta_x * sum(g[n] * exp(-2j*pi*x[n]*f[k]))``, with
    ``x[n]=(n-M//2)*delta_x`` and ``f[k]=(k-M//2)*delta_fx``. The last axis
    is transformed; leading axes and complex precision are preserved.

    Args:
        g (torch.Tensor): Input complex field [..., M]
        M (int): Number of sample points
        delta_x (float): Spatial sampling interval (m)
        delta_fx (float): Frequency sampling interval (1/m)

    Returns:
        torch.Tensor: Transformed complex field [..., M]

    Examples:
        >>> M = 1000
        >>> pitch = 2e-6
        >>> g = torch.exp(-x**2 / (2 * (100*um)**2)).to(torch.complex64)
        >>> G = sc_dft_1d(g, M, pitch, 1/(M*pitch))
    """
    return _scaled_transform_last_axis(g, M, delta_x, delta_fx)

def sc_idft_1d(G: torch.Tensor, M: int, delta_fx: float, delta_x: float) -> torch.Tensor:
    """Compute 1D scaled inverse DFT for optical field reconstruction.

    Uses the positive exponential and a ``delta_fx`` quadrature weight. It
    inverts ``sc_dft_1d`` when ``M*delta_x*delta_fx == 1``. Leading axes
    and complex precision are preserved; arbitrary sampling need not invert.

    Args:
        G (torch.Tensor): Frequency domain input [..., M]
        M (int): Number of samples
        delta_fx (float): Frequency sampling interval (1/m)
        delta_x (float): Spatial sampling interval (m)

    Returns:
        torch.Tensor: Spatial domain complex output [..., M]

    Examples:
        >>> M = 1000
        >>> G = torch.ones(M, dtype=torch.complex64)
        >>> field = sc_idft_1d(G, M, 1/(M*2e-6), 4e-6)
    """
    return _scaled_transform_last_axis(G, M, delta_x, delta_fx, inverse=True)

def sc_dft_2d(u: torch.Tensor, Mx: int, My: int, 
             delta_x: float, delta_y: float, 
             delta_fx: float, delta_fy: float) -> torch.Tensor:
    """Perform 2D scaled DFT using batched separable 1D transforms.

    Leading batch/channel axes and complex precision are preserved. Both
    spatial and frequency grids use centered integer sample coordinates.

    Args:
        u (torch.Tensor): Input field [..., My, Mx]
        Mx, My (int): Number of samples in x,y directions
        delta_x, delta_y (float): Spatial sampling intervals (m)
        delta_fx, delta_fy (float): Frequency sampling intervals (1/m)

    Returns:
        torch.Tensor: Transformed complex field [..., My, Mx]

    Examples:
        >>> field = light.get_field().squeeze()
        >>> U = sc_dft_2d(field, 1024, 1024, pitch, pitch, 1/(pitch*1024), 1/(pitch*1024))
    """
    U_intermediate = sc_dft_1d(u, Mx, delta_x, delta_fx)
    return sc_dft_1d(U_intermediate.transpose(-2, -1), My, delta_y, delta_fy).transpose(-2, -1)

def sc_idft_2d(U: torch.Tensor, Mx: int, My: int, 
              delta_x: float, delta_y: float, 
              delta_fx: float, delta_fy: float) -> torch.Tensor:
    """Perform 2D scaled inverse DFT using batched separable 1D transforms.

    Leading batch/channel axes and complex precision are preserved. The
    inverse has positive exponential signs and weight ``delta_fx*delta_fy``.

    Args:
        U (torch.Tensor): Frequency domain input [..., My, Mx]
        Mx, My (int): Number of samples in x,y directions
        delta_x, delta_y (float): Target spatial sampling intervals (m)
        delta_fx, delta_fy (float): Frequency sampling intervals (1/m)

    Returns:
        torch.Tensor: Spatial domain complex output [..., My, Mx]

    Examples:
        >>> U = sc_dft_2d(field, Mx, My, dx, dy, dfx, dfy)
        >>> field_recovered = sc_idft_2d(U, Mx, My, dx, dy, dfx, dfy)
    """
    u_intermediate = sc_idft_1d(U.transpose(-2, -1), My, delta_fy, delta_y).transpose(-2, -1)
    return sc_idft_1d(u_intermediate, Mx, delta_fx, delta_x)

def compute_scasm_transfer_function(Mx: int, My: int, 
                                   delta_fx: float, delta_fy: float, 
                                   λ: float, z: float, *,
                                   device: Optional[torch.device] = None,
                                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Compute transfer function for Scaled Angular Spectrum Method propagation.

    Positive distance uses outgoing evanescent decay. Negative distance is the
    stable adjoint: propagating phase is conjugated and evanescent modes still
    decay with abs(z). Exact grazing bins use a finite zero subgradient for the
    singular square root, matching the full angular-spectrum propagator.

    Args:
        Mx, My (int): Number of sampling points in x,y directions
        delta_fx, delta_fy (float): Frequency sampling intervals (1/m)
        λ (float): Wavelength (m)
        z (float): Propagation distance (m)
        device (torch.device, optional): Device for direct transfer-function construction
        dtype (torch.dtype): Output real or complex precision; phase geometry
            is evaluated in float64 before casting to this precision.

    Returns:
        torch.Tensor: Transfer function H(fx,fy) [My, Mx]

    Examples:
        >>> H = compute_scasm_transfer_function(1024, 1024, 1/(1024*2e-6), 1/(1024*2e-6), 633e-9, 0.1)
        >>> U_prop = torch.fft.fft2(light.get_field()) * H
    """
    if dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        raise TypeError("Transfer dtype must be float32, float64, complex64 or complex128")
    complex_dtype = torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64
    if device is None:
        device = next((value.device for value in (delta_fx, delta_fy, λ, z)
                       if isinstance(value, torch.Tensor)), None)
    delta_fx = torch.as_tensor(delta_fx, dtype=torch.float64, device=device)
    delta_fy = torch.as_tensor(delta_fy, dtype=torch.float64, device=device)
    λ = torch.as_tensor(λ, dtype=torch.float64, device=device)
    z = torch.as_tensor(z, dtype=torch.float64, device=device)
    if any(value.numel() != 1 for value in (delta_fx, delta_fy, λ, z)):
        raise ValueError("Transfer sampling intervals, wavelength and distance must be scalars")
    delta_fx, delta_fy, λ, z = (value.reshape(()) for value in (delta_fx, delta_fy, λ, z))
    fx = (torch.arange(Mx, dtype=torch.float64, device=device) - Mx // 2) * delta_fx
    fy = (torch.arange(My, dtype=torch.float64, device=device) - My // 2) * delta_fy
    from ._propagation import angular_transfer
    return angular_transfer(fx, fy, λ, z, dtype=complex_dtype)
