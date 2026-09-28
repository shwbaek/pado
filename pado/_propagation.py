"""Shared scalar propagation geometry, computed in float64 on the field device."""
import torch
from typing import Tuple


def compute_pad_width(field: torch.Tensor, linear: bool) -> Tuple[int, int, int, int]:
    """Double spatial dimensions while preserving the integer-centered origin."""
    rows, cols = field.shape[-2:]
    return (cols-cols//2, cols//2, rows-rows//2, rows//2) if linear else (0,)*4


def unpad(field_padded: torch.Tensor, pad_width: Tuple[int, int, int, int]) -> torch.Tensor:
    """Crop (left, right, top, bottom), including zero-width edges."""
    left, right, top, bottom = pad_width
    return field_padded[..., top:-bottom or None, left:-right or None]


def real_tensor(value, device):
    """Convert scalar metadata without detaching tensor-valued parameters."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float64)
    if isinstance(value, (tuple, list)) and any(isinstance(v, torch.Tensor) for v in value):
        return torch.stack([real_tensor(v, device) for v in value])
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def complex_dtype(field):
    return torch.complex128 if field.dtype in (torch.float64, torch.complex128) else torch.complex64


def wavelength_tensor(light):
    wavelengths = real_tensor(light.wvl, light.field.device).reshape(-1)
    if wavelengths.numel() not in (1, light.field.shape[1]):
        raise ValueError("wavelength must be scalar or have one value per channel")
    return wavelengths.expand(light.field.shape[1]).reshape(1, -1, 1, 1)


def frequency_axes(rows, cols, pitch, device, *, centered=True):
    """Return (fy, fx) in centered or native FFT order, preserving bin values."""
    fy = torch.arange(-(rows // 2), rows - rows // 2, dtype=torch.float64, device=device) / (rows * pitch)
    fx = torch.arange(-(cols // 2), cols - cols // 2, dtype=torch.float64, device=device) / (cols * pitch)
    return (fy, fx) if centered else (torch.fft.ifftshift(fy), torch.fft.ifftshift(fx))


def angular_transfer(fx, fy, wavelengths, distance, offset=(0, 0),
                     dtype=torch.complex128, *, fft_period=None, backend='torch', grid_pitch=None):
    """Homogeneous scalar ASM with stable adjoint propagation for negative z.

    Positive z uses outgoing evanescent decay. Negative z reverses the axial
    phase and retains decay with abs(z). Reversing both z and offset gives
    the adjoint, not the unstable inverse of decayed evanescent components. At an exact grazing
    bin the square-root derivative is singular; the discrete cutoff uses a
    finite zero subgradient rather than contaminating all gradients with NaNs.
    Optional fft_period=(Ly,Lx) applies the sampled phase-gradient Nyquist
    support to the same geometry; hard support is locally nondifferentiable.
    grid_pitch identifies axes made by frequency_axes from that pitch, allowing
    complex128 to retain exact integer-bin geometry and its pitch derivative.
    """
    device = fx.device
    distance = real_tensor(distance, device)
    if distance.numel() != 1:
        raise ValueError("z must be scalar")
    distance = distance.reshape(())
    offset = real_tensor(offset, device)
    if offset.shape != (2,):
        raise ValueError("offset must contain (y, x)")
    grid_pitch = (real_tensor(grid_pitch, device)
                  if dtype == torch.complex128 and grid_pitch is not None else None)
    if backend == 'fused' and device.type == 'cuda':
        parameters = (fx, fy, wavelengths, distance, offset) + (() if grid_pitch is None else (grid_pitch,))
        # Forward-mode AD remains active inside no_grad; Jiterator would drop
        # its tangent. Native Torch also handles torch.func/vmap transforms.
        differentiable = (torch._C._are_functorch_transforms_active()
                          or (torch.is_grad_enabled() and any(v.requires_grad for v in parameters))
                          or any(torch.autograd.forward_ad.unpack_dual(v).tangent is not None
                                 for v in parameters))
        if not differentiable:
            from ._cuda import angular_transfer as fused_transfer
            return fused_transfer(fx, fy, wavelengths, distance, offset, dtype, fft_period, grid_pitch)
    x, y = fx[None, :], fy[:, None]
    if dtype == torch.complex128:
        from ._precision import angular_transfer as precise_transfer
        transfer = precise_transfer(fx, fy, wavelengths, distance, offset, grid_pitch)
    else:
        q = 1 - ((wavelengths * x).square() + (wavelengths * y).square())
        # Avoid sqrt(0)'s infinite derivative even in masked branches.
        nonzero = q != 0
        root = torch.sqrt(torch.where(nonzero, q.abs(), torch.ones_like(q)))
        root = torch.where(nonzero, root, torch.zeros_like(root))
        kz_scale = 2 * torch.pi / wavelengths
        phase = kz_scale * distance * torch.where(q >= 0, root, 0.)
        phase = phase + 2 * torch.pi * (x * offset[1] + y * offset[0])
        amplitude = torch.exp(-kz_scale * distance.abs() * torch.where(q < 0, root, 0.))
        transfer = torch.polar(amplitude, phase).to(dtype)
        del amplitude, phase, nonzero, q, root
    if fft_period is not None:
        with torch.no_grad():
            # Preserve subtraction order: regrouping can flip a grazing bin.
            q = 1 - (wavelengths*x).square() - (wavelengths*y).square()
            gamma = q.clamp_min(0).sqrt()
            half_y, half_x = fft_period[0]*.5, fft_period[1]*.5
            mask = ((q > 0)
                    & ((offset[1]*gamma-distance*wavelengths*x).abs() <= half_x*gamma)
                    & ((offset[0]*gamma-distance*wavelengths*y).abs() <= half_y*gamma))
            mask |= ((distance == 0) & (offset[1].abs() <= half_x) & (offset[0].abs() <= half_y))
        transfer = transfer * mask
    return transfer
