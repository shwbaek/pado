"""Sampled Fresnel, far-field, and scaled angular-spectrum operators.

Coordinates use (index - size//2)*pitch on both axes. These are finite sampled
scalar models; their agreement with a continuous diffraction problem still
depends on aperture, sampling, propagation regime and padding convergence.
"""
import torch

from ._propagation import (angular_transfer, complex_dtype, compute_pad_width,
                          frequency_axes, real_tensor, unpad, wavelength_tensor)
from .light import Light
from .math import fft, ifft, sc_dft_2d, sc_idft_2d


def _output(light, field, pitch=None):
    return Light(tuple(field.shape), light.pitch if pitch is None else pitch,
                 light.wvl, field=field, device=str(field.device))


def fresnel(light, distance, linear):
    """Exact sampled paraxial Fourier operator with optional zero padding."""
    field = light.field
    pad = compute_pad_width(field, linear)
    rows = field.shape[-2]+pad[2]+pad[3]
    cols = field.shape[-1]+pad[0]+pad[1]
    fy, fx = frequency_axes(rows, cols, light.pitch, field.device, centered=False)
    wavelength = wavelength_tensor(light)
    distance = real_tensor(distance, field.device)
    if distance.numel() != 1:
        raise ValueError("z must be scalar")
    distance = distance.reshape(())
    phase = 2*torch.pi*distance/wavelength - torch.pi*wavelength*distance*(
        fy[:, None].square()+fx[None, :].square())
    transfer = torch.polar(torch.ones_like(phase), phase).to(complex_dtype(field))
    propagated = ifft(fft(field, pad_width=pad, shift=False)*transfer,
                      pad_width=pad, shift=False)
    return _output(light, propagated)


def fraunhofer(light, distance, linear):
    """Finite Fraunhofer integral on a common isotropic wavelength grid.

    linear=True doubles the reciprocal-grid sampling density, equivalent to
    zero-padding before a Fourier transform and retaining the central output
    shape. Output pitch is min(wavelength)*abs(z)/(max(R,C)*input_pitch),
    divided by two when padded. The field includes physical pixel quadrature,
    spherical carrier, 1/(i*wavelength*z), and output quadratic phase. Tensor
    z/wavelength gradients include the dependent output coordinates; Light's
    float pitch metadata is detached from that calculation.
    """
    field = light.field
    rows, cols = field.shape[-2:]
    wavelengths = wavelength_tensor(light).reshape(-1)
    distance = real_tensor(distance, field.device)
    if distance.numel() != 1 or not bool(torch.isfinite(distance)) or bool(distance == 0):
        raise ValueError("Fraunhofer z must be a finite nonzero scalar")
    distance = distance.reshape(())
    output_pitch = wavelengths.min()*distance.abs()/(max(rows, cols)*light.pitch*(2 if linear else 1))
    y = (torch.arange(rows, device=field.device, dtype=torch.float64)-rows//2)*output_pitch
    x = (torch.arange(cols, device=field.device, dtype=torch.float64)-cols//2)*output_pitch
    output=[]
    for channel, wavelength in enumerate(wavelengths):
        frequency_step = output_pitch/(wavelength*distance)
        spectrum = sc_dft_2d(field[:, channel], cols, rows, light.pitch,
                             light.pitch, frequency_step, frequency_step)
        phase = 2*torch.pi*distance/wavelength + torch.pi*(
            y[:, None].square()+x[None, :].square())/(wavelength*distance)
        prefactor = torch.polar(torch.ones_like(phase), phase)/(1j*wavelength*distance)
        output.append(spectrum*prefactor.to(complex_dtype(field)))
    return _output(light, torch.stack(output, dim=1), float(output_pitch.detach()))


def scaled_angular_spectrum(light, distance, scale, linear, *, backend='torch'):
    """Resample the angular-spectrum integral at isotropic pitch scale*pitch.

    The forward FFT is multiplied by pixel area and the inverse scaled DFT
    by reciprocal pixel area. Separate row/column frequency intervals support
    odd rectangular arrays. The same operator handles expansion and focusing.
    Negative distance uses the stable adjoint policy of angular_transfer.
    """
    if not isinstance(scale, (int, float)) or not 0 < scale < float("inf"):
        raise ValueError("b must be a finite positive scalar")
    field = light.field
    pad = compute_pad_width(field, linear)
    spectrum = fft(field, pad_width=pad)*light.pitch**2
    rows, cols = spectrum.shape[-2:]
    fy, fx = frequency_axes(rows, cols, light.pitch, field.device)
    transfer = angular_transfer(fx, fy, wavelength_tensor(light), distance,
                                dtype=complex_dtype(field), backend=backend, grid_pitch=light.pitch)
    output_pitch = scale*light.pitch
    output = sc_idft_2d(spectrum*transfer, cols, rows, output_pitch, output_pitch,
                        1/(cols*light.pitch), 1/(rows*light.pitch))
    if linear:
        output = unpad(output, pad)
    return _output(light, output, output_pitch)
