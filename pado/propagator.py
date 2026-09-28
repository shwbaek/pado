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

import torch
import torch.nn.functional as Func
from typing import Tuple, Optional, Callable, Dict

from .math import (
    fft,
    ifft, 
    wrap_phase,
    sc_dft_2d,
    sc_idft_2d,
    compute_scasm_transfer_function,
)
from .light import Light, PolarizedLight
from .kspace_light import KSpaceLight
from ._propagation import compute_pad_width, unpad


def _zero_offset(offset):
    """Recognize constant identity metadata without reading a device tensor."""
    return offset is None or isinstance(offset, bool) or (
        isinstance(offset, (tuple, list)) and len(offset) == 2
        and all(isinstance(v, (int, float)) and v == 0 for v in offset))


def _fixed_geometry(value, device):
    """Validate fixed real metadata without detaching a requested derivative."""
    from ._propagation import real_tensor

    if isinstance(value, (tuple, list)) and value:
        return torch.stack([_fixed_geometry(v, device) for v in value])
    probe = torch.as_tensor(value)
    if probe.is_complex() or probe.dtype == torch.bool:
        raise TypeError("Prepared ASM geometry must be real numeric values")
    if (probe.requires_grad
            or torch.autograd.forward_ad.unpack_dual(probe).tangent is not None):
        raise ValueError("Prepared ASM requires fixed geometry; use forward for parameter gradients")
    # Convert the original value: probing a Python float may use float32.
    return real_tensor(value, device)


class Propagator:
    def __init__(self, mode: str, polar: str = 'non', *, backend: str = 'torch'):
        """Light propagator for simulating wave propagation through free space.

        Implement common diffraction methods including Fraunhofer, Fresnel, ASM and RS.
        Support complex field calculations.

        Args:
            mode (str): Propagation method to use:
                - "Fraunhofer": Far-field diffraction
                - "Fresnel": Near-field diffraction
                - "ASM": Angular Spectrum Method
                - "RS": Rayleigh-Sommerfeld Method
            polar (str): Polarization mode ('non': scalar, 'polar': vector)
            backend (str): 'torch' uses native PyTorch on the field device.
                'fused' opts ASM into an experimental NVRTC transfer kernel
                for fixed CUDA geometry. CPU and differentiable geometry use
                PyTorch; input-field gradients remain supported in both paths.
                First CUDA use compiles the kernel through PyTorch's Jiterator.

        Examples:
            >>> # Create ASM propagator for scalar field
            >>> prop = Propagator(mode="ASM", polar="non")
            
            >>> # Create Fresnel propagator for vector field
            >>> prop = Propagator(mode="Fresnel", polar="polar")
            
            >>> # Create Fraunhofer propagator
            >>> prop = Propagator(mode="ASM")
            >>> field = torch.ones((1, 1, 1000, 1000))
            >>> light = Light(field, pitch=2e-6, wvl=660e-9)
            >>> light_prop = prop.forward(light, z=0.05)
        """
        self.mode: str = mode
        self.polar: str = polar
        if backend not in ('torch', 'fused'):
            raise ValueError("backend must be 'torch' or 'fused'")
        if backend == 'fused' and mode != 'ASM':
            raise ValueError("The fused backend is available for ASM only")
        self.backend = backend

    def prepare(self, light: Light, z: float, *, offset=(0, 0),
                linear: bool = True, band_limit: bool = True) -> Callable:
        """Prepare scalar, unscaled ASM for repeated Tensor -> Tensor calls.

        Optical conditions are fixed at preparation. The callable accepts new
        fields with the same channels, spatial shape, dtype and device; batch
        size may vary. Only H and layout metadata persist, not the input Light,
        field or geometry tensors. Changing conditions requires a new prepare.
        H occupies one complex tensor per wavelength at the padded FFT size.

        Field gradients remain supported across repeated backward calls.
        Differentiable geometry is rejected, even inside no_grad; use forward
        to optimize wavelength, pitch, distance or offset. Preparation inside
        torch.func transforms is unsupported; applying an existing operator to
        transformed fields uses ordinary Torch operations. Fixed zero distance
        and zero offset return the input field itself, including Tensor zeros.
        The selected backend is used only to prepare H, never during reuse.
        """
        if self.mode != "ASM" or self.polar != "non" or isinstance(light, PolarizedLight):
            raise NotImplementedError("prepare supports scalar unscaled ASM only")
        if torch._C._are_functorch_transforms_active():
            raise ValueError("Prepare fixed geometry outside torch.func transforms")
        field = light.field
        if (field.ndim != 4 or any(size < 1 for size in field.shape)
                or field.dtype not in (torch.float32, torch.float64,
                                       torch.complex64, torch.complex128)):
            raise ValueError("Prepared ASM requires a nonempty 4D float32/64 or complex64/128 field")
        shape, dtype, device = tuple(field.shape[1:]), field.dtype, field.device
        offset = (0, 0) if offset is None or isinstance(offset, bool) else offset
        # H must remain usable by later autograd even if the caller is doing
        # inference now. Never cache an inference tensor or a parameter graph.
        with torch.inference_mode(False), torch.no_grad():
            pitch, wavelength, distance, shift = (
                _fixed_geometry(value, device) for value in (light.pitch, light.wvl, z, offset))
            if pitch.ndim != 0 or distance.numel() != 1 or shift.shape != (2,):
                raise ValueError("prepare requires scalar pitch/z and offset=(y, x)")
            if wavelength.numel() not in (1, shape[0]):
                raise ValueError("wavelength must be scalar or have one value per channel")
            if not bool(torch.isfinite(torch.cat([
                    value.reshape(-1) for value in (pitch, wavelength, distance, shift)])).all()):
                raise ValueError("Prepared ASM geometry must be finite")
            if not bool((pitch > 0) & (wavelength > 0).all()):
                raise ValueError("pitch and wavelength must be positive")
            identity = bool((distance == 0).all() & (shift == 0).all())
            transfer, pad = (None, None) if identity else self._angular_spectrum_transfer(
                light, distance.reshape(()), offset, linear, band_limit)

        def apply(field: torch.Tensor) -> torch.Tensor:
            if (not isinstance(field, torch.Tensor) or field.ndim != 4
                    or field.shape[0] < 1 or tuple(field.shape[1:]) != shape):
                raise ValueError("Prepared ASM field must have shape (batch, channels, rows, cols) matching preparation")
            if field.dtype != dtype or field.device != device:
                raise ValueError("Prepared ASM field dtype and device must match preparation")
            if transfer is None:
                return field
            return ifft(fft(field, pad_width=pad, shift=False) * transfer,
                        pad_width=pad, shift=False)

        return apply

    def forward(self, light: Light, z: float, offset: Tuple[float, float] = (0, 0), 
                linear: bool = True, band_limit: bool = True, b: float = 1, 
                target_plane: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                sampling_ratio: int = 1, vectorized: bool = False, steps: int = 100) -> Light:
        """Propagate incident light through the propagator.

        Args:
            light (Light): Incident light field
            z (float): Propagation distance in meters
            offset (tuple): Lateral shift (y, x) in meters for off-axis propagation
            linear (bool): Flag for linear convolution (with padding) or circular convolution (no padding)
            band_limit (bool): If True, apply band-limiting for unscaled ASM (b=1)
            b (float): Scaling factor for observation plane (b>1: expansion, b<1: focusing)
            target_plane (tuple, optional): (x, y, z) coordinates for RS diffraction
            sampling_ratio (int): Spatial sampling ratio for RS computation
            vectorized (bool): If True, use vectorized implementation for RS (better performance but higher memory usage)
            steps (int): Number of chunks for vectorized RS. More chunks reduce
                temporary kernel storage, but not total autograd saved tensors.

        Returns:
            Light: Propagated light field

        Examples:
            >>> # Basic propagation
            >>> prop = Propagator(mode="ASM")
            >>> light_prop = prop.forward(light, z=0.1)
            
            >>> # Propagation with padding
            >>> light_prop = prop.forward(light, z=0.1, linear=True)
            
            >>> # Vector field propagation
            >>> prop = Propagator(mode="Fresnel", polar="polar")
            >>> light_prop = prop.forward(light, z=0.05)
            >>> # Access x,y components
            >>> x_component = light_prop.get_lightX()
            >>> y_component = light_prop.get_lightY()
            
            >>> # Vectorized RS propagation for better performance
            >>> prop = Propagator(mode="RS")
            >>> light_prop = prop.forward(light, z=0.1, vectorized=True, steps=50)
        """
        if self.mode == "180":
            # Spatial-domain Light -> centered FFT -> k-space KSpaceLight.
            # z is unused for this mode (no plane-to-plane propagation).
            return self.forward_180(light)

        if (not isinstance(z, torch.Tensor) and z == 0
                and self.mode in ("ASM", "Fresnel", "RS")
                and isinstance(b, (int, float)) and b == 1 and _zero_offset(offset)
                and not (self.mode == "RS" and target_plane is not None)):
            # Preserve the numeric identity shortcut, but do not discard a
            # tensor distance's derivative or requested resampling/translation.
            return light

        if self.polar=='non':
            return self.forward_non_polar(light, z, offset, linear, band_limit, b, target_plane, sampling_ratio, vectorized, steps)
        elif self.polar=='polar':
            x = self.forward_non_polar(light.get_lightX(), z, offset, linear, band_limit, b, target_plane, sampling_ratio, vectorized, steps)
            y = self.forward_non_polar(light.get_lightY(), z, offset, linear, band_limit, b, target_plane, sampling_ratio, vectorized, steps)
            light.set_lightX(x)
            light.set_lightY(y)
            return light
        else:
            raise NotImplementedError('Polar is not set.')

    def forward_180(self, light: Light) -> KSpaceLight:
        """Convert a spatial-domain Light into a k-space KSpaceLight.

        Perform a centered 2D FFT of the input spatial field and represent the
        result on a wave-vector (kx, ky) grid derived from the input's spatial
        pitch and field size. This is the entry point for wide-angle / 180-degree
        propagation: (kx, ky) maps naturally to propagation direction (theta, phi).

        No propagation distance is applied: this mode performs only
        spatial complex field -> centered FFT -> KSpaceLight. Autograd is
        preserved through the FFT (the field is never detached).

        Only scalar Light with polar='non' is supported. PolarizedLight and
        other polarization modes raise NotImplementedError; no polarization
        components are silently combined. Angular sampling supports the
        forward hemisphere, not backward propagation.

        Args:
            light (Light): Input spatial-domain light field

        Returns:
            KSpaceLight: k-space field on a centered (kx, ky) grid
        """
        if self.polar != "non" or isinstance(light, PolarizedLight):
            raise NotImplementedError("mode='180' supports scalar Light with polar='non' only")
        U = light.get_field()
        B, Ch, R, C = U.shape

        # Centered FFT convention: zero-frequency component at the tensor center.
        A = fft(U)

        # Spatial frequency vectors (cycles/m), centered to match the FFT shift.
        fx = torch.fft.fftshift(torch.fft.fftfreq(C, d=light.pitch, device=U.device))
        fy = torch.fft.fftshift(torch.fft.fftfreq(R, d=light.pitch, device=U.device))

        # Wave-vector coordinates (radians/m).
        kx = 2 * torch.pi * fx
        ky = 2 * torch.pi * fy

        return KSpaceLight(
            dim=U.shape,
            kx=kx,
            ky=ky,
            wvl=light.wvl,
            field=A,
            source_pitch=light.pitch,
            source_dim=light.dim,
            device=light.device,
        )

    def forward_non_polar(self, light: Light, z: float, offset: Tuple[float, float] = (0, 0), 
                          linear: bool = True, band_limit: bool = True, b: float = 1, 
                          target_plane: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                          sampling_ratio: int = 1, vectorized: bool = False, steps: int = 100) -> Light:
        """Propagate non-polarized light field using selected propagation method.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            offset (tuple): Lateral shift (y, x) in meters for off-axis propagation
            linear (bool): If True, use linear convolution with zero-padding
            band_limit (bool): If True, apply band-limiting for ASM
            b (float): Scaling factor for observation plane (b>1: expansion, b<1: focusing)
            target_plane (tuple, optional): (x, y, z) coordinates for RS diffraction
            sampling_ratio (int): Spatial sampling ratio for RS computation
            vectorized (bool): If True, use vectorized implementation for RS
            steps (int): Number of computation steps for vectorized RS

        Returns:
            Light: Propagated light field
        """
        if self.mode == 'Fraunhofer':
            return self.forward_Fraunhofer(light, z, linear)
        if self.mode == 'Fresnel':
            return self.forward_Fresnel(light, z, linear)
        if self.mode == 'FFT':
            return self.forward_FFT(light, z)
        if self.mode == 'ASM':
            return self.forward_ASM(light, z, offset, linear, band_limit, b)
        if self.mode == 'RS':
            return self.forward_RayleighSommerfeld(light, z, target_plane, sampling_ratio, vectorized, steps)
        raise NotImplementedError(f'{self.mode} propagator is not implemented')

    def forward_Fraunhofer(self, light: Light, z: float, linear: bool = True) -> Light:
        """Physical Fraunhofer integral on a common isotropic output grid.

        The returned shape is unchanged. Padding doubles Fourier sampling;
        output pitch and field coordinates depend on wavelength and distance.
        Includes pixel area, carrier, amplitude and output quadratic phase.
        Tensor distance/wavelength gradients include the moving output grid.
        """
        from ._scalar_diffraction import fraunhofer
        return fraunhofer(light, z, linear)

    def forward_Fresnel(self, light: Light, z: float, linear: bool) -> Light:
        """Sampled paraxial propagation with unchanged pixel pitch.

        Uses H=exp(i*2*pi*z/wavelength-i*pi*wavelength*z*(fx**2+fy**2)).
        This replaces the formerly amplitude-normalized impulse convolution.
        Optional zero padding reduces circular wrap-around before cropping.
        """
        from ._scalar_diffraction import fresnel
        return fresnel(light, z, linear)

    def forward_FFT(self, light: Light, z: Optional[float] = None) -> Light:
        """Propagate light using simple FFT-based propagation.
        
        Apply exp(1j * phase) before FFT without considering propagation distance z
        or padding. Used for basic Fourier transform of the input field.

        Args:
            light (Light): Input light field
            z (float, optional): Not used in this method

        Returns:
            Light: FFT of input field
        """
        field = fft(torch.exp(1j * light.field.angle()))
        return Light(tuple(field.shape), light.pitch, light.wvl, field=field)
    
    def forward_ASM(self, light: Light, z: float, offset: Tuple[float, float] = (0, 0), 
                   linear: bool = True, band_limit: bool = True, b: float = 1) -> Light:
        """Select appropriate ASM propagation method based on parameters.

        Automatically choose between standard ASM, band-limited ASM, and scaled ASM
        depending on the scaling factor b and offset requirements.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            offset (tuple): Lateral shift (y, x) in meters
            linear (bool): If True, use linear convolution
            band_limit (bool): If True, apply band-limiting
            b (float): Scaling factor (b>1: expansion, b<1: focusing)

        Returns:
            Light: Propagated light field using selected ASM method
        """
        if not isinstance(b, (int, float)) or not 0 < b < float("inf"):
            raise ValueError("b must be a finite positive scalar")
        if b != 1:
            if not _zero_offset(offset):
                from ._propagation import real_tensor
                shift = real_tensor(offset, light.field.device)
                if shift.shape != (2,) or bool((shift != 0).any()):
                    raise ValueError("Scaled ASM does not support a nonzero offset")
                if shift.requires_grad:
                    raise ValueError("Scaled ASM does not support a differentiable offset")
            return self.forward_ScASM(light, z, b, linear) if b > 1 else self.forward_ScASM_focusing(light, z, b, linear)
        
        return self.forward_shifted_BL_ASM(light, z, offset, linear) if band_limit else self.forward_standard_ASM(light, z, offset, linear)

    def forward_standard_ASM(self, light: Light, z: float, offset: Tuple[float, float] = (0, 0),
                             linear: bool = True) -> Light:
        """Scalar ASM with double-precision geometry and evanescent decay.

        Complex field dtype is preserved. Negative z applies the stable adjoint:
        propagating phases reverse, while evanescent components still decay.
        It is not the unstable inverse of a decaying evanescent field.
        """
        return self._forward_angular_spectrum(light, z, offset, linear, False)

    def _angular_spectrum_transfer(self, light, z, offset, linear, band_limit):
        from ._propagation import angular_transfer, complex_dtype, frequency_axes, wavelength_tensor

        field = light.field
        if offset is None or isinstance(offset, bool):
            offset = (0, 0)
        pad = compute_pad_width(field, linear)
        rows = field.shape[-2] + pad[2] + pad[3]
        cols = field.shape[-1] + pad[0] + pad[1]
        fy, fx = frequency_axes(rows, cols, light.pitch, field.device, centered=False)
        wavelengths = wavelength_tensor(light)
        period = (rows*light.pitch, cols*light.pitch) if band_limit else None
        transfer = angular_transfer(fx, fy, wavelengths, z, offset,
                                    complex_dtype(field), fft_period=period,
                                    backend=self.backend, grid_pitch=light.pitch)
        return transfer, pad

    def _forward_angular_spectrum(self, light, z, offset, linear, band_limit):
        field = light.field
        transfer, pad = self._angular_spectrum_transfer(light, z, offset, linear, band_limit)
        # This translation-invariant operator commutes with the spatial shift.
        # Native frequency order avoids four full-field copies, including odd grids.
        output = ifft(fft(field, pad_width=pad, shift=False) * transfer,
                      pad_width=pad, shift=False)
        return Light(tuple(output.shape), light.pitch, light.wvl,
                     field=output, device=str(output.device))
    
    def forward_shifted_BL_ASM(self, light: Light, z: float, offset: Tuple[float, float] = (0, 0),
                              linear: bool = True) -> Light:
        """Shifted ASM with a mask derived from sampled phase-gradient limits.

        offset is (y, x) in metres. The Nyquist bound uses half of the actual
        FFT period, including padding. Evanescent bins are suppressed at
        nonzero distance. Hard support changes are nondifferentiable; parameter
        gradients describe the transfer within the locally fixed support.
        """
        return self._forward_angular_spectrum(light, z, offset, linear, True)

    def forward_ScASM(self, light: Light, z: float, b: float, linear: bool = True) -> Light:
        """Scaled angular spectrum on an isotropic output pitch b*input_pitch.

        Preserves field precision and supports odd rectangular arrays. Uses
        physical forward/inverse quadrature normalization and the same scaled
        inverse operator for expansion and focusing. Negative z is the stable
        adjoint for evanescent components, rather than their unstable inverse.
        """
        from ._scalar_diffraction import scaled_angular_spectrum
        return scaled_angular_spectrum(light, z, b, linear, backend=self.backend)

    def forward_ScASM_focusing(self, light: Light, z: float, b: float, linear: bool = True) -> Light:
        """Focusing variant of the same physically normalized scaled operator."""
        from ._scalar_diffraction import scaled_angular_spectrum
        return scaled_angular_spectrum(light, z, b, linear, backend=self.backend)

    def forward_RayleighSommerfeld(self, light: 'Light', z: float,
                                  target_plane: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                                  sampling_ratio: int = 1, vectorized: bool = False,
                                  steps: int = 100) -> 'Light':
        """Evaluate the scalar Rayleigh-Sommerfeld integral by pixel quadrature.

        Source coordinates are (index - size//2)*pitch, in metres. Batch and
        wavelength channels are independent. The returned field preserves the
        input complex dtype and device. This finite-grid quadrature is not an
        exact continuous diffraction solution; sampling convergence is required.

        target_plane is an optional tuple of finite, equally shaped 2D (x,y,z)
        tensors, with nonzero z. Output target coordinates are retained as
        result.target_plane. For explicit targets, result.pitch remains the
        input pitch as a fallback: the coordinate tensors are authoritative and
        must be used for nonuniform/tilted grids, rather than feeding the result
        directly to a uniform-grid propagator.

        sampling_ratio evaluates every nth target row/column and replicates
        the samples with nearest-neighbour blocks, including partial edge
        blocks. It does not subsample the source integral. vectorized selects
        chunked rather than one-target-at-a-time evaluation; steps is the
        requested number of chunks, bounded by the sampled target count.
        Both execution paths implement the same quadrature and sampling rule.
        """
        for name, value in (("sampling_ratio", sampling_ratio), ("steps", steps)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        field = light.get_field()
        if field.ndim != 4 or not field.is_complex():
            raise ValueError("RS requires a complex [B,Ch,R,C] field")
        B, Ch, R, C = field.shape
        device = field.device
        # Use double-precision geometry even for complex64 optical fields.
        x = (torch.arange(C, dtype=torch.float64, device=device) - C//2) * light.pitch
        y = (torch.arange(R, dtype=torch.float64, device=device) - R//2) * light.pitch
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        if target_plane is None:
            distance = torch.as_tensor(z, dtype=torch.float64, device=device)
            if distance.numel() != 1:
                raise ValueError("z must be a scalar")
            distance = distance.reshape(())
            targets = (xx, yy, torch.ones_like(xx) * distance)
        else:
            if not isinstance(target_plane, (tuple, list)) or len(target_plane) != 3:
                raise ValueError("target_plane must contain x, y, z tensors")
            if any(not isinstance(t, torch.Tensor) or not t.is_floating_point()
                   or t.ndim != 2 or t.numel() == 0 for t in target_plane):
                raise ValueError("target_plane requires nonempty real floating-point 2D tensors")
            if any(t.shape != target_plane[0].shape for t in target_plane):
                raise ValueError("target_plane tensors must have identical shapes")
            targets = tuple(t.to(device=device, dtype=torch.float64) for t in target_plane)
        if any(not torch.isfinite(t).all() for t in targets) or torch.any(targets[2] == 0):
            raise ValueError("target coordinates must be finite and target z must be nonzero")
        from ._propagation import real_tensor
        wavelengths = real_tensor(light.wvl, device).reshape(-1)
        if wavelengths.numel() not in (1, Ch) or not torch.isfinite(wavelengths).all() or torch.any(wavelengths <= 0):
            raise ValueError("wavelength must be finite, positive, and scalar or per-channel")
        wavelengths = wavelengths.expand(Ch)
        target_rows, target_cols = targets[0].shape
        sampled = tuple(t[::sampling_ratio, ::sampling_ratio] for t in targets)
        sr, sc = sampled[0].shape
        X, Y, Z = (t.reshape(-1, 1) for t in sampled)
        total = X.shape[0]
        chunk_size = max(1, (total + steps - 1)//steps) if vectorized else 1
        source = field.reshape(B, Ch, -1).to(torch.complex128)
        # Keep source coordinates as 1D views. Expanding the source meshgrid
        # into flattened arrays would retain two dense float64 coordinate
        # copies (64 MiB for a 2048-square source) across the kernel peak.
        source_x, source_y = x[None, None, :], y[None, :, None]
        wave_numbers = 2 * torch.pi / wavelengths
        pieces = []
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            dz = Z[start:stop]
            radius = torch.sqrt((X[start:stop, :, None]-source_x)**2
                                + (Y[start:stop, :, None]-source_y)**2
                                + dz[..., None]**2).reshape(stop-start, -1)
            channels = []
            for c in range(Ch):
                k = wave_numbers[c]
                kernel = (dz/radius) * (1-1/(1j*k*radius)) * torch.exp(1j*k*radius) / radius
                kernel = kernel * (k/(2*torch.pi*1j)) * light.pitch**2
                channels.append(source[:, c] @ kernel.transpose(0,1))
            pieces.append(torch.stack(channels, dim=1))
        output = torch.cat(pieces, dim=-1).reshape(B, Ch, sr, sc)
        if sampling_ratio != 1:
            output = output.repeat_interleave(sampling_ratio, -2).repeat_interleave(sampling_ratio, -1)
        output = output[..., :target_rows, :target_cols].to(field.dtype)
        result = Light(tuple(output.shape), light.pitch, light.wvl, field=output, device=str(device))
        result.target_plane = targets
        return result

    def _forward_RayleighSommerfeld_vectorized(self, light: 'Light', z: float,
                                              target_plane: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                                              steps: int = 100) -> 'Light':
        """Compatibility wrapper for chunked RS quadrature."""
        return self.forward_RayleighSommerfeld(light, z, target_plane, vectorized=True, steps=steps)
