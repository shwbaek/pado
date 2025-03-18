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
from .light import Light


def compute_pad_width(field: torch.Tensor, linear: bool) -> Tuple[int, int, int, int]:
    """Compute padding width for FFT-based convolution.

    Args:
        field (torch.Tensor): Complex tensor of shape (B, Ch, R, C)
        linear (bool): Flag for linear convolution (with padding) or circular convolution (no padding)

    Returns:
        tuple: Padding width tuple
    """
    if linear:
        R, C = field.shape[-2:]
        pad_width = (C//2, C//2, R//2, R//2)
    else:
        pad_width = (0, 0, 0, 0)
    return pad_width 


def unpad(field_padded: torch.Tensor, pad_width: Tuple[int, int, int, int]) -> torch.Tensor:
    """Remove padding from a padded complex tensor.

    Args:
        field_padded (torch.Tensor): Padded complex tensor of shape (B, Ch, R, C)
        pad_width (tuple): Padding width tuple

    Returns:
        torch.Tensor: Unpadded complex tensor
    """
    field = field_padded[...,pad_width[2]:-pad_width[3],pad_width[0]:-pad_width[1]]
    return field


class Propagator:
    def __init__(self, mode: str, polar: str = 'non'):
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
            band_limit (bool): If True, apply band-limiting for ASM
            b (float): Scaling factor for observation plane (b>1: expansion, b<1: focusing)
            target_plane (tuple, optional): (x, y, z) coordinates for RS diffraction
            sampling_ratio (int): Spatial sampling ratio for RS computation
            vectorized (bool): If True, use vectorized implementation for RS (better performance but higher memory usage)
            steps (int): Number of computation steps for vectorized RS (higher values use less memory)

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
        if z == 0:
            # No propagation
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
        propagator_map: Dict[str, Callable[[], Light]] = {
            'Fraunhofer': lambda: self.forward_Fraunhofer(light, z, linear),
            'Fresnel': lambda: self.forward_Fresnel(light, z, linear),
            'FFT': lambda: self.forward_FFT(light, z),
            'ASM': lambda: self.forward_ASM(light, z, offset, linear, band_limit, b),
            'RS': lambda: self.forward_RayleighSommerfeld(light, z, target_plane, sampling_ratio, 
                                                         vectorized, steps)
        }
        
        if self.mode not in propagator_map:
            raise NotImplementedError(f'{self.mode} propagator is not implemented')
        
        return propagator_map[self.mode]()

    def forward_Fraunhofer(self, light: Light, z: float, linear: bool = True) -> Light:
        """Propagate light using Fraunhofer diffraction.

        Implement far-field diffraction with multi-wavelength support.
        The propagated field is independent of z distance, which only
        affect the output pixel pitch.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            linear (bool): If True, use linear convolution with zero-padding

        Returns:
            Light: Propagated light field
        """
        # Convert z to float if it's a tensor
        if isinstance(z, torch.Tensor):
            z = float(z.item())
        
        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * light.field.shape[1]  # Replicate the wavelength for each channel
        
        wavelengths = np.array(wavelengths)
        
        # Calculate bandwidth for original light
        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        
        # Calculate pixel pitch after propagation for each wavelength
        pitch_r_after_propagation = wavelengths * z / bw_r
        pitch_c_after_propagation = wavelengths * z / bw_c
        
        # Find minimum pixel pitch (highest resolution) among all wavelengths
        min_pitch_r = np.min(pitch_r_after_propagation)
        min_pitch_c = np.min(pitch_c_after_propagation)
        
        # Use the smaller of min_pitch_r and min_pitch_c as target pitch
        target_pitch = min(min_pitch_r, min_pitch_c)
        
        # Perform FFT for all channels at once
        field_propagated = fft(light.field)
        
        # Create propagated light object
        light_propagated = light.clone()
        light_propagated.set_field(field_propagated)
        
        # Calculate single uniform scaling factor based on smallest pitch
        if min_pitch_r <= min_pitch_c:
            scale_factor = float(min_pitch_r / target_pitch)
        else:
            scale_factor = float(min_pitch_c / target_pitch)
        
        # Apply single uniform scaling to all channels
        light_propagated.magnify(scale_factor, interp_mode='bilinear')
        light_propagated.set_pitch(target_pitch)
        
        return light_propagated

    def forward_Fresnel(self, light: Light, z: float, linear: bool) -> Light:
        """Propagate light using Fresnel diffraction.

        Implement Fresnel approximation with multi-wavelength support.
        Valid when z >> (x² + y²)_max/λ.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            linear (bool): If True, use linear convolution with zero-padding

        Returns:
            Light: Propagated light field
        """
        field_input = light.field
        light_propagated = light.clone()

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]

        # Convert wavelengths to tensor for broadcasting
        wavelengths = torch.tensor(wavelengths, device=light.device).view(1, -1, 1, 1)
        
        # Calculate pad width for all channels at once
        pad_width = compute_pad_width(field_input, linear)

        # Adjust spatial domain calculations based on 'linear'
        if linear:
            sx = light.dim[3]
            sy = light.dim[2]
            x = torch.arange(-sx, sx, 1, device=light.device)
            y = torch.arange(-sy, sy, 1, device=light.device)
        else:
            sx = light.dim[3] / 2
            sy = light.dim[2] / 2
            x = torch.arange(-sx, sx, 1, device=light.device)
            y = torch.arange(-sy, sy, 1, device=light.device)

        xx, yy = torch.meshgrid(x, y, indexing='xy')
        xx = (xx*light.pitch).to(light.device)
        yy = (yy*light.pitch).to(light.device)

        # Calculate phase and amplitude for all channels at once
        k = 2*np.pi/wavelengths  # [1, Ch, 1, 1]
        phase_u = (k*(xx**2 + yy**2)/(2*z))  # [1, Ch, R, C]
        phase_w = wrap_phase(phase_u, stay_positive=False)
        amplitude = torch.ones_like(phase_w) / z / wavelengths
        conv_kernel = amplitude * torch.exp(phase_w*1j)
        conv_kernel /= conv_kernel.abs().sum(dim=(-2, -1), keepdim=True)

        # Apply FFT and convolution for all channels at once
        H = fft(conv_kernel)
        F = fft(field_input, pad_width=pad_width)
        G = F * H
        field_propagated = ifft(G, pad_width=pad_width)

        light_propagated.set_field(field_propagated)
        return light_propagated
    
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
        field_input = light.field
        light_propagated = light.clone()

        # Apply exp(1j * phase) for all channels at once
        phase = field_input.angle()
        field_exp = torch.exp(1j * phase)

        # Perform forward FFT for all channels at once
        field_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(field_exp)))

        light_propagated.set_field(field_fft)
        return light_propagated
    
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
        if b > 1 or b < 1:
            if offset != (0, 0):
                print("Warning: b is ignored because offset is provided. Scaled ASM and shifted BL_ASM cannot be used simultaneously.")
                return self.forward_shifted_BL_ASM(light, z, offset, linear)
            return self.forward_ScASM(light, z, b, linear) if b > 1 else self.forward_ScASM_focusing(light, z, b, linear)
        
        return self.forward_shifted_BL_ASM(light, z, offset, linear) if band_limit else self.forward_standard_ASM(light, z, offset, linear)

    def forward_standard_ASM(self, light: Light, z: float, offset: Tuple[float, float] = (0, 0), 
                            linear: bool = True) -> Light:
        """Propagate light using standard Angular Spectrum Method.

        Implement basic ASM propagation with optional off-axis shift.
        Support multi-wavelength channels and linear/circular convolution.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            offset (tuple): Lateral shift (y, x) in meters
            linear (bool): If True, use linear convolution with zero-padding

        Returns:
            Light: Propagated light field
        """
        field_input = light.field
        light_propagated = light.clone()

        # Ensure offset is a valid tuple
        if offset is None or isinstance(offset, bool):
            offset = (0, 0)

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]

        # Convert wavelengths to tensor for broadcasting
        wavelengths = torch.tensor(wavelengths, device=light.device).view(1, -1, 1, 1)
        
        # Calculate pad width for all channels at once
        pad_width = compute_pad_width(field_input, linear)

        # Adjust frequency calculations based on 'linear'
        if linear:
            fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
            fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
        else:
            fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
            fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
        fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

        # Calculate transfer function for all channels at once
        k = 2 * torch.pi / wavelengths  # [1, Ch, 1, 1]
        gamma = torch.sqrt(torch.abs(1. - (wavelengths*fxx)**2 - (wavelengths*fyy)**2))
        gamma_offset = wavelengths*fxx*offset[1] + wavelengths*fyy*offset[0]
        H = torch.exp(1j*k*(z*gamma+gamma_offset))

        # Apply FFT and transfer function for all channels at once
        F = fft(field_input, pad_width=pad_width)
        G = F * H
        field_propagated = ifft(G, pad_width=pad_width)

        light_propagated.set_field(field_propagated)
        return light_propagated
    
    def forward_shifted_BL_ASM(self, light: Light, z: float, offset: Tuple[float, float] = (0, 0), 
                              linear: bool = True) -> Light:
        """Propagate light using band-limited Angular Spectrum Method.

        Implement shifted band-limited ASM for improved numerical stability in
        off-axis propagation. Based on Matsushima's method.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            offset (tuple): offset (y, x) between source and target plane in meters for off-axis propagation
            linear (bool): If True, use linear convolution with zero-padding

        Returns:
            Light: Propagated light field

        References:
            Matsushima, "Shifted angular spectrum method for off-axis numerical propagation"
        """
        field_input = light.field
        light_propagated = light.clone()

        # Ensure offset is a valid tuple
        if offset is None or isinstance(offset, bool):
            offset = (0, 0)

        # Convert z and offset to tensors
        z = torch.tensor(z, device=light.device)
        offset = torch.tensor(offset, device=light.device)

        # Check if wvl is a list or a single float, and adjust accordingly
        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * field_input.shape[1]

        # Convert wavelengths to tensor for broadcasting
        wavelengths = torch.tensor(wavelengths, device=light.device).view(1, -1, 1, 1)
        
        # Calculate pad width for all channels at once
        pad_width = compute_pad_width(field_input, linear)

        # Adjust frequency calculations based on 'linear'
        if linear:
            fx = torch.arange(-light.dim[3], light.dim[3], device=light.device) / (2*light.pitch * light.dim[3])
            fy = torch.arange(-light.dim[2], light.dim[2], device=light.device) / (2*light.pitch * light.dim[2])
        else:
            fx = torch.arange(-light.dim[3]//2, light.dim[3]//2, device=light.device) / (light.pitch * light.dim[3])
            fy = torch.arange(-light.dim[2]//2, light.dim[2]//2, device=light.device) / (light.pitch * light.dim[2])
        fxx, fyy = torch.meshgrid(fx, fy, indexing='xy')

        # Calculate transfer function for all channels at once
        k = 2 * torch.pi / wavelengths  # [1, Ch, 1, 1]
        gamma = torch.sqrt(torch.abs(1. - (wavelengths*fxx)**2 - (wavelengths*fyy)**2))

        # Define constants to compute bandlimit
        bw_x = light.dim[-1]*light.pitch
        bw_y = light.dim[-2]*light.pitch
        u_limit_plus = 1/torch.sqrt(1+torch.pow(z/(offset[-1]+bw_x+1e-7), 2))/wavelengths
        u_limit_minus = 1/torch.sqrt(1+torch.pow(z/(offset[-1]-bw_x+1e-7), 2))/wavelengths
        v_limit_plus = 1/torch.sqrt(1+torch.pow(z/(offset[-2]+bw_y+1e-7), 2))/wavelengths
        v_limit_minus = 1/torch.sqrt(1+torch.pow(z/(offset[-2]-bw_y+1e-7), 2))/wavelengths

        # Calculate bounds for all channels at once
        if offset[-1] > bw_x:
            fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_plus,2))
            fxx_lower_bound = fxx >= torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_minus,2))
        elif offset[-1] <= bw_x and offset[-1] >= -bw_x:
            fxx_upper_bound = fxx <= torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_plus,2)) - (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-1]==-bw_x else 0)
            fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_minus,2)) + (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-1]==bw_x else 0)
        else:
            fxx_upper_bound = fxx <= -torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_plus,2))
            fxx_lower_bound = fxx >= -torch.sqrt((1-torch.pow(fyy*wavelengths,2))*torch.pow(u_limit_minus,2))

        # fyy bound
        if offset[-2] > bw_y:
            fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_plus,2))
            fyy_lower_bound = fyy >= torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_minus,2))
        elif offset[-2] <= bw_y and offset[-2] >= -bw_y:
            fyy_upper_bound = fyy <= torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_plus,2)) - (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-2]==-bw_y else 0)
            fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_minus,2)) + (torch.sin(torch.tensor(np.radians(0.1), device=light.device))/wavelengths if offset[-2]==bw_y else 0)
        else:
            fyy_upper_bound = fyy <= -torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_plus,2))
            fyy_lower_bound = fyy >= -torch.sqrt((1-torch.pow(fxx*wavelengths,2))*torch.pow(v_limit_minus,2))

        uv_filter = fxx_upper_bound & fxx_lower_bound & fyy_upper_bound & fyy_lower_bound

        # Calculate transfer function with band-limiting
        gamma_offset = wavelengths*fxx*offset[1] + wavelengths*fyy*offset[0]
        H = torch.exp(1j*k*(z*gamma+gamma_offset)) * uv_filter

        # Apply FFT and transfer function for all channels at once
        F = fft(field_input, pad_width=pad_width)
        G = F * H
        field_propagated = ifft(G, pad_width=pad_width)

        light_propagated.set_field(field_propagated)
        return light_propagated

    def forward_ScASM(self, light: Light, z: float, b: float, linear: bool = True) -> Light:
        """Propagate light using scaled Angular Spectrum Method.

        This function perform a scaled forward angular spectrum propagation. It take an
        input optical field 'light', propagate it over a distance 'z', and scale the observation
        plane by a factor 'b' relative to the source plane. If 'linear' is True, zero-padding is
        applied to avoid wrap-around effects from FFT-based convolutions. 
        refer to M. Abedi, H. Saghafifar, and L. Rahimi, 
        "Improvement of optical wave propagation simulations: the scaled angular spectrum method 
        for far-field and focal analysis," Opt. Continuum 3, 935-947 (2024)

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            b (float): Scaling factor for observation plane (b>1)
            linear (bool): If True, use linear convolution with zero-padding

        Returns:
            Light: Propagated light field with scaled observation plane

        References:
            Abedi et al., "Improvement of optical wave propagation simulations: 
            the scaled angular spectrum method for far-field and focal analysis"
        """
        field_input = light.field
        B, Ch, R, C = field_input.shape

        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * Ch

        Lsrc = light.pitch * C
        Lobs = b * Lsrc
        delta_xobs = Lobs / C
        delta_yobs = Lobs / R


        field_output_all_channels = torch.zeros((B, Ch, R, C), dtype=torch.complex64, device=field_input.device)

        if linear:
            pad_width = (C//2, C//2, R//2, R//2)
            Rp = 2 * R
            Cp = 2 * C
        else:
            pad_width = (0,0,0,0)
            Rp = R
            Cp = C

        for chan in range(Ch):
            λ = wavelengths[chan]
            field_input_ch = field_input[0, chan, :, :].to(torch.complex64)

            if linear:
                field_input_ch = torch.nn.functional.pad(field_input_ch.unsqueeze(0).unsqueeze(0), pad=pad_width)
                field_input_ch = field_input_ch[0,0]

            Lsrc_padded = 2 * Lsrc if linear else Lsrc
            delta_xsrc_p = Lsrc_padded / Cp
            delta_ysrc_p = Lsrc_padded / Rp
            delta_fx_p = 1 / Lsrc_padded
            delta_fy_p = 1 / Lsrc_padded

            U = sc_dft_2d(field_input_ch, Cp, Rp, delta_xsrc_p, delta_ysrc_p, delta_fx_p, delta_fy_p)
            H = compute_scasm_transfer_function(Cp, Rp, delta_fx_p, delta_fy_p, λ, z).to(field_input.device)
            U_prop = U * H
            field_output_ch = sc_idft_2d(U_prop, Cp, Rp, delta_xobs, delta_yobs, delta_fx_p, delta_fy_p)

            if linear:
                field_output_ch = field_output_ch[R//2:R//2+R, C//2:C//2+C]
            field_output_all_channels[0, chan, :, :] = field_output_ch

        light_propagated = light.clone()
        light_propagated.field = field_output_all_channels
        light_propagated.set_pitch(delta_xobs)

        return light_propagated

    def forward_ScASM_focusing(self, light: 'Light', z: float, b: float, linear: bool = True) -> 'Light':
        """Propagate light using scaled ASM for focusing.

        Implement scaled ASM propagation for focusing to a smaller observation plane.
        Use FFT to transform to frequency domain, apply transfer function, then 
        use Sc-IDFT to resample field at smaller observation plane. Support
        multi-wavelength channels and linear/circular convolution.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters  
            b (float): Scaling factor (b<1 for focusing to smaller plane)
            linear (bool): If True, use linear convolution with zero-padding

        Returns:
            Light: Propagated light field at focusing plane
        """
        field_input = light.field
        B, Ch, R, C = field_input.shape

        if hasattr(light.wvl, "__iter__") and not isinstance(light.wvl, str):
            wavelengths = light.wvl
        else:
            wavelengths = [light.wvl] * Ch

        Lsrc = light.pitch * C
        Lobs = b * Lsrc

        if linear:
            Rp = 2 * R
            Cp = 2 * C
            pad_width = (C//2, C//2, R//2, R//2)
        else:
            Rp = R
            Cp = C
            pad_width = (0,0,0,0)

        field_output_all_channels = torch.zeros((B, Ch, R, C), dtype=torch.complex64, device=field_input.device)

        for chan in range(Ch):
            λ = wavelengths[chan]
            field_ch = field_input[0, chan, :, :]

            if linear:
                field_ch = torch.nn.functional.pad(field_ch.unsqueeze(0).unsqueeze(0), pad=pad_width)
                field_ch = field_ch[0,0]

            U = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field_ch)))

            Lsrc_padded = 2 * Lsrc if linear else Lsrc
            delta_fx_p = 1 / Lsrc_padded
            delta_fy_p = 1 / Lsrc_padded
            delta_xobs = Lobs / C
            delta_yobs = Lobs / R

            H = compute_scasm_transfer_function(Cp, Rp, delta_fx_p, delta_fy_p, λ, z).to(field_input.device)
            G = U * H
            u_obs = sc_idft_2d(G, Cp, Rp, delta_xobs, delta_yobs, delta_fx_p, delta_fy_p)

            if linear:
                u_obs = u_obs[R//2:R//2+R, C//2:C//2+C]

            field_output_all_channels[0, chan, :, :] = u_obs

        light_focus = light.clone()
        light_focus.set_field(field_output_all_channels)
        light_focus.set_pitch(Lobs / C)

        return light_focus

    def forward_RayleighSommerfeld(self, light: 'Light', z: float, target_plane: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                                  sampling_ratio: int = 1, vectorized: bool = False, steps: int = 100) -> 'Light':
        """Propagate light using Rayleigh-Sommerfeld diffraction.

        Implement exact scalar diffraction calculation. Computationally intensive
        but accurate for both near and far field. Support arbitrary target plane geometry.

        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            target_plane (tuple, optional): (x, y, z) coordinates of target plane points
            sampling_ratio (int): Spatial sampling ratio for computation (>1 for faster calculation)
            vectorized (bool): If True, use vectorized implementation for better performance
            steps (int): Number of computation steps for vectorized mode (higher values use less memory)

        Returns:
            Light: Propagated light field
        """
        if vectorized:
            return self._forward_RayleighSommerfeld_vectorized(light, z, target_plane, steps)
        
        field_input = light.field
        
        R, C = light.dim[2], light.dim[3]
        x = torch.linspace(-C//2, C//2, C, dtype=torch.float64) * light.pitch
        y = torch.linspace(-R//2, R//2, R, dtype=torch.float64) * light.pitch
        xx, yy = torch.meshgrid(x, y)
        xx = xx.to(light.device)
        yy = yy.to(light.device)
        
        if target_plane:
            xx_t, yy_t, zz_t = target_plane 
            R, C = xx_t.shape[0], xx_t.shape[1]
        else: 
            xx_t, yy_t, zz_t = xx.clone(), yy.clone(), torch.full(xx.shape, z, dtype=torch.float64, device=light.device) 
        
        light_propagated = Light((1, light.dim[1], R, C), pitch=(abs(xx_t[0, 0]-xx_t[-1, 0])/len(xx_t[0])).item(), 
                                 wvl=light.wvl, device=light.device)
        field_output = torch.zeros((1, light.dim[1], R//sampling_ratio, C//sampling_ratio), dtype=torch.complex128, device=light.device)

        # Iterate over each pixel in the output image
        for idx_y in range(0, R, sampling_ratio):
            for idx_x in range(0, C, sampling_ratio):
                X, Y, Z = xx_t[idx_y, idx_x], yy_t[idx_y, idx_x], zz_t[idx_y, idx_x]
                r = torch.sqrt((X-xx)**2 + (Y-yy)**2 + Z**2)

                # Handle multiple channels with different wavelengths
                for c in range(light.dim[1]):
                    k = 2 * np.pi / light.wvl[c] if isinstance(light.wvl, list) else 2 * np.pi / light.wvl
                    contribution = (Z/r) * (1-1./(1j*k*r)) * torch.exp(1j*k*r) / r * field_input[:, c:c+1, ...]
                    field_output[:, c:c+1, idx_y//sampling_ratio, idx_x//sampling_ratio] = torch.sum(contribution)
        
        # Apply wavelength-dependent scaling for each channel
        for c in range(light.dim[1]):
            k = 2 * np.pi / light.wvl[c] if isinstance(light.wvl, list) else 2 * np.pi / light.wvl
            field_output[:, c:c+1, ...] = k / (2*torch.pi*1j) * light.pitch**2 * field_output[:, c:c+1, ...]

        light_propagated.field.real = Func.interpolate(field_output.real, scale_factor=sampling_ratio)
        light_propagated.field.imag = Func.interpolate(field_output.imag, scale_factor=sampling_ratio)
        return light_propagated

    def _forward_RayleighSommerfeld_vectorized(self, light: 'Light', z: float, 
                                              target_plane: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None, 
                                              steps: int = 100) -> 'Light':
        """Vectorized implementation of RayleighSommerfeld propagation.
        
        Args:
            light (Light): Input light field
            z (float): Propagation distance in meters
            target_plane (tuple, optional): (x, y, z) coordinates of target plane points
            steps (int): How many steps to divide computation into to manage memory usage
            
        Returns:
            Light: Propagated light field
        """
        field_input = light.field
        light_propagated = light.clone()

        R, C = light.dim[2], light.dim[3]
        x = torch.linspace(-C//2, C//2, C, dtype=torch.float64, device=light.device) * light.pitch
        y = torch.linspace(-R//2, R//2, R, dtype=torch.float64, device=light.device) * light.pitch
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        xx = xx.to(light.device)
        yy = yy.to(light.device)

        if target_plane:
            xx_t, yy_t, zz_t = target_plane 
            R, C = xx_t.shape[0], xx_t.shape[1]
        else: 
            R, C = light.dim[2], light.dim[3]
            xx_t, yy_t = xx.clone(), yy.clone()
            zz_t = torch.full(xx.shape, z, dtype=torch.float64, device=light.device)
        
        # Flatten the target plane coordinates
        X = xx_t.flatten().to(light.device)
        Y = yy_t.flatten().to(light.device)
        Z = zz_t.flatten().to(light.device)

        # Expand the coordinate grids for broadcasting
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)
        X = X.unsqueeze(1).unsqueeze(1)
        Y = Y.unsqueeze(1).unsqueeze(1)
        Z = Z.unsqueeze(1).unsqueeze(1)

        field_output = torch.zeros((R*C, light.dim[1]), dtype=torch.complex128, device=light.device)
        step_length = R*C//steps

        for step in range(steps):
            start = step * step_length
            end = (step + 1) * step_length if step < steps - 1 else R * C
            r_partial = (torch.sqrt((X[start: end] - xx)**2
                                    + (Y[start: end] - yy)**2
                                    + Z[start: end]**2))

            # Handle multiple channels with different wavelengths
            for c in range(light.dim[1]):
                k = 2 * np.pi / light.wvl[c] if isinstance(light.wvl, list) else 2 * np.pi / light.wvl
                contribution_partial = ((Z[start: end] / r_partial) *
                                        (1 - 1./(1j * k * r_partial)) *
                                        torch.exp(1j * k * r_partial) / r_partial * field_input[:, c:c+1, ...])
                contribution_partial = contribution_partial.sum(dim=(-2, -1)).view((len(r_partial),))
                field_output[start:end, c] = contribution_partial

        field_output = field_output.view((R, C, light.dim[1])).permute(2, 0, 1).unsqueeze(0)
        
        # Apply wavelength-dependent scaling for each channel
        for c in range(light.dim[1]):
            k = 2 * np.pi / light.wvl[c] if isinstance(light.wvl, list) else 2 * np.pi / light.wvl
            field_output[:, c:c+1, ...] = k / (2 * torch.pi * 1j) * light.pitch**2 * field_output[:, c:c+1, ...]
        
        light_propagated.field = field_output
        
        if target_plane:
            new_pitch = (abs(xx_t[0, 0]-xx_t[-1, 0])/len(xx_t[0])).item()
            if new_pitch <= 0:
                new_pitch = light.pitch
                print(f"Warning: Calculated pitch was 0, keeping original pitch: {new_pitch}m")
        else:
            new_pitch = light.pitch
        
        light_propagated.set_pitch(new_pitch)
        return light_propagated