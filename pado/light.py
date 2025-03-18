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
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from typing import Tuple, Union, List, Optional

import torch
import torch.nn.functional as F
import os

class Light:
    """Light wave with complex field wavefront.
    
    Represent a light wave with complex field wavefront that can be manipulated
    through various optical operations.
    """

    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, 
                 wvl: Union[float, List[float]], field: Optional[torch.Tensor] = None, 
                 device: str = 'cpu'):
        """Create light wave instance with complex field wavefront.

        Args:
            dim (tuple): Field dimensions (B, Ch, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            wvl (float or list): Wavelength in meters. Can be single value or list for multi-wavelength
            field (torch.Tensor, optional): Initial complex field [B, Ch, R, C]
            device (str): Device for computation ('cpu', 'cuda:0', etc.)

        Examples:
            >>> light = Light(dim=(1, 1, 1024, 1024), pitch=6.4e-6, wvl=633e-9)
            >>> light = Light(dim=(2, 3, 512, 512), pitch=2e-6, wvl=[450e-9, 550e-9, 650e-9])
        """
        if not isinstance(dim, tuple) or len(dim) != 4:
            raise ValueError(f"dim must be a 4-element tuple (B,Ch,R,C), got {dim}")
        if dim[0] < 1 or dim[1] < 1 or dim[2] < 1 or dim[3] < 1:
            raise ValueError(f"All dimensions must be positive, got {dim}")
        if not isinstance(pitch, (int, float)) or pitch <= 0:
            raise ValueError(f"pitch must be a positive number, got {pitch}")
        if not (isinstance(wvl, (int, float)) or (hasattr(wvl, "__iter__") and len(wvl) == dim[1])):
            raise ValueError(f"wvl must be a number or a list with length equal to channels ({dim[1]})")
        if hasattr(wvl, "__iter__") and any(w <= 0 for w in wvl):
            raise ValueError(f"All wavelengths must be positive")
        if isinstance(wvl, (int, float)) and wvl <= 0:
            raise ValueError(f"Wavelength must be positive, got {wvl}")
        
        self.dim: Tuple[int, int, int, int] = dim
        self.pitch: float = pitch
        self.device: str = device
        self.wvl: Union[float, List[float]] = wvl
    
        if field is None:
            field = torch.ones(dim, device=device, dtype=torch.cfloat)
        self.field: torch.Tensor = field

    def crop(self, crop_width: Tuple[int, int, int, int]) -> None:
        """Crop light wavefront.

        Args:
            crop_width (tuple): Crop dimensions (left, right, top, bottom)

        Examples:
            >>> light.crop((32, 32, 32, 32))
        """
        if not isinstance(crop_width, tuple) or len(crop_width) != 4:
            raise ValueError(f"crop_width must be a 4-element tuple (left, right, top, bottom), got {crop_width}")
        if any(w < 0 for w in crop_width):
            raise ValueError(f"All crop widths must be non-negative, got {crop_width}")
        if crop_width[0] + crop_width[1] >= self.dim[3] or crop_width[2] + crop_width[3] >= self.dim[2]:
            raise ValueError(f"Crop width {crop_width} too large for dimensions {self.dim}")
        
        self.field = self.field[..., crop_width[2]:-crop_width[3], crop_width[0]:-crop_width[1]]
        # Update dim as a new tuple since tuples are immutable
        dim_list = list(self.dim)
        dim_list[2], dim_list[3] = self.field.shape[2], self.field.shape[3]
        self.dim = tuple(dim_list)

    def clone(self) -> 'Light':
        """Create deep copy of light instance.

        Returns:
            Light: New light instance with copied attributes

        Examples:
            >>> light_copy = light.clone()
        """
        return Light(self.dim, self.pitch, self.wvl, self.field.clone(), device=self.device)

    def pad(self, pad_width: Tuple[int, int, int, int], padval: int = 0) -> None:
        """Pad light field with constant value.

        Args:
            pad_width (tuple): Padding dimensions (left, right, top, bottom)
            padval (int): Padding value (only 0 supported)

        Examples:
            >>> light.pad((16, 16, 16, 16))  # Add 16 pixels padding on all sides
        """
        if not isinstance(pad_width, tuple) or len(pad_width) != 4:
            raise ValueError(f"pad_width must be a 4-element tuple (left, right, top, bottom), got {pad_width}")
        if any(w < 0 for w in pad_width):
            raise ValueError(f"All padding widths must be non-negative, got {pad_width}")
        
        if padval == 0:
            self.field = torch.nn.functional.pad(self.field, pad_width)
        else:
            raise NotImplementedError('only zero padding supported')
        dim_list = list(self.dim)
        dim_list[2] = dim_list[2] + pad_width[0] + pad_width[1]
        dim_list[3] = dim_list[3] + pad_width[2] + pad_width[3]
        self.dim = tuple(dim_list)

    def set_real(self, real: torch.Tensor, c: Optional[int] = None) -> None:
        """Set real part of light wavefront.

        Args:
            real (torch.Tensor): Real part in rectangular representation.
                If c is None: Expected shape is (B,Ch,R,C) matching self.field.real.shape
                If c is provided: Expected shape is (B,R,C) matching self.field[:, c, ...].real.shape
                where B=batch size, Ch=channels, R=rows, C=columns
            c (int, optional): Channel index to modify. If provided, only that specific channel will be updated.

        Examples:
            >>> # Set real part for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> real_part = torch.ones((1, 1, 1024, 1024))  # (B,Ch,R,C)
            >>> light.set_real(real_part)
            >>> 
            >>> # Set real part for only channel 0 (B=1, R=1024, C=1024)
            >>> real_part_channel0 = torch.ones((1, 1024, 1024))  # (B,R,C)
            >>> light.set_real(real_part_channel0, c=0)
        """
        if not isinstance(real, torch.Tensor):
            raise TypeError(f"real must be a torch.Tensor, got {type(real)}")
        
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            if real.shape != self.field[:, c, ...].real.shape:
                raise ValueError(f"Expected real tensor of shape {self.field[:, c, ...].real.shape}, got {real.shape}")
            imag_part = self.field[:, c, ...].imag.detach()
            self.field[:, c, ...] = torch.complex(real, imag_part)
        else:
            if real.shape != self.field.real.shape:
                raise ValueError(f"Expected real tensor of shape {self.field.real.shape}, got {real.shape}")
            imag_part = self.field.imag.detach()
            self.field = torch.complex(real, imag_part)

    def set_imag(self, imag: torch.Tensor, c: Optional[int] = None) -> None:
        """Set imaginary part of light wavefront.

        Args:
            imag (torch.Tensor): Imaginary part in rectangular representation.
                If c is None: Expected shape is (B,Ch,R,C) matching self.field.imag.shape
                If c is provided: Expected shape is (B,R,C) matching self.field[:, c, ...].imag.shape
                where B=batch size, Ch=channels, R=rows, C=columns
            c (int, optional): Channel index to modify. If provided, only that specific channel will be updated.

        Examples:
            >>> # Set imaginary part for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> imag_part = torch.ones((1, 1, 1024, 1024))  # (B,Ch,R,C)
            >>> light.set_imag(imag_part)
            >>> 
            >>> # Set imaginary part for only channel 0 (B=1, R=1024, C=1024)
            >>> imag_part_channel0 = torch.ones((1, 1024, 1024))  # (B,R,C)
            >>> light.set_imag(imag_part_channel0, c=0)
        """
        if not isinstance(imag, torch.Tensor):
            raise TypeError(f"imag must be a torch.Tensor, got {type(imag)}")
        
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            if imag.shape != self.field[:, c, ...].imag.shape:
                raise ValueError(f"Expected imag tensor of shape {self.field[:, c, ...].imag.shape}, got {imag.shape}")

            real_part = self.field[:, c, ...].real.detach()
            self.field[:, c, ...] = torch.complex(real_part, imag)
        else:
            if imag.shape != self.field.imag.shape:
                raise ValueError(f"Expected imag tensor of shape {self.field.imag.shape}, got {imag.shape}")
            real_part = self.field.real.detach()
            self.field = torch.complex(real_part, imag)
        
    def set_amplitude(self, amplitude: torch.Tensor, c: Optional[int] = None) -> None:
        """Set amplitude of light wavefront (keeps phase unchanged) without maintaining computation graph.
        
        Args:
            amplitude (torch.Tensor): Amplitude in polar representation.
            c (int, optional): Channel index to modify.
        """
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError(f"amplitude must be a torch.Tensor, got {type(amplitude)}")
        
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            if amplitude.shape != self.field[:, c, ...].shape:
                raise ValueError(f"Expected amplitude tensor of shape {self.field[:, c, ...].shape}, got {amplitude.shape}")
            
            phase = self.field[:, c, ...].angle().detach()
            self.field[:, c, ...] = amplitude * torch.exp(phase * 1j)
        else:
            if amplitude.shape != self.field.shape:
                raise ValueError(f"Expected amplitude tensor of shape {self.field.shape}, got {amplitude.shape}")
            
            phase = self.field.angle().detach()
            self.field = amplitude * torch.exp(phase * 1j)

    def set_phase(self, phase: torch.Tensor, c: Optional[int] = None) -> None:
        """Set phase of light wavefront (keeps amplitude unchanged).

        Args:
            phase (torch.Tensor): Phase in polar representation (in radians).
                If c is None: Expected shape is (B,Ch,R,C) matching self.field.shape
                If c is provided: Expected shape is (B,R,C) matching self.field[:, c, ...].shape
                where B=batch size, Ch=channels, R=rows, C=columns
            c (int, optional): Channel index to modify. If provided, only that specific channel will be updated.

        Examples:
            >>> # Set phase for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> phase = torch.zeros((1, 1, 1024, 1024))  # (B,Ch,R,C)
            >>> light.set_phase(phase)
            >>> 
            >>> # Set phase for only channel 0 (B=1, R=1024, C=1024)
            >>> phase_channel0 = torch.zeros((1, 1024, 1024))  # (B,R,C)
            >>> light.set_phase(phase_channel0, c=0)
        """
        if not isinstance(phase, torch.Tensor):
            raise TypeError(f"phase must be a torch.Tensor, got {type(phase)}")
        
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            if phase.shape != self.field[:, c, ...].shape:
                raise ValueError(f"Expected phase tensor of shape {self.field[:, c, ...].shape}, got {phase.shape}")
            
            amplitude = self.field[:, c, ...].abs().detach()
            self.field[:, c, ...] = amplitude * torch.exp(phase * 1j)
        else:
            if phase.shape != self.field.shape:
                raise ValueError(f"Expected phase tensor of shape {self.field.shape}, got {phase.shape}")
            
            amplitude = self.field.abs().detach()
            self.field = amplitude * torch.exp(phase * 1j)

    def set_field(self, field: torch.Tensor, c: Optional[int] = None) -> None:
        """Set complex field of light wavefront.

        Args:
            field (torch.Tensor): Complex field tensor.
                If c is None: Expected shape is (B,Ch,R,C) matching self.field.shape
                If c is provided: Expected shape is (B,R,C) matching self.field[:, c, ...].shape
                where B=batch size, Ch=channels, R=rows, C=columns
            c (int, optional): Channel index to modify. If provided, only that specific channel will be updated.

        Examples:
            >>> # Set field for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> field = torch.ones((1, 1, 1024, 1024), dtype=torch.complex64)  # (B,Ch,R,C)
            >>> light.set_field(field)
            >>> 
            >>> # Set field for only channel 0 (B=1, R=1024, C=1024)
            >>> field_channel0 = torch.ones((1, 1024, 1024), dtype=torch.complex64)  # (B,R,C)
            >>> light.set_field(field_channel0, c=0)
        """
        if not isinstance(field, torch.Tensor):
            raise TypeError(f"field must be a torch.Tensor, got {type(field)}")
        if not field.is_complex():
            raise TypeError(f"field must be a complex tensor (dtype=torch.cfloat)")
        
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            if field.shape != self.field[:, c, ...].shape:
                raise ValueError(f"Expected field tensor of shape {self.field[:, c, ...].shape}, got {field.shape}")
            self.field[:, c, ...] = field
        else:
            if field.shape != self.field.shape:
                raise ValueError(f"Expected field tensor of shape {self.field.shape}, got {field.shape}")
                
            self.field = field

    def set_pitch(self, pitch: float) -> None:
        """Set pixel pitch of light field.

        Args:
            pitch (float): New pixel pitch in meters

        Examples:
            >>> light.set_pitch(6.4e-6)  # Set 6.4μm pitch
        """
        if not isinstance(pitch, (int, float)) or pitch <= 0:
            raise ValueError(f"pitch must be a positive number, got {pitch}")
        self.pitch = pitch

    def get_channel(self) -> int:
        """Return number of channels in light field.

        Returns:
            int: Number of channels

        Examples:
            >>> channels = light.get_channel()
        """
        return self.dim[1]

    def get_amplitude(self, c: Optional[int] = None) -> torch.Tensor:
        """Get amplitude of light wavefront.

        Args:
            c (int, optional): Channel index to retrieve. If provided, only that specific channel will be returned.

        Returns:
            torch.Tensor: Amplitude in polar representation.
                If c is None: Shape is (B,Ch,R,C)
                If c is provided: Shape is (B,R,C)
                where B=batch size, Ch=channels, R=rows, C=columns

        Examples:
            >>> # Get amplitude for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> amp = light.get_amplitude()  # Shape: (1, 1, 1024, 1024)
            >>> 
            >>> # Get amplitude for only channel 0 (B=1, R=1024, C=1024)
            >>> amp_channel0 = light.get_amplitude(c=0)  # Shape: (1, 1024, 1024)
        """
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            return self.field[:, c, ...].abs()
        else:
            return self.field.abs()

    def get_phase(self, c: Optional[int] = None) -> torch.Tensor:
        """Get phase of light wavefront.

        Args:
            c (int, optional): Channel index to retrieve. If provided, only that specific channel will be returned.

        Returns:
            torch.Tensor: Phase in polar representation (in radians).
                If c is None: Shape is (B,Ch,R,C)
                If c is provided: Shape is (B,R,C)
                where B=batch size, Ch=channels, R=rows, C=columns

        Examples:
            >>> # Get phase for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> phase = light.get_phase()  # Shape: (1, 1, 1024, 1024)
            >>> 
            >>> # Get phase for only channel 0 (B=1, R=1024, C=1024)
            >>> phase_channel0 = light.get_phase(c=0)  # Shape: (1, 1024, 1024)
        """
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            return self.field[:, c, ...].angle()
        else:
            return self.field.angle()
        
    def get_intensity(self, c: Optional[int] = None) -> torch.Tensor:
        """Get intensity (amplitude squared) of light wavefront.

        Args:
            c (int, optional): Channel index to retrieve. If provided, only that specific channel will be returned.

        Returns:
            torch.Tensor: Intensity values.
                If c is None: Shape is (B,Ch,R,C)
                If c is provided: Shape is (B,R,C)
                where B=batch size, Ch=channels, R=rows, C=columns

        Examples:
            >>> # Get intensity for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> intensity = light.get_intensity()  # Shape: (1, 1, 1024, 1024)
            >>> 
            >>> # Get intensity for only channel 0 (B=1, R=1024, C=1024)
            >>> intensity_channel0 = light.get_intensity(c=0)  # Shape: (1, 1024, 1024)
        """
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            return (self.field[:, c, ...] * torch.conj(self.field[:, c, ...])).real
        else:
            return (self.field * torch.conj(self.field)).real

    def get_field(self, c: Optional[int] = None) -> torch.Tensor:
        """Get complex field of light wavefront.

        Args:
            c (int, optional): Channel index to retrieve. If provided, only that specific channel will be returned.

        Returns:
            torch.Tensor: Complex field tensor.
                If c is None: Shape is (B,Ch,R,C)
                If c is provided: Shape is (B,R,C)
                where B=batch size, Ch=channels, R=rows, C=columns

        Examples:
            >>> # Get field for all channels (B=1, Ch=1, R=1024, C=1024)
            >>> field = light.get_field()  # Shape: (1, 1, 1024, 1024), dtype=torch.complex64
            >>> 
            >>> # Get field for only channel 0 (B=1, R=1024, C=1024)
            >>> field_channel0 = light.get_field(c=0)  # Shape: (1, 1024, 1024), dtype=torch.complex64
        """
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            return self.field[:, c, ...]
        else:
            return self.field
        
    def get_device(self) -> str:
        """Return device of light field.

        Returns:
            str: Device name ('cpu', 'cuda:0', etc.)

        Examples:
            >>> device = light.get_device()
        """
        return self.device

    def get_bandwidth(self) -> Tuple[float, float]:
        """Return spatial bandwidth of light wavefront.

        Returns:
            tuple: Spatial height and width of wavefront in meters

        Examples:
            >>> height, width = light.get_bandwidth()
        """
        return self.pitch*self.dim[2], self.pitch*self.dim[3]
    
    def get_ideal_angle_limit(self) -> float:
        """Return ideal angle limit of light wavefront based on optical axis.

        Calculate the maximum diffraction angle supported by the current sampling,
        based on the Nyquist sampling criterion: sin(θ_max) = λ/(2·pitch).
        Use the shortest wavelength for multi-wavelength cases.
        
        Returns:
            float: Ideal angle limit in degrees

        Examples:
            >>> angle_limit = light.get_ideal_angle_limit()
        """
        if hasattr(self.wvl, "__iter__") and not isinstance(self.wvl, str):
            min_wvl = min(self.wvl)
        else:
            min_wvl = self.wvl

        sin_val = (min_wvl / self.pitch) * 0.5
        ideal_angle_limit = np.arcsin(sin_val) * 180 / np.pi
        
        return ideal_angle_limit - 0.0001

    def magnify(self, scale_factor: float, interp_mode: str = 'nearest', c: Optional[int] = None) -> None:
        """Change wavefront resolution without changing pixel pitch.

        Args:
            scale_factor (float): Scale factor for interpolation
            interp_mode (str): Interpolation method ('bilinear' or 'nearest')
            c (int, optional): Channel index to modify

        Examples:
            >>> light.magnify(2.0)  # Double resolution
            >>> light.magnify(0.5, interp_mode='bilinear')  # Half resolution
        """
        if not isinstance(scale_factor, (int, float)):
            raise TypeError(f"scale_factor must be a number, got {type(scale_factor)}")
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be positive, got {scale_factor}")
        if interp_mode not in ['nearest', 'bilinear']:
            raise ValueError(f"interp_mode must be 'nearest' or 'bilinear', got {interp_mode}")
        
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")
            self.field.real[:, c, ...] = F.interpolate(self.field.real[:, c, ...], 
                                                      scale_factor=scale_factor, mode=interp_mode)
            self.field.imag[:, c, ...] = F.interpolate(self.field.imag[:, c, ...], 
                                                      scale_factor=scale_factor, mode=interp_mode)
        else:
            self.field.real = F.interpolate(self.field.real, scale_factor=scale_factor, mode=interp_mode)
            self.field.imag = F.interpolate(self.field.imag, scale_factor=scale_factor, mode=interp_mode)
        self.dim = (self.dim[0], self.dim[1], self.field.shape[2], self.field.shape[3])

    def resize(self, target_pitch: float, interp_mode: str = 'nearest') -> None:
        """Resize wavefront by changing pixel pitch.

        Args:
            target_pitch (float): New pixel pitch in meters
            interp_mode (str): Interpolation method ('bilinear' or 'nearest')

        Examples:
            >>> light.resize(4e-6)  # Change pitch to 4μm
        """
        if not isinstance(target_pitch, (int, float)):
            raise TypeError(f"target_pitch must be a number, got {type(target_pitch)}")
        if target_pitch <= 0:
            raise ValueError(f"target_pitch must be positive, got {target_pitch}")
        if interp_mode not in ['nearest', 'bilinear']:
            raise ValueError(f"interp_mode must be 'nearest' or 'bilinear', got {interp_mode}")
        
        scale_factor = self.pitch / target_pitch
        self.magnify(scale_factor, interp_mode)
        self.set_pitch(target_pitch)

    def set_spherical_light(self, z: float, dx: float = 0.0, dy: float = 0.0) -> None:
        """Set spherical wavefront from point source.

        Args:
            z (float): Distance from source along optical axis
            dx (float): Lateral x-offset of source
            dy (float): Lateral y-offset of source

        Examples:
            >>> light.set_spherical_light(z=0.1)  # Source at 10cm
            >>> light.set_spherical_light(z=0.05, dx=1e-3)  # Offset source
        """
        if not isinstance(z, (int, float)):
            raise TypeError(f"z must be a number, got {type(z)}")
        if z == 0:
            raise ValueError("z cannot be zero (would cause division by zero)")
        if not isinstance(dx, (int, float)):
            raise TypeError(f"dx must be a number, got {type(dx)}")
        if not isinstance(dy, (int, float)):
            raise TypeError(f"dy must be a number, got {type(dy)}")
        
        # Create coordinate grids directly in PyTorch
        y = torch.arange(-self.dim[2]//2, self.dim[2]//2, device=self.device, dtype=torch.float32)
        x = torch.arange(-self.dim[3]//2, self.dim[3]//2, device=self.device, dtype=torch.float32)
        
        # Create 2D grid
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # Scale by pitch
        x_grid = x_grid * self.pitch
        y_grid = y_grid * self.pitch
        
        # Calculate distance from source to each point
        r = torch.sqrt((x_grid - dx) ** 2 + (y_grid - dy) ** 2 + z ** 2)
        
        # Calculate phase
        theta = 2 * torch.pi * r / self.wvl
        # Ensure 4D tensor shape [B, Ch, R, C]
        theta = theta.unsqueeze(0).unsqueeze(0) % (2*torch.pi)
        
        # Create magnitude (all ones)
        mag = torch.ones_like(theta)
        
        # Set the field
        self.set_field(mag * torch.exp(theta*1j))

    def set_plane_light(self, theta: float = 0) -> None:
        """Set plane wave with unit amplitude.

        Args:
            theta (float): Incident angle in degrees

        Examples:
            >>> light.set_plane_light(theta=15)  # 15° incident angle
        """
        if not isinstance(theta, (int, float)):
            raise TypeError(f"theta must be a number, got {type(theta)}")
        
        R, C = self.dim[-2], self.dim[-1]
        amplitude = torch.ones((1, 1, R, C), device=self.device)
        phase = torch.zeros((1, 1, R, C), device=self.device)

        # Create coordinate grids with correct dimensions
        x = torch.linspace(-C * self.pitch / 2, C * self.pitch / 2, C).to(self.device)
        y = torch.linspace(-R * self.pitch / 2, R * self.pitch / 2, R).to(self.device)
        
        # Use indexing='xy' to match expected behavior
        y_grid, x_grid = torch.meshgrid(y, x, indexing='xy')
        
        # Add batch and channel dimensions
        x_grid = x_grid[None, None, :, :]
        y_grid = y_grid[None, None, :, :]

        # Calculate phase term for plane wave at angle theta
        theta_rad = np.deg2rad(theta)
        term = -2 * torch.pi * x_grid * np.sin(theta_rad) / self.wvl
        phase = phase - term.to(torch.float32)

        self.set_field(amplitude * torch.exp(phase * 1j))

    def set_amplitude_ones(self) -> None:
        """Set amplitude to ones.

        Examples:
            >>> light.set_amplitude_ones()
        """
        self.set_amplitude(torch.ones_like(self.get_amplitude()))

    def set_amplitude_zeros(self) -> None:
        """Set amplitude to zeros.

        Examples:
            >>> light.set_amplitude_zeros()
        """
        self.set_amplitude(torch.zeros_like(self.get_amplitude()))

    def set_phase_zeros(self) -> None:
        """Set phase to zeros.

        Examples:
            >>> light.set_phase_zeros()
        """
        self.set_phase(torch.zeros_like(self.get_phase()))

    def set_phase_random(self, std: float = np.pi/2, distribution: str = 'gaussian', c: Optional[int] = None) -> None:
        """Set random phase with specified distribution.

        Args:
            std (float, optional): Standard deviation for phase randomness.
                - For gaussian: Standard deviation in radians
                - For uniform: Half-width of uniform distribution in radians
                - For von_mises: Inverse of concentration parameter (1/κ)
                Defaults to π/2.
            distribution (str, optional): Type of random distribution.
                Must be one of ['gaussian', 'uniform', 'von_mises'].
                Defaults to 'gaussian'.
            c (int, optional): Channel index to modify. If None, all channels are modified.

        Raises:
            ValueError: If distribution is not one of the supported types.

        Examples:
            >>> light.set_phase_random()  # Default gaussian with π/2 std for all channels
            >>> light.set_phase_random(std=np.pi/4)  # Reduced randomness
            >>> light.set_phase_random(distribution='uniform')  # Uniform distribution
            >>> light.set_phase_random(std=0.25, distribution='von_mises')  # von Mises with κ=4
            >>> light.set_phase_random(c=0)  # Only modify first channel
        """
        if not isinstance(std, (int, float)):
            raise TypeError(f"std must be a number, got {type(std)}")
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        
        if distribution not in ['gaussian', 'uniform', 'von_mises']:
            raise ValueError("distribution must be one of ['gaussian', 'uniform', 'von_mises']")

        # Check channel index if provided
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= self.dim[1]:
                raise IndexError(f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels")

        # Determine which channels to process
        channels = [c] if c is not None else range(self.dim[1])
        
        for channel in channels:
            # Get the shape needed for this specific channel
            if c is not None:
                # For specific channel, we need [B, H, W] 
                target_shape = (self.dim[0], self.dim[2], self.dim[3])
            else:
                # For all channels, we keep full tensor shape [B, Ch, H, W]
                target_shape = self.dim
            
            if distribution == 'uniform':
                # Create uniform random values between -std and std
                if c is not None:
                    phase = (torch.rand(target_shape, device=self.device) - 0.5) * (2 * std)
                else:
                    phase = (torch.rand(target_shape, device=self.device) - 0.5) * (2 * std)
                
            elif distribution == 'gaussian':
                # Create Gaussian random values with std deviation
                if c is not None:
                    phase = torch.randn(target_shape, device=self.device) * std
                else:
                    phase = torch.randn(target_shape, device=self.device) * std
                
            elif distribution == 'von_mises':
                # von Mises distribution implementation
                # μ = 0 (mean direction)
                # κ = 1/std (concentration parameter)
                kappa = torch.tensor(1/std, device=self.device)
                
                if c is not None:
                    u1 = torch.rand(target_shape, device=self.device)
                    u2 = torch.rand(target_shape, device=self.device)
                else:
                    u1 = torch.rand(target_shape, device=self.device)
                    u2 = torch.rand(target_shape, device=self.device)
                
                a = 1 + torch.sqrt(1 + 4 * kappa**2)
                b = (a - torch.sqrt(2 * a)) / (2 * kappa)
                r = (1 + b**2) / (2 * b)
                
                while True:
                    z = torch.cos(np.pi * u1)
                    f = (1 + r * z) / (r + z)
                    c_param = kappa * (r - f)
                    
                    accept = c_param * (2 - c_param) - u2 > 0
                    if accept.all():
                        break
                    
                    mask = ~accept
                    u1[mask] = torch.rand_like(u1[mask])
                    u2[mask] = torch.rand_like(u2[mask])
                
                phase = torch.sign(u2 - 0.5) * torch.arccos(f)

            if c is not None:
                # For a specific channel, we need to correctly reshape the phase
                # to match the expected shape in set_phase method
                self.field[:, channel, ...] = self.field[:, channel, ...].abs() * torch.exp(phase * 1j)
            else:
                self.set_phase(phase)

    def save(self, fn: str) -> None:
        """Save light field to file.

        Args:
            fn (str): Filename (.pt, .pth, .npy or .mat)

        Examples:
            >>> light.save("field.pt")  # Save as PyTorch format
            >>> light.save("field.npy")  # Save as NumPy format
            >>> light.save("field.mat")  # Save as MATLAB format
        """
        if not isinstance(fn, str):
            raise TypeError(f"Filename must be a string, got {type(fn)}")
        if not fn:
            raise ValueError("Filename cannot be empty")
        
        if fn[-2:] == 'pt' or fn[-3:] == 'pth':
            # Save in PyTorch format
            torch.save({
                'field': self.field,
                'dim': self.dim,
                'pitch': self.pitch,
                'wvl': self.wvl,
                'device': self.device
            }, fn)
        else:
            # Save in NumPy/MATLAB format
            field_np = self.get_field().data.cpu().numpy()
            if fn[-3:] == 'npy':
                np.save(fn, field_np)
            elif fn[-3:] == 'mat':
                savemat(fn, {'field':field_np})
            else:
                raise ValueError(f'Unsupported file extension in {fn}. Use .pt, .pth, .npy, or .mat')
        
        print(f'light saved to {fn}\n')

    def adjust_amplitude_to_other_light(self, other_light: 'Light') -> None:
        """Scale amplitude to match average of another light field.

        Args:
            other_light (Light): Reference light field

        Examples:
            >>> target = Light(dim=(1,1,1024,1024), pitch=6.4e-6, wvl=633e-9)
            >>> light.adjust_amplitude_to_other_light(target)
        """
        if not isinstance(other_light, Light):
            raise TypeError(f"other_light must be a Light instance, got {type(other_light)}")
        
        other_avg = torch.mean(other_light.get_amplitude())
        current_avg = torch.mean(self.get_amplitude())
        
        # Check for division by zero
        if current_avg == 0 or torch.isclose(current_avg, torch.tensor(0.0, device=current_avg.device)):
            raise ValueError("Current amplitude average is zero or very close to zero. Cannot scale.")
            
        scale_factor = other_avg / current_avg
        self.set_amplitude(self.get_amplitude() * scale_factor)

    def load_image(self, image_path: str, random_phase: bool = False, std: float = np.pi, distribution: str = 'uniform', batch_idx: Optional[int] = None) -> None:
        """Load image as amplitude pattern with optional random phase.

        Args:
            image_path (str): Path to image file.
            random_phase (bool, optional): Whether to apply random phase.
                Defaults to False.
            std (float, optional): Standard deviation for phase randomness.
                For gaussian: Standard deviation in radians
                For uniform: Half-width of uniform distribution in radians  
                For von_mises: Inverse of concentration parameter (1/κ)
                Defaults to π.
            distribution (str, optional): Type of random distribution.
                Must be one of ['gaussian', 'uniform', 'von_mises'].
                Defaults to 'uniform'.
            batch_idx (int, optional): Specific batch index to load the image into.
                If None, loads the image into all batches. Defaults to None.

        Examples:
            >>> light.load_image("target.png")  # No random phase, all batches
            >>> light.load_image("target.png", random_phase=True)  # Default uniform, all batches
            >>> light.load_image("target.png", random_phase=True, std=np.pi/4)
            >>> light.load_image("target.png", random_phase=True, distribution='gaussian')
            >>> light.load_image("target.png", batch_idx=0)  # Load only into first batch
        """
        if not isinstance(image_path, str):
            raise TypeError(f"image_path must be a string, got {type(image_path)}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not isinstance(random_phase, bool):
            raise TypeError(f"random_phase must be a boolean, got {type(random_phase)}")
        if not isinstance(std, (int, float)):
            raise TypeError(f"std must be a number, got {type(std)}")
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        if distribution not in ['gaussian', 'uniform', 'von_mises']:
            raise ValueError("distribution must be one of ['gaussian', 'uniform', 'von_mises']")
        if batch_idx is not None and (not isinstance(batch_idx, int) or batch_idx < 0 or batch_idx >= self.dim[0]):
            raise ValueError(f"batch_idx must be None or an integer in range [0, {self.dim[0]-1}], got {batch_idx}")
        
        try:
            img = plt.imread(image_path)
        except Exception as e:
            raise IOError(f"Error reading image file: {e}")
        
        img_tensor = torch.tensor(img, device=self.device, dtype=torch.float32)

        if img_tensor.max() > 1:
            img_tensor /= 255.0

        if img_tensor.shape[-1] == 4:
            img_tensor = img_tensor[..., :3]

        if self.dim[1] == 1:
            img_tensor = img_tensor.mean(dim=-1, keepdim=True)

        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor_resized = F.interpolate(img_tensor, size=(self.dim[2], self.dim[3]),
                                         mode='bilinear', align_corners=False)

        amplitude = torch.sqrt(img_tensor_resized)
        
        # Get current amplitude
        current_amplitude = self.get_amplitude()
        
        # Apply to specific batch or all batches
        if batch_idx is not None:
            # Apply to specific batch
            current_amplitude[batch_idx] = amplitude[0]
        else:
            # Apply to all batches by repeating the image
            amplitude = amplitude.repeat(self.dim[0], 1, 1, 1)
            current_amplitude = amplitude
            
        self.set_amplitude(current_amplitude)

        if random_phase:
            if batch_idx is not None:
                # Get current phase
                current_phase = self.get_phase()
                # Generate random phase for specific batch
                random_phase_tensor = self._generate_random_phase(std, distribution)
                current_phase[batch_idx] = random_phase_tensor[0]
                self.set_phase(current_phase)
            else:
                # Generate random phase for all batches
                self.set_phase_random(std=std, distribution=distribution)
        else:
            if batch_idx is not None:
                # Get current phase
                current_phase = self.get_phase()
                # Set zero phase for specific batch
                current_phase[batch_idx] = torch.zeros_like(amplitude[0])
                self.set_phase(current_phase)
            else:
                # Set zero phase for all batches
                self.set_phase(torch.zeros_like(amplitude))

    def visualize(self, b: int = 0, c: Optional[int] = None, uniform_scale: bool = False, 
                 vmin: Optional[float] = None, vmax: Optional[float] = None, 
                 fix_noise: bool = True, amp_threshold: float = 1e-5) -> None:
        """Visualize amplitude, phase, and intensity of light field with improved noise handling.

        Args:
            b (int): Batch index
            c (int, optional): Channel index
            uniform_scale (bool): Use same scale for all channels
            vmin, vmax (float, optional): Intensity plot range
            fix_noise (bool): Whether to fix numerical noise in amplitude visualization
            amp_threshold (float): Threshold for detecting uniform amplitude with noise

        Examples:
            >>> light.visualize()  # Show all channels
            >>> light.visualize(c=0)  # Show only first channel
            >>> light.visualize(uniform_scale=True)  # Use uniform scaling
            >>> light.visualize(fix_noise=False)  # Don't fix numerical noise
            >>> light.visualize(b=1)  # Visualize the second batch
        """
        if not isinstance(b, int) or b < 0 or b >= self.dim[0]:
            raise ValueError(f"Batch index b must be in range [0, {self.dim[0]-1}], got {b}")
        if c is not None and (not isinstance(c, int) or c < 0 or c >= self.dim[1]):
            raise ValueError(f"Channel index c must be in range [0, {self.dim[1]-1}], got {c}")
        if not isinstance(uniform_scale, bool):
            raise TypeError(f"uniform_scale must be a boolean, got {type(uniform_scale)}")
            
        bw = self.get_bandwidth()
        
        # Handle c parameter - ensure it's either None or an integer
        if c is not None and not isinstance(c, int):
            raise TypeError("Channel index 'c' must be an integer or None")
            
        channels = [c] if c is not None else range(self.get_channel())
        vmin_amplitude, vmax_amplitude = None, None

        # Move data to CPU once to avoid multiple transfers
        amplitude_all = self.get_amplitude().cpu()
        phase_all = self.get_phase().cpu()
        intensity_all = self.get_intensity().cpu()
        
        # Fix numerical noise in amplitude if requested
        if fix_noise and self.device != 'cpu':
            for chan in channels:
                # Check amplitude data
                amp = amplitude_all[b, chan].numpy().squeeze()
                
                # Check if amplitude is nearly uniform (std/mean below threshold)
                amp_mean = np.mean(amp)
                amp_std = np.std(amp)
                
                if amp_mean > 0 and amp_std / amp_mean < amp_threshold:
                    # Nearly uniform amplitude - remove noise
                    amplitude_all[b, chan] = torch.ones_like(amplitude_all[b, chan]) * amp_mean

        for chan in channels:
            plt.figure(figsize=(20, 5))
            
            # Amplitude
            plt.subplot(131)
            amplitude = amplitude_all[b, chan].numpy().squeeze() if c is not None else amplitude_all[b, chan].numpy().squeeze()
            if uniform_scale and vmin_amplitude is not None and vmax_amplitude is not None:
                plt.imshow(amplitude, extent=[0, float(bw[1]*1e3), 0, float(bw[0]*1e3)],
                          cmap='inferno', vmin=vmin_amplitude, vmax=vmax_amplitude)
            else:
                vmin_amplitude, vmax_amplitude = amplitude.min(), amplitude.max()
                plt.imshow(amplitude, extent=[0, float(bw[1]*1e3), 0, float(bw[0]*1e3)],
                          cmap='inferno', vmin=vmin_amplitude, vmax=vmax_amplitude)
            plt.title(f'Amplitude (Batch {b}, Channel {chan})\n[{vmin_amplitude:.2e}, {vmax_amplitude:.2e}]')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.colorbar()

            # Phase
            plt.subplot(132)
            phase = phase_all[b, chan].numpy().squeeze() if c is not None else phase_all[b, chan].numpy().squeeze()
            plt.imshow(phase, extent=[0, float(bw[1]*1e3), 0, float(bw[0]*1e3)],
                      cmap='hsv', vmin=-np.pi, vmax=np.pi)
            plt.title(f'Phase (Batch {b}, Channel {chan})')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.colorbar()

            # Intensity
            plt.subplot(133)
            intensity = intensity_all[b, chan].numpy().squeeze() if c is not None else intensity_all[b, chan].numpy().squeeze()
            vmin_intensity = intensity.min()
            vmax_intensity = intensity.max()
            plt.imshow(intensity, extent=[0, float(bw[1]*1e3), 0, float(bw[0]*1e3)],
                      cmap='inferno',
                      vmin=vmin_intensity, vmax=vmax_intensity)
            plt.title(f'Intensity (Batch {b}, Channel {chan})\n[{vmin_intensity:.2e}, {vmax_intensity:.2e}]')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.colorbar()
            
            wvl_text = f'{self.wvl[chan]*1e9:.2f} [nm]' if isinstance(self.wvl, list) else f'{self.wvl*1e9:.2f} [nm]'
            pitch_text = f'{float(self.pitch)*1e6:.2f}' if isinstance(self.pitch, (np.ndarray, list)) else f'{self.pitch*1e6:.2f}'
            plt.suptitle(f'Batch {b}: ({self.dim[2]},{self.dim[3]}), pitch:{pitch_text} [um], wvl:{wvl_text}, device:{self.device}')
            plt.tight_layout()
            plt.show()

    def visualize_image(self, b: int = 0) -> None:
        """Visualize amplitude, phase, and intensity as RGB images.

        Args:
            b (int): Batch index to visualize

        Examples:
            >>> light.visualize_image()
            >>> light.visualize_image(b=1)  # Visualize the second batch
        """
        if not isinstance(b, int) or b < 0 or b >= self.dim[0]:
            raise ValueError(f"Batch index b must be in range [0, {self.dim[0]-1}], got {b}")
            
        bw = self.get_bandwidth()

        # Move data to CPU once to avoid multiple transfers
        amplitude_cpu = self.get_amplitude().data.cpu()
        phase_cpu = self.get_phase().data.cpu()
        intensity_cpu = self.get_intensity().data.cpu()

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        plt.subplots_adjust(wspace=0.25)

        # Amplitude as RGB, clamped to [0, 1]
        amplitude = amplitude_cpu[b, ...].permute(1, 2, 0).squeeze()
        amplitude_clamped = torch.clamp(amplitude, min=0, max=1)  # Clamp values to [0, 1]
        img0 = axes[0].imshow(amplitude_clamped, extent=[0, bw[1]*1e3, 0, bw[0]*1e3])
        axes[0].set_title(f'Amplitude as RGB Image (Batch {b})')
        axes[0].set_xlabel('mm')
        axes[0].set_ylabel('mm')

        # Phase as RGB, normalized from -π to π
        phase = phase_cpu[b, ...].permute(1, 2, 0).squeeze()
        # Normalize from 0 to 1 for color mapping
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        img1 = axes[1].imshow(phase_normalized, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], 
                              cmap='hsv')
        axes[1].set_title(f'Phase as RGB Image (Batch {b})')
        axes[1].set_xlabel('mm')

        # Intensity as RGB, clamped to [0, 1]
        intensity = intensity_cpu[b, ...].permute(1, 2, 0).squeeze()
        intensity_clamped = torch.clamp(intensity, min=0, max=1)  # Clamp values to [0, 1]
        img2 = axes[2].imshow(intensity_clamped, extent=[0, bw[1]*1e3, 0, bw[0]*1e3])
        axes[2].set_title(f'Intensity as RGB Image (Batch {b})')
        axes[2].set_xlabel('mm')
        axes[2].set_ylabel('mm')

        # Format wavelengths for display
        if isinstance(self.wvl, list):
            wvl_text = ', '.join([f'{w/1e-9:.2f}[nm]' for w in self.wvl])
        else:
            wvl_text = f'{self.wvl/1e-9:.2f}[nm]'

        plt.suptitle(
            f'Batch {b}: ({self.dim[2]},{self.dim[3]}), ' 
            f'pitch: {self.pitch/1e-6:.2f}[um], '
            f'wvl: {wvl_text}, '
            f'device: {self.device}'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        plt.show()
        
    def shape(self) -> Tuple[int, int, int, int]:
        """Return shape of light wavefront.

        Returns:
            tuple: Shape of the field tensor

        Examples:
            >>> shape = light.shape()
        """
        return self.field.shape

    def load(self, fn: str) -> None:
        """Load light field from file.
        
        Args:
            fn (str): Filename (.pt, .pth, .npy or .mat)
        
        Examples:
            >>> light.load("field.pt")  # Load PyTorch format
            >>> light.load("field.npy")  # Load NumPy format
            >>> light.load("field.mat")  # Load MATLAB format
        """
        if not isinstance(fn, str):
            raise TypeError(f"Filename must be a string, got {type(fn)}")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"File not found: {fn}")
        
        try:
            if fn[-2:] == 'pt' or fn[-3:] == 'pth':
                # Load PyTorch format
                checkpoint = torch.load(fn, map_location=self.device)
                self.field = checkpoint['field']
                self.dim = checkpoint['dim']
                self.pitch = checkpoint['pitch']
                self.wvl = checkpoint['wvl']
            else:
                # Load NumPy/MATLAB format
                if fn[-3:] == 'npy':
                    field_np = np.load(fn)
                elif fn[-3:] == 'mat':
                    field_np = loadmat(fn)['field']
                else:
                    raise ValueError(f'Unknown file extension: {fn}')
                
                self.field = torch.tensor(field_np, device=self.device)
        except Exception as e:
            raise IOError(f"Error loading file {fn}: {e}")
        
        print(f'light loaded from {fn}\n')

class PolarizedLight(Light):
    """Light wave with polarized complex field wavefront.
    
    Represent a light wave with X and Y polarization components that can be 
    manipulated through various optical operations.
    """

    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, wvl: float, 
                 fieldX: Optional[torch.Tensor] = None, fieldY: Optional[torch.Tensor] = None, 
                 device: str = 'cuda:0') -> None:
        """Create polarized light wave instance with X and Y field components.

        Args:
            dim (tuple): Field dimensions (B, Ch, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            wvl (float): Wavelength in meters
            fieldX (torch.Tensor, optional): Initial X-polarized field [B, Ch, R, C]
            fieldY (torch.Tensor, optional): Initial Y-polarized field [B, Ch, R, C]
            device (str): Device for computation ('cpu', 'cuda:0', etc.)

        Examples:
            >>> light = PolarizedLight(dim=(1, 1, 1024, 1024), pitch=6.4e-6, wvl=633e-9)
            >>> light = PolarizedLight(dim=(2, 1, 512, 512), pitch=2e-6, wvl=532e-9,
            ...                       fieldX=torch.ones(2,1,512,512), 
            ...                       fieldY=torch.zeros(2,1,512,512))
        """
        if not isinstance(dim, tuple) or len(dim) != 4:
            raise ValueError(f"dim must be a 4-element tuple (B,Ch,R,C), got {dim}")
        if dim[0] < 1 or dim[1] < 1 or dim[2] < 1 or dim[3] < 1:
            raise ValueError(f"All dimensions must be positive, got {dim}")
        if not isinstance(pitch, (int, float)) or pitch <= 0:
            raise ValueError(f"pitch must be a positive number, got {pitch}")
        if not isinstance(wvl, (int, float)) or wvl <= 0:
            raise ValueError(f"wvl must be a positive number, got {wvl}")
        
        self.dim = dim
        self.pitch = pitch
        self.device = device
        self.wvl = wvl
        
        # Check fieldX and fieldY if provided
        if fieldX is not None:
            if not isinstance(fieldX, torch.Tensor):
                raise TypeError(f"fieldX must be a torch.Tensor, got {type(fieldX)}")
            if fieldX.shape != dim:
                raise ValueError(f"fieldX shape {fieldX.shape} must match dim {dim}")
            if not fieldX.is_complex():
                raise TypeError("fieldX must be a complex tensor")
        
        if fieldY is not None:
            if not isinstance(fieldY, torch.Tensor):
                raise TypeError(f"fieldY must be a torch.Tensor, got {type(fieldY)}")
            if fieldY.shape != dim:
                raise ValueError(f"fieldY shape {fieldY.shape} must match dim {dim}")
            if not fieldY.is_complex():
                raise TypeError("fieldY must be a complex tensor")
        
        fieldX = torch.ones(dim, device=device, dtype=torch.cfloat) if fieldX is None else fieldX 
        fieldY = torch.ones(dim, device=device, dtype=torch.cfloat) if fieldY is None else fieldY 
        self.lightX = Light(dim, pitch, wvl, fieldX, device)
        self.lightY = Light(dim, pitch, wvl, fieldY, device)

    def clone(self) -> 'PolarizedLight':
        """Create deep copy of polarized light instance.

        Returns:
            PolarizedLight: New polarized light instance with copied attributes

        Examples:
            >>> light_copy = light.clone()
        """
        return PolarizedLight(self.dim, self.pitch, self.wvl, self.get_fieldX().clone(), self.get_fieldX().clone(), device=self.device)

    def crop(self, crop_width: Tuple[int, int, int, int]) -> None:
        """Crop light wavefront.

        Args:
            crop_width (tuple): Crop dimensions (left, right, top, bottom)

        Examples:
            >>> light.crop((32, 32, 32, 32))
        """
        self.lightX.crop(crop_width)
        self.lightY.crop(crop_width)
        
        self.dim[2], self.dim[3] = self.lightX.dim[2], self.lightX.dim[3]

    def get_amplitude(self) -> torch.Tensor:
        """Return total amplitude of polarized field.
        
        Returns:
            torch.Tensor: Total amplitude
        """
        return torch.sqrt(self.get_intensityX() + self.get_intensityY())

    def get_amplitudeX(self) -> torch.Tensor:
        """Return amplitude of X-polarized field.
        
        Returns:
            torch.Tensor: Amplitude of X-polarized field
        """
        return self.lightX.get_amplitude()

    def get_amplitudeY(self) -> torch.Tensor:
        """Return amplitude of Y-polarized field.
        
        Returns:
            torch.Tensor: Amplitude of Y-polarized field
        """
        return self.lightY.get_amplitude()

    def get_field(self) -> torch.Tensor:
        """Return complex field.

        Returns:
            torch.Tensor: Complex field values for X and Y components

        Examples:
            >>> field = light.get_field()
        """
        x = self.lightX.get_field()
        y = self.lightY.get_field()
        return torch.stack((x, y), -1)

    def get_fieldX(self) -> torch.Tensor:
        """Return X-polarized field component.

        Returns:
            torch.Tensor: Complex field tensor for X polarization

        Examples:
            >>> x_field = light.get_fieldX()
        """
        return self.lightX.get_field()

    def get_fieldY(self) -> torch.Tensor:
        """Return Y-polarized field component.

        Returns:
            torch.Tensor: Complex field tensor for Y polarization

        Examples:
            >>> y_field = light.get_fieldY() 
        """
        return self.lightY.get_field()

    def get_imag(self) -> torch.Tensor:
        """Get imaginary part of field for both components.

        Returns:
            torch.Tensor: Imaginary part values for X and Y components

        Examples:
            >>> imag_parts = light.get_imag()
        """
        x = self.lightX.get_field().imag
        y = self.lightY.get_field().imag
        return torch.stack((x, y), -1)

    def get_intensity(self) -> torch.Tensor:
        """Return total intensity of polarized field.
        
        Returns:
            torch.Tensor: Total intensity
        """
        return self.get_intensityX() + self.get_intensityY()

    def get_intensityX(self) -> torch.Tensor:
        """Return intensity of X-polarized field.
        
        Returns:
            torch.Tensor: Intensity of X-polarized field
        """
        return self.lightX.get_intensity()

    def get_intensityY(self) -> torch.Tensor:
        """Return intensity of Y-polarized field.
        
        Returns:
            torch.Tensor: Intensity of Y-polarized field
        """
        return self.lightY.get_intensity()

    def get_lightX(self) -> Light:
        """Return X-polarized Light instance.

        Returns:
            Light: Light instance for X component

        Examples:
            >>> x_light = light.get_lightX()
        """
        return self.lightX

    def get_lightY(self) -> Light:
        """Return Y-polarized Light instance.

        Returns:
            Light: Light instance for Y component

        Examples:
            >>> y_light = light.get_lightY()
        """
        return self.lightY

    def get_phase(self) -> torch.Tensor:
        """Return phase of field.

        Returns:
            torch.Tensor: Phase values for X and Y components

        Examples:
            >>> phase = light.get_phase()
        """
        x = self.lightX.get_phase()
        y = self.lightY.get_phase()
        return torch.stack((x, y), -1)

    def get_phaseX(self) -> torch.Tensor:
        """Return phase of X-polarized field.
        
        Returns:
            torch.Tensor: Phase of X-polarized field
        """
        return self.lightX.get_phase()

    def get_phaseY(self) -> torch.Tensor:
        """Return phase of Y-polarized field.
        
        Returns:
            torch.Tensor: Phase of Y-polarized field
        """
        return self.lightY.get_phase()

    def get_real(self) -> torch.Tensor:
        """Get real part of field for both components.

        Returns:
            torch.Tensor: Real part values for X and Y components

        Examples:
            >>> real_parts = light.get_real()
        """
        x = self.lightX.get_field().real
        y = self.lightY.get_field().real
        return torch.stack((x, y), -1)

    def magnify(self, scale_factor: float, interp_mode: str = 'nearest') -> None:
        """Change wavefront resolution without changing pixel pitch.

        Args:
            scale_factor (float): Scale factor for interpolation
            interp_mode (str): Interpolation method ('bilinear', 'nearest')

        Examples:
            >>> light.magnify(2.0, 'bilinear')
        """
        self.lightX.magnify(scale_factor, interp_mode)
        self.lightY.magnify(scale_factor, interp_mode)
        self.dim[2], self.dim[3] = self.lightX.dim[2], self.lightY.dim[3]

    def pad(self, pad_width: Tuple[int, int, int, int], padval: complex = 0) -> None:
        """Pad light field.
        
        Args:
            pad_width (tuple): Padding width (left, right, top, bottom)
            padval (complex): Padding value (default: 0+0j)
        
        Examples:
            >>> light.pad((10, 10, 10, 10))
        """
        self.lightX.pad(pad_width, padval)
        self.lightY.pad(pad_width, padval)
        self.dim = self.lightX.dim

    def resize(self, new_pitch: float, interp_mode: str = 'nearest') -> None:
        """Resize light field to match new pixel pitch.
        
        Args:
            new_pitch (float): New pixel pitch in meters
            interp_mode (str): Interpolation mode ('nearest', 'bilinear')
        
        Examples:
            >>> light.resize(8e-6)
        """
        self.lightX.resize(new_pitch, interp_mode)
        self.lightY.resize(new_pitch, interp_mode)
        self.pitch = new_pitch
        self.dim = self.lightX.dim

    def set_amplitude(self, amplitude: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Set amplitude for both X and Y components.

        Args:
            amplitude (torch.Tensor or tuple): Amplitude values for both components
                                              or tuple of (amplitudeX, amplitudeY)

        Examples:
            >>> light.set_amplitude(torch.ones(1,1,512,512))
            >>> light.set_amplitude((x_amplitude, y_amplitude))
        """
        if isinstance(amplitude, tuple) and len(amplitude) == 2:
            self.set_amplitudeX(amplitude[0])
            self.set_amplitudeY(amplitude[1])
        else:
            self.set_amplitudeX(amplitude)
            self.set_amplitudeY(amplitude)
    
    def set_amplitudeX(self, amplitude: torch.Tensor) -> None:
        """Set amplitude for X component.

        Args:
            amplitude (torch.Tensor): Amplitude values for X component

        Examples:
            >>> light.set_amplitudeX(torch.ones(1,1,512,512))
        """
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError(f"amplitude must be a torch.Tensor, got {type(amplitude)}")
        if amplitude.shape != self.lightX.field.shape:
            raise ValueError(f"Expected amplitude shape {self.lightX.field.shape}, got {amplitude.shape}")
        
        phase = self.lightX.get_phase()
        self.lightX.set_field(amplitude * torch.exp(1j * phase))
    
    def set_amplitudeY(self, amplitude: torch.Tensor) -> None:
        """Set amplitude for Y component.

        Args:
            amplitude (torch.Tensor): Amplitude values for Y component

        Examples:
            >>> light.set_amplitudeY(torch.ones(1,1,512,512))
        """
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError(f"amplitude must be a torch.Tensor, got {type(amplitude)}")
        if amplitude.shape != self.lightY.field.shape:
            raise ValueError(f"Expected amplitude shape {self.lightY.field.shape}, got {amplitude.shape}")
        
        phase = self.lightY.get_phase()
        self.lightY.set_field(amplitude * torch.exp(1j * phase))

    def set_field(self, field: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set both X and Y field components.

        Args:
            field (tuple): Complex field values for X and Y components

        Examples:
            >>> light.set_field((x_field, y_field))
        """
        if not isinstance(field, tuple) or len(field) != 2:
            raise TypeError("field must be a tuple of (fieldX, fieldY)")
        
        fieldX, fieldY = field
        
        if not isinstance(fieldX, torch.Tensor) or not isinstance(fieldY, torch.Tensor):
            raise TypeError("Both field components must be torch.Tensor objects")
        
        if not fieldX.is_complex() or not fieldY.is_complex():
            raise TypeError("Both field components must be complex tensors")
        
        if fieldX.shape != self.lightX.field.shape or fieldY.shape != self.lightY.field.shape:
            raise ValueError(f"Expected field shapes {self.lightX.field.shape}, got {fieldX.shape} and {fieldY.shape}")
        
        self.lightX.set_field(field[0])
        self.lightY.set_field(field[1])

    def set_fieldX(self, field: torch.Tensor) -> None:
        """Set X-polarized field.

        Args:
            field (torch.Tensor): Complex field values

        Examples:
            >>> light.set_fieldX(torch.ones(1,1,512,512, dtype=torch.cfloat))
        """
        if not isinstance(field, torch.Tensor):
            raise TypeError(f"field must be a torch.Tensor, got {type(field)}")
        if not field.is_complex():
            raise TypeError("field must be a complex tensor")
        if field.shape != self.lightX.field.shape:
            raise ValueError(f"Expected field shape {self.lightX.field.shape}, got {field.shape}")
        
        self.lightX.set_field(field)

    def set_fieldY(self, field: torch.Tensor) -> None:
        """Set Y-polarized field.

        Args:
            field (torch.Tensor): Complex field values

        Examples:
            >>> light.set_fieldY(torch.ones(1,1,512,512, dtype=torch.cfloat))
        """
        if not isinstance(field, torch.Tensor):
            raise TypeError(f"field must be a torch.Tensor, got {type(field)}")
        if not field.is_complex():
            raise TypeError("field must be a complex tensor")
        if field.shape != self.lightY.field.shape:
            raise ValueError(f"Expected field shape {self.lightY.field.shape}, got {field.shape}")
        
        self.lightY.set_field(field)

    def set_imag(self, imag: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Set imaginary part for both X and Y components.

        Args:
            imag (torch.Tensor or tuple): Imaginary values for both components
                                         or tuple of (imagX, imagY)

        Examples:
            >>> light.set_imag(torch.zeros(1,1,512,512))
            >>> light.set_imag((x_imag, y_imag))
        """
        if isinstance(imag, tuple) and len(imag) == 2:
            self.set_imagX(imag[0])
            self.set_imagY(imag[1])
        else:
            self.set_imagX(imag)
            self.set_imagY(imag)

    def set_imagX(self, imag: torch.Tensor) -> None:
        """Set imaginary part of X-polarized field.

        Args:
            imag (torch.Tensor): Imaginary component values

        Examples:
            >>> light.set_imagX(torch.zeros(1,1,512,512))
        """
        self.lightX.set_imag(imag)

    def set_imagY(self, imag: torch.Tensor) -> None:
        """Set imaginary part of Y-polarized field.

        Args:
            imag (torch.Tensor): Imaginary component values

        Examples:
            >>> light.set_imagY(torch.zeros(1,1,512,512))
        """
        self.lightY.set_imag(imag)

    def set_lightX(self, light: Light) -> None:
        """Set X-polarized Light instance.

        Args:
            light (Light): Light instance for X component

        Examples:
            >>> light.set_lightX(x_light)
        """
        self.lightX = light

    def set_lightY(self, light: Light) -> None:
        """Set Y-polarized Light instance.

        Args:
            light (Light): Light instance for Y component

        Examples:
            >>> light.set_lightY(y_light)
        """
        self.lightY = light

    def set_phase(self, phase: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Set phase for both X and Y components.

        Args:
            phase (torch.Tensor or tuple): Phase values for both components
                                           or tuple of (phaseX, phaseY)

        Examples:
            >>> light.set_phase(torch.zeros(1,1,512,512))
            >>> light.set_phase((x_phase, y_phase))
        """
        if isinstance(phase, tuple) and len(phase) == 2:
            self.set_phaseX(phase[0])
            self.set_phaseY(phase[1])
        else:
            self.set_phaseX(phase)
            self.set_phaseY(phase)

    def set_phaseX(self, phase: torch.Tensor) -> None:
        """Set phase of X-polarized field.

        Args:
            phase (torch.Tensor): Phase values in radians

        Examples:
            >>> light.set_phaseX(torch.zeros(1,1,512,512))
        """
        self.lightX.set_phase(phase)

    def set_phaseY(self, phase: torch.Tensor) -> None:
        """Set phase of Y-polarized field.

        Args:
            phase (torch.Tensor): Phase values in radians

        Examples:
            >>> light.set_phaseY(torch.zeros(1,1,512,512))
        """
        self.lightY.set_phase(phase)

    def set_pitch(self, pitch: float) -> None:
        """Set pixel pitch.
        
        Args:
            pitch (float): Pixel pitch in meters
        """
        if not isinstance(pitch, (int, float)) or pitch <= 0:
            raise ValueError(f"pitch must be a positive number, got {pitch}")
        self.pitch = pitch
        self.lightX.set_pitch(pitch)
        self.lightY.set_pitch(pitch)

    def set_plane_light(self) -> None:
        """Set plane wave with unit amplitude and zero phase.

        Examples:
            >>> light.set_plane_light()
        """
        self.lightX.set_plane_light()
        self.lightY.set_plane_light()

    def set_real(self, real: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Set real part for both X and Y components.

        Args:
            real (torch.Tensor or tuple): Real values for both components
                                         or tuple of (realX, realY)

        Examples:
            >>> light.set_real(torch.ones(1,1,512,512))
            >>> light.set_real((x_real, y_real))
        """
        if isinstance(real, tuple) and len(real) == 2:
            self.set_realX(real[0])
            self.set_realY(real[1])
        else:
            self.set_realX(real)
            self.set_realY(real)

    def set_realX(self, real: torch.Tensor) -> None:
        """Set real part of X-polarized field.

        Args:
            real (torch.Tensor): Real component values

        Examples:
            >>> light.set_realX(torch.ones(1,1,512,512))
        """
        self.lightX.set_real(real)

    def set_realY(self, real: torch.Tensor) -> None:
        """Set real part of Y-polarized field.

        Args:
            real (torch.Tensor): Real component values

        Examples:
            >>> light.set_realY(torch.ones(1,1,512,512))
        """
        self.lightY.set_real(real)

    def set_spherical_light(self, z: float, dx: float = 0, dy: float = 0) -> None:
        """Set spherical wavefront from point source.

        Args:
            z (float): Z distance of source in meters
            dx (float): X offset of source in meters
            dy (float): Y offset of source in meters

        Examples:
            >>> light.set_spherical_light(0.1, dx=1e-3)
        """
        self.lightX.set_spherical_light(z, dx, dy)
        self.lightY.set_spherical_light(z, dx, dy)

    def shape(self) -> Tuple[int, int, int, int]:
        """Return shape of light wavefront.

        Returns:
            tuple: Shape of the field tensor

        Examples:
            >>> shape = light.shape()
        """
        return self.lightX.get_field().shape

    def visualize(self, b: int = 0, c: int = 0) -> None:
        """Visualize polarized light components.

        Args:
            b (int): Batch index to visualize
            c (int): Channel index to visualize

        Examples:
            >>> light.visualize(b=0, c=0)
        """
        bw = self.get_bandwidth()
        std = 3
        
        # Transfer data to CPU once for all visualizations
        amplitude_x_cpu = self.get_amplitudeX().data.cpu()[b, c, ...].squeeze()
        phase_x_cpu = self.get_phaseX().data.cpu()[b, c, ...].squeeze()
        intensity_x_cpu = self.get_intensityX().data.cpu()[b, c, ...].squeeze()
        
        amplitude_y_cpu = self.get_amplitudeY().data.cpu()[b, c, ...].squeeze()
        phase_y_cpu = self.get_phaseY().data.cpu()[b, c, ...].squeeze()
        intensity_y_cpu = self.get_intensityY().data.cpu()[b, c, ...].squeeze()
        
        amplitude_cpu = self.get_amplitude().data.cpu()[b, c, ...].squeeze()
        intensity_cpu = self.get_intensity().data.cpu()[b, c, ...].squeeze()
        
        # Calculate ratio once
        ratio = amplitude_x_cpu / amplitude_y_cpu
        
        plt.figure(figsize=(15, 11))
        
        plt.subplot(331)
        plt.imshow(amplitude_x_cpu, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno', 
                   vmin=0, vmax=amplitude_x_cpu.mean() + amplitude_x_cpu.std() * std)
        plt.title('amplitude X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(332)
        plt.imshow(phase_x_cpu.squeeze(),
                   extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(333)
        plt.imshow(intensity_x_cpu, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity_x_cpu.mean() + intensity_x_cpu.std() * std)
        plt.title('intensity X')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.subplot(334)
        plt.imshow(amplitude_y_cpu, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=amplitude_y_cpu.mean() + amplitude_y_cpu.std() * std)
        plt.title('amplitude Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(335)
        plt.imshow(phase_y_cpu.squeeze(),
                   extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(336)
        plt.imshow(intensity_y_cpu, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity_y_cpu.mean() + intensity_y_cpu.std() * std)
        plt.title('intensity Y')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.subplot(337)
        plt.imshow(amplitude_cpu, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=amplitude_cpu.mean() + amplitude_cpu.std() * std)
        plt.title('amplitude')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(338)
        plt.imshow(ratio.squeeze(),
                   extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='gray',
                   vmin=ratio.mean()-ratio.std()*5, vmax=ratio.mean()+ratio.std()*5)  # cyclic colormap
        plt.title('Ratio (X / Y)')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(339)
        plt.imshow(intensity_cpu, extent=[0, bw[1]*1e3, 0, bw[0]*1e3], cmap='inferno',
                   vmin=0, vmax=intensity_cpu.mean() + intensity_cpu.std() * std)
        plt.title('intensity')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()
        
        plt.suptitle(
            '(%d,%d), pitch:%.2f[um], wvl:%.2f[nm], device:%s' 
            % (self.dim[2], self.dim[3], self.pitch/1e-6, self.wvl/1e-9, self.device))
        plt.tight_layout()
        plt.show()