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
from numbers import Integral as _Integral
from typing import Tuple, Optional, Union, List, TYPE_CHECKING as _TYPE_CHECKING, Any as _Any

import torch
import torch.nn.functional as F

from .math import wrap_phase
from .math import nm, um, mm, cm, m
from .light import Light, _channel_wavelengths, _scalar_parameter
from .material import Material

if _TYPE_CHECKING:
    from numpy import ndarray as _NDArray
else:
    _NDArray = _Any


def _wavelengths_match(light, element):
    """Compare scalar/channel metadata without ambiguous tensor truth values."""
    if isinstance(light.wvl, (int, float)) and isinstance(element.wvl, (int, float)):
        return light.wvl == element.wvl
    channels = max(light.dim[1], element.dim[1])
    left = _channel_wavelengths(light.wvl, light.dim[1], light.device).expand(channels)
    right = _channel_wavelengths(element.wvl, element.dim[1], light.device).expand(channels)
    return torch.equal(left, right)


class OpticalElement:
    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, wvl: float, 
                 field_change: Optional[torch.Tensor] = None, device: str = 'cpu', 
                 name: str = "not defined", polar: str = 'non') -> None:
        """Base class for optical elements that modify incident light wavefront.

        The wavefront modification is stored as amplitude and phase tensors.
        Note that the number of channels is one for wavefront modulation.

        Args:
            dim (tuple): Dimensions (B, 1, R, C) for batch size, channels, rows, columns
            pitch (float): Pixel pitch in meters
            wvl (float): Wavelength of light in meters
            field_change (torch.Tensor, optional): Wavefront modification tensor [B, C, H, W]
            device (str): Device to store wavefront ('cpu', 'cuda:0', etc.)
            name (str): Name identifier for this optical element
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.field_change.shape
            torch.Size([1, 1, 100, 100])
        """
        self.name = name
        self.dim = dim
        self.pitch = pitch
        self.device = device
        if field_change is None:
            self.field_change = torch.ones(dim, dtype=torch.cfloat, device=device)
        else:
            self.field_change = field_change
        self.wvl = wvl
        self.polar = polar

    def _match_pitch(self, light, interp_mode):
        """Resample the coarser field onto the finer pitch."""
        if light.pitch > self.pitch:
            light.resize(self.pitch, interp_mode)
            light.set_pitch(self.pitch)
        elif light.pitch < self.pitch:
            self.resize(light.pitch, interp_mode)
            self.set_pitch(light.pitch)

    def _match_spatial_shape(self, light):
        """Pad rows then columns, aligning each floor(size/2) optical origin."""
        for axis in (2, 3):
            if light.dim[axis] == self.dim[axis]:
                continue
            smaller, larger = (light, self) if light.dim[axis] < self.dim[axis] else (self, light)
            leading = larger.dim[axis]//2 - smaller.dim[axis]//2
            trailing = larger.dim[axis] - smaller.dim[axis] - leading
            padding = (0, 0, leading, trailing) if axis == 2 else (leading, trailing, 0, 0)
            smaller.pad(padding)

    def forward(self, light: 'Light', interp_mode: str = 'nearest') -> 'Light':
        """Propagate incident light through the optical element.

        Args:
            light (Light): Input light field
            interp_mode (str): Interpolation method for resizing ('bilinear', 'nearest')

        Returns:
            Light: Light field after interaction with optical element

        Examples:
            >>> element = OpticalElement(dim=(1, 1, 64, 64), pitch=2e-6)
            >>> light = Light(dim=(1, 1, 64, 64), pitch=2e-6)
            >>> output = element.forward(light)
        """
        self._match_pitch(light, interp_mode)

        if self.polar=='non':
            return self.forward_non_polar(light, interp_mode)
        elif self.polar=='polar':
            x = self.forward_non_polar(light.get_lightX(), interp_mode)
            y = self.forward_non_polar(light.get_lightY(), interp_mode)
            light.set_lightX(x)
            light.set_lightY(y)
            return light
        else:
            raise NotImplementedError('Polar is not set.')

    def forward_non_polar(self, light: 'Light', interp_mode: str = 'nearest') -> 'Light':
        """Propagate non-polarized light through the optical element.

        Handles resolution matching between light and optical element by resizing and padding
        as needed. Applies the optical element's field modulation to the input light.

        Args:
            light (Light): Input light field to propagate through the element
            interp_mode (str): Interpolation method for resizing ('bilinear', 'nearest')

        Returns:
            Light: Modified light field after interaction with optical element

        Raises:
            ValueError: If wavelengths of light and element don't match
        """
        if not _wavelengths_match(light, self):
            raise ValueError(f'Wavelength mismatch: light wavelength {light.wvl} != element wavelength {self.wvl}')

        self._match_spatial_shape(light)

        light.set_field(light.field*self.field_change)

        return light

    def get_amplitude_change(self) -> torch.Tensor:
        """Return amplitude change of the wavefront.

        Returns:
            torch.Tensor: Amplitude change of the wavefront

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> amp = element.get_amplitude_change()
        """
        return self.field_change.abs()

    def get_device(self) -> str:
        """Returns the device on which tensors are stored.

        Returns:
            str: The device identifier (e.g., 'cpu', 'cuda:0').
        """
        return self.device

    def get_field_change(self) -> torch.Tensor:
        """Returns the field_change tensor.

        Returns:
            torch.Tensor: The field_change tensor representing amplitude and phase changes.
        """
        return self.field_change

    def get_name(self) -> str:
        """Returns the name of the optical element.

        Returns:
            str: The name identifier.
        """
        return self.name

    def get_phase_change(self) -> torch.Tensor:
        """Return phase change of the wavefront.

        Returns:
            torch.Tensor: Phase change of the wavefront

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> phase = element.get_phase_change()
        """
        return self.field_change.angle()

    def get_pitch(self) -> float:
        """Returns the pixel pitch.

        Returns:
            float: The pixel pitch in meters.
        """
        return self.pitch

    def get_polar(self) -> str:
        """Returns the polarization mode.

        Returns:
            str: The polarization mode ('non' for scalar, 'polar' for vector).
        """
        return self.polar

    def get_wvl(self) -> float:
        """Returns the wavelength.

        Returns:
            float: The wavelength in meters.
        """
        return self.wvl

    def pad(self, pad_width: Tuple[int, int, int, int], padval: float = 0) -> None:
        """Pad the wavefront change with constant value.

        Args:
            pad_width (tuple): Padding width following torch.nn.functional.pad format
            padval (float): Value to pad with, only 0 supported currently

        Raises:
            NotImplementedError: If padval is not 0

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.pad((10,10,10,10))  # Add 10 pixels padding on all sides
        """
        if padval == 0:
            # Create a new padded tensor instead of modifying in-place
            padded_field_change = torch.nn.functional.pad(self.field_change, pad_width)
            self.field_change = padded_field_change
        else:
            raise NotImplementedError('only zero padding supported')

        # Create a new dim tuple instead of modifying in-place
        self.dim = tuple(self.field_change.shape)

    def resize(self, target_pitch: float, interp_mode: str = 'nearest') -> None:
        """Resize the wavefront change by changing the pixel pitch.

        Args:
            target_pitch (float): New pixel pitch to use
            interp_mode (str): Interpolation method used in torch.nn.functional.interpolate
                - 'bilinear': Bilinear interpolation
                - 'nearest': Nearest neighbor interpolation

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.resize(1e-6)  # Resize to 1μm pitch
        """
        if not isinstance(target_pitch, (int, float)) or not _math.isfinite(target_pitch) or target_pitch <= 0:
            raise ValueError("target_pitch must be finite and positive")
        scale_factor = self.pitch / target_pitch
        self.field_change = torch.complex(
            F.interpolate(self.field_change.real, scale_factor=scale_factor, mode=interp_mode),
            F.interpolate(self.field_change.imag, scale_factor=scale_factor, mode=interp_mode))
        self.dim = tuple(self.field_change.shape)
        self.set_pitch(target_pitch)

    def set_amplitude_change(self, amplitude: torch.Tensor, c: Optional[int] = None) -> None:
        """Set amplitude change for specific or all channels.

        Args:
            amplitude (torch.Tensor): Amplitude change in polar representation
            c (int, optional): Channel index. If None, applies to all channels

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> amp = torch.ones((1,1,100,100))
            >>> element.set_amplitude_change(amp)
        """
        if c is not None:
            phase = self.field_change[:, c, ...].angle()
            # Create a new field change tensor instead of modifying in-place
            new_field_change = self.field_change.clone()
            new_field_change[:, c, ...] = amplitude * torch.exp(phase * 1j)
            self.field_change = new_field_change
        else:
            phase = self.field_change.angle()
            # Create a completely new tensor
            new_field_change = amplitude * torch.exp(phase * 1j)
            self.field_change = new_field_change

    def set_field_change(self, field_change: torch.Tensor, c: Optional[int] = None) -> None:
        """Set field change for specific or all channels.

        Args:
            field_change (torch.Tensor): Field change in complex tensor
            c (int, optional): Channel index. If None, applies to all channels

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> field = torch.ones((1,1,100,100), dtype=torch.cfloat)
            >>> element.set_field_change(field)
        """
        if c is not None:
            value = field_change.squeeze(1) if field_change.ndim == 4 and field_change.shape[1] == 1 else field_change
            new_field_change = self.field_change.clone()
            new_field_change[:, c, ...] = value
            self.field_change = new_field_change
        else:
            value = field_change.unsqueeze(1) if field_change.ndim == 3 else field_change
            self.field_change = torch.broadcast_to(value, self.dim).to(self.field_change.dtype).clone()

    def set_name(self, name: str) -> None:
        """Sets the name of the optical element.

        Args:
            name (str): The name identifier.
        """
        self.name = name

    def set_phase_change(self, phase: torch.Tensor, c: Optional[int] = None) -> None:
        """Set phase change for specific or all channels.

        Args:
            phase (torch.Tensor): Phase change in polar representation
            c (int, optional): Channel index. If None, applies to all channels

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> phase = torch.zeros((1,1,100,100))
            >>> element.set_phase_change(phase)
        """
        if c is not None:
            phase = phase.squeeze(1) if phase.ndim == 4 and phase.shape[1] == 1 else phase
            amplitude = self.field_change[:, c, ...].abs()
            new_field_change = self.field_change.clone()
            new_field_change[:, c, ...] = amplitude * torch.exp(phase * 1j)
            self.field_change = new_field_change
        else:
            phase = phase.unsqueeze(1) if phase.ndim == 3 else phase
            phase = torch.broadcast_to(phase, self.dim)
            self.field_change = (self.field_change.abs() * torch.exp(phase * 1j)).to(self.field_change.dtype)

    def set_pitch(self, pitch: float) -> None:
        """Set the pixel pitch of the complex tensor.

        Args:
            pitch (float): Pixel pitch in meters

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.set_pitch(1e-6)  # Set 1μm pitch
        """
        if pitch <= 0:
            raise ValueError(f"Pitch must be positive, got {pitch}")
        self.pitch = pitch

    def set_polar(self, polar: str) -> None:
        """Set polarization mode for the optical element.

        Args:
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.set_polar('polar')  # Set vector field mode
        """
        self.polar = polar

    def set_wvl(self, wvl: float) -> None:
        """Sets the wavelength.

        Args:
            wvl (float): The wavelength in meters.
        """
        self.wvl = wvl

    def shape(self) -> Tuple[int, int, int, int]:
        """Return shape of light-wavefront modulation.

        The number of channels is one for wavefront modulation.

        Returns:
            tuple: Dimensions (B, 1, R, C) for batch size, channels, rows, columns

        Examples:
            >>> element = OpticalElement((1,1,100,100), pitch=2e-6, wvl=500e-9)
            >>> element.shape()
            (1, 1, 100, 100)
        """
        return self.dim

    def visualize(self, b: int = 0, c: Optional[int] = None) -> None:
        """Visualize the wavefront modulation of the optical element.

        Displays amplitude and phase changes of the optical element's wavefront modulation.
        Creates subplots showing amplitude change and phase change for specified channels.

        Args:
            b (int, optional): Batch index to visualize. Defaults to 0.
            c (int, optional): Channel index to visualize. If None, visualizes all channels.
                Defaults to None.

        Examples:
            >>> lens = RefractiveLens((1,1,512,512), 2e-6, 0.1, 633e-9, 'cpu')
            >>> lens.visualize()  # Visualize first batch, all channels
            >>> lens.visualize(b=0, c=0)  # Visualize first batch, first channel
        """
        import matplotlib.pyplot as plt
        channels = [c] if c is not None else range(self.dim[1])

        for chan in channels:
            plt.figure(figsize=(13,6))
            plt.subplot(121)
            plt.imshow(self.get_amplitude_change().detach().cpu()[b,chan,...].squeeze(), cmap='inferno', vmin=0, vmax=1)
            plt.title('amplitude change')
            plt.colorbar()
            
            plt.subplot(122)
            plt.imshow(self.get_phase_change().detach().cpu()[b,chan,...].squeeze(), cmap='hsv', vmin=-_math.pi, vmax=_math.pi)
            plt.title('phase change')
            plt.colorbar()
            
            wvl_text = f'{self.wvl[chan]/nm:.2f}[nm]' if isinstance(self.wvl, list) else f'{self.wvl/nm:.2f}[nm]'
            plt.suptitle(
                f'{self.name}, '
                f'({self.dim[2]},{self.dim[3]}), '
                f'pitch:{self.pitch/um:.2f}[um], '
                f'wvl:{wvl_text}, '
                f'device:{self.device}'
            )

class RefractiveLens(OpticalElement):
    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, focal_length: float, 
                 wvl: Union[float, List[float]], device: str, polar: str = 'non', 
                 designated_wvl: Optional[float] = None) -> None:
        """Create a thin refractive lens optical element.

        Simulates a thin refractive lens that modifies the phase of incident light
        based on its focal length and wavelength.

        Args:
            dim (tuple): Shape of the lens field (B, Ch, R, C) where:
                B: Batch size
                Ch: Number of channels
                R: Number of rows
                C: Number of columns
            pitch (float): Pixel pitch in meters
            focal_length (float): Focal length of the lens in meters
            wvl (float or list): Wavelength(s) of light in meters. Can be single value or list for multi-channel
            device (str): Device to store the lens field ('cpu', 'cuda:0', etc.)
            polar (str, optional): Polarization mode. Defaults to 'non'
            designated_wvl (float, optional): Override wavelength for all channels. Defaults to None

        Examples:
            >>> # Create single channel lens
            >>> lens = RefractiveLens((1,1,512,512), 2e-6, 0.1, 633e-9, 'cpu')
            
            >>> # Create multi-channel lens with different wavelengths
            >>> lens = RefractiveLens((1,3,512,512), 2e-6, 0.1, [633e-9,532e-9,450e-9], 'cuda:0')
        """
        super().__init__(dim, pitch, wvl, None, device, name="refractive_lens", polar=polar)

        self.focal_length: Optional[float] = None

        if focal_length is None:
            raise ValueError("focal_length cannot be None")
            
        self.set_focal_length(focal_length)

        self.designated_wvl = designated_wvl
        wavelengths = _channel_wavelengths(self.wvl if designated_wvl is None else designated_wvl, dim[1], self.device)
        phase = self.compute_phase(wavelengths)
        self.set_field_change(torch.exp(1j * phase))

    def set_focal_length(self, focal_length: float) -> None:
        """Set the focal length of the lens.

        Args:
            focal_length (float): New focal length in meters

        Examples:
            >>> lens.set_focal_length(0.2)  # Set 20cm focal length
        """
        value = _scalar_parameter(focal_length, "focal_length", self.device)
        if value == 0:
            raise ValueError("focal_length cannot be zero")
        self.focal_length = value
        if hasattr(self, "designated_wvl"):
            wavelengths = _channel_wavelengths(self.wvl if self.designated_wvl is None else self.designated_wvl, self.dim[1], self.device)
            self.set_field_change(torch.exp(1j * self.compute_phase(wavelengths)))

    def compute_phase(self, wvl: float, shift_x: float = 0, shift_y: float = 0) -> torch.Tensor:
        """Compute the phase modulation for the lens.

        Calculates the phase change introduced by the lens based on its focal length,
        wavelength and any lateral shifts.

        Args:
            wvl (float): Wavelength of light in meters
            shift_x (float, optional): Displacement along rows in meters (historical convention). Defaults to 0
            shift_y (float, optional): Displacement along columns in meters (historical convention). Defaults to 0

        Returns:
            torch.Tensor: Phase modulation pattern of the lens

        Examples:
            >>> phase = lens.compute_phase(633e-9)  # Centered lens
            >>> phase = lens.compute_phase(633e-9, shift_x=10e-6)  # Shifted lens
        """
        # Historical shift_x/shift_y refer to rows/columns, respectively.
        x = (torch.arange(self.dim[2], device=self.device, dtype=torch.float64) - self.dim[2]//2) * self.pitch
        y = (torch.arange(self.dim[3], device=self.device, dtype=torch.float64) - self.dim[3]//2) * self.pitch
        radius_squared = (x[:, None] - shift_x)**2 + (y[None, :] - shift_y)**2
        wavelength = torch.as_tensor(wvl, device=self.device, dtype=torch.float64).reshape(-1)
        if wavelength.numel() not in (1, self.dim[1]) or not torch.isfinite(wavelength).all() or (wavelength <= 0).any():
            raise ValueError("wvl must contain positive finite scalar or channel wavelengths")
        phase = -torch.pi * radius_squared[None, None] / (wavelength[None, :, None, None] * self.focal_length)
        return wrap_phase(phase)


class CosineSquaredLens(OpticalElement):
    def __init__(self, dim: Tuple[int, int, int, int], pitch: float, focal_length: float, 
                 wvl: float, device: str, polar: str = 'non') -> None:
        """Lens with cosine squared phase distribution.

        Creates a lens with phase distribution of form [1+cos(k*r^2)]/2.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) - batch size, channels, rows, columns
            pitch (float): Pixel pitch in meters
            focal_length (float): Focal length in meters
            wvl (float): Wavelength in meters
            device (str): Device to store wavefront ('cpu', 'cuda:0', ...)
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> # Create basic cosine squared lens
            >>> lens = CosineSquaredLens((1,1,1024,1024), 2e-6, 0.1, 633e-9, 'cpu')
        """
        super().__init__(dim, pitch, wvl, None, device, name="cosine_squared_lens", polar=polar)
        
        self.focal_length: float = focal_length
        self.compute_and_set_phase_change()

    def compute_and_set_phase_change(self) -> None:
        """Compute and set the phase change induced by the lens.

        Calculates and applies phase change in range [0, π] to the lens.

        Examples:
            >>> lens.compute_and_set_phase_change()
        """
        wavelength = _channel_wavelengths(self.wvl, self.dim[1], self.device)
        k = 20 * torch.pi / wavelength  # Preserve the documented phase model.
        x = (torch.arange(self.dim[2], device=self.device, dtype=torch.float64) - self.dim[2]//2) * self.pitch
        y = (torch.arange(self.dim[3], device=self.device, dtype=torch.float64) - self.dim[3]//2) * self.pitch
        r_squared = x[:, None]**2 + y[None, :]**2
        phase = torch.pi * (1 + torch.cos(k[None, :, None, None] * r_squared[None, None])) / 2
        self.set_phase_change(phase)


def height2phase(height: float, wvl: float, RI: float, wrap: bool = True) -> torch.Tensor:
    """Convert material height to corresponding phase shift.

    Calculates phase shift from material height using wavelength and refractive index.

    Args:
        height (float): Height of material in meters
        wvl (float): Wavelength of light in meters
        RI (float): Refractive index of material at given wavelength
        wrap (bool): If True, wraps phase to [0,2π] range

    Returns:
        torch.Tensor: Phase change induced by material height

    Examples:
        >>> height = 500e-9  # 500nm height
        >>> phase = height2phase(height, 633e-9, 1.5)
    """
    dRI = RI - 1
    wv_n = 2. * _math.pi / wvl
    phi = wv_n * dRI * height
    if wrap:
        phi = wrap_phase(phi, stay_positive=True)
    return phi

def phase2height(phase_u: torch.Tensor, wvl: float, RI: float, minh: float = 0) -> torch.Tensor:
    """Convert phase change to material height.

    Note that phase to height mapping is not one-to-one.
    There exists an integer phase wrapping factor:
        height = wvl/(RI-1) * (phase_u/(2π) + i), where i is integer
    This function uses minimum height minh to constrain the conversion.
    Minimal height is chosen such that height is always >= minh.

    Args:
        phase_u (torch.Tensor): Phase change of light
        wvl (float): Wavelength of light in meters
        RI (float): Refractive index of material at given wavelength
        minh (float): Minimum height constraint in meters

    Returns:
        torch.Tensor: Material height that induces the phase change

    Examples:
        >>> phase = torch.ones((1,1,1024,1024)) * torch.pi
        >>> height = phase2height(phase, 633e-9, 1.5, minh=100e-9)  # 100nm min height
    """
    dRI = RI - 1
    if torch.any(torch.as_tensor(dRI) == 0):
        raise ValueError("Height cannot be recovered when RI equals the ambient index 1")
    period = wvl / dRI
    cycles = phase_u / (2 * torch.pi)
    if minh is not None:
        offset = minh / period - cycles
        i = torch.where(torch.as_tensor(period, device=cycles.device) > 0, torch.ceil(offset), torch.floor(offset))
    else:
        i = 0
    return period * (cycles + i)



class DOE(OpticalElement):
    def __init__(self, dim: tuple, pitch: float, material: 'Material', wvl: float, device: str, height: Optional[torch.Tensor] = None, phase_change: Optional[torch.Tensor] = None, polar: str = 'non'):
        """Diffractive optical element (DOE) that modifies incident light wavefront.

        The wavefront modification is determined by the material height profile.
        Supports both height and phase change specifications.

        Args:
            dim (tuple): Dimensions (B, 1, R, C) for batch size, channels, rows, columns
            pitch (float): Pixel pitch in meters
            material (Material): Material properties of the DOE
            wvl (float): Wavelength of light in meters
            device (str): Device to store wavefront ('cpu', 'cuda:0', etc.)
            height (torch.Tensor, optional): Height profile in meters
            phase_change (torch.Tensor, optional): Phase change profile
            polar (str): Polarization mode ('non': scalar, 'polar': vector)

        Examples:
            >>> # Create DOE with specified height profile
            >>> height = torch.ones((1,1,100,100)) * 500e-9  # 500nm height
            >>> doe = DOE(height.shape, 2e-6, material, 500e-9, 'cpu', height=height)
            
            >>> # Create DOE with specified phase profile
            >>> phase = torch.ones((1,1,100,100)) * torch.pi  # π phase
            >>> doe = DOE(phase.shape, 2e-6, material, 500e-9, 'cpu', phase_change=phase)
        """
        super().__init__(dim=dim, pitch=pitch, wvl=wvl, device=device, name="doe", polar=polar)

        self.material: 'Material' = material
        self.height: Optional[torch.Tensor] = None

        # initial DOE is tranparent and induces 0 phase delay
        super().set_field_change(torch.ones(dim,device=device)*torch.exp(1*torch.zeros(dim,device=device)))

        if (height is None) and (phase_change is not None):
            self.set_phase_change(phase_change, sync_height=True)
        elif (height is not None) and (phase_change is None):
            self.set_height(height, sync_phase=True)
        elif (height is None) and (phase_change is None):
            phase = torch.zeros(dim, device=device)
            self.set_phase_change(phase, sync_height=True)

    def visualize(self, b: int = 0, c: int = 0) -> None:
        """Visualize the DOE wavefront modulation.

        Displays amplitude change, phase change and height profile.

        Args:
            b (int): Batch index to visualize, defaults to 0
            c (int): Channel index to visualize, defaults to 0

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.visualize()  # Shows modulation plots
            >>> doe.visualize(b=1, c=0)  # Shows plots for batch index 1
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.imshow(self.get_amplitude_change().detach().cpu()[b,c,...].squeeze(),
                   cmap='inferno', vmin=0, vmax=1)
        plt.title('amplitude change')
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(self.get_phase_change().detach().cpu()[b,c,...].squeeze(),
                   cmap='hsv', vmin=-_math.pi, vmax=_math.pi)
        plt.title('phase change')
        plt.colorbar()
        
        plt.subplot(133)
        plt.imshow(self.get_height().detach().cpu()[b,c,...].squeeze()*1e6,
                   cmap='hot')
        plt.title('height [um]')
        plt.colorbar()
        
        plt.suptitle(
            f'{self.name}, '
            f'({self.dim[2]},{self.dim[3]}), '
            f'pitch:{self.pitch/1e-6:.2f}[um], '
            f'wvl:{self.wvl/1e-9:.2f}[nm], '
            f'device:{self.device}'
        )
        plt.show()

    def set_diffraction_grating_1d(self, slit_width: float, minh: float, maxh: float) -> None:
        """Set the wavefront modulation as a 1D diffraction grating.

        Create alternating height regions to form a binary phase grating.

        Args:
            slit_width (float): Width of each slit in meters
            minh (float): Minimum height in meters
            maxh (float): Maximum height in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_diffraction_grating_1d(10e-6, 0, 500e-9)  # 10μm slits
            >>> doe.visualize()  # Shows 1D grating pattern
        """
        slit_width_px = round(slit_width / self.pitch)
        if slit_width_px < 1:
            raise ValueError("slit_width must span at least one pixel after rounding")
        columns = torch.arange(self.dim[3], device=self.device) // slit_width_px
        high = (columns[None, :] % 2 == 0).expand(self.dim[-2:])
        low_height = _scalar_parameter(minh, "minh", self.device)
        high_height = _scalar_parameter(maxh, "maxh", self.device)
        height = torch.where(high, high_height, low_height)[None, None].expand(self.dim).clone()
        self.set_height(height, sync_phase=True)

    def set_diffraction_grating_2d(self, slit_width: float, minh: float, maxh: float) -> None:
        """Set the wavefront modulation as a 2D diffraction grating.

        Create a checkerboard pattern of alternating height regions.

        Args:
            slit_width (float): Width of each slit in meters
            minh (float): Minimum height in meters
            maxh (float): Maximum height in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_diffraction_grating_2d(10e-6, 0, 500e-9)  # 10μm slits
            >>> doe.visualize()  # Shows 2D grating pattern
        """
        slit_width_px = round(slit_width / self.pitch)
        if slit_width_px < 1:
            raise ValueError("slit_width must span at least one pixel after rounding")
        columns = torch.arange(self.dim[3], device=self.device) // slit_width_px
        rows = torch.arange(self.dim[2], device=self.device) // slit_width_px
        high = (rows[:, None] + columns[None, :]) % 2 == 0
        low_height = _scalar_parameter(minh, "minh", self.device)
        high_height = _scalar_parameter(maxh, "maxh", self.device)
        height = torch.where(high, high_height, low_height)[None, None].expand(self.dim).clone()
        self.set_height(height, sync_phase=True)

    def set_Fresnel_lens(self, focal_length: float, wvl: float, shift_x: float = 0, shift_y: float = 0) -> None:
        """Set the wavefront modulation as a Fresnel lens.

        Create a phase profile that focus light to a point.

        Args:
            focal_length (float): Focal length in meters
            wvl (float): Wavelength in meters
            shift_x (float): Horizontal shift in meters. Defaults to 0
            shift_y (float): Vertical shift in meters. Defaults to 0

        Examples:
            >>> doe = DOE((1,1,1024,1024), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_Fresnel_lens(0.1, 500e-9)  # f=10cm lens
            >>> doe.set_Fresnel_lens(0.1, 500e-9, shift_x=50e-6)  # Shifted lens
        """
        x = (torch.arange(self.dim[3], device=self.device, dtype=torch.float64) - self.dim[3]//2) * self.pitch
        y = (torch.arange(self.dim[2], device=self.device, dtype=torch.float64) - self.dim[2]//2) * self.pitch
        focal = _scalar_parameter(focal_length, "focal_length", self.device)
        if focal == 0:
            raise ValueError("focal_length cannot be zero")
        r2 = (x[None, :] - shift_x)**2 + (y[:, None] - shift_y)**2
        # Rationalization avoids catastrophic cancellation for r << abs(f).
        optical_path = torch.sign(focal) * r2 / (torch.sqrt(r2 + focal**2) + focal.abs())
        phase = -2 * torch.pi * optical_path / wvl
        self.set_phase_change(wrap_phase(phase)[None, None], sync_height=True)

    def set_Fresnel_zone_plate_lens(self, focal_length: float, wvl: float, shift_x: float = 0, shift_y: float = 0) -> None:
        """Set binary Fresnel zone plate pattern.

        Creates a binary phase plate with alternating 0 and π phase zones.
        Transmission amplitude remains one; this is not an opaque-zone mask.

        Args:
            focal_length (float): Focal length in meters
            wvl (float): Wavelength in meters
            shift_x (float): Horizontal shift in meters. Defaults to 0
            shift_y (float): Vertical shift in meters. Defaults to 0

        Examples:
            >>> doe = DOE((1,1,1024,1024), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_Fresnel_zone_plate_lens(0.1, 500e-9)  # f=10cm lens
            >>> doe.set_Fresnel_zone_plate_lens(0.1, 500e-9, shift_x=50e-6)  # Shifted lens
        """
        x = (torch.arange(self.dim[3], device=self.device, dtype=torch.float64) - self.dim[3]//2) * self.pitch
        y = (torch.arange(self.dim[2], device=self.device, dtype=torch.float64) - self.dim[2]//2) * self.pitch
        focal = _scalar_parameter(focal_length, "focal_length", self.device)
        if focal == 0:
            raise ValueError("focal_length cannot be zero")
        r2 = (x[None, :] - shift_x)**2 + (y[:, None] - shift_y)**2
        original_phase = -torch.pi * r2 / (wvl * focal)
        phase = torch.pi * (torch.cos(original_phase) >= 0).to(torch.float64)
        self.set_phase_change(phase[None, None], sync_height=True)

    def change_wvl(self, wvl: float) -> None:
        """Change the wavelength and update phase change.

        Args:
            wvl (float): New wavelength in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.change_wvl(633e-9)  # Change to 633nm wavelength
        """
        height = self.get_height()
        # Store the new wavelength
        self.wvl = wvl
        # Calculate the phase change for the new wavelength
        phase = height2phase(height, self.wvl, self.material.get_RI(self.wvl))
        # Create a new field change tensor
        new_field_change = torch.exp(phase*1j)
        self.set_field_change(new_field_change, sync_height=False)

            
    def sync_height_with_phase(self) -> None:
        """Synchronize height profile with current phase profile.

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_phase_change(phase, sync_height=False)
            >>> doe.sync_height_with_phase()  # Update height to match phase
        """
        height = phase2height(self.get_phase_change(), self.wvl, self.material.get_RI(self.wvl))
        self.set_height(height, sync_phase=False)

    def sync_phase_with_height(self) -> None:
        """Synchronize phase profile with current height profile.

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.set_height(height, sync_phase=False)
            >>> doe.sync_phase_with_height()  # Update phase to match height
        """
        phase = height2phase(self.get_height(), self.wvl, self.material.get_RI(self.wvl))
        self.set_phase_change(phase, sync_height=False)

    def resize(self, target_pitch: float) -> None:
        """Resize DOE with a new pixel pitch.

        Resize field from which DOE height is recomputed.

        Args:
            target_pitch (float): New pixel pitch in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> doe.resize(1e-6)  # Change pitch to 1μm
        """
        super().resize(target_pitch)  # this changes the field change 
        self.sync_height_with_phase()

    def get_height(self) -> torch.Tensor:
        """Return the height map of the DOE.

        Returns:
            torch.Tensor: Height map in meters

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> height = doe.get_height()  # Get current height profile
        """
        return self.height
    
    def set_phase_change(self, phase_change: torch.Tensor, sync_height: bool = True) -> None:
        """Set phase change induced by the DOE.

        Args:
            phase_change (torch.Tensor): Phase change profile
            sync_height (bool): If True, syncs height profile

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> phase = torch.ones((1,1,100,100)) * torch.pi
            >>> doe.set_phase_change(phase, sync_height=True)
        """
        super().set_phase_change(phase_change)
        if sync_height:
            self.sync_height_with_phase()

    def set_field_change(self, field_change: torch.Tensor, sync_height: bool = True) -> None:
        """Change the field change of the DOE.

        Args:
            field_change (torch.Tensor): Complex field change tensor
            sync_height (bool): If True, syncs height profile

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> field = torch.exp(1j * torch.ones((1,1,100,100)))
            >>> doe.set_field_change(field, sync_height=True)
        """
        super().set_field_change(field_change)
        if sync_height:
            self.sync_height_with_phase()

    def set_height(self, height: torch.Tensor, sync_phase: bool = True) -> None:
        """Set the height map of the DOE.

        Args:
            height (torch.Tensor): Height map in meters
            sync_phase (bool): If True, syncs phase profile

        Examples:
            >>> doe = DOE((1,1,100,100), 2e-6, material, 500e-9, 'cpu')
            >>> height = torch.ones((1,1,100,100)) * 500e-9
            >>> doe.set_height(height, sync_phase=True)
        """
        self.height = height
        if sync_phase:  
            self.sync_phase_with_height()      


class SLM(OpticalElement):
    def __init__(self, dim: tuple, pitch: float, wvl: float, device: str, polar: str = 'non'):
        """Spatial Light Modulator (SLM) optical element.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            wvl (float): Wavelength in meters
            device (str): Device for computation ('cpu', 'cuda:0', etc.)
            polar (str): Polarization mode ('non' or 'polar')

        Examples:
            >>> slm = SLM(dim=(1,1,1024,1024), pitch=6.4e-6, wvl=633e-9, device='cuda:0')
        """
        super().__init__(dim, pitch, wvl, device=device, name="SLM", polar=polar)

    def set_lens(self, focal_length: float, shift_x: float = 0, shift_y: float = 0) -> None:
        """Set phase profile to implement a thin lens.

        Args:
            focal_length (float): Focal length in meters
            shift_x (float): Lateral shift in x direction in meters
            shift_y (float): Lateral shift in y direction in meters

        Examples:
            >>> slm.set_lens(focal_length=0.5, shift_x=100e-6)  # 500mm focal length, 100μm x-shift
        """
        # Preserve the historical positive quadratic phase sign for this SLM
        # API; unlike RefractiveLens, it specifies a programmed phase pattern.
        focal = _scalar_parameter(focal_length, "focal_length", self.device)
        if focal == 0:
            raise ValueError("focal_length cannot be zero")
        x = (torch.arange(self.dim[3], dtype=torch.float64, device=self.device)-self.dim[3]//2)*self.pitch
        y = (torch.arange(self.dim[2], dtype=torch.float64, device=self.device)-self.dim[2]//2)*self.pitch
        wavelength = _channel_wavelengths(self.wvl, self.dim[1], self.device)
        r2 = (x[None, :]-shift_x)**2 + (y[:, None]-shift_y)**2
        phase = torch.pi*r2[None, None]/(wavelength[None, :, None, None]*focal)
        self.set_phase_change(wrap_phase(phase), self.wvl)

    def set_amplitude_change(self, amplitude: torch.Tensor, wvl: float) -> None:
        """Set amplitude modulation profile of the SLM.

        Args:
            amplitude (torch.Tensor): Amplitude modulation profile [B, 1, R, C]
            wvl (float): Operating wavelength in meters

        Examples:
            >>> amp = torch.ones((1,1,1024,1024)) * 0.8  # 80% transmission
            >>> slm.set_amplitude_change(amp, wvl=633e-9)
        """
        self.wvl = wvl
        super().set_amplitude_change(amplitude)

    def set_phase_change(self, phase_change: torch.Tensor, wvl: float) -> None:
        """Set phase modulation profile of the SLM.

        Args:
            phase_change (torch.Tensor): Phase modulation profile [B, 1, R, C] in radians
            wvl (float): Operating wavelength in meters

        Examples:
            >>> phase = torch.ones((1,1,1024,1024)) * torch.pi  # π phase shift
            >>> slm.set_phase_change(phase, wvl=633e-9)
        """
        self.wvl = wvl
        super().set_phase_change(phase_change)
        
        
class PolarizedSLM(OpticalElement):
    def __init__(self, dim: tuple, pitch: float, wvl: float, device: str):
        """SLM which can control phase & amplitude of each polarization component.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            wvl (float): Wavelength in meters
            device (str): Device for computation ('cpu', 'cuda:0', etc.)

        Examples:
            >>> slm = PolarizedSLM(dim=(1,1,1024,1024), pitch=6.4e-6, wvl=633e-9, device='cuda:0')
        """
        super().__init__(dim, pitch, wvl, device=device, name="Metasurface", polar='polar')
        # One authoritative complex modulation tensor; last axis is X/Y.
        self.field_change = self.field_change[..., None].expand(dim+(2,)).clone()

    @property
    def amplitude_change(self):
        return self.get_amplitude_change()

    @amplitude_change.setter
    def amplitude_change(self, value):
        self.set_amplitude_change(value, self.wvl)

    @property
    def phase_change(self):
        return self.get_phase_change()

    @phase_change.setter
    def phase_change(self, value):
        self.set_phase_change(value, self.wvl)

    def set_field_change(self, field_change: torch.Tensor, c: Optional[int] = None) -> None:
        if c is None:
            self.field_change = torch.broadcast_to(field_change, self.dim+(2,)).to(self.field_change.dtype).clone()
        else:
            result = self.field_change.clone()
            result[:, c] = field_change
            self.field_change = result

    def set_amplitude_change(self, amplitude: torch.Tensor, wvl: float) -> None:
        """Set amplitude change for both polarization components.

        Args:
            amplitude (torch.Tensor): Amplitude change [B, 1, R, C, 2] in polar representation
            wvl (float): Wavelength in meters

        Examples:
            >>> amp = torch.ones((1,1,1024,1024,2)) * 0.8  # 80% transmission for both polarizations
            >>> slm.set_amplitude_change(amp, wvl=633e-9)
        """
        self.wvl = wvl
        amplitude = torch.broadcast_to(amplitude, self.dim+(2,))
        self.field_change = (amplitude * torch.exp(1j*self.get_phase_change())).to(self.field_change.dtype)

    def set_phase_change(self, phase_change: torch.Tensor, wvl: float) -> None:
        """Set phase change for both polarization components.

        Args:
            phase_change (torch.Tensor): Phase change [B, 1, R, C, 2] in polar representation
            wvl (float): Wavelength in meters

        Examples:
            >>> phase = torch.ones((1,1,1024,1024,2)) * torch.pi  # π phase shift for both polarizations
            >>> slm.set_phase_change(phase, wvl=633e-9)
        """
        self.wvl = wvl
        phase_change = torch.broadcast_to(phase_change, self.dim+(2,))
        self.field_change = (self.get_amplitude_change() * torch.exp(1j*phase_change)).to(self.field_change.dtype)

    def _replace_component(self, component, value):
        # Preserve the untouched component's identity derivative, including zero.
        field = self.field_change.clone()
        field[..., component] = value
        self.field_change = field

    def set_amplitudeX_change(self, amplitude: torch.Tensor, wvl: float) -> None:
        """Set amplitude change for X polarization component.

        Args:
            amplitude (torch.Tensor): Amplitude change [B, 1, R, C] for X component
            wvl (float): Wavelength in meters

        Examples:
            >>> ampX = torch.ones((1,1,1024,1024)) * 0.8  # 80% transmission for X polarization
            >>> slm.set_amplitudeX_change(ampX, wvl=633e-9)
        """
        self.wvl = wvl
        self._replace_component(0, amplitude * torch.exp(1j*self.field_change[..., 0].angle()))

    def set_amplitudeY_change(self, amplitude: torch.Tensor, wvl: float) -> None:
        """Set amplitude change for Y polarization component.

        Args:
            amplitude (torch.Tensor): Amplitude change [B, 1, R, C] for Y component
            wvl (float): Wavelength in meters

        Examples:
            >>> ampY = torch.ones((1,1,1024,1024)) * 0.6  # 60% transmission for Y polarization
            >>> slm.set_amplitudeY_change(ampY, wvl=633e-9)
        """
        self.wvl = wvl
        self._replace_component(1, amplitude * torch.exp(1j*self.field_change[..., 1].angle()))

    def set_phaseX_change(self, phase_change: torch.Tensor, wvl: float) -> None:
        """Set phase change for X polarization component.

        Args:
            phase_change (torch.Tensor): Phase change [B, 1, R, C] for X component
            wvl (float): Wavelength in meters

        Examples:
            >>> phaseX = torch.ones((1,1,1024,1024)) * torch.pi/2  # π/2 phase shift for X polarization
            >>> slm.set_phaseX_change(phaseX, wvl=633e-9)
        """
        self.wvl = wvl
        self._replace_component(0, self.field_change[..., 0].abs() * torch.exp(1j*phase_change))

    def set_phaseY_change(self, phase_change: torch.Tensor, wvl: float) -> None:
        """Set phase change for Y polarization component.

        Args:
            phase_change (torch.Tensor): Phase change [B, 1, R, C] for Y component
            wvl (float): Wavelength in meters

        Examples:
            >>> phaseY = torch.ones((1,1,1024,1024)) * torch.pi  # π phase shift for Y polarization
            >>> slm.set_phaseY_change(phaseY, wvl=633e-9)
        """
        self.wvl = wvl
        self._replace_component(1, self.field_change[..., 1].abs() * torch.exp(1j*phase_change))

    def get_phase_changeX(self) -> torch.Tensor:
        """Return phase change for X polarization component.

        Returns:
            torch.Tensor: Phase change [B, 1, R, C] for X component

        Examples:
            >>> phaseX = slm.get_phase_changeX()  # Get X polarization phase profile
        """
        return self.get_phase_change()[:,:,:,:,0]

    def get_phase_changeY(self) -> torch.Tensor:
        """Return phase change for Y polarization component.

        Returns:
            torch.Tensor: Phase change [B, 1, R, C] for Y component

        Examples:
            >>> phaseY = slm.get_phase_changeY()  # Get Y polarization phase profile
        """
        return self.get_phase_change()[:,:,:,:,1]

    def get_amplitude_changeX(self) -> torch.Tensor:
        """Return amplitude change for X polarization component.

        Returns:
            torch.Tensor: Amplitude change [B, 1, R, C] for X component

        Examples:
            >>> ampX = slm.get_amplitude_changeX()  # Get X polarization amplitude profile
        """
        return self.get_amplitude_change()[:,:,:,:,0]

    def get_amplitude_changeY(self) -> torch.Tensor:
        """Return amplitude change for Y polarization component.

        Returns:
            torch.Tensor: Amplitude change [B, 1, R, C] for Y component

        Examples:
            >>> ampY = slm.get_amplitude_changeY()  # Get Y polarization amplitude profile
        """
        return self.get_amplitude_change()[:,:,:,:,1]
        
    def forward(self, light: 'Light', interp_mode: str = 'nearest') -> 'Light':
        """Apply polarization-dependent modulation to input light.

        Args:
            light (Light): Input light field
            interp_mode (str): Interpolation mode for resizing ('nearest', 'bilinear', etc.)

        Returns:
            Light: Modulated light field

        Examples:
            >>> modulated_light = slm.forward(input_light)  # Apply polarization modulation
            >>> modulated_light = slm.forward(input_light, interp_mode='bilinear')  # Use bilinear interpolation
        """
        if not _wavelengths_match(light, self):
            raise ValueError(f'Wavelength mismatch: light wavelength {light.wvl} != element wavelength {self.wvl}')
        
        self._match_pitch(light, interp_mode)
            
        self._match_spatial_shape(light)
        
        # Multiplication preserves independent input phases and the complete
        # input/modulator autograd graph, including zero-amplitude samples.
        light.set_fieldX(light.get_fieldX() * self.field_change[..., 0])
        light.set_fieldY(light.get_fieldY() * self.field_change[..., 1])
        return light

    def pad(self, pad_width: tuple, padval: int = 0) -> None:
        """Pad spatial axes using (left, right, top, bottom), retaining X/Y."""
        if padval != 0:
            raise NotImplementedError('only zero padding supported')
        self.field_change = F.pad(self.field_change, (0, 0)+tuple(pad_width))
        self.dim = tuple(self.field_change.shape[:-1])

    def resize(self, target_pitch: float, interp_mode: str = 'nearest') -> None:
        """Interpolate each complex polarization on the spatial axes."""
        if not isinstance(target_pitch, (int, float)) or not _math.isfinite(target_pitch) or target_pitch <= 0:
            raise ValueError("target_pitch must be finite and positive")
        batch, channels, rows, cols = self.dim
        field = self.field_change.permute(0, 1, 4, 2, 3).reshape(batch, 2*channels, rows, cols)
        scale = self.pitch / target_pitch
        field = torch.complex(F.interpolate(field.real, scale_factor=scale, mode=interp_mode),
                              F.interpolate(field.imag, scale_factor=scale, mode=interp_mode))
        rows, cols = field.shape[-2:]
        self.field_change = field.reshape(batch, channels, 2, rows, cols).permute(0, 1, 3, 4, 2)
        self.dim = (batch, channels, rows, cols)
        self.pitch = target_pitch

    def visualize(self, b: int = 0) -> None:
        """Visualize amplitude and phase modulation for both polarizations.

        Args:
            b (int): Batch index to visualize, default 0

        Examples:
            >>> slm.visualize()  # Visualize first batch
            >>> slm.visualize(b=1)  # Visualize second batch
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13,8))
        
        plt.subplot(221)
        plt.imshow(self.get_amplitude_changeX().detach().cpu()[b,...].squeeze(), cmap='inferno')
        plt.title('amplitude change X')
        plt.colorbar()
        
        plt.subplot(222)
        plt.imshow(self.get_phase_changeX().detach().cpu()[b,...].squeeze(), cmap='hsv')
        plt.title('phase change X')
        plt.colorbar()
        
        plt.subplot(223)
        plt.imshow(self.get_amplitude_changeY().detach().cpu()[b,...].squeeze(), cmap='inferno')
        plt.title('amplitude change Y')
        plt.colorbar()
        
        plt.subplot(224)
        plt.imshow(self.get_phase_changeY().detach().cpu()[b,...].squeeze(), cmap='hsv')
        plt.title('phase change Y')
        plt.colorbar()
        
        plt.suptitle(
            f'{self.name}, '
            f'({self.dim[2]},{self.dim[3]}), '
            f'pitch:{self.pitch/1e-6:.2f}[um], '
            f'wvl:{self.wvl/1e-9:.2f}[nm], '
            f'device:{self.device}'
        )
        plt.show()


class Aperture(OpticalElement):
    """Aperture optical element for amplitude modulation.
    
    Implement square or circular aperture that modulate light amplitude.
    Support both polarized and non-polarized light.
    """

    def __init__(self, dim: tuple, pitch: float, aperture_diameter: float, aperture_shape: str, wvl: float, device: str = 'cpu', polar: str = 'non'):
        """Create aperture optical element instance.

        Args:
            dim (tuple): Field dimensions (B, 1, R, C) for batch, channels, rows, cols
            pitch (float): Pixel pitch in meters
            aperture_diameter (float): Diameter of aperture in meters
            aperture_shape (str): Shape of aperture ('square' or 'circle')
            wvl (float): Wavelength in meters
            device (str): Device for computation ('cpu', 'cuda:0', etc.)
            polar (str): Polarization mode ('non', 'x', 'y', 'xy')

        Examples:
            >>> aperture = Aperture(dim=(1,1,1024,1024), pitch=6.4e-6, 
            ...                    aperture_diameter=1e-3, aperture_shape='circle',
            ...                    wvl=633e-9)
        """
        super().__init__(dim, pitch, wvl, device=device, name="aperture", polar=polar)

        self.aperture_diameter = aperture_diameter
        self.aperture_shape = aperture_shape
        self.amplitude_change = torch.zeros((self.dim[2], self.dim[3]), device=device)
        if self.aperture_shape == 'square':
            self.set_square()
        elif self.aperture_shape == 'circle':
            self.set_circle()
        else:
            return NotImplementedError

    def set_square(self) -> None:
        """Set square aperture amplitude modulation.

        Create square aperture mask centered on optical axis.

        Examples:
            >>> aperture.set_square()
        """
        self.aperture_shape = 'square'

        x = torch.arange(self.dim[2], device=self.device, dtype=torch.float64) - self.dim[2]//2
        y = torch.arange(self.dim[3], device=self.device, dtype=torch.float64) - self.dim[3]//2
        r = (self.pitch * torch.maximum(x[:, None].abs(), y[None, :].abs()))[None, None]

        max_val = self.aperture_diameter / 2
        amp = (r <= max_val).to(torch.float32)
        amp[amp == 0] = 1e-20  # to enable stable learning
        self.set_field_change(amp)

    def set_circle(self, cx: float = 0, cy: float = 0, dia: float = None) -> None:
        """Set circular aperture amplitude modulation.

        Create circular aperture mask with optional offset and diameter.

        Args:
            cx (float): Center row offset in pixels (historical convention)
            cy (float): Center column offset in pixels (historical convention)
            dia (float, optional): Circle diameter in meters

        Examples:
            >>> aperture.set_circle()  # Centered circle
            >>> aperture.set_circle(cx=10, cy=-10, dia=2e-3)  # Offset circle
        """
        # Preserve historical cx=row and cy=column offsets, measured in pixels.
        x = torch.arange(self.dim[2], device=self.device, dtype=torch.float64) - self.dim[2]//2
        y = torch.arange(self.dim[3], device=self.device, dtype=torch.float64) - self.dim[3]//2
        r = self.pitch * torch.sqrt((x[:, None]-cx)**2 + (y[None, :]-cy)**2)
        if dia is not None:
            self.aperture_diameter = dia
        self.aperture_shape = 'circle'
        amp = (r <= self.aperture_diameter / 2).to(torch.float32)[None, None]
        amp = torch.where(amp == 0, amp.new_tensor(1e-20), amp)
        self.set_field_change(amp)



def quantize(x: Union[torch.Tensor, _NDArray], levels: int, vmin: float = None, vmax: float = None, include_vmax: bool = True) -> Union[torch.Tensor, _NDArray]:
    """Quantize floating point array.

    Discretize input array into specified number of levels.

    Args:
        x (torch.Tensor or np.ndarray): Input array to quantize
        levels (int): Nonnegative number of levels; zero disables quantization.
            One level returns the lower bound. A constant input remains finite.
        vmin (float, optional): Lower bound when include_vmax=False.
        vmax (float, optional): Upper bound when include_vmax=False.
        include_vmax (bool): Historical bin convention. True uses the observed
            input range with spacing (max-min)/levels and excludes its maximum;
            False uses spacing (vmax-vmin)/(levels-1), including both bounds.

    Returns:
        torch.Tensor or np.ndarray: Quantized array

    Examples:
        >>> x = torch.randn(100)
        >>> x_quant = quantize(x, levels=8)
        >>> x_quant = quantize(x, levels=16, vmin=-1, vmax=1, include_vmax=False)
    """
    is_tensor = isinstance(x, torch.Tensor)
    if not is_tensor:
        import numpy as np
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a torch.Tensor or numpy.ndarray")

    if isinstance(levels, bool) or not isinstance(levels, _Integral):
        raise TypeError("levels must be a nonnegative integer")
    if levels < 0:
        raise ValueError("levels must be nonnegative")
    if levels == 0:
        return x

    if include_vmax is False:
        if vmin is None:
            vmin = x.min()
        if vmax is None:
            vmax = x.max()
        if levels == 1:
            result = x * 0 + vmin
            return result if is_tensor else np.asarray(result)

        normalized = (x - vmin) / (vmax - vmin + 1e-16)
        if not is_tensor:
            levelized = np.floor(normalized * levels) / (levels - 1)
        else:
            levelized = (normalized * levels).floor() / (levels - 1)
        result = levelized * (vmax - vmin) + vmin
        
        if not is_tensor:
            return np.asarray(np.clip(result, vmin, vmax))
        else:  # torch.Tensor
            # For tensors, we use clamp which returns a new tensor
            return torch.clamp(result, min=vmin, max=vmax)
    
    elif include_vmax is True:
        vmin = x.min()
        span = x.max() - vmin
        # Use a finite divisor for flat fields without a CUDA scalar read.
        space = (torch.where(span == 0, torch.ones_like(span), span) if is_tensor
                 else np.where(span == 0, 1, span)) / levels
        vmax = vmin + space*(levels-1)
        if not is_tensor:
            result = (np.floor((x-vmin)/space))*space + vmin
            return np.asarray(np.clip(result, vmin, vmax))
        else:
            result = (((x-vmin)/space).floor())*space + vmin
            # For tensors, we use clamp which returns a new tensor
            return torch.clamp(result, min=vmin, max=vmax)
