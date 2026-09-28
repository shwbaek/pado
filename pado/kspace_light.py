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
from typing import Tuple, Union, List, Optional

import torch

from .light import _channel_wavelengths, _stack_wavelengths

__all__ = ["KSpaceLight"]


def _validate_k_axis(axis: torch.Tensor, name: str) -> None:
    """Validate a real, finite, strictly increasing uniform coordinate axis."""
    if not isinstance(axis, torch.Tensor) or not axis.is_floating_point():
        raise TypeError(f"{name} must be a real floating-point tensor")
    if axis.ndim != 1 or axis.numel() == 0:
        raise ValueError(f"{name} must be a nonempty 1D tensor")
    if not torch.isfinite(axis).all():
        raise ValueError(f"{name} must contain finite coordinates")
    if axis.numel() > 1:
        if not torch.all(axis[1:] > axis[:-1]):
            raise ValueError(f"{name} must be strictly increasing")
        expected = torch.linspace(axis[0], axis[-1], axis.numel(),
                                  device=axis.device, dtype=axis.dtype)
        # Account for coordinate rounding at large optical wavenumbers, not
        # a fixed absolute tolerance that would accept tiny nonuniform axes.
        tolerance = 8 * torch.finfo(axis.dtype).eps * axis.abs().max()
        if not torch.all((axis - expected).abs() <= tolerance):
            raise ValueError(f"{name} must be uniformly spaced")


class KSpaceLight:
    """Complex optical field defined on a k-space (kx, ky) grid.

    Unlike :class:`Light`, which represents a complex field on a spatial (x, y)
    plane and carries spatial concepts such as ``pitch`` and ``bandwidth``,
    ``KSpaceLight`` represents a field that already lives in k-space, i.e. on a
    grid of wave-vector coordinates ``(kx, ky)`` measured in radians per meter.

    This is a lightweight, differentiable container. It does **not** perform any
    FFT internally, and it does **not** implement propagation or rendering logic.
    It only stores the complex k-space field and exposes complex-field accessors
    (shared in spirit with :class:`Light`) together with k-space coordinate
    utilities useful for wide-angle propagation and angular rendering.
    """

    def __init__(
        self,
        dim: Tuple[int, int, int, int],
        kx: torch.Tensor,
        ky: torch.Tensor,
        wvl: Union[float, List[float]],
        field: Optional[torch.Tensor] = None,
        source_pitch: Optional[float] = None,
        source_dim: Optional[Tuple[int, int, int, int]] = None,
        device: str = "cpu",
    ):
        """Create a k-space light container.

        Args:
            dim (tuple): Field dimensions (B, Ch, R, C) for batch, channels, rows, cols.
            kx (torch.Tensor): 1D tensor of shape (C,), wave-vector coordinate along
                the x axis in radians per meter. Must be real floating-point,
                finite, strictly increasing and uniform (a singleton is valid).
            ky (torch.Tensor): 1D tensor of shape (R,), wave-vector coordinate along
                the y axis in radians per meter, with the same constraints as kx.
            wvl (float or list): Wavelength in meters. Either a single value or a list
                with length equal to the channel dimension.
            field (torch.Tensor, optional): Complex field [B, Ch, R, C]. If None, it is
                initialized to complex ones. Stored directly without detaching so that
                an existing computation graph is preserved.
            source_pitch (float, optional): Spatial pitch of the original spatial Light
                before Fourier conversion (metadata only).
            source_dim (tuple, optional): Original spatial Light dimensions (metadata only).
            device (str): Device for computation ('cpu', 'cuda:0', etc.).

        Examples:
            >>> R, C = 256, 256
            >>> kx = torch.linspace(-1e7, 1e7, C)
            >>> ky = torch.linspace(-1e7, 1e7, R)
            >>> klight = KSpaceLight(dim=(1, 1, R, C), kx=kx, ky=ky, wvl=633e-9)
        """
        # ---- dim validation ----
        if not isinstance(dim, tuple) or len(dim) != 4:
            raise ValueError(f"dim must be a 4-element tuple (B,Ch,R,C), got {dim}")
        if any((not isinstance(d, _Integral) or d < 1) for d in dim):
            raise ValueError(f"All dimensions must be positive integers, got {dim}")
        B, Ch, R, C = dim

        # ---- kx / ky validation ----
        if not isinstance(kx, torch.Tensor):
            raise TypeError(f"kx must be a torch.Tensor, got {type(kx)}")
        if not isinstance(ky, torch.Tensor):
            raise TypeError(f"ky must be a torch.Tensor, got {type(ky)}")
        if kx.dim() != 1:
            raise ValueError(f"kx must be 1D, got shape {tuple(kx.shape)}")
        if ky.dim() != 1:
            raise ValueError(f"ky must be 1D, got shape {tuple(ky.shape)}")
        if kx.shape[0] != C:
            raise ValueError(f"kx must have length C={C}, got {kx.shape[0]}")
        if ky.shape[0] != R:
            raise ValueError(f"ky must have length R={R}, got {ky.shape[0]}")
        _validate_k_axis(kx, "kx")
        _validate_k_axis(ky, "ky")

        # ---- wvl validation ----
        if isinstance(wvl, (int, float)):
            if not _math.isfinite(wvl) or wvl <= 0:
                raise ValueError("Wavelengths must be finite and positive")
        else:
            validation_device = wvl.device if isinstance(wvl, torch.Tensor) else next(
                (value.device for value in wvl if isinstance(value, torch.Tensor)), 'cpu')
            _channel_wavelengths(wvl, Ch, validation_device)

        # ---- field validation ----
        if field is not None:
            if not isinstance(field, torch.Tensor):
                raise TypeError(f"field must be a torch.Tensor, got {type(field)}")
            if not field.is_complex():
                raise TypeError("field must be a complex tensor (e.g. dtype=torch.cfloat)")
            if tuple(field.shape) != dim:
                raise ValueError(f"field shape {tuple(field.shape)} must match dim {dim}")
            # Like Light, a supplied field owns the storage device.
            device = str(field.device)

        self.dim: Tuple[int, int, int, int] = dim
        self.wvl: Union[float, List[float]] = wvl
        self.device: str = str(device)
        self.source_pitch: Optional[float] = source_pitch
        self.source_dim: Optional[Tuple[int, int, int, int]] = source_dim

        # Place coordinates on the field device without detaching their graphs.
        self.kx: torch.Tensor = kx.to(device)
        self.ky: torch.Tensor = ky.to(device)

        # Store the field directly to preserve any existing computation graph.
        if field is None:
            field = torch.ones(dim, device=device, dtype=torch.cfloat)
        self.field: torch.Tensor = field

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _check_channel(self, c: int) -> None:
        if not isinstance(c, int):
            raise TypeError(f"Channel index c must be an integer, got {type(c)}")
        if c < 0 or c >= self.dim[1]:
            raise IndexError(
                f"Channel index {c} out of bounds for tensor with {self.dim[1]} channels"
            )

    def _k0_scalar(self, c: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Return one wavenumber for a coordinate grid, preserving tensor gradients."""
        value = self.get_k0(c)
        if isinstance(value, torch.Tensor) and value.ndim != 0:
            raise ValueError("wvl is multi-wavelength; please specify a channel index c")
        return value

    # ------------------------------------------------------------------
    # complex-field accessors (shared semantics with Light)
    # ------------------------------------------------------------------
    def get_field(self, c: Optional[int] = None) -> torch.Tensor:
        """Return the complex field, optionally for a single channel."""
        if c is not None:
            self._check_channel(c)
            return self.field[:, c, ...]
        return self.field

    def set_field(self, field: torch.Tensor, c: Optional[int] = None) -> None:
        """Set the complex field, optionally for a single channel.

        Autograd is preserved: the incoming tensor is not detached.
        """
        if not isinstance(field, torch.Tensor):
            raise TypeError(f"field must be a torch.Tensor, got {type(field)}")
        if not field.is_complex():
            raise TypeError("field must be a complex tensor (e.g. dtype=torch.cfloat)")

        if c is not None:
            self._check_channel(c)
            if field.shape != self.field[:, c, ...].shape:
                raise ValueError(
                    f"Expected field of shape {self.field[:, c, ...].shape}, got {field.shape}"
                )
            new_field = self.field.clone()
            new_field[:, c, ...] = field
            self.field = new_field
        else:
            if field.shape != self.field.shape:
                raise ValueError(
                    f"Expected field of shape {self.field.shape}, got {field.shape}"
                )
            self.field = field
            self.device = str(field.device)
            self.kx = self.kx.to(field.device)
            self.ky = self.ky.to(field.device)

    def get_real(self, c: Optional[int] = None) -> torch.Tensor:
        """Return the real part of the field, optionally for a single channel."""
        if c is not None:
            self._check_channel(c)
            return self.field[:, c, ...].real
        return self.field.real

    def set_real(self, real: torch.Tensor, c: Optional[int] = None) -> None:
        """Set the real part of the field (keeps imaginary part). Autograd preserved."""
        if not isinstance(real, torch.Tensor):
            raise TypeError(f"real must be a torch.Tensor, got {type(real)}")

        if c is not None:
            self._check_channel(c)
            if real.shape != self.field[:, c, ...].real.shape:
                raise ValueError(
                    f"Expected real of shape {self.field[:, c, ...].real.shape}, got {real.shape}"
                )
            imag = self.field[:, c, ...].imag
            new_field = self.field.clone()
            new_field[:, c, ...] = torch.complex(real, imag)
            self.field = new_field
        else:
            if real.shape != self.field.real.shape:
                raise ValueError(
                    f"Expected real of shape {self.field.real.shape}, got {real.shape}"
                )
            self.field = torch.complex(real, self.field.imag)

    def get_imag(self, c: Optional[int] = None) -> torch.Tensor:
        """Return the imaginary part of the field, optionally for a single channel."""
        if c is not None:
            self._check_channel(c)
            return self.field[:, c, ...].imag
        return self.field.imag

    def set_imag(self, imag: torch.Tensor, c: Optional[int] = None) -> None:
        """Set the imaginary part of the field (keeps real part). Autograd preserved."""
        if not isinstance(imag, torch.Tensor):
            raise TypeError(f"imag must be a torch.Tensor, got {type(imag)}")

        if c is not None:
            self._check_channel(c)
            if imag.shape != self.field[:, c, ...].imag.shape:
                raise ValueError(
                    f"Expected imag of shape {self.field[:, c, ...].imag.shape}, got {imag.shape}"
                )
            real = self.field[:, c, ...].real
            new_field = self.field.clone()
            new_field[:, c, ...] = torch.complex(real, imag)
            self.field = new_field
        else:
            if imag.shape != self.field.imag.shape:
                raise ValueError(
                    f"Expected imag of shape {self.field.imag.shape}, got {imag.shape}"
                )
            self.field = torch.complex(self.field.real, imag)

    def get_amplitude(self, c: Optional[int] = None) -> torch.Tensor:
        """Return the amplitude of the field, optionally for a single channel."""
        if c is not None:
            self._check_channel(c)
            return self.field[:, c, ...].abs()
        return self.field.abs()

    def set_amplitude(self, amplitude: torch.Tensor, c: Optional[int] = None) -> None:
        """Set the amplitude of the field (keeps phase). Autograd preserved."""
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError(f"amplitude must be a torch.Tensor, got {type(amplitude)}")

        if c is not None:
            self._check_channel(c)
            if amplitude.shape != self.field[:, c, ...].shape:
                raise ValueError(
                    f"Expected amplitude of shape {self.field[:, c, ...].shape}, got {amplitude.shape}"
                )
            phase = self.field[:, c, ...].angle()
            new_field = self.field.clone()
            new_field[:, c, ...] = amplitude * torch.exp(1j * phase)
            self.field = new_field
        else:
            if amplitude.shape != self.field.shape:
                raise ValueError(
                    f"Expected amplitude of shape {self.field.shape}, got {amplitude.shape}"
                )
            phase = self.field.angle()
            self.field = amplitude * torch.exp(1j * phase)

    def get_phase(self, c: Optional[int] = None) -> torch.Tensor:
        """Return the phase of the field (radians), optionally for a single channel."""
        if c is not None:
            self._check_channel(c)
            return self.field[:, c, ...].angle()
        return self.field.angle()

    def set_phase(self, phase: torch.Tensor, c: Optional[int] = None) -> None:
        """Set the phase of the field (keeps amplitude). Autograd preserved."""
        if not isinstance(phase, torch.Tensor):
            raise TypeError(f"phase must be a torch.Tensor, got {type(phase)}")

        if c is not None:
            self._check_channel(c)
            if phase.shape != self.field[:, c, ...].shape:
                raise ValueError(
                    f"Expected phase of shape {self.field[:, c, ...].shape}, got {phase.shape}"
                )
            amplitude = self.field[:, c, ...].abs()
            new_field = self.field.clone()
            new_field[:, c, ...] = amplitude * torch.exp(1j * phase)
            self.field = new_field
        else:
            if phase.shape != self.field.shape:
                raise ValueError(
                    f"Expected phase of shape {self.field.shape}, got {phase.shape}"
                )
            amplitude = self.field.abs()
            self.field = amplitude * torch.exp(1j * phase)

    def get_intensity(self, c: Optional[int] = None) -> torch.Tensor:
        """Return the intensity (|field|^2) of the field, optionally for a single channel."""
        if c is not None:
            self._check_channel(c)
            f = self.field[:, c, ...]
            return (f * torch.conj(f)).real
        return (self.field * torch.conj(self.field)).real

    # ------------------------------------------------------------------
    # container utilities
    # ------------------------------------------------------------------
    def clone(self) -> "KSpaceLight":
        """Create a deep copy. The field clone preserves the computation graph."""
        return KSpaceLight(
            dim=self.dim,
            kx=self.kx.clone(),
            ky=self.ky.clone(),
            wvl=self.wvl,
            field=self.field.clone(),
            source_pitch=self.source_pitch,
            source_dim=self.source_dim,
            device=self.device,
        )

    def shape(self) -> torch.Size:
        """Return the shape of the field tensor."""
        return self.field.shape

    def get_channel(self) -> int:
        """Return the number of channels."""
        return self.dim[1]

    def get_device(self) -> str:
        """Return the device of the k-space light."""
        return self.device

    def to(self, device: str) -> "KSpaceLight":
        """Move field and coordinate tensors to the given device (in place).

        Returns self to allow chaining. The field move preserves autograd.
        """
        self.field = self.field.to(device)
        self.kx = self.kx.to(device)
        self.ky = self.ky.to(device)
        self.device = device
        return self

    # ------------------------------------------------------------------
    # k-space coordinate utilities
    # ------------------------------------------------------------------
    def get_kx_ky(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the 1D kx and ky coordinate tensors."""
        return self.kx, self.ky

    def get_k_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return 2D meshgrids (kx_grid, ky_grid), each of shape (R, C)."""
        ky_grid, kx_grid = torch.meshgrid(self.ky, self.kx, indexing="ij")
        return kx_grid, ky_grid

    def get_k_limits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (kx_min, kx_max, ky_min, ky_max)."""
        return self.kx.min(), self.kx.max(), self.ky.min(), self.ky.max()

    def get_k_spacing(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (dkx, dky), assuming uniform spacing.

        For a degenerate axis of length 1, the corresponding spacing is 0.
        """
        if self.kx.numel() > 1:
            dkx = self.kx[1] - self.kx[0]
        else:
            dkx = torch.zeros((), device=self.device, dtype=self.kx.dtype)
        if self.ky.numel() > 1:
            dky = self.ky[1] - self.ky[0]
        else:
            dky = torch.zeros((), device=self.device, dtype=self.ky.dtype)
        return dkx, dky

    def get_k0(self, c: Optional[int] = None) -> Union[float, torch.Tensor]:
        """Return k0 = 2*pi/wavelength, preserving tensor dtype and gradients.

        A numeric scalar returns a float. Numeric lists retain the historical
        float32 vector output. Tensor wavelengths, including scalar tensors or
        lists of tensors, retain their precision and graph on the field device.
        """
        if isinstance(self.wvl, (int, float)):
            return 2 * _math.pi / self.wvl
        if isinstance(self.wvl, torch.Tensor):
            wavelengths = self.wvl.to(self.device)
            if wavelengths.ndim == 0:
                return 2 * _math.pi / wavelengths
            if c is not None:
                self._check_channel(c)
                wavelengths = wavelengths[c]
            return 2 * _math.pi / wavelengths
        if c is not None:
            self._check_channel(c)
            wavelength = self.wvl[c]
            if isinstance(wavelength, torch.Tensor):
                wavelength = wavelength.to(self.device)
            return 2 * _math.pi / wavelength
        if any(isinstance(w, torch.Tensor) for w in self.wvl):
            return 2 * _math.pi / _stack_wavelengths(self.wvl, self.device)
        return torch.tensor([2 * _math.pi / w for w in self.wvl],
                            device=self.device, dtype=torch.float32)

    def get_k_radius_grid(self) -> torch.Tensor:
        """Return the sampled k-space radius sqrt(kx^2 + ky^2), shape (R, C)."""
        kx_grid, ky_grid = self.get_k_grid()
        return torch.sqrt(kx_grid ** 2 + ky_grid ** 2)

    def get_valid_mask(self, c: Optional[int] = None) -> torch.Tensor:
        """Return the propagating-wave mask kx^2 + ky^2 <= k0^2, shape (R, C)."""
        k0 = self._k0_scalar(c)
        kx_grid, ky_grid = self.get_k_grid()
        return (kx_grid ** 2 + ky_grid ** 2) <= (k0 ** 2)

    def get_direction_grid(
        self, c: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unit direction components (dir_x, dir_y, dir_z, valid_mask).

        dir_x = kx/k0, dir_y = ky/k0, dir_z = sqrt(max(1 - dir_x^2 - dir_y^2, 0)).
        valid_mask marks directions inside the unit circle (propagating waves).
        """
        k0 = self._k0_scalar(c)
        kx_grid, ky_grid = self.get_k_grid()
        dir_x = kx_grid / k0
        dir_y = ky_grid / k0
        valid_mask = (dir_x ** 2 + dir_y ** 2) <= 1
        dir_z = torch.sqrt(torch.clamp(1 - dir_x ** 2 - dir_y ** 2, min=0))
        return dir_x, dir_y, dir_z, valid_mask

    def get_theta_phi_grid(
        self, c: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return angular coordinates (theta, phi, valid_mask).

        theta = asin(clamp(sqrt(dir_x^2 + dir_y^2), 0, 1)) is the polar angle from
        the optical axis; phi = atan2(dir_y, dir_x) is the azimuth.
        """
        dir_x, dir_y, _, valid_mask = self.get_direction_grid(c)
        rho = torch.sqrt(dir_x ** 2 + dir_y ** 2)
        theta = torch.asin(torch.clamp(rho, 0, 1))
        phi = torch.atan2(dir_y, dir_x)
        return theta, phi, valid_mask
