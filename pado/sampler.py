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

from typing import Optional

import torch
import torch.nn.functional as F
from .kspace_light import _validate_k_axis

__all__ = ["Sampler"]


def _normalize_k_coordinate(query: torch.Tensor, axis: torch.Tensor,
                            align_corners: bool) -> torch.Tensor:
    """Map a uniformly sampled k axis to grid_sample pixel coordinates.

    A singleton axis only defines its stored coordinate, not an interpolation
    interval. Accept that coordinate and reject off-axis queries rather than
    inventing a spacing (including for border/reflection padding).
    """
    if axis.numel() == 1:
        if not torch.all(torch.isclose(query, axis[0])):
            raise ValueError("Cannot sample away from a singleton k-space axis")
        return query * 0

    fraction = (query - axis[0]) / (axis[-1] - axis[0])
    if align_corners:
        return 2 * fraction - 1
    # Pixel centers are at (2*i + 1)/size - 1 when align_corners=False.
    return (2 * fraction * (axis.numel() - 1) + 1) / axis.numel() - 1


def _complex_grid_sample(field: torch.Tensor, grid: torch.Tensor,
                         mode: str = "bilinear", padding_mode: str = "zeros",
                         align_corners: bool = True) -> torch.Tensor:
    """Sample complex components using native real grid_sample operations.

    PyTorch ``grid_sample`` does not support complex tensors, so the real and
    imaginary parts are represented as real channels and recombined. A single
    batch/channel uses one operation on a strided view; larger layouts retain
    separate component calls to bound forward temporary storage. Autograd is
    preserved.

    Args:
        field (torch.Tensor): Complex tensor [N, C, H, W].
        grid (torch.Tensor): Sampling grid [N, H_out, W_out, 2] with last dim (x, y).
        mode (str): Interpolation mode for grid_sample.
        padding_mode (str): Out-of-range padding mode for grid_sample.
        align_corners (bool): grid_sample align_corners flag.

    Returns:
        torch.Tensor: Sampled complex tensor [N, C, H_out, W_out].
    """
    if field.shape[:2] != (1, 1) or field.shape[-2:] == (1, 1):
        # Packing larger/broadcast layouts can increase forward peak memory.
        # A 1x1 view also has incompatible backward strides on Torch 2.10.
        real = F.grid_sample(field.real.contiguous(), grid, mode=mode,
                             padding_mode=padding_mode, align_corners=align_corners)
        imag = F.grid_sample(field.imag.contiguous(), grid, mode=mode,
                             padding_mode=padding_mode, align_corners=align_corners)
        return torch.complex(real, imag)
    components = torch.view_as_real(field.resolve_conj()).movedim(-1, 1).flatten(1, 2)
    sampled = F.grid_sample(components, grid, mode=mode,
                            padding_mode=padding_mode, align_corners=align_corners)
    real, imag = sampled.chunk(2, dim=1)
    return torch.complex(real, imag)


class Sampler:
    """Sample a k-space (KSpaceLight) complex field at angular / direction queries.

    The k-space field is stored on a ``(kx, ky)`` grid in radians per meter.
    Given query angular coordinates ``(theta, phi)`` or 3D direction vectors,
    this sampler converts them to ``(kx_query, ky_query)``, normalizes to the
    ``[-1, 1]`` coordinate system expected by ``torch.nn.functional.grid_sample``,
    and samples the complex field (real/imag separately to preserve autograd).

    Note:
        ``sample_direction`` samples the angular / k-space field along a
        direction. It does NOT compute the coherent field at a physical 3D
        point; that would require an integral over k-space and is left for a
        separate, later implementation.
    """

    def sample_theta_phi(self, source_field, theta: torch.Tensor, phi: torch.Tensor,
                         c: Optional[int] = None, mode: str = "bilinear",
                         padding_mode: str = "zeros", align_corners: bool = True,
                         polar_axis: str = "z") -> torch.Tensor:
        """Sample the k-space field at angular coordinates (theta, phi).

        Args:
            source_field (KSpaceLight): k-space field to sample.
            theta (torch.Tensor): Query polar angles in radians, shape [H, W] or [B, H, W].
            phi (torch.Tensor): Query azimuthal angles in radians, same shape as theta.
            c (int, optional): Channel index for multi-wavelength fields.
            mode (str): grid_sample interpolation mode.
            padding_mode (str): grid_sample out-of-range padding mode.
            align_corners (bool): grid_sample align_corners flag.
            polar_axis (str): "z" (default) uses theta from the optical z axis
                and phi from +x toward +y, matching KSpaceLight.get_theta_phi_grid.
                "y" uses theta from +y and phi from +x toward +z, preserving the
                validation notebook's angular layout.

        Returns:
            torch.Tensor: Sampled complex field [N, C_out, H, W].
        """
        if not isinstance(theta, torch.Tensor) or not isinstance(phi, torch.Tensor):
            raise TypeError("theta and phi must be real floating-point tensors")
        if not theta.is_floating_point() or not phi.is_floating_point():
            raise TypeError("theta and phi must be real floating-point tensors")
        if theta.ndim not in (2, 3) or theta.numel() == 0:
            raise ValueError("theta and phi must be nonempty [H,W] or [B,H,W] tensors")
        if not (torch.isfinite(theta).all() & torch.isfinite(phi).all()):
            raise ValueError("theta and phi must be finite")
        if theta.shape != phi.shape:
            raise ValueError(
                f"theta and phi must have the same shape, got {tuple(theta.shape)} and {tuple(phi.shape)}"
            )

        # Nominally unit direction from spherical angles.
        sin_theta = torch.sin(theta)
        dir_x = sin_theta * torch.cos(phi)
        if polar_axis == "z":
            dir_y = sin_theta * torch.sin(phi)
            dir_z = torch.cos(theta)
        elif polar_axis == "y":
            dir_y = torch.cos(theta)
            dir_z = sin_theta * torch.sin(phi)
        else:
            raise ValueError("polar_axis must be 'z' or 'y'")
        dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)

        return self.sample_direction(
            source_field, dirs, c=c, mode=mode, padding_mode=padding_mode,
            # Use the same normalization as direct direction queries. Trig
            # identities are only approximate at finite precision.
            align_corners=align_corners, normalize=True,
        )

    def sample_direction(self, source_field, dirs: torch.Tensor, c: Optional[int] = None,
                         mode: str = "bilinear", padding_mode: str = "zeros",
                         align_corners: bool = True, normalize: bool = True) -> torch.Tensor:
        """Sample the k-space field along 3D direction vectors.

        Coordinate axes must be uniformly sampled. If an axis has length one,
        queries must match that stored coordinate (torch.isclose tolerance);
        off-axis queries raise ValueError for every padding mode because no
        interpolation spacing is defined.

        Only the forward hemisphere (dir_z >= 0) is supported. Rear-facing,
        zero and nonfinite vectors raise ValueError. With normalize=False,
        vectors must already have unit length. Values within 8 dtype eps of
        the horizon are treated as horizon roundoff. Validation on CUDA may
        synchronize with the host.

        Args:
            source_field (KSpaceLight): k-space field to sample.
            dirs (torch.Tensor): Direction vectors with last dim 3, shape
                [H, W, 3] or [B, H, W, 3], representing (dir_x, dir_y, dir_z).
            c (int, optional): Channel index for multi-wavelength fields.
            mode (str): grid_sample interpolation mode.
            padding_mode (str): grid_sample out-of-range padding mode.
            align_corners (bool): grid_sample align_corners flag.
            normalize (bool): If True, normalize each direction vector to unit length.

        Returns:
            torch.Tensor: Sampled complex field [N, C_out, H, W].
        """
        if not isinstance(dirs, torch.Tensor) or not dirs.is_floating_point():
            raise TypeError("dirs must be a real floating-point tensor")
        if dirs.ndim not in (3, 4) or dirs.numel() == 0:
            raise ValueError("dirs must be nonempty [H,W,3] or [B,H,W,3] tensors")
        if dirs.shape[-1] != 3:
            raise ValueError(f"dirs last dimension must be 3, got {dirs.shape[-1]}")
        if dirs.dim() == 3:        # [H, W, 3] -> add batch dim
            dirs = dirs.unsqueeze(0)
        elif dirs.dim() != 4:      # expect [B, H, W, 3]
            raise ValueError(
                f"dirs must be [H, W, 3] or [B, H, W, 3], got shape {tuple(dirs.shape)}"
            )

        field = source_field.get_field()           # [B, Ch, H, W], complex
        if c is not None:
            if not isinstance(c, int):
                raise TypeError(f"Channel index c must be an integer, got {type(c)}")
            if c < 0 or c >= field.shape[1]:
                raise IndexError(
                    f"Channel index {c} out of bounds for {field.shape[1]} channels"
                )
            field = field[:, c:c + 1, ...]         # keep channel dim -> [B, 1, H, W]

        device = field.device
        dirs = dirs.to(device)
        if not torch.isfinite(dirs).all():
            raise ValueError("dirs must contain finite values")
        scale = dirs.abs().amax(dim=-1, keepdim=True)
        if torch.any(scale == 0):
            raise ValueError("dirs must contain nonzero direction vectors")
        tolerance = 8 * torch.finfo(dirs.dtype).eps
        if normalize:
            # Scale first to avoid overflow/underflow for finite vectors.
            scaled = dirs / scale
            dirs = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        elif not torch.allclose(torch.linalg.vector_norm(dirs, dim=-1),
                                torch.ones_like(scale[..., 0]), rtol=tolerance, atol=tolerance):
            raise ValueError("normalize=False requires unit direction vectors")
        if torch.any(dirs[..., 2] < -tolerance):
            raise ValueError("Only the forward hemisphere (dir_z >= 0) is supported")
        dir_x, dir_y = dirs[..., 0], dirs[..., 1]

        # k-space query coordinates (radians/m). The spatial pitch is already
        # encoded in the stored kx/ky vectors, so it must NOT be reapplied here.
        # A vector wavelength must not broadcast into the query's spatial axes.
        k0 = source_field._k0_scalar(c)
        kx_query = k0 * dir_x                       # [N, H, W]
        ky_query = k0 * dir_y

        kx, ky = source_field.get_kx_ky()
        # Axes are public tensors and may have been mutated since construction.
        _validate_k_axis(kx, "kx")
        _validate_k_axis(ky, "ky")
        gx = _normalize_k_coordinate(kx_query, kx, align_corners)
        gy = _normalize_k_coordinate(ky_query, ky, align_corners)

        # grid_sample expects the last dim ordered as (x, y): x<->kx, y<->ky.
        grid = torch.stack([gx, gy], dim=-1)       # [N, H, W, 2]
        grid = grid.to(device=device, dtype=field.real.dtype)

        # Reconcile batch sizes: query batch must be 1 or equal to the field batch.
        B = field.shape[0]
        N = grid.shape[0]
        if N == 1 and B > 1:
            grid = grid.expand(B, *grid.shape[1:])
        elif B == 1 and N > 1:
            field = field.expand(N, *field.shape[1:])
        elif N != B:
            raise ValueError(
                f"Query batch ({N}) must be 1 or equal to field batch ({B})"
            )

        return _complex_grid_sample(
            field, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners,
        )
