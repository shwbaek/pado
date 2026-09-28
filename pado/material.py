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
from typing import Literal, TYPE_CHECKING as _TYPE_CHECKING, Union, Any as _Any

import torch

if _TYPE_CHECKING:
    from numpy import ndarray as _NDArray
else:
    _NDArray = _Any


def _sqrt(value):
    """Dispatch without moving tensor data to the host or loading NumPy."""
    if isinstance(value, torch.Tensor):
        return torch.sqrt(value)
    if type(value) in (int, float):
        # Preserve NumPy's real-valued NaN result outside the model domain.
        return _math.sqrt(value) if value >= 0 else float("nan")
    # NumPy arrays remain supported at the compatibility boundary.
    import numpy as np
    return np.sqrt(value)


class Material:
    def __init__(self, material_name: Literal["PDMS", "FUSED_SILICA", "VACUUM", "NOA61"]):
        """Create optical material instance with specified refractive index.

        Args:
            material_name (str): Material name: PDMS, FUSED_SILICA, VACUUM, or NOA61

        Examples:
            >>> glass = Material("FUSED_SILICA")
            >>> ri = glass.get_RI(500e-9)  # Get RI at 500nm
        """
        self.material_name: str = material_name

    def get_RI(self, wvl: Union[float, torch.Tensor, _NDArray]) -> Union[float, torch.Tensor, _NDArray]:
        """Return refractive index at specified wavelength.

        Args:
            wvl (float, torch.Tensor, or numpy.ndarray): Wavelength in meters.
                Tensor inputs are evaluated entirely in PyTorch, retaining the
                input device, float32/float64 dtype, shape, and autograd connection.
                Values must be real, finite, and strictly positive.

        Returns:
            float, torch.Tensor, or numpy.ndarray: Refractive index. Scalars
                return a scalar and arrays are evaluated elementwise. VACUUM
                preserves the legacy scalar 1.0 return for non-tensor inputs;
                tensor inputs return ones with a zero wavelength derivative.

        Examples:
            >>> pdms = Material("PDMS")
            >>> n = pdms.get_RI(633e-9)  # Get RI at 633nm
        """
        if isinstance(wvl, torch.Tensor):
            if wvl.dtype not in (torch.float32, torch.float64):
                raise TypeError("wvl tensors must have dtype float32 or float64")
            valid = torch.isfinite(wvl) & (wvl > 0)
        elif type(wvl) in (int, float):
            valid = _math.isfinite(wvl) and wvl > 0
        else:
            import numpy as np
            if (not isinstance(wvl, (np.ndarray, np.generic))
                    or not np.issubdtype(wvl.dtype, np.number)
                    or np.iscomplexobj(wvl)):
                raise TypeError("wvl must contain real numeric wavelengths")
            valid = np.isfinite(wvl) & (wvl > 0)
        if not (valid if isinstance(valid, bool) else valid.all()):
            raise ValueError("wvl must be finite and strictly positive")

        wvl_nm = wvl / 1e-9

        if self.material_name == "PDMS":
            # Coefficients use micrometres; the public input remains in metres.
            # This equivalent form avoids cancellation in wavelength gradients.
            wvl_um = wvl / 1e-6
            return _sqrt(1 + 1.0057 / (1 - 0.013217 / wvl_um ** 2))
        
        if self.material_name == "FUSED_SILICA":
            wvl_um: float = wvl_nm * 1e-3
            return _sqrt(
                1 + 0.6961663 / (1 - (0.0684043 / wvl_um) ** 2)
                + 0.4079426 / (1 - (0.1162414 / wvl_um) ** 2)
                + 0.8974794 / (1 - (9.896161 / wvl_um) ** 2)
            )
            # Malitson (1965): 0.21-3.71 micrometres, 20 degrees Celsius.
            # https://doi.org/10.1364/JOSA.55.001205

        if self.material_name == "NOA61":
            # Preserve public PADO 12c57df's model (wavelength in micrometres).
            # Norland NOA 61 TDS, 25 degrees Celsius; no fitted interval stated.
            # https://norlandproducts.com/wp-content/uploads/2025/02/Norland-Products-NOA-61-TDS.pdf
            wvl_um = wvl_nm * 1e-3
            return 1.5375 + 0.00829045 * wvl_um**-2 - 0.000211046 * wvl_um**-4

        if self.material_name == "VACUUM":
            return wvl * 0 + 1 if isinstance(wvl, torch.Tensor) else 1.0
        
        raise NotImplementedError(f"{self.material_name} is not in the RI list.")
