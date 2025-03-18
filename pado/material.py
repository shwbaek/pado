########################################################
# The MIT License (MIT)
#
# PADO (Pytorch Automatic Differentiable Optics)
# Copyright (c) 2025 by POSTECH Computer Graphics Lab
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
from typing import Literal


class Material:
    def __init__(self, material_name: Literal["PDMS", "FUSED_SILICA", "VACUUM"]):
        """Create optical material instance with specified refractive index.

        Args:
            material_name (str): Material name: PDMS, FUSED_SILICA, or VACUUM

        Examples:
            >>> glass = Material("FUSED_SILICA")
            >>> ri = glass.get_RI(500e-9)  # Get RI at 500nm
        """
        self.material_name: str = material_name

    def get_RI(self, wvl: float) -> float:
        """Return refractive index at specified wavelength.

        Args:
            wvl (float): Wavelength in meters

        Returns:
            float: Refractive index

        Examples:
            >>> pdms = Material("PDMS")
            >>> n = pdms.get_RI(633e-9)  # Get RI at 633nm
        """
        wvl_nm: float = wvl / 1e-9

        if self.material_name == "PDMS":
            return np.sqrt(1 + (1.0057 * (wvl_nm ** 2)) / (wvl_nm ** 2 - 0.013217))
        
        if self.material_name == "FUSED_SILICA":
            wvl_um: float = wvl_nm * 1e-3
            return np.sqrt(
                1 + 0.6961663 / (1 - (0.0684043 / wvl_um) ** 2)
                + 0.4079426 / (1 - (0.1162414 / wvl_um) ** 2)
                + 0.8974794 / (1 - (9.896161 / wvl_um) ** 2)
            )
            # Source: https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
        
        if self.material_name == "VACUUM":
            return 1.0
        
        raise NotImplementedError(f"{self.material_name} is not in the RI list.")
