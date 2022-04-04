import numpy as np


class Material:
    def __init__(self, material_name):
        """
        Material of optical elements and its refractive index.
        Args:
            material name: name of the material. So far, we provide PDMS, FUSED_SILICA, VACCUM.
        """
        self.material_name = material_name

    def get_RI(self, wvl):
        """
        Return the refractive index of the current material for a wavelength
        Args:
            wvl: wavelength in meter
        Returns:
            RI: Refractive index at wvl
        """

        wvl_nm = wvl / 1e-9
        if self.material_name == 'PDMS':
            RI = np.sqrt(1 + (1.0057 * (wvl_nm**2))/(wvl_nm**2 - 0.013217))
        elif self.material_name == 'FUSED_SILICA':
            wvl_um = wvl_nm*1e-3
            RI = (1 + 0.6961663 / (1 - (0.0684043 / wvl_um) ** 2) + 0.4079426 / (1 - (0.1162414 / wvl_um) ** 2) + 0.8974794 / (1 - (9.896161 / wvl_um) ** 2)) ** .5 
            # https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
        elif self.material_name == 'VACUUM':
            RI = 1.0
        else:
            return NotImplementedError('%s is not in the RI list'%self.material_name)

        return RI

