pado.optical_element.phase2height
========================================

.. currentmodule:: pado.optical_element

.. py:function:: phase2height(phase_u, wvl, RI, minh=0)

   Convert phase to the least equivalent material height at or above ``minh``.
   The ambient refractive index is 1. Whole phase periods are added as needed;
   the phase-to-height mapping is not unique without this minimum-height rule.

   :param phase_u: Phase change tensor, in radians.
   :param wvl: Wavelength in metres.
   :param RI: Refractive index of the material. ``RI == 1`` raises ValueError.
   :param minh: Minimum height in metres; defaults to zero.
   :rtype: torch.Tensor
