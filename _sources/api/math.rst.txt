Mathematical Utilities
========================================

.. currentmodule:: pado.math

Constants
---------------------------------------------------

.. data:: nm
   
   :type: float
   :value: 1e-9

   Nanometer unit (1e-9 meters)

.. data:: um
   
   :type: float
   :value: 1e-6

   Micrometer unit (1e-6 meters)

.. data:: mm
   
   :type: float
   :value: 1e-3

   Millimeter unit (1e-3 meters)

.. data:: cm
   
   :type: float
   :value: 1e-2

   Centimeter unit (1e-2 meters)

.. data:: m
   
   :type: float
   :value: 1

   Meter unit (1 meter)

.. data:: s
   
   :type: float
   :value: 1

   Second unit (1 second)

.. data:: ms
   
   :type: float
   :value: 1e-3

   Millisecond unit (1e-3 seconds)

.. data:: us
   
   :type: float
   :value: 1e-6

   Microsecond unit (1e-6 seconds)

.. data:: ns
   
   :type: float
   :value: 1e-9

   Nanosecond unit (1e-9 seconds)

Functions
--------------------------------

.. autosummary::
   :toctree: _autosummary
   :template: function.rst
   :recursive:

   wrap_phase
   fft
   ifft
   calculate_psnr
   calculate_ssim
   gaussian_window
   sc_dft_1d
   sc_idft_1d
   sc_dft_2d
   sc_idft_2d
   compute_scasm_transfer_function
