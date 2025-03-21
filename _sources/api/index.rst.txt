API Reference
==========================

This section provides detailed API documentation for all PADO components.

Core Components
----------------------------

.. toctree::
   :maxdepth: 2
   :caption: Core Components

   light
   optical_element
   propagator
   material
   math

Module Overview
----------------------------

PADO consists of the following main modules:

1. **Light Module** (`pado.light`)
   
   Handles light wavefronts and their properties:
   - Wavefront generation and manipulation
   - Polarization handling
   - Intensity and phase calculations

2. **Optical Elements** (`pado.optical_element`)
   
   Defines various optical components:
   - Lenses and mirrors
   - Diffractive elements
   - Apertures and stops
   - Custom optical elements

3. **Propagator** (`pado.propagator`)
   
   Manages light propagation:
   - Angular spectrum method
   - Fresnel propagation
   - Rayleigh-Sommerfeld diffraction
   - Custom propagation methods

4. **Materials** (`pado.material`)
   
   Defines optical properties of materials:
   - Refractive indices
   - Dispersion relations
   - Absorption coefficients

5. **Math Utilities** (`pado.math`)
   
   Provides mathematical tools:
   - Fourier transforms
   - Special functions
   - Numerical methods
   - Optimization utilities 