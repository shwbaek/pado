Migrating to PADO 1.1
=====================

This page describes PADO 1.1.0. The distribution remains
``pado-optics`` and the Python import remains ``pado``. Existing classes and
methods remain available, but several numerical results change. A successful
import is not sufficient validation of a downstream optical calculation.

Fields and physical conventions
-------------------------------

Lengths are in metres; fields have shape ``(batch, channel, rows, columns)``.
Offsets are ``(y, x)`` and the time convention is ``Re(U exp(-i*omega*t))``.

* **Fresnel:** The old absolute-sum kernel normalization introduced an incorrect
  distance/grid-dependent field amplitude. PADO 1.1 uses scalar paraxial
  Fourier propagation and retains the carrier. Omitting a carrier can be a
  legitimate envelope convention; the old amplitude normalization was a
  separate defect. ``linear=True`` pads the periodic Fourier grid and crops;
  it does not automatically establish convergence to infinite free space.
* **Fraunhofer:** Results include pixel-area quadrature, the carrier,
  ``1/(i*lambda*z)`` and the output quadratic phase. Distance affects amplitude
  and phase as well as output pitch. Padding changes the returned sample pitch.
* **Rayleigh-Sommerfeld:** Coordinates are ``(index-size//2)*pitch``, with rows
  corresponding to y and columns to x. Source samples and batch members remain
  distinct. Explicit output coordinates define tilted or nonuniform planes.
* **ASM:** Evanescent components decay. Negative distance retains stable decay
  rather than amplifying evanescent components. With zero offset this is the
  adjoint of positive-distance propagation; a nonzero offset must also reverse
  sign to obtain the adjoint. Reversing distance alone keeps the same shift.
* **PDMS:** Wavelength inputs remain metres; coefficient evaluation converts
  to micrometres. The coefficients and material name are retained. Re-evaluate
  height-to-phase calibration. This unit correction does not validate a
  particular product or curing condition experimentally.
* **Zero distance:** Plain numeric zero without a requested transform retains
  the ordinary identity shortcut for ASM/Fresnel/RS. Tensor zero retains
  distance derivatives; offsets still translate; explicit RS planes are
  evaluated; FFT still transforms. Fraunhofer rejects zero distance.

Fresnel and Fraunhofer require their paraxial and far-field approximation
regimes. Sampling pitch, field extent and padding must be converged for the
application. Scalar propagation, local-periodic RCWA and measured optical
validation are separate scopes. This release does not integrate RCWA.

Devices, dependencies and autograd
----------------------------------

``PolarizedLight`` retains the public ``device="cuda:0"`` default for implicit
field creation. Pass ``device="cpu"`` explicitly for CPU construction. Supplied
field tensors determine the device; a missing polarization component follows
the supplied field's device and dtype. NumPy, Matplotlib
and SciPy are optional for their corresponding APIs. Import them directly;
incidental ``pado.np``, ``pado.plt``, ``pado.loadmat`` and ``pado.savemat``
exports are removed. See :doc:`installation` for extras.

Phase/amplitude setters keep the other component's value with its graph
detached, as in the previous public release. For joint optimization compose
the field explicitly:

.. code-block:: python

   import torch
   from pado import Light, Propagator

   device = "cuda" if torch.cuda.is_available() else "cpu"
   amplitude = torch.ones((1, 1, 128, 128), device=device, requires_grad=True)
   phase = torch.zeros_like(amplitude, requires_grad=True)
   light = Light(tuple(amplitude.shape), 2e-6, 532e-9,
                 field=amplitude * torch.exp(1j * phase))
   output = Propagator("ASM").forward(light, 1e-3).field
   loss = output.abs().square()[..., 48:80, 48:80].mean()
   loss.backward()

Replacing one channel preserves untouched-channel gradients and avoids
modifying a borrowed autograd leaf in place. New frequency-domain APIs are
documented in :doc:`api/kspace_light` and :doc:`api/sampler`.

Repeated geometry and precision
-------------------------------

Use ``Propagator("ASM").prepare(light, z=1e-3, linear=True, band_limit=True)``
to create a callable for repeated input fields at fixed wavelength, pitch,
distance, offset, shape, dtype and device. Batch size may vary. Field autograd
is retained. Use ordinary ``forward`` when optimizing geometry. The callable
retains the transfer H; release it when no longer needed. This is explicit
reuse, not a global cache.

Complex64 is the performance-oriented path. Complex128 ASM compensates geometry
and phase construction to preserve bits lost at long distances. This increases
setup time and temporary memory. Preparation amortizes setup over repeated
fields; it does not eliminate the initial cost. The default backend is Torch;
the optional fused CUDA backend is experimental. Parameter derivatives and
``torch.func`` transformations use Torch. Complex128 input alone does not mean
every optical-element buffer or operation is evaluated at that precision.

Validation and measured costs
-----------------------------

Installed-package checks and numerical-model validation cover different
properties. CUDA measurements below use one RTX 5070 and CUDA 12.8; they do not
establish performance or support for untested GPUs, macOS/MPS or ROCm.
The numerical and timing figures cover the stated implementations and sampled
conditions, rather than arbitrary optical systems.

The 2048-square ASM audit passes 120 sampled conditions per backend over
400/450/532/633/700 nm, pitches 2 micrometres and 0.4 wavelength, distances
0.1/1/100 mm, ordinary/band-limited ASM and complex64/128. Maximum field phase
residual above 1e-4 of reference peak amplitude is 1.160e-3 rad for complex64
and 2.368e-12 rad for complex128. References use MP80 transfer values for
difficult geometries and calibrated extended geometry elsewhere with extended
FFTs. These are not 80-digit FFTs, and the residual is not a certified true
phase-error bound. Illuminating all bins of a finite periodic grid is not
continuous-band or arbitrary-aperture validation.

A contained-Gaussian optical pipeline at 400/532/700 nm refines from 2048 to
4096 at fixed extent with full-coarse-grid relative field L2 2.24e-8 to
2.42e-8, without fitting phase or scale. This is one specified input and
geometry family. Six 2048 cases have exact ordinary/prepared fields, losses
and gradients; 12 fresh graphs show no retained-allocation growth.

For warmed complex128 transfer construction at 2048, 532 nm, 2-micrometre
pitch and 100 mm on RTX 5070 / Windows / Torch 2.11+cu128, median setup is
29.65 ms Torch / 8.57 ms fused, versus 2.56 ms for the previous less accurate
construction. Fixed peak extra allocation is 680.2 / 128.0 MiB; H retains
64 MiB per unpadded channel, or 256 MiB when both dimensions are doubled.
These timings exclude FFT application and cold compilation. Trainable geometry
uses Torch and requires additional saved autograd storage. Performance depends
on dtype, geometry, padding, device and reuse.
