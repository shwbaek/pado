# PADO

Pytorch Automatic Differentiable Optics, developed by the POSTECH Computer Graphics Lab.

[Documentation](https://shwbaek.github.io/pado) · [Source](https://github.com/shwbaek/pado)

This is the 1.1.0 release. It keeps the `pado-optics`
distribution and `pado` import names. Scalar optical propagation runs in PyTorch
on the input device, with CPU and CUDA support and automatic differentiation
through supported Tensor parameters. K-space sampling is included; RCWA is not.

## Installation

Python 3.10 or later and PyTorch 2.10 or later are required. Install the PyTorch
build appropriate to your CPU or CUDA system, then install PADO. NumPy array/NPY
operations, plotting and MAT files have optional dependencies:

```bash
python -m pip install "https://github.com/shwbaek/pado/releases/download/1.1.0/pado_optics-1.1.0-py3-none-any.whl"
python -m pip install "pado-optics[array,viz,mat] @ https://github.com/shwbaek/pado/releases/download/1.1.0/pado_optics-1.1.0-py3-none-any.whl"  # optional APIs
```

The core dependency is PyTorch. Optional extras are `array` (NumPy), `viz`
(Matplotlib) and `mat` (SciPy). These packages are still needed when their
corresponding APIs are used.
PyPI publication is separate; an unpinned PyPI install may still select an
earlier release.

## Minimal differentiable propagation

```python
import torch
from pado import Light, Propagator

device = "cuda" if torch.cuda.is_available() else "cpu"
phase = torch.zeros((1, 1, 128, 128), device=device, requires_grad=True)
field = torch.exp(1j * phase)
light = Light(tuple(field.shape), pitch=2e-6, wvl=532e-9, field=field)
output = Propagator("ASM").forward(light, z=1e-3).field
loss = output.abs().square()[..., 48:80, 48:80].mean()
loss.backward()
```

Complex64 is the performance-oriented field type. Complex128 ASM uses
compensated transfer geometry and phase, which costs additional construction
time and memory. For repeated fields at fixed geometry, `Propagator.prepare`
reuses the transfer; it does not remove the initial construction cost.
Performance depends on dtype, grid, padding, device and whether geometry is reused.


Lengths are in metres and fields have shape `(batch, channel, row, column)`.
Fresnel and Fraunhofer have their respective paraxial/far-field approximations;
sampling, aperture and padding must be converged for the intended problem.
Numerical agreement does not establish physical measurement accuracy.

## Migrating from 1.0.1

This release intentionally changes some numerical results: Fresnel kernel
normalization is corrected, Fraunhofer includes pixel area and physical
amplitude/phase, RS uses the declared sample pitch, and PDMS coefficient units
are corrected. Revalidate downstream field/intensity comparisons. PolarizedLight
retains its `device="cuda:0"` default for newly allocated fields; pass
`device="cpu"` for explicit CPU construction. Supplied fields retain their actual
device and dtype, including when the other polarization component is created.
Phase/amplitude setters preserve the other component's value with a detached
graph. For joint optimization use `light.set_field(amplitude * torch.exp(1j*phase))`.
The optional fused ASM backend is experimental; the default backend is Torch.

## License and citation

PADO is distributed under the [MIT license](https://github.com/shwbaek/pado/blob/main/LICENSE).
Copyright and author credits remain in the source and package metadata.

```bibtex
@misc{Pado,
   Author = {Seung-Hwan Baek, Dong-Ha Shin, Yujin Jeon, Seung-Woo Yoon, Eunsue Choi, Gawoon Ban, Hyunmo Kang},
   Year = {2025},
   Note = {https://github.com/shwbaek/pado},
   Title = {Pado: Pytorch Automatic Differentiable Optics}
}
```
