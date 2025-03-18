<div align="center">
  <img src="docs/images/banner_1.0.0.svg" width="100%">
</div>

<h1 align="center">PADO</h1>
<h3 align="center">Pytorch Automatic Differentiable Optics</h3>

<p align="center">
  <a href="#-installation">⚙️ Installation</a> •
  <a href="#-quickstart">🚀 Quickstart</a> •
  <a href="#-features">✨ Features</a> •
  <a href="#-license">📄 License</a>
</p>

<p align="center">
  <img alt="Python Version" src="https://img.shields.io/badge/python-3.6%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.10.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.16.0%2B-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.3.0%2B-FF5733?style=for-the-badge&logo=matplotlib&logoColor=white">
  <img alt="SciPy" src="https://img.shields.io/badge/SciPy-1.0.0%2B-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-F7DF1E?style=for-the-badge">
</p>

---

## 📋 Overview

🌊**PADO** (파도) is a cutting-edge framework for differentiable optical simulations powered by PyTorch. Inspired by the Korean word for "wave," PADO enables seamless and fully differentiable simulation workflows, perfect for researchers and developers in optical physics, computational imaging, and beyond.

---

## ✨ Features

- 🔥 **Fully Differentiable:** Integrates effortlessly with PyTorch Autograd.
- 🏎️ **CUDA Acceleration:** Leverages GPU hardware for ultra-fast simulations.
- 🧩 **Modular Components:** Easily customizable optical elements and simulation environments.
- 📊 **Visualization Tools:** Rich visualization with Matplotlib.
- ⚡ **Easy-to-use API:** Beginner-friendly API for rapid experimentation.

---

## ⚙️ Installation

You can install PADO via pip:

```bash
pip install pado
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/shwbaek/pado.git
```

For development installation:

```bash
git clone https://github.com/shwbaek/pado.git
cd pado
```

---

## 🚀 Quickstart

PADO includes a comprehensive set of example notebooks organized by topic:

### Exploring Examples

Browse our examples by category:

- **[1. Basics](./example/1_Basics/)**
  - [1.1 Pado fundamentals](./example/1_Basics/1.1_Pado_fundamentals.ipynb) - Learn about core components and building blocks
  - [1.2 RGB multi-wavelength](./example/1_Basics/1.2_RGB_multi_wavelength.ipynb) - Working with multiple wavelengths
  - [1.3 4-F with batch](./example/1_Basics/1.3_4-F_with_batch.ipynb) - Batch processing in 4-F systems
  - [1.4 How to use ASM options](./example/1_Basics/1.4_How2use_ASM_options.ipynb) - Angular Spectrum Method configuration

- **[2. Computer Generated Holography](./example/2_Computer_Generated_Holography/)**
  - [2.1 DPAC](./example/2_Computer_Generated_Holography/2.1_DPAC.ipynb) - Double Phase Amplitude Coding
  - [2.2 Multi-depth CGH](./example/2_Computer_Generated_Holography/2.2_multi_depth_cgh.ipynb) - Multi-plane holography
  - [2.3 CGH optimization](./example/2_Computer_Generated_Holography/2.3_cgh_optimization_gs_sgd_adam.ipynb) - GS, SGD, and Adam methods
  - [2.4 Multi-depth hologram with Adam](./example/2_Computer_Generated_Holography/2.4_multi_depth_hologram_generation_using_adam.ipynb) - Complex loss-based optimization
  - [2.5 Phase-only SLM optimization](./example/2_Computer_Generated_Holography/2.5_cgh_optimization_with_phase_only_slm.ipynb) - Optimization with phase-only spatial light modulators
  - [2.6 Multi-depth hologram with phase-only SLM](./example/2_Computer_Generated_Holography/2.6_multi_depth_hologram_generation_using_adam_with_phase_only_slm.ipynb) - Multi-plane optimization with phase-only SLMs

- **[3. Coded Imaging](./example/3_Coded_Imaging/)**
  - [3.1 Lens comparison](./example/3_Coded_Imaging/3.1_lens_comparison.ipynb) - Different lens models and wavefront observation
  - [3.2 Coded aperture comparison](./example/3_Coded_Imaging/3.2_coded_aperture_comparison.ipynb) - Coded aperture techniques
  - [3.3 Seeing through DOE](./example/3_Coded_Imaging/3.3_seeing_through_doe.ipynb) - Imaging through diffractive optical elements

- **[4. Polarization Imaging](./example/4_Polarization_Imaging/)**
  - [4.1 Polarization light](./example/4_Polarization_Imaging/4.1_polarization_light.ipynb) - Polarized light simulation

- **[5. Advanced Applications](./example/5_Advanced_Applications/)**
  - [5.1 Chromatic aberration singlet](./example/5_Advanced_Applications/5.1_chromatic_aberration_singlet.ipynb) - Chromatic aberration simulation

### Getting Started

For beginners, we recommend starting with [1.1 Pado fundamentals](./example/1_Basics/1.1_Pado_fundamentals.ipynb) to understand the core concepts and API.

```bash
jupyter notebook ./example/1_Basics/1.1_Pado_fundamentals.ipynb
```

---

## 📚 About

Developed and maintained by the [POSTECH Computer Graphics Lab](http://cg.postech.ac.kr/).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 📝 Citation

If you use Pado in your research, please cite Pado using the following BibText template:

```bib
@misc{Pado,
   Author = {Seung-Hwan Baek, Dong-Ha Shin, Yujin Jeon, Seung-Woo Yoon, Gawoon Ban, Hyunmo Kang},
   Year = {2025},
   Note = {https://github.com/shwbaek/pado},
   Title = {Pado: Pytorch Automatic Differentiable Optics}
}
```

<div align="center">
  <img src="docs/images/footer_1.0.0.svg" width="100%">
</div>
