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
  <img alt="License" src="https://img.shields.io/badge/license-Custom-F7DF1E?style=for-the-badge">
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
pip install -e .
```

---

## 🚀 Quickstart

Start exploring optics with our interactive Jupyter notebook tutorial:

```bash
jupyter notebook ./example/tutorial.ipynb
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
   Year = {2023},
   Note = {https://github.com/shwbaek/pado},
   Title = {Pado: Pytorch Automatic Differentiable Optics}
}
```

<div align="center">
  <img src="docs/images/footer_1.0.0.svg" width="100%">
</div>
