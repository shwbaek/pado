<img src="./docs/images/logo.png" width="120" alt="pado logo">

# Pado

Pado is a differentiable wave-optics library written in Pytorch. Objects and operations in Pado enjoy the automatic differentiation of Pytorch, allowing us to differentiate light-wave simulation. Pado can be intergrated into any pytorch-based network systems. 

Pado provides high-level abstraction of light-wave simulation which is particularly useful for users lacking of knowledge in wave optics. Pado achieves this with three main components: light waves, optical elements, and propagation. Constructing your own light-wave simulator with the three components can be done without caring too much about wave optics. This could improve your working efficiency by focusing on the core problem you want to tackle instead of studying and implementing wave optics from scratch.

Pado is a Korean word for wave.

# How to use Pado
We provide a jupyter notebook (`tutorial.ipynb`). More examples will be added in the future.

# Prerequisites
- Python 3
- Pytorch 1.10.0 
- Numpy

# About
Pado was created by [Seung-Hwan Baek](http://www.shbaek.com) at [POSTECH Computer Graphics Lab](http://cg.postech.ac.kr/). Pado was started as a personal project for differentiable cameras and displays. 
