<p align="center">
<img src="./docs/images/logo.png" width="120" alt="pado logo">
</p>

# Pado: Differentiable Light-wave Simulation

Pado is a differentiable wave-optics library written in PyTorch. Thus, objects and operations defined in Pado are differentiable via PyTorch's automatic differentiation. Pado allows us to differentiate light-wave simulation and it can be integrated into any PyTorch-based network system. This makes Pado particularly useful for research in learning-based computational imaging and display.

Pado provides high-level abstractions of light-wave simulation, which is useful for users lacking knowledge in wave optics. Pado achieves this with three main objects: light, optical element, and propagator. Constructing your own light-wave simulator with the three objects could improve your working efficiency by focusing on the core problem you want to tackle instead of studying and implementing wave optics from scratch.

Pado is a Korean word for wave.

# How to use Pado
We provide a jupyter notebook (`./example/tutorial.ipynb`). More examples will be added later.

# Prerequisites
- Python 
- Pytorch 
- Numpy
- Matplotlib
- Scipy

# About
Pado is maintained and developed by [Seung-Hwan Baek](http://www.shbaek.com) at [POSTECH Computer Graphics Lab](http://cg.postech.ac.kr/). 
If you use Pado in your research, please cite Pado using the following BibText template:

```bib
@misc{Pado,
   Author = {Seung-Hwan baek},
   Year = {2022},
   Note = {https://github.com/shwbaek/pado},
   Title = {Pado: Differentiable Light-wave Simulation}
}
```
