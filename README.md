<p align="center">
<img src="./docs/images/logo.png" width="120" alt="pado logo">
</p>

# Pado: Differentiable Light-wave Simulation

Pado is a differentiable wave-optics library written in PyTorch. Thus, objects and operations defined in Pado are differentiable via PyTorch's automatic differentiation. Pado allows us to differentiate light-wave simulation and it can be integrated into any PyTorch-based network system. This makes Pado particularly useful for research in learning-based computational imaging and display.

Pado provides high-level abstractions of light-wave simulation, which is useful for users lacking knowledge in wave optics. Pado achieves this with three main objects: light, optical element, and propagator. Constructing your own light-wave simulator with the three objects could improve your working efficiency by focusing on the core problem you want to tackle instead of studying and implementing wave optics from scratch.

Pado is a Korean word for wave.

# How to use Pado
We are planning to provide jupyter notebooks on how to use Pado. 

# Prerequisites
- Python 
- Pytorch 
- Numpy
- Matplotlib
- Scipy

# About
Pado is maintained and developed by [Seung-Hwan Baek](http://www.shbaek.com) at [POSTECH Computer Graphics Lab](http://cg.postech.ac.kr/). Pado was started as a personal project for differentiable cameras and displays. 

If you use Pado in your research, please cite Pado using the following BibText template:

```bib
@misc{Pado,
   Author = {Seung-Hwan baek},
   Year = {2022},
   Note = {https://github.com/shwbaek/pado},
   Title = {Pado: Differentiable Light-wave Simulation}
}
```

# License
Seung-Hwan Baek have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:

Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.

The use of the software is for Non-Commercial Purposes only. As used in this Agreement, “Non-Commercial Purpose” means for the purpose of education or research in a non-commercial organization only. “Non-Commercial Purpose” excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [shwbaek@postech.ac.kr].

Warranty: POSTECH-CGLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.

Please refer to [here](./LICENSE) for more details.