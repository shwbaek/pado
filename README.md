<p align="center">
<img src="./docs/images/logo.png" width="120" alt="pado logo">
</p>

# Pado: Differentiable Light-wave Simulation

Pado is a differentiable wave-optics library written in PyTorch. Objects and operations in Pado enjoy the automatic differentiation of Pytorch, allowing us to differentiate light-wave simulation. Pado can be integrated into any PyTorch-based network system.

Pado provides high-level abstractions of light-wave simulation, which is particularly useful for users lacking knowledge in wave optics. Pado achieves this with three main components: light waves, optical elements, and propagation. Constructing a light-wave simulator with the three components can be done without caring too much about wave optics. Thus, Pado could improve your working efficiency by focusing on the core problem you want to tackle instead of studying and implementing wave optics from scratch.

Pado is a Korean word for wave.

# How to use Pado
We provide a jupyter notebook (`tutorial.ipynb`). More examples will be added in the future.

# Prerequisites
- Python 3
- Pytorch 1.10.0 
- Numpy

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