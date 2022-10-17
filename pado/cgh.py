import torch
import numpy as np
from .optical_element import SLM, RefractiveLens
from .light import Light
from .propagator import Propagator
from torch import optim
import torch.nn as nn
from torch.nn import functional as F


class CGH:
    def __init__(self):
        """
        Optimizing algorithm for CGH (computer generated holography)
        """

        self.f = None
        self.pitch = None
        self.wvl = None
        self.device = None

    def set_property(self, f, pitch, wvl, device):
        """
        Set property of SLM, light, and lens
        Args:
            f: focal length for lens and propagator
            pitch: pitch for SLM and light
            wvl: wavelength for SLM and light
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
        Returns:
            SLM: SLM generating desired image
        """

        self.f = f
        self.pitch = pitch
        self.wvl = wvl
        self.device = device

    def forward(self, img, epochs=1000, lr=1e-1):
        """
        Forward the desired image to make proper SLM
        Args:
            epochs: the number of iterating
            img: desired image
            epochs: number of iteration
            lr: learning rate for optimizing SLM
        Returns:
            phase parameter: modulated phase generating desired image
        """

        R, C = img.shape[-2:]
        phase_params = torch.nn.parameter.Parameter(
            data=torch.zeros((1, 1, R, C), device=self.device, requires_grad=True))

        optimizer = optim.Adam([phase_params], lr=lr)

        slm = SLM(R, C, self.pitch, self.wvl, self.device)
        light_ = Light(R, C, self.pitch, self.wvl, self.device)
        lens = RefractiveLens(R, C, self.pitch, self.f, self.wvl, self.device)
        prop = Propagator("auto")

        losses = []

        for i in range(epochs):
            slm.set_phase_change(phase_params, self.wvl)
            light = light_.clone()
            propagated_light = prop.forward(lens.forward(slm.forward(light)), self.f).get_intensity()

            loss = F.mse_loss(propagated_light, img).sqrt()
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return phase_params.data
