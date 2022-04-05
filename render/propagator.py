import torch
import numpy as np
from pado.math import fourier
from pado.math.complex import Complex
from pado.math.conv import conv_fft


def compute_pad_width(field, linear):
    """
    Compute the pad width of an array for FFT-based convolution
    Args:
        field: (B,Ch,R,C) complex tensor
        linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
    Returns:
        pad_width: pad-width tensor
    """

    if linear:
        R,C = field.shape()[-2:]
        pad_width = (C//2, C//2, R//2, R//2)
    else:
        pad_width = (0,0,0,0)
    return pad_width 

def unpad(field_padded, pad_width):
    """
    Unpad the already-padded complex tensor 
    Args:
        field_padded: (B,Ch,R,C) padded complex tensor 
        pad_width: pad-width tensor
    Returns:
        field: unpadded complex tensor
    """

    field = field_padded[...,pad_width[2]:-pad_width[3],pad_width[0]:-pad_width[1]]
    return field

class Propagator:
    def __init__(self, mode):
        """
        Free-space propagator of light waves
        One can simulate the propagation of light waves on free space (no medium change at all).
        Args:
            mode: type of propagator. currently, we support "Fraunhofer" propagation or "Fresnel" propagation. Use Fraunhofer for far-field propagation and Fresnel for near-field propagation. 
        """
        self.mode = mode

    def forward(self, light, z, linear=True):
        """
        Forward the incident light with the propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        if self.mode == 'Fraunhofer':
            return self.forward_Fraunhofer(light, z, linear)
        if self.mode == 'Fresnel':
            return self.forward_Fresnel(light, z, linear)
        else:
            return NotImplementedError('%s propagator is not implemented'%self.mode)


    def forward_Fraunhofer(self, light, z, linear=True):
        """
        Forward the incident light with the Fraunhofer propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter. 
                The propagated wavefront is independent w.r.t. the travel distance z.
                The distance z only affects the size of the "pixel", effectively adjusting the entire image size.
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """

        pad_width = compute_pad_width(light.field, linear)
        field_propagated = fourier.fft(light.field, pad_width=pad_width)
        field_propagated = unpad(field_propagated, pad_width)

        # based on the Fraunhofer reparametrization (u=x/wvl*z) and the Fourier frequency sampling (1/bandwidth)
        bw_r = light.get_bandwidth()[0]
        bw_c = light.get_bandwidth()[1]
        pitch_r_after_propagation = light.wvl*z/bw_r
        pitch_c_after_propagation = light.wvl*z/bw_c

        light_propagated = light.clone()

        # match the x-y pixel pitch using resampling
        if pitch_r_after_propagation >= pitch_c_after_propagation:
            scale_c = 1
            scale_r = pitch_r_after_propagation/pitch_c_after_propagation
            pitch_after_propagation = pitch_c_after_propagation
        elif pitch_r_after_propagation < pitch_c_after_propagation:
            scale_r = 1
            scale_c = pitch_c_after_propagation/pitch_r_after_propagation
            pitch_after_propagation = pitch_r_after_propagation

        light_propagated.set_field(field_propagated)
        light_propagated.magnify((scale_r,scale_c))
        light_propagated.set_pitch(pitch_after_propagation)

        return light_propagated

    def forward_Fresnel(self, light, z, linear):
        """
        Forward the incident light with the Fresnel propagator. 
        Args:
            light: incident light 
            z: propagation distance in meter. 
            linear: True or False, flag for linear convolution (zero padding) or circular convolution (no padding)
        Returns:
            light: light after propagation
        """
        field_input = light.field

        # compute the convolutional kernel 
        sx = light.C / 2
        sy = light.R / 2
        x = np.arange(-sx, sx, 1)
        y = np.arange(-sy, sy, 1)
        xx, yy = np.meshgrid(x,y)
        xx = torch.from_numpy(xx*light.pitch).to(light.device)
        yy = torch.from_numpy(yy*light.pitch).to(light.device)
        k = 2*np.pi/light.wvl  # wavenumber
        phase = (k*(xx**2 + yy**2)/(2*z))
        amplitude = torch.ones_like(phase) / z / light.wvl
        conv_kernel = Complex(mag=amplitude, ang=phase) 
        
        # Propagation with the convolution kernel
        pad_width = compute_pad_width(field_input, linear)
        
        #field_input.visualize()
        #conv_kernel.visualize()
        
        field_propagated = conv_fft(field_input, conv_kernel, pad_width)

        # return the propagated light
        light_propagated = light.clone()
        light_propagated.set_field(field_propagated)

        return light_propagated

