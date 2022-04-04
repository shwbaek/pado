import torch
from pado.math.complex import Complex
import matplotlib.pyplot as plt


def fft(arr_c, normalized=False, pad_width=None, padval=0, shift=True):
    """
    Compute the Fast Fourier transform of a complex tensor 
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
        normalized: Normalize the FFT output. default: False
        pad_width: (tensor) pad width for the last spatial dimensions. 
        shift: flag for shifting the input data to make the zero-frequency located at the center of the arrc
    Returns:
        arr_c_fft: [B,Ch,R,C] FFT of the input complex tensor
    """

    arr_c = Complex(mag=arr_c.get_mag().clone(), ang=arr_c.get_ang().clone())
    if pad_width is not None:
        if padval == 0:
            arr_c.pad_zero(pad_width)
        else:
            return NotImplementedError('zero padding is only implemented for now')
    if shift:
        arr_c_shifted = ifftshift(arr_c)
    else:
        arr_c_shifted = arr_c
    arr_c_shifted.to_rect()

    #arr_c_shifted_stack = arr_c_shifted.get_stack()
    if normalized is False:
        normalized = "backward"
    else:
        normalized = "forward"
    arr_c_shifted_fft = torch.fft.fft2(arr_c_shifted.get_ctorch() , norm=normalized)
    arr_c_shifted_fft_c = Complex(real=arr_c_shifted_fft.real, imag=arr_c_shifted_fft.imag)
    if shift:
        arr_c_fft = fftshift(arr_c_shifted_fft_c)
    else:
        arr_c_fft = arr_c_shifted_fft_c

    return arr_c_fft



def ifft(arr_c, normalized=False, pad_width=None, shift=True):
    """
    Compute the inverse Fast Fourier transform of a complex tensor 
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
        normalized: Normalize the FFT output. default: False
        pad_width: (tensor) pad width for the last spatial dimensions. 
        shift: flag for inversely shifting the input data 
    Returns:
        arr_c_fft: [B,Ch,R,C] inverse FFT of the input complex tensor
    """

    arr_c = Complex(mag=arr_c.get_mag().clone(), ang=arr_c.get_ang().clone())
    if shift:
        arr_c_shifted = ifftshift(arr_c)
    else:
        arr_c_shifted = arr_c

    arr_c_shifted.to_rect()
    if normalized is False:
        normalized = "backward"
    else:
        normalized = "forward"
    arr_c_shifted_fft = torch.fft.ifft2(arr_c_shifted.get_ctorch(), norm=normalized)
    arr_c_shifted_fft_c = Complex(real=arr_c_shifted_fft.real, imag=arr_c_shifted_fft.imag)
    if shift:
        arr_c_fft = fftshift(arr_c_shifted_fft_c)
    else:
        arr_c_fft = arr_c_shifted_fft_c
    
    if pad_width is not None:
        arr_c_fft.crop(pad_width)

    return arr_c_fft

def fftshift(arr_c, invert=False):
    """
    Shift the complex tensor so that the  zero-frequency signal located at the center of the input
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
        invert: flag for inversely shifting the input data 
    Returns:
        arr_c: [B,Ch,R,C] shifted tensor
    """

    arr_c.to_rect()
    shift_adjust = 0 if invert else 1

    arr_c_shape = arr_c.shape()
    C = arr_c_shape[-1]
    R = arr_c_shape[-2]

    shift_len = (C + shift_adjust) // 2
    arr_c = arr_c[...,shift_len:].cat(arr_c[...,:shift_len], -1)

    shift_len = (R + shift_adjust) // 2
    arr_c = arr_c[...,shift_len:,:].cat(arr_c[...,:shift_len,:], -2)

    return arr_c


def ifftshift(arr_c):
    """
    Inversely shift the complex tensor 
    Args:
        arr_c: [B,Ch,R,C] complex tensor 
    Returns:
        arr_c: [B,Ch,R,C] shifted tensor
    """

    return fftshift(arr_c, invert=True)
