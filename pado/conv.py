import torch
import torch.nn.functional as F
from .fourier import fft, ifft

def conv_fft(a, b, pad_width=None):
    """
    Compute the convolution of an image with a convolution kernel using FFT
    Args:
        a: [B,Ch,R,C] input as Complex
        b: [B,Ch,R,C] convolution kernel as Complex
        pad_width: (tensor) pad width for the last spatial dimensions. should be (0,0,0,0) for circular convolution. for linear convolution, pad zero by the size of the original image
    Returns:
        im_conv: [B,Ch,R,C] convolved Complex
    """

    a_fft = fft(a, pad_width=pad_width)
    b_fft = fft(b, pad_width=pad_width)
    return ifft(a_fft * b_fft, pad_width=pad_width)


def conv_fft_img_psf(img: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """
    Compute the convolution of an image with a PSF (Point Spread Function) using FFT
    Args:
        img: [B, Ch, R, C] image tensor
        psf: [pB, 1, pR, pC] PSF tensor that is desired to be convolved with the image
    Returns:
        im_conv: [B, Ch, R, C] convolved image
    """
    B, Ch, R, C = img.shape
    pB, pCh, pR, pC = psf.shape
    assert (pB, pCh) == (B, 1)
    if Ch != 1:
        psf = torch.cat([psf for i in range(Ch)], axis=1)
    if (pR, pC) != (R, C):
        w1 = (R - pR) // 2
        w2 = R - pR - w1
        h1 = (C - pC) // 2
        h2 = C - pC - h1
        psf = F.pad(psf, (h1, h2, w1, w2))
        
    i = torch.fft.fftshift(torch.fft.fft2(img))
    p = torch.fft.fftshift(torch.fft.fft2(psf))
    
    im_conv = abs(torch.fft.ifftshift(torch.fft.ifft2(i * p)))
    
    return im_conv