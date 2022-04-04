from pado.math.fourier import fft, ifft

def conv_fft(img_c, kernel_c, pad_width=None):
    """
    Compute the convolution of an image with a convolution kernel using FFT
    Args:
        img_c: [B,Ch,R,C] image as a complex tensor 
        kernel_c: [B,Ch,R,C] convolution kernel as a complex tensor
        pad_width: (tensor) pad width for the last spatial dimensions. should be (0,0,0,0) for circular convolution. for linear convolution, pad zero by the size of the original image
    Returns:
        im_conv: [B,Ch,R,C] blurred image
    """

    img_fft = fft(img_c, pad_width=pad_width)
    kernel_fft = fft(kernel_c, pad_width=pad_width)
    return ifft( img_fft * kernel_fft, pad_width=pad_width)

