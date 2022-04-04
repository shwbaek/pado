import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.io import savemat

class Complex:
    
    def __init__(self, real=None, imag=None, mag=None, ang=None, native=None):
        """
        Tensor of complex numbers. We support three typese of complex-number modes which are all interchangable. 
            1. Rect
                In rect mode, we define the complex number x with real and imaginary parts as 
                    x = real + imag*1j
            2. Polar
                In polar mode, we define the complex number x with magnitude and angle as
                    x = mag*exp(1j*ang)
            3. Native PyTorch
                PyTorch recently started supporting complex numbers. We support using the native complex number in Pytorch.
                    x = complex
            You should select only one mode out of the three options and provide the required components
        Args:
            real: [batch_size, # of channels, row, column] tensor of real values. x = real + 1j * imag
            imag: [batch_size, # of channels, row, column] tensor of imaginary values. x = real + 1j * imag
            mag: [batch_size, # of channels, row, column] tensor of magnitude values. x = mag*exp(1j*ang)
            ang: [batch_size, # of channels, row, column] tensor of angle values. x = mag*exp(1j*ang)
            native: [batch_size, # of channels, row, column] tensor of native complex values. 
        """

        self.real, self.imag, self.mag, self.ang, self.native = None, None, None, None, None

        if real is None and imag is None and mag is not None and ang is not None and native is None:
            self.mag, self.ang = mag, ang
            self.mode = 'polar'

        elif real is not None and imag is not None and mag is None and ang is None and native is None:
            self.real, self.imag = real, imag
            self.mode = 'rect'
        elif real is None and imag is None and mag is None and ang is None and native is not None:
            self.native = native
            self.mode = 'native'
        else:
            return NotImplementedError('it should be either polar, rect, or native')

    def get_vis(self, amin=None, amax=None):
        """
        Return the colorized complex values in numpy based on amplitude and phase. 
        Args:
            amin: minimum amplitude to visualize
            amax: maximum amplitude to visualize
        Returns:
            vis: numpy-based rgb visualization of the complex tensor 
        """

        amp = self.get_mag().data.cpu().numpy()
        phase = self.get_ang().data.cpu().numpy()
        
        return vis_complex(amp*np.exp(1j*phase), amin, amax)
        

    def dim(self):
        """
        Return the dimension of the tensor
        Returns:
            dim
        """

        if self.mode == 'rect':
            return self.real.dim()
        elif self.mode == 'polar':
            return self.mag.dim()
        elif self.mode == 'native':
            return self.native.dim()

    def get_mag(self):
        """
        Return the magnitude of the complex tensor
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """

        if self.mode == 'native':
            return torch.abs(self.native)
        if self.mode == 'rect':
            self.to_polar()
        return self.mag

    def get_ang(self):
        """
        Return the angle of the complex tensor
        Returns:
            ang: angle in the polar representation of the complex number
        """
        if self.mode == 'native':
            return torch.ang(self.native)
        elif self.mode == 'rect':
            self.to_polar()
            return self.ang
        elif self.mode == 'polar':
            return self.ang

    def get_real(self):
        """
        Return the real part of the complex tensor
        Returns:
            real: real part in the rect representation of the complex number
        """
        if self.mode == 'polar':
            self.to_rect()
            return self.real
        elif self.mode == 'native':
            return self.native.real
        elif self.mode == 'rect':
            return self.real

    def get_imag(self):
        """
        Return the imaginary part of the complex tensor
        Returns:
            imag: imaginary part in the rect representation of the complex number
        """
        if self.mode == 'polar':
            self.to_rect()
            return self.imag
        elif self.mode == 'native':
            return self.native.imag
        elif self.mode == 'rect':
            return self.imag

    def get_native(self):
        """
        Return the native complex number of pytorch
        Returns:
            native
        """
        if self.mode is not 'native':
            self.to_native()
        return self.native


    def set_mag(self, mag):
        """
        Set the magnitude of the complex tensor
        Args:
            mag: magnitude in the polar representation of the complex number 
        """
        if self.mode == 'rect':
            self.to_polar()
            self.mag = mag
        elif self.mode == 'native':
            self.native = mag*torch.exp(1j*self.get_ang())
        elif self.mode == 'polar':
            self.mag = mag

    def set_ang(self, ang):
        """
        Set the angle of the complex tensor
        Args:
            ang: angle in the polar representation of the complex number 
        """
        if self.mode == 'rect':
            self.to_polar()
            self.ang = ang
        elif self.mode == 'native':
            self.native = self.get_mag()*torch.exp(1j*ang)
        elif self.mode == 'rect':
            self.ang = ang

    def set_real(self, real):
        """
        Set the real part of the complex tensor
        Args:
            real: real part in the rect representation of the complex number 
        """
        if self.mode == 'polar':
            self.to_rect()
            self.real = real
        elif self.mode == 'rect':
            self.real = real
        elif self.mode == 'native':
            self.native = real + self.get_imag()*1j

    def set_imag(self, imag):
        """
        Set the imaginary part of the complex tensor
        Args:
            imag: imaginary part in the rect representation of the complex number 
        """
        if self.mode == 'polar':
            self.to_rect()
            self.imag = imag
        elif self.mode == 'rect':
            self.imag = imag
        elif self.mode == 'native':
            self.native = self.get_real() + imag*1j

    def to_rect(self):
        """
        Change the current mode to rect 
        """
        if self.mode == 'polar':
            self.real, self.imag = polar2rect(self.mag, self.ang)
            self.mag, self.ang = None, None
        elif self.mode == 'native':
            self.real, self.imag = self.native.real, self.native.imag
            self.native = None
        self.mode = 'rect'

    def to_polar(self):
        """
        Change the current mode to polar 
        """
        if self.mode == 'rect':
            self.mag, self.ang = rect2polar(self.real, self.imag)
            self.real, self.imag = None, None
        elif self.mode == 'native':
            self.mag, self.ang = torch.abs(self.native), torch.ang(self.native)
            self.native = None
        self.mode = 'polar'

    def to_native(self):
        """
        Change the current mode to native 
        """
        if self.mode == 'rect':
            self.native = self.real+1j*self.imag
            self.real, self.imag = None, None
        elif self.mode == 'polar':
            self.native = self.mag*torch.exp(1j*self.ang)
            self.mag, self.ang = None, None
        self.mode = 'native'

    def shape(self):
        """
        Returns the shape of the complex tensor
        """
        if self.mode == 'polar':
            return self.mag.shape
        elif self.mode == 'rect':
            return self.real.shape
        elif self.mode == 'native':
            return self.native.shape
            
    def size(self, dim=None):
        """
        Returns the size of the complex tensor  
        Args:
            dim: size of a specific dimension or the entire dimension
        """
        if dim is None:
            return self.shape()
        else:
            shape = self.shape()
            return shape[dim]

    def pad_zero(self, pad_width):
        """
        Pad zeros to the complex tensor by pad_Width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
        """
        if self.mode == 'polar':
            self.mag = torch.nn.functional.pad(self.mag, pad_width)
            self.ang = torch.nn.functional.pad(self.ang, pad_width)
        elif self.mode == 'rect':
            self.real = torch.nn.functional.pad(self.real, pad_width)
            self.imag = torch.nn.functional.pad(self.imag, pad_width)
        elif self.mode == 'native':
            self.native = torch.nn.functional.pad(self.native, pad_width)

    def crop(self, crop_width):
        """
        Crop the complex tensor by crop_width
        Args:
            crop_width: (tuple) crop width of the tensor following torch functional pad 
        """
        if (crop_width[2] + crop_width[3] == 0) or (crop_width[0] + crop_width[1] == 0):
            return
            
        if self.mode == 'polar':
            self.mag = self.mag[...,crop_width[2]:-crop_width[3], crop_width[0]:-crop_width[1]]
            self.ang = self.ang[...,crop_width[2]:-crop_width[3], crop_width[0]:-crop_width[1]]
        elif self.mode == 'rect':
            self.real = self.real[...,crop_width[2]:-crop_width[3], crop_width[0]:-crop_width[1]]
            self.imag = self.imag[...,crop_width[2]:-crop_width[3], crop_width[0]:-crop_width[1]]


    def __getitem__(self, key):
        """
        indexing the complex tensor with key
        Args:
            key: index to the complex number
        Returns:
            complex number: complex number to return
        """

        if self.mode == 'polar':
            return Complex(mag=self.mag[key], ang=self.ang[key])
        elif self.mode == 'rect':
            return Complex(real=self.real[key], imag=self.imag[key])
        elif self.mode == 'native':
            return Complex(native=self.native[key])

    def reshape(self, new_shape):
        """
        Reshape the complex tensor 
        Args:
            new_shape: shape 
        """
        if self.mode == 'polar':
            self.mag = self.mag.reshape(new_shape)
            self.ang = self.ang.reshape(new_shape)
        elif self.mode == 'rect':
            self.real = self.real.reshape(new_shape)
            self.imag = self.imag.reshape(new_shape)
        elif self.mode == 'native':
            self.native = self.native.reshape(new_shape)
    
    def resize(self, scale_factor, interp_method='nearest'):
        """
        Resize the complex tensor
        Args:
            scale_factor: scale factor for the interpolation
            interp_method: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        """
        if self.mode == 'polar':
            self.mag = F.interpolate(self.mag, scale_factor=scale_factor,
                                                  mode=interp_method)
            self.ang = F.interpolate(self.ang, scale_factor=scale_factor,
                                              mode=interp_method)
        elif self.mode == 'rect':
            self.real = F.interpolate(self.real, scale_factor=scale_factor,
                                                  mode=interp_method)
            self.imag = F.interpolate(self.imag, scale_factor=scale_factor,
                                              mode=interp_method)
        elif self.mode == 'native':
            self.native = F.interpolate(self.native, scale_factor=scale_factor,
                                                mode=interp_method)

    def cat(self, other, dim):
        """
        Concatenate another complex tensor to the current complex tensor 
        Args:
            other: another complex tensor to be concatenated
            dim: dimension to concatenate
        Returns:
            self
        """

        if self.mode == 'polar':
            other.to_polar()
            self.mag = torch.cat((self.mag, other.mag), dim)
            self.ang = torch.cat((self.ang, other.ang), dim)

        elif self.mode == 'rect':
            other.to_rect()
            self.real = torch.cat((self.real, other.real), dim)
            self.imag = torch.cat((self.imag, other.imag), dim)

        elif self.mode == 'native':
            other.to_native()
            self.native = torch.cat((self.native, other.native), dim)

        return self

    def __matmul__(self, other):
        """
        Matrix multiplication of the current complex tensor with another one
        Args:
            other: another complex tensor to be matrix-multiplied
        Returns:
            tensor
        """
        # this is the matrix multiplication 
        if other.mode is not 'rect':
            other.to_rect()
        if self.mode is not 'rect':
            self.to_rect()
        
        return Complex(real=self.real@other.real - self.imag@other.imag,
                       imag=self.real@other.imag + self.imag@other.real)
    
    def __mul__(self, other):
        """
        element-wise multiplication of the current complex tensor with another one
        Args:
            other: another complex tensor to be multiplied
        Returns:
            tensor
        """
        if other.mode is not 'rect':
            other.to_rect()
        if self.mode is not 'rect':
            self.to_rect()
        
        return Complex(real=self.real*other.real - self.imag*other.imag,
                       imag=self.real*other.imag + self.imag*other.real)
    
    def __add__(self,other):
        """
        element-wise addition of the current complex tensor with another one
        Args:
            other: another complex tensor to be added
        Returns:
            tensor
        """
        if other.mode is not 'rect':
            other.to_rect()
        if self.mode is not 'rect':
            self.to_rect()
        
        return Complex(real=self.real + other.real,
                       imag=self.imag - other.real)   

    def __truediv__(self, other):
        """
        element-wise division of the current complex tensor with another one
        Args:
            other: another complex tensor to be added
        Returns:
            tensor
        """
        denominator = other.get_mag()**2
        if other.mode is not 'rect':
            other.to_rect()
        if self.mode is not 'rect':
            self.to_rect()
        real = (self.real*other.real + self.imag*other.imag)/denominator
        imag = (self.imag*other.real - self.real*other.imag)/denominator
        return Complex(real=real,
                       imag=imag)             

    def get_intensity(self):
        """
        intensity of the current tensor based on its polar coordinate 
        Returns:
            tensor
        """
        return (self * self.conj()).get_real()


    def conj(self):
        """
        complex conjugate of the tensor
        Returns:
            tensor
        """
        self.to_rect()
        return Complex(real=self.real, imag=-self.imag)

    def visualize(self, b=0, c=0):
        """
        Visualize the complex tensor 
        Args:
            b: batch index to visualize default is 0
            c: channel index to visualize. default is 0
        """

        if self.mode == 'polar':

            plt.figure(figsize=(14,5))

            plt.subplot(131)
            if len(self.mag.shape) == 2:
                plt.imshow(self.mag.data.cpu().squeeze())
            elif len(self.mag.shape) == 4:
                plt.imshow(self.mag[b,c,...].data.cpu().squeeze())
            plt.colorbar()
            plt.title('magnitude, b%d, c%d'%(b,c))

            plt.subplot(132)
            if len(self.ang.shape) == 2:
                plt.imshow(self.ang.data.cpu().squeeze())
            elif len(self.ang.shape) == 4:
                plt.imshow(self.ang[b,c,...].data.cpu().squeeze())
            plt.colorbar()
            plt.title('angle, b%d, c%d'%(b,c))

            plt.subplot(133)
            vis = self.get_vis()
            if len(self.ang.shape) == 2:
                plt.imshow(vis)
            elif len(self.ang.shape) == 4:
                plt.imshow(vis[b,c,...].squeeze())
            plt.title('complex, b%d, c%d'%(b,c))

        elif self.mode == 'rect':

            plt.figure(figsize=(14,5))

            plt.subplot(131)
            if len(self.real.shape) == 2:
                plt.imshow(self.real.data.cpu().squeeze())
            elif len(self.real.shape) == 4:
                plt.imshow(self.real[b,c,...].data.cpu().squeeze())
            plt.colorbar()
            plt.title('real, b%d, c%d'%(b,c))

            plt.subplot(132)
            if len(self.imag.shape) == 2:
                plt.imshow(self.imag.data.cpu().squeeze())
            elif len(self.imag.shape) == 4:
                plt.imshow(self.imag[b,c,...].data.cpu().squeeze())
            plt.colorbar()
            plt.title('imag, b%d, c%d'%(b,c))

            plt.subplot(133)
            vis = self.get_vis()
            if len(self.ang.shape) == 2:
                plt.imshow(vis)
            elif len(self.ang.shape) == 4:
                plt.imshow(vis[b,c,...].squeeze())
            plt.title('complex, b%d, c%d'%(b,c))

        plt.show()

    def save(self, fn):
        """
        save the complex tensor as a file 
        Args:
            fn: filename to save. the format should be either "npy" or "mat"
        """
        amp = self.get_mag().data.cpu().numpy()
        phase = self.get_ang().data.cpu().numpy()

        if fn[-3:] == 'npy':
            np.save(fn, amp, phase)
        elif fn[-3:] == 'mat':
            savemat(fn, {'amplitude':amp, 'phase':phase})
        else:
            print('extension in %s is unknown'%fn)
        print('light saved to %s\n'%fn)




import pandas as pd
import matplotlib

# colormap of a complex value
cmap_path = './pado/math/cmap_phase.txt'
data_frame = pd.read_csv(cmap_path, header=None)  

dat = data_frame.values
cmap_p = np.zeros((dat.shape[0], 3))
for i in range(dat.shape[0]):
    c = dat[i][0].split(' ')
    for j in range(3):
        cmap_p[i, j] = float(c[j])
cmap_a = np.linspace(0, 1, dat.shape[0])[..., np.newaxis].repeat(3, axis=1)
cmap_phase = matplotlib.colors.ListedColormap(cmap_p, name='phase')

def vis_complex(arr_c, amin=None, amax=None):
    """
    visualize the complex tensor in numpy based on amplitude and phase. 
    Args:
        amin: minimum amplitude to visualize
        amax: maximum amplitude to visualize
    Returns:
        vis: numpy-based rgb visualization of the complex tensor 
    """
    # arr_c: complex array
    amp = np.abs(arr_c)
    if amin is None:
        amin = amp.min()
    if amax is None:
        amax = amp.max()
    amp_n = (amp - amin) / (amax - amin)
    amp_n[amp_n<0]=0
    amp_n[amp_n>1]=1
    phase = np.angle(arr_c)
    phase_n = np.mod(phase + np.pi, 2*np.pi) / (2 * np.pi)
    return cmap_a[(amp_n * 255).astype(np.uint8), :] * cmap_p[(phase_n * 255).astype(np.uint8), :]


def polar2rect(mag, ang):
    """
    convert polar representation to the rect representation 
    Args:
        mag: magnitude of the complex tensor
        ang: angle of the complex tensor
    Returns:
        real: real part of the complex tensor
        imag: imaginary part of the complex tensor
    """
    real = mag * torch.cos(ang)
    imag = mag * torch.sin(ang)

    return real, imag


def rect2polar(real, imag):
    """
    convert rect representation to the polar representation 
    Args:
        real: real part of the complex tensor
        imag: imaginary part of the complex tensor
    Returns:
        mag: magnitude of the complex tensor
        ang: angle of the complex tensor
    """

    mag_sq = (real ** 2 + imag ** 2)

    real2 = real.clone()
    imag2 = imag.clone()

    # we need this to prevent nan error of the gradient value for the backpropagation when mag_sq becomes zero
    eps = torch.finfo(mag_sq.dtype).tiny
    real2[mag_sq<10*eps] = 10*eps
    imag2[mag_sq<10*eps] = 10*eps

    mag = (real2**2 + imag2**2)**0.5
    ang = torch.atan2(imag2, real2)

    return mag, ang


if __name__ == '__main__':


    a = torch.zeros((10,10),requires_grad=True)
    b = torch.zeros((10,10),requires_grad=True)

    optimizer = torch.optim.Adam([a], lr=1e-3)

    for i in range(1000):
        c = Complex(real=a,imag=b)
        c.to_polar()
        # c_mag = c.get_mag()
        c_mag = (c * c.conj()).get_real()

        # c_mag = torch.sqrt((a+1e-10)**2 + (b+1e-10)**2)
        # ang = torch.atan2(b, a)

        # c_mag = c.get_real()

        optimizer.zero_grad()
        d = torch.ones((10,10))
        loss = (c_mag - d).abs().sum()
        loss.backward()
        optimizer.step()



