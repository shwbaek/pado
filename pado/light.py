import torch
import torch.nn.functional as F
from .complex import Complex
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

class Light:
    def __init__(self, R, C, pitch, wvl, device, amplitude=None, phase=None, real=None, imag=None, B=1):
        """
        Light wave that has a complex field (B,Ch,R,C) as a wavefront
        It takes the input wavefront in one of the following types
            1. amplitude and phase
            2. real and imaginary
            3. None ==> we initialize light with amplitude of one and phase of zero

        Args:
            R: row
            C: column
            pitch: pixel pitch in meter
            wvl: wavelength of light in meter
            device: device to store the wavefront of light. 'cpu', 'cuda:0', ...
            amplitude: [batch_size, # of channels, row, column] tensor of wavefront amplitude, default is None
            phase: [batch_size, # of channels, row, column] tensor of wavefront amplitude, default is None
            real: [batch_size, # of channels, row, column] tensor of wavefront real part, default is None
            imag: [batch_size, # of channels, row, column] tensor of wavefront imaginary part, default is None
            B: batch size
        """

        self.B = B
        self.R = R
        self.C = C
        self.pitch = pitch
        self.device = device
        self.wvl = wvl

        if (amplitude==None) and (phase == None) and (real != None) and (imag != None):
            self.field = Complex(real=real, imag=imag)
            
        elif (amplitude!=None) and (phase != None) and (real == None) and (imag == None):
            self.field = Complex(mag=amplitude, ang=phase)
            
        elif (amplitude==None) and (phase == None) and (real == None) and (imag == None):
            amplitude = torch.ones((B, 1, self.R, self.C), device=self.device)
            phase = torch.zeros((B, 1, self.R, self.C), device=self.device)
            self.field = Complex(mag=amplitude, ang=phase)
        
        else:
            NotImplementedError('nope!')
            
        
    def crop(self, crop_width):
        """
        Crop the light wavefront by crop_width
        Args:
            crop_width: (tuple) crop width of the tensor following torch functional pad 
        """

        self.field.crop(crop_width)
        self.R = self.field.size(2)
        self.C = self.field.size(3)

    def clone(self):
        """
        Clone the light and return it
        """

        return Light(self.R, self.C,
                     self.pitch, self.wvl, self.device,
                     amplitude=self.field.get_mag().clone(), phase=self.field.get_ang().clone(),
                     B=self.B)


    def pad(self, pad_width, padval=0):
        """
        Pad the light wavefront with a constant value by pad_width
        Args:
            pad_width: (tuple) pad width of the tensor following torch functional pad 
            padval: value to pad. default is zero
        """

        if padval == 0:
            self.set_amplitude(torch.nn.functional.pad(self.get_amplitude(), pad_width))
            self.set_phase(torch.nn.functional.pad(self.get_phase(), pad_width))
        else:
            return NotImplementedError('only zero padding supported')

        self.R += pad_width[0] + pad_width[1]
        self.C += pad_width[2] + pad_width[3]

    def set_real(self, real):
        """
        Set the real part of the light wavefront
        Args:
            real: real part in the rect representation of the complex number 
        """

        self.field.set_real(real)
        
    def set_imag(self, imag):
        """
        Set the imaginary part of the light wavefront
        Args:
            imag: imaginary part in the rect representation of the complex number 
        """

        self.field.set_imag(imag)
        
    def set_amplitude(self, amplitude):
        """
        Set the amplitude of the light wavefront
        Args:
            amplitude: amplitude in the polar representation of the complex number 
        """
        self.field.set_mag(amplitude)

    def set_phase(self, phase):
        """
        Set the phase of the complex tensor
        Args:
            phase: phase in the polar representation of the complex number 
        """
        self.field.set_ang(phase)


    def set_field(self, field):
        """
        Set the wavefront modulation of the complex tensor
        Args:
            field: wavefront as a complex number
        """
        self.field = field

    def set_pitch(self, pitch):
        """
        Set the pixel pitch of the complex tensor
        Args:
            pitch: pixel pitch in meter
        """
        self.pitch = pitch


    def get_amplitude(self):
        """
        Return the amplitude of the wavefront
        Returns:
            mag: magnitude in the polar representation of the complex number 
        """

        return self.field.get_mag()

    def get_phase(self):
        """
        Return the phase of the wavefront
        Returns:
            ang: angle in the polar representation of the complex number
        """

        return self.field.get_ang()

    def get_field(self):
        """
        Return the complex wavefront
        Returns:
            field: complex wavefront
        """

        return self.field

    def get_intensity(self):
        """
        Return the intensity of light wavefront
        Returns:
            intensity: intensity of light
        """
        return self.field.get_intensity()

    def get_bandwidth(self):
        """
        Return the bandwidth of light wavefront
        Returns:
            R_m: spatial height of the wavefront 
            C_m: spatial width of the wavefront 
        """

        return self.pitch*self.R, self.pitch*self.C

    def magnify(self, scale_factor, interp_mode='nearest'):
        '''
        Change the wavefront resolution without changing the pixel pitch
        Args:
            scale_factor: scale factor for interpolation used in tensor.nn.functional.interpolate
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        self.field.resize(scale_factor, interp_mode)
        self.R = self.field.mag.shape[-2]
        self.C = self.field.mag.shape[-1]


    def resize(self, target_pitch, interp_mode='nearest'):
        '''
        Resize the wavefront by changing the pixel pitch. 
        Args:
            target_pitch: new pixel pitch to use
            interp_mode: interpolation method used in torch.nn.functional.interpolate 'bilinear', 'nearest'
        '''
        scale_factor = self.pitch / target_pitch
        self.magnify(scale_factor, interp_mode)
        self.set_pitch(target_pitch)

    def set_spherical_light(self, z, dx=0, dy=0):
        '''
        Set the wavefront as spherical one coming from the position of (dx,dy,z). 
        Args:
            z: z distance of the spherical light source from the current light position
            dx: x distance of the spherical light source from the current light position
            dy: y distance of the spherical light source from the current light position
        '''

        [x, y] = np.mgrid[-self.C // 2:self.C // 2, -self.R // 2:self.R // 2].astype(np.float64)
        x = x * self.pitch
        y = y * self.pitch
        r = np.sqrt((x - dx) ** 2 + (y - dy) ** 2 + z ** 2)  # this is computed in double precision
        theta = 2 * np.pi * r / self.wvl
        theta = np.expand_dims(np.expand_dims(theta, axis=0), axis=0)%(2*np.pi)
        theta = theta.astype(np.float32)

        theta = torch.tensor(theta, device=self.device)
        mag = torch.ones_like(theta)

        self.set_phase(theta)
        self.set_amplitude(mag)

    def set_plane_light(self):
        '''
        Set the wavefront as a plane wave with zero phase and amptliude of one
        '''
        amplitude = torch.ones((1, 1, self.R, self.C), device=self.device)
        phase = torch.zeros((1, 1, self.R, self.C), device=self.device)
        self.set_amplitude(amplitude)
        self.set_phase(phase)

    def save(self, fn):
        '''
        Save the amplitude and phase of the light wavefront as a file
        Args:
            fn: filename to save. the format should be either "npy" or "mat"

        '''

        amp = self.get_amplitude().data.cpu().numpy()
        phase = self.get_phase().data.cpu().numpy()

        if fn[-3:] == 'npy':
            np.save(fn, amp, phase)
        elif fn[-3:] == 'mat':
            savemat(fn, {'amplitude':amp, 'phase':phase})
        else:
            print('extension in %s is unknown'%fn)
        print('light saved to %s\n'%fn)

    def visualize(self,b=0,c=0):
        """
        Visualize the light wave 
        Args:
            b: batch index to visualize default is 0
            c: channel index to visualize. default is 0
        """

        bw = self.get_bandwidth()
        
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        amplitude_b = self.get_amplitude().data.cpu()[b,c,...].squeeze()
        plt.imshow(amplitude_b,
                   extent=[0,bw[0]*1e3, 0, bw[1]*1e3], cmap='inferno')
        plt.title('amplitude')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(132)
        phase = self.get_phase().data.cpu()[b,c,...].squeeze()
        plt.imshow(self.get_phase().data.cpu()[b,c,...].squeeze(),
                   extent=[0,bw[0]*1e3, 0, bw[1]*1e3], cmap='hsv', vmin=-np.pi, vmax=np.pi)  # cyclic colormap
        plt.title('phase')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.subplot(133)
        intensity_b = self.get_intensity().data.cpu()[b,c,...].squeeze()
        plt.imshow(intensity_b,
                   extent=[0,bw[0]*1e3, 0, bw[1]*1e3], cmap='inferno')
        plt.title('intensity')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.colorbar()

        plt.suptitle('(%d,%d), pitch:%.2f[um], wvl:%.2f[nm], device:%s'%(self.R, self.C,
                                                                         self.pitch/1e-6, self.wvl/1e-9, self.device))
        plt.show()

    def shape(self):
        """
        Returns the shape of light wavefront
        Returns:
            shape
        """
        return self.field.shape()
