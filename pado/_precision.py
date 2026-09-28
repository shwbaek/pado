"""Compensated complex128 ASM geometry; ordinary Torch preserves parameter AD.

Two-double sums and Dekker products retain phase bits lost by million-radian
binary64 intermediates. Only transfer construction pays this precision cost.
"""
import torch


def _pair(x):
    return x, torch.zeros_like(x)


def _sum(a, b):
    s = a + b
    v = s - a
    return s, (a - (s - v)) + (b - v)


def _add(a, b):
    h, l = _sum(a[0], b[0])
    return _sum(h, l + (a[1] + b[1]))


def _neg(a):
    return -a[0], -a[1]


def _mul(a, b):
    h = a[0] * b[0]
    ca, cb = 134217729. * a[0], 134217729. * b[0]
    ah, bh = ca - (ca - a[0]), cb - (cb - b[0])
    al, bl = a[0] - ah, b[0] - bh
    error = ((ah * bh - h) + ah * bl + al * bh) + al * bl
    return _sum(h, error + (a[0] * b[1] + a[1] * b[0]))


def _div(a, b):
    h = a[0] / b[0]
    r = _add(a, _neg(_mul(b, _pair(h))))
    return _sum(h, (r[0] + r[1]) / b[0])


def angular_transfer(fx, fy, wavelength, distance, offset, pitch):
    axes = []
    for f in (fx, fy):
        if pitch is None:
            axes.append(_pair(f))
        else:
            # Recover exact integer bins before division, including native FFT
            # order. The provided pitch carries the geometry derivative.
            indices = (f * (f.numel() * pitch)).round()
            period = _mul(_pair(torch.full_like(pitch, f.numel())), _pair(pitch))
            axes.append(_div(_pair(indices), period))
    x, y = tuple(v[None, :] for v in axes[0]), tuple(v[:, None] for v in axes[1])
    wx, wy = _mul(_pair(wavelength), x), _mul(_pair(wavelength), y)
    q = _add(_pair(torch.ones_like(wavelength)), _neg(_add(_mul(wx, wx), _mul(wy, wy))))
    negative, nonzero = q[0] < 0, (q[0] != 0) | (q[1] != 0)
    qa = tuple(torch.where(negative, -v, v) for v in q)
    rh = torch.sqrt(torch.where(nonzero, qa[0], 1.))
    error = _add(qa, _neg(_mul(_pair(rh), _pair(rh))))
    root = _sum(rh, (error[0] + error[1]) / (2 * rh))
    root = tuple(torch.where(nonzero, v, 0.) for v in root)
    del q, qa, rh, error, nonzero, wx, wy
    tau = (torch.full_like(wavelength, 6.283185307179586),
           torch.full_like(wavelength, 2.4492935982947064e-16))
    axial = _mul(_div(_mul(tau, _pair(distance)), _pair(wavelength)), root)
    shift = _mul(tau, _add(_mul(x, _pair(offset[1])), _mul(y, _pair(offset[0]))))
    del root, axes, x, y
    ph, pl = _add(tuple(torch.where(negative, 0., v) for v in axial), shift)
    # sign(0)=0 preserves abs(z)'s zero subgradient for evanescent decay.
    attenuation = tuple(torch.where(negative, v * distance.sign(), 0.) for v in axial)
    del axial, shift, negative
    amplitude = torch.exp(-(attenuation[0] + attenuation[1]))
    del attenuation
    high = torch.polar(amplitude, ph)
    return high * torch.polar(torch.ones_like(pl), pl)
