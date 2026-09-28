"""Optional forward fusion using PyTorch's beta Jiterator/NVRTC interface.

The caller selects this only when transfer-parameter autograd is unnecessary.
There are no tensor caches or additional runtime/compiler dependencies.
"""
from functools import lru_cache

import torch


_CODE = r"""template <typename T> void pado_angular_transfer(T x, T y, T w, T z, T oy, T ox, T p, T hy, T hx, T nx, T ny, T grid, T& out_r, T& out_i) {
  const double tau = 6.283185307179586;
  double wx = __dmul_rn(w, x), wy = __dmul_rn(w, y);
  double a = __dmul_rn(wx, wx), b = __dmul_rn(wy, wy);
  double q = __dsub_rn(1.0, __dadd_rn(a, b));
  double root = sqrt(fabs(q));
  bool mask = true;
  if (hx >= 0.0) {
    double qm = __dsub_rn(__dsub_rn(1.0, a), b);
    double gamma = sqrt(qm > 0.0 ? qm : 0.0);
    double boundx = fabs(__dsub_rn(__dmul_rn(ox,gamma), __dmul_rn(__dmul_rn(z,w),x)));
    double boundy = fabs(__dsub_rn(__dmul_rn(oy,gamma), __dmul_rn(__dmul_rn(z,w),y)));
    mask = qm > 0.0 && boundx <= __dmul_rn(hx,gamma) && boundy <= __dmul_rn(hy,gamma);
    mask = mask || (z == 0.0 && fabs(ox) <= hx && fabs(oy) <= hy);
  }
  if (!__PRECISE__) {
    // Preserve the original complex64 operation order, including reciprocal.
    double k = __dmul_rn(__ddiv_rn(1.0, w), tau);
    double phase = __dmul_rn(__dmul_rn(k, z), q >= 0.0 ? root : 0.0);
    double shift = __dadd_rn(__dmul_rn(x, ox), __dmul_rn(y, oy));
    phase = __dadd_rn(phase, __dmul_rn(tau, shift));
    double amplitude = exp(__dmul_rn(__dmul_rn(-k, fabs(z)), q < 0.0 ? root : 0.0));
    out_r = __dmul_rn(__dmul_rn(amplitude, cos(phase)), mask ? 1.0 : 0.0);
    out_i = __dmul_rn(__dmul_rn(amplitude, sin(phase)), mask ? 1.0 : 0.0);
    return;
  }
  struct D { double h; double l; };
  auto sum = [](double a, double b) {
    double s = __dadd_rn(a,b), v = __dsub_rn(s,a);
    return D{s, __dadd_rn(__dsub_rn(a,__dsub_rn(s,v)), __dsub_rn(b,v))};
  };
  auto add = [&](D a, D b) {
    D s = sum(a.h,b.h); return sum(s.h, __dadd_rn(s.l,__dadd_rn(a.l,b.l)));
  };
  auto neg = [](D a) { return D{-a.h,-a.l}; };
  auto mul = [&](D a, D b) {
    double h = __dmul_rn(a.h,b.h);
    double e = __dadd_rn(fma(a.h,b.h,-h),__dadd_rn(__dmul_rn(a.h,b.l),__dmul_rn(a.l,b.h)));
    return sum(h,e);
  };
  auto div = [&](D a, D b) {
    double h = __ddiv_rn(a.h,b.h); D r = add(a,neg(mul(b,D{h,0.0})));
    return sum(h,__ddiv_rn(__dadd_rn(r.h,r.l),b.h));
  };
  D fx{x,0.0}, fy{y,0.0};
  if (grid != 0.0) {
    fx = div(D{nearbyint(x*(nx*p)),0.0},mul(D{nx,0.0},D{p,0.0}));
    fy = div(D{nearbyint(y*(ny*p)),0.0},mul(D{ny,0.0},D{p,0.0}));
  }
  D ax = mul(D{w,0.0},fx), ay = mul(D{w,0.0},fy);
  D qq = add(D{1.0,0.0},neg(add(mul(ax,ax),mul(ay,ay))));
  bool evanescent = qq.h < 0.0;
  if (evanescent) qq = neg(qq);
  D rr{0.0,0.0};
  if (qq.h != 0.0 || qq.l != 0.0) {
    double r = sqrt(qq.h); D e = add(qq,neg(mul(D{r,0.0},D{r,0.0})));
    rr = sum(r,__ddiv_rn(__dadd_rn(e.h,e.l),2.0*r));
  }
  D tau2{tau,2.4492935982947064e-16};
  D axial = mul(div(mul(tau2,D{z,0.0}),D{w,0.0}),rr);
  D shift = mul(tau2,add(mul(fx,D{ox,0.0}),mul(fy,D{oy,0.0})));
  D phase = add(evanescent ? D{0.0,0.0} : axial,shift);
  double sign = z < 0.0 ? -1.0 : 1.0;
  double amplitude = evanescent ? exp(-(axial.h+axial.l)*sign) : 1.0;
  double sh = sin(phase.h), ch = cos(phase.h), sl = sin(phase.l), cl = cos(phase.l);
  out_r = amplitude*fma(-sh,sl,ch*cl)*(mask ? 1.0 : 0.0);
  out_i = amplitude*fma(ch,sl,sh*cl)*(mask ? 1.0 : 0.0);
}"""


@lru_cache(maxsize=2)
def _kernel(precise=False):
    from torch.cuda.jiterator import _create_multi_output_jit_fn

    if not precise:
        # Omit unused tensor/keyword arguments as well as the compensated body;
        # merely making the branch constant still costs TensorIterator work.
        code = _CODE.split("  struct D", 1)[0] + "}"
        code = code.replace(" T p,", "").replace(" T nx, T ny, T grid,", "")
        return _create_multi_output_jit_fn(code.replace("__PRECISE__", "false"),
                                         num_outputs=2, hy=-1.0, hx=-1.0)
    code = _CODE.replace("__PRECISE__", "true" if precise else "false")
    return _create_multi_output_jit_fn(code, num_outputs=2, hy=-1.0, hx=-1.0,
                                     nx=1.0, ny=1.0, grid=0.0)


def angular_transfer(fx, fy, wavelengths, distance, offset, dtype, fft_period, grid_pitch=None):
    """Fuse complex64's existing order or compensated complex128 geometry.

    Real/imaginary outputs are joined using native Torch to preserve c64/c128.
    The BL period is normally Python metadata; CUDA scalar period tensors incur
    scalar extraction here. Parameter-grad calls are dispatched to Torch before
    reaching this module, while subsequent field operations retain autograd.
    """
    hy, hx = (-1.0, -1.0) if fft_period is None else (float(value)*.5 for value in fft_period)
    try:
        if dtype == torch.complex128:
            pitch = distance if grid_pitch is None else grid_pitch
            real, imag = _kernel(True)(fx[None, :], fy[:, None], wavelengths, distance,
                                      offset[0], offset[1], pitch, hy=hy, hx=hx,
                                      nx=float(fx.numel()), ny=float(fy.numel()),
                                      grid=float(grid_pitch is not None))
        else:
            real, imag = _kernel()(fx[None, :], fy[:, None], wavelengths, distance,
                                   offset[0], offset[1], hy=hy, hx=hx)
    except (ImportError, AttributeError, TypeError, RuntimeError) as error:
        raise RuntimeError(
            "The fused CUDA backend could not run PyTorch Jiterator/NVRTC. "
            "Use backend='torch' or a PyTorch CUDA build with compatible "
            "Jiterator/NVRTC support."
        ) from error
    return torch.complex(real, imag).to(dtype)
