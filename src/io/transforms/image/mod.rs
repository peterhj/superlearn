use super::{Transform};

use densearray::prelude::*;
use sharedmem::*;

use ipp::*;
use rng::xorshift::*;

use rand::{Rng, SeedableRng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use std::cmp::{min};
use std::marker::{PhantomData};

#[derive(Default)]
pub struct ImageCast<T, U> {
  _marker:  PhantomData<fn (T, U)>,
}

impl Transform for ImageCast<u8, f32> {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<f32, SharedMem<f32>>;

  fn transform(&mut self, src: Self::Src) -> Self::Dst {
    let mut buf = Vec::with_capacity(src.dim().flat_len());
    unsafe { buf.set_len(src.dim().flat_len()) };
    buf.flatten_mut().cast_from_u8(src.as_view().flatten());
    Array3d::from_storage(src.dim(), SharedMem::new(buf))
  }
}

impl Transform for ImageCast<f32, u8> {
  type Src = Array3d<f32, SharedMem<f32>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Self::Src) -> Self::Dst {
    let mut buf = Vec::with_capacity(src.dim().flat_len());
    unsafe { buf.set_len(src.dim().flat_len()) };
    buf.flatten_mut().round_clamp_from_f32(src.as_view().flatten());
    Array3d::from_storage(src.dim(), SharedMem::new(buf))
  }
}

pub struct PlanarImageZeroPad {
  pad_w:    usize,
  pad_h:    usize,
}

impl PlanarImageZeroPad {
  pub fn new(pad_w: usize, pad_h: usize) -> Self {
    PlanarImageZeroPad{
      pad_w:    pad_w,
      pad_h:    pad_h,
    }
  }
}

impl Transform for PlanarImageZeroPad {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let (src_w, src_h, chan_dim) = src.dim();
    let dst_w = src_w + 2 * self.pad_w;
    let dst_h = src_h + 2 * self.pad_h;
    let mut buf = Vec::with_capacity(dst_w * dst_h * chan_dim);
    buf.resize(dst_w * dst_h * chan_dim, 0);
    let src_buf = src.as_slice();
    for c in 0 .. chan_dim {
      for y in 0 .. src_h {
        for x in 0 .. src_w {
          let px = x + self.pad_w;
          let py = y + self.pad_h;
          buf[px + dst_w * (py + dst_h * c)] = src_buf[x + src_w * (y + src_h * c)];
        }
      }
    }
    Array3d::from_storage((dst_w, dst_h, chan_dim), SharedMem::new(buf))
  }
}

pub struct PlanarImageAddColorNoise {
  rng:      Xorshiftplus128Rng,
}

pub struct PlanarImageAddPixelPCANoise {
  scale:    f64,
  svals:    Vec<f64>,
  svecs:    Vec<Vec<f64>>,
  dist:     Normal,
  rng:      Xorshiftplus128Rng,
}

impl PlanarImageAddPixelPCANoise {
  pub fn new<R>(scale: f64, svals: Vec<f64>, svecs: Vec<Vec<f64>>, seed_rng: &mut R) -> Self where R: Rng {
    PlanarImageAddPixelPCANoise{
      scale:    scale,
      svals:    svals,
      svecs:    svecs,
      dist:     Normal::new(0.0, scale),
      rng:      Xorshiftplus128Rng::from_seed(seed_rng),
    }
  }
}

impl Transform for PlanarImageAddPixelPCANoise {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let (src_w, src_h, chan_dim) = src.dim();
    assert_eq!(3, chan_dim); // TODO
    let mut alphas = Vec::with_capacity(chan_dim);
    for _ in 0 .. chan_dim {
      alphas.push(self.dist.ind_sample(&mut self.rng));
    }
    let mut noise = Vec::with_capacity(chan_dim);
    for row in 0 .. chan_dim {
      let mut z = 0.0;
      for col in 0 .. chan_dim {
        z += alphas[col] * self.svals[col] * self.svecs[col][row];
      }
      noise.push(z);
    }
    let mut buf = Vec::with_capacity(src_w * src_h * chan_dim);
    let src_buf = src.as_slice();
    for c in 0 .. chan_dim {
      for p in 0 .. src_w * src_h {
        let u = src_buf[p + src_w * src_h * c] as f64;
        let mut v: f64 = (u + noise[c] + 0.5);
        if v < 0.0 {
          v = 0.0;
        } else if v > 255.0 {
          v = 255.0;
        }
        buf.push(v as u8);
      }
    }
    Array3d::from_storage((src_w, src_h, chan_dim), SharedMem::new(buf))
  }
}

pub struct PlanarImageLinearResize<T> {
  src_dim:  (usize, usize),
  dst_dim:  (usize, usize),
  _marker:  PhantomData<fn (T)>,
}

impl<T> PlanarImageLinearResize<T> {
  pub fn new(src_dim: (usize, usize), dst_dim: (usize, usize)) -> Self {
    PlanarImageLinearResize{
      src_dim:  src_dim,
      dst_dim:  dst_dim,
      _marker:  PhantomData,
    }
  }
}

impl Transform for PlanarImageLinearResize<f32> {
  type Src = Array3d<f32, SharedMem<f32>>;
  type Dst = Array3d<f32, SharedMem<f32>>;

  fn transform(&mut self, src: Array3d<f32, SharedMem<f32>>) -> Array3d<f32, SharedMem<f32>> {
    // TODO: use `self.src_dim`.
    let (src_w, src_h) = (src.dim().0, src.dim().1);
    let (dst_w, dst_h) = self.dst_dim;
    if (dst_w, dst_h) == (src_w, src_h) {
      return src;
    }
    let mut buf = Vec::<f32>::with_capacity(dst_w * dst_h * src.dim().2);
    buf.resize(dst_w * dst_h * src.dim().2, 0.0);
    let mut src_buf = IppImageBuf::alloc(src_w, src_h);
    let mut dst_buf = IppImageBuf::alloc(dst_w, dst_h);
    let mut resizer = IppImageResize::create(IppImageResizeKind::Linear, src_w, src_h, dst_w, dst_h).unwrap();
    for c in 0 .. src.dim().2 {
      src_buf.write(&src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h]);
      resizer.resize(&src_buf, &mut dst_buf);
      dst_buf.read(&mut buf[c * dst_w * dst_h .. (c+1) * dst_w * dst_h]);
    }
    Array3d::from_storage((dst_w, dst_h, src.dim().2), SharedMem::new(buf))
  }
}

#[derive(Default)]
pub struct ImageTranspose;

impl Transform for ImageTranspose {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let mut tpixels = Vec::with_capacity(src.dim().flat_len());
    //unsafe { tpixels.set_len(src.dim().flat_len()); }
    tpixels.resize(src.dim().flat_len(), 0);
    let (chan, width, height) = src.dim();
    let mut p = 0;
    let pixels = src.as_slice();
    for y in 0 .. height {
      for x in 0 .. width {
        for c in 0 .. chan {
          let q = x + width * (y + height * c);
          tpixels[q] = pixels[p];
          p += 1;
        }
      }
    }
    Array3d::from_storage((width, height, chan), SharedMem::new(tpixels))
  }
}

pub struct ImageRandomRescale {
  lo_side:  usize,
  hi_side:  usize,
  rng:      Xorshiftplus128Rng,
  //pyramid:  IppImageDownsamplePyramid<u8>,
  //downsamp: HashMap<usize, IppImageResize<u8>>,
  //upsamp:   HashMap<usize, IppImageResize<u8>>,
}

impl ImageRandomRescale {
  pub fn new<R>(lo_side: usize, hi_side: usize, seed_rng: &mut R) -> Self where R: Rng {
    ImageRandomRescale{
      lo_side:  lo_side,
      hi_side:  hi_side,
      rng:      Xorshiftplus128Rng::from_seed(seed_rng),
      //pyramid:  IppImageDownsamplePyramid::new(),
    }
  }
}

impl Transform for ImageRandomRescale {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let rescale_side = self.rng.gen_range(self.lo_side, self.hi_side + 1);
    let (src_w, src_h) = (src.dim().0, src.dim().1);
    let lesser_side = min(src_w, src_h);
    let scale = rescale_side as f64 / lesser_side as f64;
    let (dst_w, dst_h) =
        if lesser_side == src_w {
          (rescale_side, (scale * src_h as f64).round() as usize)
        } else if lesser_side == src_h {
          ((scale * src_w as f64).round() as usize, rescale_side)
        } else {
          unreachable!();
        };
    /*if 0 == self.rng.gen_range(0, 1000) {
      println!("DEBUG: random rescale: rescale: {} lesser: {} src: {} x {} dst: {} x {}",
          rescale_side, lesser_side, src_w, src_h, dst_w, dst_h);
    }*/
    if (dst_w, dst_h) == (src_w, src_h) {
      return src;
    }
    let mut buf = Vec::<u8>::with_capacity(dst_w * dst_h * src.dim().2);
    buf.resize(dst_w * dst_h * src.dim().2, 0);
    if scale > 1.0 {
      let mut src_buf = IppImageBuf::alloc(src_w, src_h);
      let mut dst_buf = IppImageBuf::alloc(dst_w, dst_h);
      let mut upsample = IppImageResize::create(IppImageResizeKind::Cubic{b: 0.0, c: 0.5}, src_w, src_h, dst_w, dst_h).unwrap();
      for c in 0 .. src.dim().2 {
        src_buf.write_strided(src_w, src_h, &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h]);
        upsample.resize(&src_buf, &mut dst_buf);
        dst_buf.read_strided(dst_w, dst_h, &mut buf[c * dst_w * dst_h .. (c+1) * dst_w * dst_h]);
      }
    } else if scale < 1.0 {
      let mut tmp_buf = Vec::<u8>::with_capacity(src_w * src_h * src.dim().2);
      tmp_buf.resize(src_w * src_h * src.dim().2, 0);
      let mut src_buf = IppImageBuf::alloc(src_w, src_h);
      let mut dst_buf = IppImageBuf::alloc(src_w, src_h);
      let mut prev_w = src_w;
      let mut prev_h = src_h;
      while prev_w > dst_w || prev_h > dst_h {
        let next_w = if prev_w >= 2 * dst_w {
          (prev_w + 1) / 2
        } else {
          dst_w
        };
        let next_h = if prev_h >= 2 * dst_h {
          (prev_h + 1) / 2
        } else {
          dst_h
        };
        // FIXME(20170325)
        //let downsample_kind = if next_w == dst_w && next_h == dst_h && prev_w != 2 * next_w && prev_h != 2 * next_h {
        let downsample_kind = if next_w == dst_w && next_h == dst_h && (prev_w < 2 * next_w || prev_h < 2 * next_h) {
          IppImageResizeKind::Lanczos{nlobes: 2}
        } else {
          IppImageResizeKind::Linear
        };
        let mut downsample = IppImageResize::create(downsample_kind, prev_w, prev_h, next_w, next_h).unwrap();
        for c in 0 .. src.dim().2 {
          if prev_w == src_w && prev_h == src_h {
            src_buf.write_strided(src_w, src_h, &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h]);
          } else {
            src_buf.write_strided(prev_w, prev_h, &tmp_buf[c * prev_w * prev_h .. (c+1) * prev_w * prev_h]);
          }
          downsample.resize(&src_buf, &mut dst_buf);
          if next_w == dst_w && next_h == dst_h {
            dst_buf.read_strided(dst_w, dst_h, &mut buf[c * dst_w * dst_h .. (c+1) * dst_w * dst_h]);
          } else {
            dst_buf.read_strided(next_w, next_h, &mut tmp_buf[c * next_w * next_h .. (c+1) * next_w * next_h]);
          }
        }
        prev_w = next_w;
        prev_h = next_h;
      }
      assert_eq!(prev_w, dst_w);
      assert_eq!(prev_h, dst_h);
    } else {
      unreachable!();
    }
    Array3d::from_storage((dst_w, dst_h, src.dim().2), SharedMem::new(buf))
  }
}

pub struct PlanarImageRandomRescaleF32 {
  lo_side:  usize,
  hi_side:  usize,
  rng:      Xorshiftplus128Rng,
}

impl PlanarImageRandomRescaleF32 {
  pub fn new<R>(lo_side: usize, hi_side: usize, seed_rng: &mut R) -> Self where R: Rng {
    PlanarImageRandomRescaleF32{
      lo_side:  lo_side,
      hi_side:  hi_side,
      rng:      Xorshiftplus128Rng::from_seed(seed_rng),
    }
  }
}

impl Transform for PlanarImageRandomRescaleF32 {
  type Src = Array3d<f32, SharedMem<f32>>;
  type Dst = Array3d<f32, SharedMem<f32>>;

  fn transform(&mut self, src: Array3d<f32, SharedMem<f32>>) -> Array3d<f32, SharedMem<f32>> {
    let rescale_side = self.rng.gen_range(self.lo_side, self.hi_side + 1);
    let (src_w, src_h) = (src.dim().0, src.dim().1);
    let lesser_side = min(src_w, src_h);
    let scale = rescale_side as f64 / lesser_side as f64;
    let (dst_w, dst_h) =
        if lesser_side == src_w {
          (rescale_side, (scale * src_h as f64).round() as usize)
        } else if lesser_side == src_h {
          ((scale * src_w as f64).round() as usize, rescale_side)
        } else {
          unreachable!();
        };
    if (dst_w, dst_h) == (src_w, src_h) {
      return src;
    }
    let mut buf = Vec::<f32>::with_capacity(dst_w * dst_h * src.dim().2);
    buf.resize(dst_w * dst_h * src.dim().2, 0.0);
    if scale > 1.0 {
      let mut src_buf = IppImageBuf::alloc(src_w, src_h);
      let mut dst_buf = IppImageBuf::alloc(dst_w, dst_h);
      let mut upsample = IppImageResize::create(IppImageResizeKind::Cubic{b: 0.0, c: 0.5}, src_w, src_h, dst_w, dst_h).unwrap();
      for c in 0 .. src.dim().2 {
        src_buf.write_strided(src_w, src_h, &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h]);
        upsample.resize(&src_buf, &mut dst_buf);
        dst_buf.read_strided(dst_w, dst_h, &mut buf[c * dst_w * dst_h .. (c+1) * dst_w * dst_h]);
      }
    } else if scale < 1.0 {
      let mut tmp_buf = Vec::<f32>::with_capacity(src_w * src_h * src.dim().2);
      tmp_buf.resize(src_w * src_h * src.dim().2, 0.0);
      let mut src_buf = IppImageBuf::alloc(src_w, src_h);
      let mut dst_buf = IppImageBuf::alloc(src_w, src_h);
      let mut prev_w = src_w;
      let mut prev_h = src_h;
      while prev_w > dst_w || prev_h > dst_h {
        let next_w = if prev_w >= 2 * dst_w {
          (prev_w + 1) / 2
        } else {
          dst_w
        };
        let next_h = if prev_h >= 2 * dst_h {
          (prev_h + 1) / 2
        } else {
          dst_h
        };
        // FIXME(20170325)
        //let downsample_kind = if next_w == dst_w && next_h == dst_h && prev_w != 2 * next_w && prev_h != 2 * next_h {
        let downsample_kind = if next_w == dst_w && next_h == dst_h && (prev_w < 2 * next_w || prev_h < 2 * next_h) {
          //IppImageResizeKind::Lanczos{nlobes: 2}
          IppImageResizeKind::Cubic{b: 0.0, c: 0.5}
        } else {
          IppImageResizeKind::Linear
        };
        let mut downsample = IppImageResize::create(downsample_kind, prev_w, prev_h, next_w, next_h).unwrap();
        for c in 0 .. src.dim().2 {
          if prev_w == src_w && prev_h == src_h {
            src_buf.write_strided(src_w, src_h, &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h]);
          } else {
            src_buf.write_strided(prev_w, prev_h, &tmp_buf[c * prev_w * prev_h .. (c+1) * prev_w * prev_h]);
          }
          downsample.resize(&src_buf, &mut dst_buf);
          if next_w == dst_w && next_h == dst_h {
            dst_buf.read_strided(dst_w, dst_h, &mut buf[c * dst_w * dst_h .. (c+1) * dst_w * dst_h]);
          } else {
            dst_buf.read_strided(next_w, next_h, &mut tmp_buf[c * next_w * next_h .. (c+1) * next_w * next_h]);
          }
        }
        prev_w = next_w;
        prev_h = next_h;
      }
      assert_eq!(prev_w, dst_w);
      assert_eq!(prev_h, dst_h);
    } else {
      unreachable!();
    }
    Array3d::from_storage((dst_w, dst_h, src.dim().2), SharedMem::new(buf))
  }
}

pub struct ImageRandomCrop {
  crop_w:   usize,
  crop_h:   usize,
  rng:      Xorshiftplus128Rng,
}

impl ImageRandomCrop {
  pub fn new<R>(crop_w: usize, crop_h: usize, seed_rng: &mut R) -> Self where R: Rng {
    ImageRandomCrop{
      crop_w:   crop_w,
      crop_h:   crop_h,
      rng:      Xorshiftplus128Rng::from_seed(seed_rng),
    }
  }
}

impl Transform for ImageRandomCrop {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let (src_w, src_h, _) = src.dim();
    assert!(self.crop_w <= src_w);
    assert!(self.crop_h <= src_h);
    let offset_x = self.rng.gen_range(0, src_w - self.crop_w + 1);
    let offset_y = self.rng.gen_range(0, src_h - self.crop_h + 1);
    let mut buf = Vec::with_capacity(self.crop_w * self.crop_h * src.dim().2);
    buf.resize(self.crop_w * self.crop_h * src.dim().2, 0);
    for c in 0 .. src.dim().2 {
      /*ipp_copy2d_u8(
          self.crop_w, self.crop_h,
          offset_x, offset_y, src_w,
          &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h],
          0, 0, self.crop_w,
          &mut buf[c * self.crop_w * self.crop_h .. (c+1) * self.crop_w * self.crop_h],
      );*/
      let src = src.as_slice();
      let src_start = c * src_w * src_h;
      let dst_start = c * self.crop_w * self.crop_h;
      for y in 0 .. self.crop_h {
        for x in 0 .. self.crop_w {
          buf[dst_start + x + self.crop_w * y] = src[src_start + offset_x + x + src_w * (offset_y + y)];
        }
      }
    }
    Array3d::from_storage((self.crop_w, self.crop_h, src.dim().2), SharedMem::new(buf))
  }
}

pub struct ImageCenterCrop {
  crop_w:   usize,
  crop_h:   usize,
}

impl ImageCenterCrop {
  pub fn new(crop_w: usize, crop_h: usize) -> Self {
    ImageCenterCrop{
      crop_w:   crop_w,
      crop_h:   crop_h,
    }
  }
}

impl Transform for ImageCenterCrop {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let (src_w, src_h, _) = src.dim();
    assert!(self.crop_w <= src_w);
    assert!(self.crop_h <= src_h);
    let offset_x = (src_w - self.crop_w) / 2;
    let offset_y = (src_h - self.crop_h) / 2;
    let mut buf = Vec::with_capacity(self.crop_w * self.crop_h * src.dim().2);
    buf.resize(self.crop_w * self.crop_h * src.dim().2, 0);
    for c in 0 .. src.dim().2 {
      /*ipp_copy2d_u8(
          self.crop_w, self.crop_h,
          offset_x, offset_y, src_w,
          &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h],
          0, 0, self.crop_w,
          &mut buf[c * self.crop_w * self.crop_h .. (c+1) * self.crop_w * self.crop_h],
      );*/
      let src = src.as_slice();
      let src_start = c * src_w * src_h;
      let dst_start = c * self.crop_w * self.crop_h;
      for y in 0 .. self.crop_h {
        for x in 0 .. self.crop_w {
          buf[dst_start + x + self.crop_w * y] = src[src_start + offset_x + x + src_w * (offset_y + y)];
        }
      }
    }
    Array3d::from_storage((self.crop_w, self.crop_h, src.dim().2), SharedMem::new(buf))
  }
}

pub struct ImageRandomFlipX {
  rng:  Xorshiftplus128Rng,
}

impl ImageRandomFlipX {
  pub fn new<R>(seed_rng: &mut R) -> Self where R: Rng {
    ImageRandomFlipX{
      rng:  Xorshiftplus128Rng::from_seed(seed_rng),
    }
  }
}

impl Transform for ImageRandomFlipX {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn transform(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let (src_w, src_h, chan_dim) = src.dim();
    let mut buf = Vec::with_capacity(src_w * src_h * chan_dim);
    buf.resize(src_w * src_h * chan_dim, 0);
    let src_buf = src.as_slice();
    match self.rng.gen_range(0, 2) {
      0 => {
        buf.copy_from_slice(src_buf);
      }
      1 => {
        for c in 0 .. chan_dim {
          for y in 0 .. src_h {
            for x in 0 .. src_w {
              let xp = src_w - 1 - x;
              buf[x + src_w * (y + src_h * c)] = src_buf[xp + src_w * (y + src_h * c)];
            }
          }
        }
      }
      _ => unreachable!(),
    }
    Array3d::from_storage(src.dim(), SharedMem::new(buf))
  }
}
