use super::{Transform};

use densearray::prelude::*;
use sharedmem::*;

use ipp::*;
use rng::xorshift::*;

use rand::{Rng, SeedableRng};
use std::cmp::{min};

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
    if 0 == self.rng.gen_range(0, 1000) {
      println!("DEBUG: random rescale: rescale: {} lesser: {} src: {} x {} dst: {} x {}",
          rescale_side, lesser_side, src_w, src_h, dst_w, dst_h);
    }
    if (dst_w, dst_h) == (src_w, src_h) {
      return src;
    }
    let mut buf = Vec::<u8>::with_capacity(dst_w * dst_h * src.dim().2);
    buf.resize(dst_w * dst_h * src.dim().2, 0);
    if scale > 1.0 {
      let mut src_buf = IppImageBuf::alloc(src_w, src_h);
      let mut dst_buf = IppImageBuf::alloc(dst_w, dst_h);
      let mut upsample = IppImageResize::new(IppImageResizeKind::Cubic{b: 0.0, c: 0.5}, src_w, src_h, dst_w, dst_h);
      for c in 0 .. src.dim().2 {
        src_buf.load_packed(src_w, src_h, &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h]);
        upsample.resize(&src_buf, &mut dst_buf);
        dst_buf.store_packed(dst_w, dst_h, &mut buf[c * dst_w * dst_h .. (c+1) * dst_w * dst_h]);
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
        let downsample_kind = if next_w == dst_w && next_h == dst_h && prev_w != 2 * next_w && prev_h != 2 * next_h {
          IppImageResizeKind::Lanczos{nlobes: 2}
        } else {
          IppImageResizeKind::Linear
        };
        let mut downsample = IppImageResize::new(downsample_kind, prev_w, prev_h, next_w, next_h);
        for c in 0 .. src.dim().2 {
          if prev_w == src_w && prev_h == src_h {
            src_buf.load_packed(src_w, src_h, &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h]);
          } else {
            src_buf.load_packed(prev_w, prev_h, &tmp_buf[c * prev_w * prev_h .. (c+1) * prev_w * prev_h]);
          }
          downsample.resize(&src_buf, &mut dst_buf);
          if next_w == dst_w && next_h == dst_h {
            dst_buf.store_packed(dst_w, dst_h, &mut buf[c * dst_w * dst_h .. (c+1) * dst_w * dst_h]);
          } else {
            dst_buf.store_packed(next_w, next_h, &mut tmp_buf[c * next_w * next_h .. (c+1) * next_w * next_h]);
          }
        }
        prev_w = next_w;
        prev_h = next_h;
      }
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
      ipp_copy2d_u8(
          self.crop_w, self.crop_h,
          offset_x, offset_y, src_w,
          &src.as_slice()[c * src_w * src_h .. (c+1) * src_w * src_h],
          0, 0, self.crop_w,
          &mut buf[c * self.crop_w * self.crop_h .. (c+1) * self.crop_w * self.crop_h],
      );
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
