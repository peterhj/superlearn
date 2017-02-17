use byteorder::*;
use densearray::prelude::*;
use ipp::*;
use rng::xorshift::*;
use sharedmem::*;

use rand::{Rng, SeedableRng};
use std::cmp::{min};
use std::io::{Cursor};

pub mod jpeg;

pub trait Decoder {
  type Src;
  type Dst;

  fn decode(&mut self, src: Self::Src) -> Self::Dst;
}

//pub fn default_decoder<F, Dec>() -> F where F: FnMut(Dec::Src) -> Dec::Dst, Dec: Decoder + Default {
pub fn default_decoder<Dec>() -> impl FnMut(Dec::Src) -> Dec::Dst where Dec: Decoder + Default {
  let mut decoder = <Dec as Default>::default();
  move |src: Dec::Src| decoder.decode(src)
}

#[derive(Default)]
pub struct LabelSuffixDecoder;

impl Decoder for LabelSuffixDecoder {
  type Src = SharedMem<u8>;
  type Dst = (SharedMem<u8>, u32);

  fn decode(&mut self, src: SharedMem<u8>) -> (SharedMem<u8>, u32) {
    let buf_len = src.len();
    assert!(buf_len >= 4);
    let mut label_reader = Cursor::new(&src[buf_len - 4 .. ]);
    let label: u32 = match label_reader.read_u32::<LittleEndian>() {
      Err(e) => panic!("failed to decode u32 label suffix: '{:?}'", e),
      Ok(x) => x,
    };
    (src.slice_v2( .. buf_len - 4), label)
  }
}

#[derive(Default)]
pub struct ImageTransposeDecoder;

impl Decoder for ImageTransposeDecoder {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn decode(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let mut tpixels = Vec::with_capacity(src.dim().flat_len());
    unsafe { tpixels.set_len(src.dim().flat_len()); }
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

pub struct RandomImageRescale {
  lo_side:  usize,
  hi_side:  usize,
  rng:      Xorshiftplus128Rng,
  //pyramid:  IppImageDownsamplePyramid<u8>,
  //downsamp: HashMap<usize, IppImageResize<u8>>,
  //upsamp:   HashMap<usize, IppImageResize<u8>>,
}

impl RandomImageRescale {
  pub fn new<R>(lo_side: usize, hi_side: usize, seed_rng: &mut R) -> Self where R: Rng {
    RandomImageRescale{
      lo_side:  lo_side,
      hi_side:  hi_side,
      rng:      Xorshiftplus128Rng::from_seed([seed_rng.next_u64(), seed_rng.next_u64()]),
      //pyramid:  IppImageDownsamplePyramid::new(),
    }
  }
}

impl Decoder for RandomImageRescale {
  type Src = Array3d<u8, SharedMem<u8>>;
  type Dst = Array3d<u8, SharedMem<u8>>;

  fn decode(&mut self, src: Array3d<u8, SharedMem<u8>>) -> Array3d<u8, SharedMem<u8>> {
    let rescale_side = self.rng.gen_range(self.lo_side, self.hi_side + 1);
    let lesser_side = min(src.dim().0, src.dim().1);
    let scale = rescale_side as f64 / lesser_side as f64;
    let (new_w, new_h) = if lesser_side == src.dim().0 {
      (lesser_side, (scale * src.dim().1 as f64).round() as usize)
    } else if lesser_side == src.dim().1 {
      ((scale * src.dim().0 as f64).round() as usize, lesser_side)
    } else {
      unreachable!();
    };
    if (new_w, new_h) == (src.dim().0, src.dim().1) {
      return src;
    }
    let mut buf = Vec::<u8>::with_capacity(new_w * new_h * src.dim().2);
    for c in 0 .. src.dim().2 {
      if scale > 1.0 {
        // TODO
        let lanczos = IppImageResize::new(IppImageResizeKind::Lanczos{nlobes: 2}, src.dim().0, src.dim().1, new_w, new_h);
      } else if scale < 1.0 {
        // TODO
        let cubic = IppImageResize::new(IppImageResizeKind::Cubic{b: 0.0, c: 0.5}, src.dim().0, src.dim().1, new_w, new_h);
      } else {
        unreachable!();
      }
    }
    unimplemented!();
  }
}
