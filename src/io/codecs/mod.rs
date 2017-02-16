use byteorder::*;
use densearray::prelude::*;
use sharedmem::*;

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
