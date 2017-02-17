use byteorder::*;
use densearray::prelude::*;
use sharedmem::*;

use rand::{Rng, SeedableRng};
use std::io::{Cursor};

pub mod image;
pub mod jpeg;

pub trait Transform {
  type Src;
  type Dst;

  fn transform(&mut self, src: Self::Src) -> Self::Dst;
}

/*//pub fn default_decoder<F, Dec>() -> F where F: FnMut(Dec::Src) -> Dec::Dst, Dec: Decoder + Default {
pub fn default_decoder<Dec>() -> impl FnMut(Dec::Src) -> Dec::Dst where Dec: Decoder + Default {
  let mut decoder = <Dec as Default>::default();
  move |src: Dec::Src| decoder.decode(src)
}*/

#[derive(Default)]
pub struct LabelSuffixDecoder;

impl Transform for LabelSuffixDecoder {
  type Src = SharedMem<u8>;
  type Dst = (SharedMem<u8>, u32);

  fn transform(&mut self, src: SharedMem<u8>) -> (SharedMem<u8>, u32) {
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
