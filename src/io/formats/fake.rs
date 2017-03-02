use io::*;

use densearray::prelude::*;
use sharedmem::{MemoryMap, SharedMem};

pub struct Fake3dData {
  dim:  (usize, usize, usize),
  buf:  SharedMem<u8>,
}

impl Fake3dData {
  pub fn new(dim: (usize, usize, usize)) -> Self {
    let buf_len = dim.flat_len();
    let mut buf = Vec::with_capacity(buf_len);
    buf.resize(buf_len, 0);
    let buf = SharedMem::new(buf);
    Fake3dData{dim: dim, buf: buf}
  }
}

impl IndexedData for Fake3dData {
  type Item = (Array3d<u8, SharedMem<u8>>, u32);

  fn len(&self) -> usize {
    50000
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    (Array3d::from_storage(self.dim, self.buf.clone()), 0)
  }
}
