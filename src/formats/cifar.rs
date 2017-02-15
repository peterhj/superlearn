use io::*;

use densearray::prelude::*;

pub struct CifarData {
}

impl IndexedData for CifarData {
  type Item = (Array3d<u8>, u32);

  fn len(&self) -> usize {
    unimplemented!();
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    unimplemented!();
  }
}
