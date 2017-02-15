use io::*;

use densearray::prelude::*;

pub struct MnistData {
}

impl IndexedData for MnistData {
  type Item = (Array2d<u8>, u32);

  fn len(&self) -> usize {
    unimplemented!();
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    unimplemented!();
  }
}
