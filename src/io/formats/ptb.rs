use io::*;

use densearray::prelude::*;

pub struct MikolovPtbTokenData {
}

impl IndexedData for MikolovPtbTokenData {
  type Item = Vec<Vec<u8>>;

  fn len(&self) -> usize {
    unimplemented!();
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    unimplemented!();
  }
}
