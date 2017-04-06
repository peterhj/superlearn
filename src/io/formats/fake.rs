/*
Copyright 2017 the superlearn authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

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
