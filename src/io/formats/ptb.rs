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
