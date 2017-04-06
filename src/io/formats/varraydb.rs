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

use sharedmem::*;
use varraydb::shared::{SharedVarrayDb};

use std::path::{PathBuf};

pub struct SharedVarrayData {
  db:   SharedVarrayDb,
}

impl SharedVarrayData {
  pub fn open(prefix: PathBuf) -> SharedVarrayData {
    let db = SharedVarrayDb::open(&prefix).unwrap();
    SharedVarrayData{
      db:   db,
    }
  }
}

impl IndexedData for SharedVarrayData {
  type Item = SharedMem<u8>;

  fn len(&self) -> usize {
    self.db.len()
  }

  fn get(&mut self, idx: usize) -> SharedMem<u8> {
    let item = self.db.get(idx);
    item
  }
}
