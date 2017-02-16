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
