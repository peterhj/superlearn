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

use std::fs::{File};
use std::marker::{PhantomData};
use std::path::{PathBuf};

pub trait KrizhevskyCifarFlavor {
  fn item_size() -> usize;
  fn label10_offset() -> usize;
  fn frame_offset() -> usize;
}

pub struct KrizhevskyCifar10Flavor;

impl KrizhevskyCifarFlavor for KrizhevskyCifar10Flavor {
  fn item_size() -> usize { 3073 }
  fn label10_offset() -> usize { 0 }
  fn frame_offset() -> usize { 1 }
}

pub struct KrizhevskyCifar100Flavor;

impl KrizhevskyCifarFlavor for KrizhevskyCifar100Flavor {
  fn item_size() -> usize { 3074 }
  fn label10_offset() -> usize { 1 }
  fn frame_offset() -> usize { 2 }
}

pub type Cifar10Data  = KrizhevskyCifarData<KrizhevskyCifar10Flavor>;
pub type Cifar100Data = KrizhevskyCifarData<KrizhevskyCifar100Flavor>;

#[derive(Clone)]
pub struct KrizhevskyCifarData<Flavor> {
  len:      usize,
  item_sz:  usize,
  label_p:  usize,
  frame_p:  usize,
  frame_d:  (usize, usize, usize),
  data_m:   SharedMem<u8>,
  _marker:  PhantomData<fn (Flavor)>,
}

impl<Flavor> KrizhevskyCifarData<Flavor> where Flavor: KrizhevskyCifarFlavor {
  pub fn open(data_path: PathBuf) -> KrizhevskyCifarData<Flavor> {
    let data_file = File::open(&data_path).unwrap();
    let file_meta = data_file.metadata().unwrap();
    let file_sz = file_meta.len() as usize;
    let item_sz = <Flavor as KrizhevskyCifarFlavor>::item_size();
    assert_eq!(0, file_sz % item_sz);
    let len = file_sz / item_sz;
    let label_p = <Flavor as KrizhevskyCifarFlavor>::label10_offset();
    let frame_p = <Flavor as KrizhevskyCifarFlavor>::frame_offset();
    let buf = match MemoryMap::open_with_offset(data_file, 0, file_sz) {
      Ok(buf) => buf,
      Err(e) => panic!("failed to mmap cifar batch file: {:?}", e),
    };
    KrizhevskyCifarData{
      len:      len,
      item_sz:  item_sz,
      label_p:  label_p,
      frame_p:  frame_p,
      frame_d:  (32, 32, 3),
      data_m:   SharedMem::new(buf),
      _marker:  PhantomData,
    }
  }
}

impl<Flavor> IndexedData for KrizhevskyCifarData<Flavor> where Flavor: KrizhevskyCifarFlavor {
  type Item = (Array3d<u8, SharedMem<u8>>, u32);

  fn len(&self) -> usize {
    self.len
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    let label = self.data_m[idx * self.item_sz + self.label_p] as u32;
    let frame_buf = self.data_m.slice_v2(idx * self.item_sz + self.frame_p .. (idx+1) * self.item_sz);
    (Array3d::from_storage(self.frame_d, frame_buf), label)
  }
}
