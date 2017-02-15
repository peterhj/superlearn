use io::*;

use densearray::prelude::*;
use sharedmem::{MemoryMap, SharedMem};

use byteorder::{ReadBytesExt, BigEndian};
use std::fs::{File};
use std::path::{PathBuf};

pub fn mmap_mnist_idx_file(mut file: File) -> (usize, Vec<usize>, MemoryMap<u8>) {
  let magic: u32 = file.read_u32::<BigEndian>().unwrap();
  let magic2 = (magic >> 8) as u8;
  let magic3 = (magic >> 0) as u8;
  assert_eq!(magic2, 0x08);
  let ndims = magic3 as usize;
  let mut dims = vec![];
  for _ in 0 .. ndims {
    dims.push(file.read_u32::<BigEndian>().unwrap() as usize);
  }
  let n = dims[0] as usize;
  let mut frame_size = 1;
  for d in 1 .. ndims {
    frame_size *= dims[d] as usize;
  }
  let buf = match MemoryMap::open_with_offset(file, (1 + ndims) * 4, frame_size * n) {
    Ok(buf) => buf,
    Err(e) => panic!("failed to mmap buffer: {:?}", e),
  };
  let mut colmaj_dims = vec![];
  for d in (1 .. ndims).rev() {
    colmaj_dims.push(dims[d]);
  }
  (n, colmaj_dims, buf)
}

#[derive(Clone)]
pub struct MnistData {
  len:      usize,
  frame_sz: usize,
  frame_d:  Vec<usize>,
  frames_m: SharedMem<u8>,
  labels_m: SharedMem<u8>,
}

impl MnistData {
  pub fn open(frames_path: PathBuf, labels_path: PathBuf) -> MnistData {
    let frames_file = File::open(&frames_path).unwrap();
    let labels_file = File::open(&labels_path).unwrap();
    let (f_n, frame_dim, frames_mmap) = mmap_mnist_idx_file(frames_file);
    let (l_n, label_dim, labels_mmap) = mmap_mnist_idx_file(labels_file);
    assert_eq!(f_n, l_n);
    assert_eq!(2, frame_dim.len());
    assert_eq!(0, label_dim.len());
    MnistData{
      len:      f_n,
      frame_sz: frame_dim[0] * frame_dim[1],
      frame_d:  frame_dim,
      frames_m: SharedMem::new(frames_mmap),
      labels_m: SharedMem::new(labels_mmap),
    }
  }
}

impl IndexedData for MnistData {
  type Item = (Array2d<u8, SharedMem<u8>>, u32);

  fn len(&self) -> usize {
    self.len
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    let frame_dim = (self.frame_d[0], self.frame_d[1]);
    let frame_buf = self.frames_m.slice_v2(idx * self.frame_sz .. (idx+1) * self.frame_sz);
    let label = self.labels_m[idx] as u32;
    (Array2d::from_storage(frame_dim, frame_buf), label)
  }
}
