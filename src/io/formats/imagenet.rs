use io::*;

use sharedmem::{MemoryMap, SharedMem};
use tar::{Archive};

//use byteorder::{ReadBytesExt, BigEndian};
use std::collections::{HashMap};
use std::fs::{File};
use std::io::{Cursor};
use std::path::{PathBuf};

#[derive(Clone, Copy)]
pub struct ImagenetEntry {
  offset:   usize,
  length:   usize,
  label:    Option<u32>,
}

#[derive(Clone)]
pub struct ImagenetTrainData {
  wnid_to_label:    HashMap<String, u32>,
  //wnid_to_id:       HashMap<String, i32>,
  entries:  Vec<ImagenetEntry>,
  data_buf: SharedMem<u8>,
}

impl ImagenetTrainData {
  pub fn open(archive_path: PathBuf) -> ImagenetTrainData {
    let archive_file = File::open(&archive_path).unwrap();
    let file_meta = archive_file.metadata().unwrap();
    let file_sz = file_meta.len() as usize;
    let archive_buf = SharedMem::new(MemoryMap::open_with_offset(archive_file, 0, file_sz).unwrap());

    let mut wnid_to_label = HashMap::new();
    let mut entries = Vec::new();

    let reader = Cursor::new(archive_buf.clone());
    let mut archive = Archive::new(reader);
    for (wnid_idx, wnid_entry) in archive.entries().unwrap().enumerate() {
      let wnid_entry = wnid_entry.unwrap();
      let wnid_pos = wnid_entry.raw_file_position();
      let wnid_size = wnid_entry.header().entry_size().unwrap();
      assert_eq!(wnid_size, wnid_entry.header().size().unwrap());

      let mut wnid_archive = Archive::new(wnid_entry);
      for (im_idx, im_entry) in wnid_archive.entries().unwrap().enumerate() {
        let im_entry = im_entry.unwrap();
        let im_pos = im_entry.raw_file_position();
        let im_size = im_entry.header().entry_size().unwrap();
        assert_eq!(im_size, im_entry.header().size().unwrap());
        assert!(wnid_pos + im_pos >= wnid_pos);
        assert!(wnid_pos + im_pos < wnid_pos + wnid_size);
        assert!(wnid_pos + im_pos + im_size <= wnid_pos + wnid_size);

        let im_path = im_entry.header().path().unwrap().into_owned();
        let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
        let im_stem_toks: Vec<_> = im_path_toks[0].splitn(2, "_").collect();
        let im_wnid = im_stem_toks[0].to_owned();

        // FIXME(20170215): in fact we should use the "ILSVRC2012_ID" for the labels,
        // which requires a preprocessed metadata file.
        let maybe_new_label = wnid_to_label.len() as u32;
        let im_label = *wnid_to_label.entry(im_wnid).or_insert(maybe_new_label);

        let entry = ImagenetEntry{
          offset:   (wnid_pos + im_pos) as usize,
          length:   im_size as usize,
          label:    Some(im_label),
        };
        if wnid_idx == 0 && im_idx < 5 {
          println!("DEBUG: imagenet data: offset: {} length: {} label: {:?}",
              entry.offset, entry.length, entry.label);
        }
        entries.push(entry);
      }
    }

    ImagenetTrainData{
      wnid_to_label:    wnid_to_label,
      entries:  entries,
      data_buf: archive_buf,
    }
  }

  pub fn num_categories(&self) -> usize {
    self.wnid_to_label.len()
  }
}

impl IndexedData for ImagenetTrainData {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.entries.len()
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    let entry = &self.entries[idx];
    let frame_buf = self.data_buf.slice_v2(entry.offset .. entry.offset + entry.length);
    (frame_buf, entry.label.unwrap())
  }
}

#[derive(Clone)]
pub struct ImagenetValidData {
  //wnid_to_label:    HashMap<String, u32>,
  entries:  Vec<ImagenetEntry>,
  data_buf: SharedMem<u8>,
}

impl IndexedData for ImagenetValidData {
  type Item = (SharedMem<u8>, u32);

  fn len(&self) -> usize {
    self.entries.len()
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    let entry = &self.entries[idx];
    let frame_buf = self.data_buf.slice_v2(entry.offset .. entry.offset + entry.length);
    (frame_buf, entry.label.unwrap())
  }
}

#[derive(Clone)]
pub struct ImagenetTestData {
  entries:  Vec<ImagenetEntry>,
  data_buf: SharedMem<u8>,
}

impl IndexedData for ImagenetTestData {
  type Item = SharedMem<u8>;

  fn len(&self) -> usize {
    self.entries.len()
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    let entry = &self.entries[idx];
    let frame_buf = self.data_buf.slice_v2(entry.offset .. entry.offset + entry.length);
    frame_buf
  }
}
