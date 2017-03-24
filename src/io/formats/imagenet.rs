use io::*;

use csv::{Reader as CsvReader};
use extar::{TarBufferExt, TarBuffer};
use sharedmem::{MemoryMap, SharedMem};
use tar::{Archive};

//use byteorder::{ReadBytesExt, BigEndian};
use std::collections::{HashMap};
use std::fs::{File};
//use std::io::{Read, Seek, BufRead, BufReader, Cursor, SeekFrom, Result as IoResult};
use std::io::{BufRead, BufReader, Cursor};
use std::path::{PathBuf};

/*pub struct BytesCursor<A> {
  inner:    Cursor<A>,
}

impl<A> BytesCursor<A> {
  pub fn new(inner: Cursor<A>) -> Self {
    BytesCursor{inner: inner}
  }
}

impl<A> AsRef<[u8]> for BytesCursor<A> where A: AsRef<[u8]> {
  fn as_ref(&self) -> &[u8] {
    self.inner.get_ref().as_ref()
  }
}

impl<A> Seek for BytesCursor<A> where A: AsRef<[u8]> {
  fn seek(&mut self, style: SeekFrom) -> IoResult<u64> {
    self.inner.seek(style)
  }
}

impl<A> Read for BytesCursor<A> where A: AsRef<[u8]> {
  fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
    self.inner.read(buf)
  }
}*/

#[derive(Clone)]
pub struct WordnetIlsvrc2012IdMap {
  wnid_to_id:   HashMap<String, i64>,
  id_to_wnid:   HashMap<i64, String>,
}

impl WordnetIlsvrc2012IdMap {
  pub fn from_metadata(path: PathBuf) -> Self {
    panic!("unimplemented: loading 'mat' files is unsupported");
  }

  pub fn from_csv(path: PathBuf) -> Self {
    let file = File::open(&path).unwrap();
    let mut reader = CsvReader::from_reader(file).has_headers(true);
    let mut wnid_to_id = HashMap::new();
    let mut id_to_wnid = HashMap::new();
    #[derive(RustcDecodable)]
    struct Row {
      wordnet_id:       String,
      ilsvrc2012_id:    i64,
    }
    for row in reader.decode() {
      let row: Row = row.unwrap();
      wnid_to_id.insert(row.wordnet_id.clone(), row.ilsvrc2012_id);
      id_to_wnid.insert(row.ilsvrc2012_id, row.wordnet_id);
    }
    WordnetIlsvrc2012IdMap{
      wnid_to_id:   wnid_to_id,
      id_to_wnid:   id_to_wnid,
    }
  }

  pub fn len(&self) -> usize {
    let n1 = self.wnid_to_id.len();
    let n2 = self.id_to_wnid.len();
    assert_eq!(n1, n2);
    n1
  }
}

#[derive(Clone)]
pub struct Ilsvrc2012ValidGroundTruth {
  ids:  Vec<i64>,
}

impl Ilsvrc2012ValidGroundTruth {
  pub fn open(path: PathBuf) -> Self {
    let file = File::open(&path).unwrap();
    let reader = BufReader::new(file);
    let mut ids = vec![];
    for line in reader.lines() {
      let line = line.unwrap();
      let id: i64 = line.parse().unwrap();
      ids.push(id);
    }
    Ilsvrc2012ValidGroundTruth{ids: ids}
  }
}

#[derive(Clone, Copy)]
pub struct Entry {
  offset:   usize,
  length:   usize,
  label:    Option<u32>,
}

#[derive(Clone)]
pub struct Ilsvrc2012TrainData {
  wnid_id_map:  WordnetIlsvrc2012IdMap,
  entries:  Vec<Entry>,
  data_buf: SharedMem<u8>,
}

impl Ilsvrc2012TrainData {
  pub fn open(wnid_id_map: WordnetIlsvrc2012IdMap, archive_path: PathBuf) -> Ilsvrc2012TrainData {
    let archive_file = File::open(&archive_path).unwrap();
    let file_meta = archive_file.metadata().unwrap();
    let file_sz = file_meta.len() as usize;
    let archive_buf = SharedMem::new(MemoryMap::open_with_offset(archive_file, 0, file_sz).unwrap());

    let mut entries = Vec::new();

    /*let reader = Cursor::new(archive_buf.clone());
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

        let im_id = *wnid_id_map.wnid_to_id.get(&im_wnid).unwrap();
        assert!(im_id >= 1);
        let im_label = (im_id - 1) as u32;
        assert!(im_label < 1000);

        let entry = Entry{
          offset:   (wnid_pos + im_pos) as usize,
          length:   im_size as usize,
          label:    Some(im_label),
        };
        /*if wnid_idx == 0 && im_idx < 5 {
          println!("DEBUG: imagenet data: offset: {} length: {} label: {:?}",
              entry.offset, entry.length, entry.label);
        }*/
        entries.push(entry);
      }
    }*/

    let reader = Cursor::new(archive_buf.clone());
    let mut archive = TarBuffer::new(reader);
    for (wnid_idx, wnid_entry) in archive.raw_entries().unwrap().enumerate() {
      let wnid_entry = wnid_entry.unwrap();
      let wnid_pos = wnid_entry.raw_file_position();
      let wnid_size = wnid_entry.file_size();
      //assert_eq!(wnid_size, wnid_entry.header().size().unwrap());

      //let mut wnid_archive = Archive::new(wnid_entry);
      let mut wnid_archive = TarBuffer::new(Cursor::new(archive_buf.slice_v2(wnid_pos as usize .. (wnid_pos + wnid_size) as usize)));
      for (im_idx, im_entry) in wnid_archive.raw_entries().unwrap().enumerate() {
        let im_entry = im_entry.unwrap();
        let im_pos = im_entry.raw_file_position();
        let im_size = im_entry.file_size();
        //assert_eq!(im_size, im_entry.header().size().unwrap());
        assert!(wnid_pos + im_pos >= wnid_pos);
        assert!(wnid_pos + im_pos < wnid_pos + wnid_size);
        assert!(wnid_pos + im_pos + im_size <= wnid_pos + wnid_size);

        //let im_path = im_entry.header().path().unwrap().into_owned();
        let im_path = im_entry.path.clone();
        let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
        let im_stem_toks: Vec<_> = im_path_toks[0].splitn(2, "_").collect();
        let im_wnid = im_stem_toks[0].to_owned();

        let im_id = *wnid_id_map.wnid_to_id.get(&im_wnid).unwrap();
        assert!(im_id >= 1);
        let im_label = (im_id - 1) as u32;
        assert!(im_label < 1000);

        let entry = Entry{
          offset:   (wnid_pos + im_pos) as usize,
          length:   im_size as usize,
          label:    Some(im_label),
        };
        /*if wnid_idx == 0 && im_idx < 5 {
          println!("DEBUG: imagenet data: offset: {} length: {} label: {:?}",
              entry.offset, entry.length, entry.label);
        }*/
        entries.push(entry);
      }
    }

    Ilsvrc2012TrainData{
      wnid_id_map:  wnid_id_map,
      entries:  entries,
      data_buf: archive_buf,
    }
  }

  pub fn num_categories(&self) -> usize {
    self.wnid_id_map.len()
  }
}

impl IndexedData for Ilsvrc2012TrainData {
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
pub struct Ilsvrc2012ValidData {
  truth:    Ilsvrc2012ValidGroundTruth,
  entries:  Vec<Entry>,
  data_buf: SharedMem<u8>,
}

impl Ilsvrc2012ValidData {
  pub fn open(truth: Ilsvrc2012ValidGroundTruth, archive_path: PathBuf) -> Self {
    let archive_file = File::open(&archive_path).unwrap();
    let file_meta = archive_file.metadata().unwrap();
    let file_sz = file_meta.len() as usize;
    let archive_buf = SharedMem::new(MemoryMap::open_with_offset(archive_file, 0, file_sz).unwrap());

    let mut entries = Vec::new();

    /*let reader = Cursor::new(archive_buf.clone());
    let mut archive = Archive::new(reader);
    for (_, im_entry) in archive.entries().unwrap().enumerate() {
      let im_entry = im_entry.unwrap();
      let im_pos = im_entry.raw_file_position();
      let im_size = im_entry.header().entry_size().unwrap();
      assert_eq!(im_size, im_entry.header().size().unwrap());

      let im_path = im_entry.header().path().unwrap().into_owned();
      let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
      let im_stem_toks: Vec<_> = im_path_toks[0].splitn(3, "_").collect();
      assert_eq!("val", im_stem_toks[1]);
      let im_rank_tok = im_stem_toks[2].to_owned();

      let im_rank: i64 = im_rank_tok.parse().unwrap();
      assert!(im_rank >= 1);
      let im_idx = (im_rank - 1) as usize;
      let im_id = truth.ids[im_idx];
      assert!(im_id >= 1);
      let im_label = (im_id - 1) as u32;
      assert!(im_label < 1000);

      let entry = Entry{
        offset:   im_pos as usize,
        length:   im_size as usize,
        label:    Some(im_label),
      };
      entries.push(entry);
    }*/

    let reader = Cursor::new(archive_buf.clone());
    let mut archive = TarBuffer::new(reader);
    for (_, im_entry) in archive.raw_entries().unwrap().enumerate() {
      let im_entry = im_entry.unwrap();
      let im_pos = im_entry.raw_file_position();
      let im_size = im_entry.file_size();
      //assert_eq!(im_size, im_entry.header().size().unwrap());

      let im_path = im_entry.path.clone();
      let im_path_toks: Vec<_> = im_path.to_str().unwrap().splitn(2, ".").collect();
      let im_stem_toks: Vec<_> = im_path_toks[0].splitn(3, "_").collect();
      assert_eq!("val", im_stem_toks[1]);
      let im_rank_tok = im_stem_toks[2].to_owned();

      let im_rank: i64 = im_rank_tok.parse().unwrap();
      assert!(im_rank >= 1);
      let im_idx = (im_rank - 1) as usize;
      let im_id = truth.ids[im_idx];
      assert!(im_id >= 1);
      let im_label = (im_id - 1) as u32;
      assert!(im_label < 1000);

      let entry = Entry{
        offset:   im_pos as usize,
        length:   im_size as usize,
        label:    Some(im_label),
      };
      entries.push(entry);
    }

    assert_eq!(truth.ids.len(), entries.len());

    Ilsvrc2012ValidData{
      truth:    truth,
      entries:  entries,
      data_buf: archive_buf,
    }
  }
}

impl IndexedData for Ilsvrc2012ValidData {
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
pub struct Ilsvrc2012TestData {
  entries:  Vec<Entry>,
  data_buf: SharedMem<u8>,
}

impl IndexedData for Ilsvrc2012TestData {
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
