use rng::xorshift::*;

use rand::{Rng, SeedableRng};
use rand::chacha::{ChaChaRng};

pub mod codecs;
pub mod formats;

pub trait IndexedData {
  type Item;

  fn len(&self) -> usize;
  fn get(&mut self, idx: usize) -> Self::Item;

  fn slice(self, lower: usize, upper: usize) -> SliceData<Self> where Self: Sized {
    SliceData{
      lower:    lower,
      upper:    upper,
      inner:    self,
    }
  }

  fn cycle(self) -> CycleData<Self> where Self: Sized {
    CycleData{
      counter:  0,
      inner:    self,
    }
  }

  fn randomly_sample(self, mut seed_rng: ChaChaRng) -> RandomSampleData<Self> where Self: Sized {
    RandomSampleData{
      rng:      Xorshiftplus128Rng::from_seed([seed_rng.next_u64(), seed_rng.next_u64()]),
      inner:    self,
    }
  }
}

#[derive(Clone)]
pub struct SliceData<Inner> where Inner: IndexedData {
  lower:    usize,
  upper:    usize,
  inner:    Inner,
}

impl<Inner> IndexedData for SliceData<Inner> where Inner: IndexedData {
  type Item = Inner::Item;

  fn len(&self) -> usize {
    self.upper - self.lower
  }

  fn get(&mut self, idx: usize) -> Self::Item {
    assert!(idx < self.len());
    self.inner.get(self.lower + idx)
  }
}

#[derive(Clone)]
pub struct CycleData<Inner> where Inner: IndexedData {
  counter:  usize,
  inner:    Inner,
}

impl<Inner> Iterator for CycleData<Inner> where Inner: IndexedData {
  type Item = Inner::Item;

  fn next(&mut self) -> Option<Self::Item> {
    let idx = self.counter;
    self.counter += 1;
    if self.counter >= self.inner.len() {
      self.counter = 0;
    }
    let item = self.inner.get(idx);
    Some(item)
  }
}

#[derive(Clone)]
pub struct RandomSampleData<Inner> where Inner: IndexedData {
  rng:      Xorshiftplus128Rng,
  inner:    Inner,
}

impl<Inner> Iterator for RandomSampleData<Inner> where Inner: IndexedData {
  type Item = Inner::Item;

  fn next(&mut self) -> Option<Self::Item> {
    let idx = self.rng.gen_range(0, self.inner.len());
    let item = self.inner.get(idx);
    Some(item)
  }
}
