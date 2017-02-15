pub trait IndexedData {
  type Item;

  fn len(&self) -> usize;
  fn get(&mut self, idx: usize) -> Self::Item;
}

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

pub struct RandomSampleData<Inner> where Inner: IndexedData {
  inner:    Inner,
}

impl<Inner> Iterator for RandomSampleData<Inner> where Inner: IndexedData {
  type Item = Inner::Item;

  fn next(&mut self) -> Option<Self::Item> {
    unimplemented!();
  }
}
