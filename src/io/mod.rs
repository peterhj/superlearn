use rng::xorshift::*;

use rand::{Rng, SeedableRng};
use rand::chacha::{ChaChaRng};
use std::collections::{VecDeque};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread::{JoinHandle, spawn};

pub mod formats;
pub mod transforms;

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

  fn randomly_sample(self, seed_rng: &mut ChaChaRng) -> RandomSampleData<Self> where Self: Sized {
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

pub fn async_bounded_queue<Inner>(capacity: usize, inner: Inner) -> AsyncBoundedQueue<<Inner as Iterator>::Item> where Inner: 'static + Iterator + Send, <Inner as Iterator>::Item: Send {
  let (tx, rx) = sync_channel(capacity);
  let h = spawn(move || {
    let mut worker = AsyncBoundedQueueWorker{
      inner:    inner,
      tx:       tx,
    };
    worker._run_loop();
  });
  AsyncBoundedQueue{
    capacity:   capacity,
    queue:      VecDeque::with_capacity(capacity),
    h:          h,
    rx:         rx,
    closed:     false,
  }
}

struct AsyncBoundedQueueWorker<Inner> where Inner: Iterator {
  inner:    Inner,
  tx:       SyncSender<Option<Inner::Item>>,
}

impl<Inner> AsyncBoundedQueueWorker<Inner> where Inner: Iterator {
  pub fn _run_loop(&mut self) {
    loop {
      match self.inner.next() {
        None => {
          self.tx.send(None).unwrap();
          break;
        }
        Some(item) => {
          self.tx.send(Some(item)).unwrap();
        }
      }
    }
  }
}

pub struct AsyncBoundedQueue<Item> {
  capacity: usize,
  queue:    VecDeque<Item>,
  h:        JoinHandle<()>,
  rx:       Receiver<Option<Item>>,
  closed:   bool,
}

impl<Item> Iterator for AsyncBoundedQueue<Item> {
  type Item = Item;

  fn next(&mut self) -> Option<Self::Item> {
    if self.queue.is_empty() {
      if self.closed {
        return None;
      }
      match self.rx.recv().unwrap() {
        None => {
          self.closed = true;
          return None;
        }
        Some(item) => {
          self.queue.push_back(item);
        }
      }
    }
    if !self.closed {
      while self.queue.len() < self.capacity {
        match self.rx.try_recv() {
          Err(_) => {
            break;
          }
          Ok(None) => {
            self.closed = true;
            break;
          }
          Ok(Some(item)) => {
            self.queue.push_back(item);
          }
        }
      }
    }
    assert!(!self.queue.is_empty());
    let item = self.queue.pop_front().unwrap();
    Some(item)
  }
}
