use rng::xorshift::*;

use rand::{Rng, SeedableRng};
use rand::chacha::{ChaChaRng};
use std::cmp::{min};
use std::collections::{VecDeque};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel};
use std::thread::{JoinHandle, spawn};

pub mod formats;
pub mod transforms;

pub trait IndexedData {
  type Item;

  fn len(&self) -> usize;
  fn get(&mut self, idx: usize) -> Self::Item;

  fn range(self, lower: usize, upper: usize) -> RangeData<Self> where Self: Sized {
    assert!(upper <= self.len());
    assert!(lower <= upper);
    RangeData{
      lower:    lower,
      upper:    upper,
      inner:    self,
    }
  }

  fn partition(self, part: usize, num_parts: usize) -> RangeData<Self> where Self: Sized {
    assert!(part < num_parts);
    let data_len = self.len();
    let max_part_sz = (data_len + num_parts - 1) / num_parts;
    let lower = part * max_part_sz;
    let part_sz = min(max_part_sz, data_len - lower);
    let upper = lower + part_sz;
    RangeData{
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
pub struct RangeData<Inner> where Inner: IndexedData {
  lower:    usize,
  upper:    usize,
  inner:    Inner,
}

impl<Inner> IndexedData for RangeData<Inner> where Inner: IndexedData {
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
  type Item = (usize, Inner::Item);

  fn next(&mut self) -> Option<Self::Item> {
    let idx = self.counter;
    self.counter += 1;
    if self.counter >= self.inner.len() {
      self.counter = 0;
    }
    let item = self.inner.get(idx);
    Some((idx, item))
  }
}

#[derive(Clone)]
pub struct RandomSampleData<Inner> where Inner: IndexedData {
  rng:      Xorshiftplus128Rng,
  inner:    Inner,
}

impl<Inner> Iterator for RandomSampleData<Inner> where Inner: IndexedData {
  type Item = (usize, Inner::Item);

  fn next(&mut self) -> Option<Self::Item> {
    let idx = self.rng.gen_range(0, self.inner.len());
    let item = self.inner.get(idx);
    Some((idx, item))
  }
}

pub fn async_queue<Inner>(capacity: usize, inner: Inner) -> AsyncQueue<<Inner as Iterator>::Item> where Inner: 'static + Iterator + Send, <Inner as Iterator>::Item: Send {
  let (tx, rx) = sync_channel(capacity);
  let h = spawn(move || {
    let mut worker = AsyncQueueWorker{
      inner:    inner,
      tx:       tx,
    };
    worker._run_loop();
  });
  AsyncQueue{
    capacity:   capacity,
    queue:      VecDeque::with_capacity(capacity),
    h:          h,
    rx:         rx,
    closed:     false,
  }
}

struct AsyncQueueWorker<Inner> where Inner: Iterator {
  inner:    Inner,
  tx:       SyncSender<Option<Inner::Item>>,
}

impl<Inner> AsyncQueueWorker<Inner> where Inner: Iterator {
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

pub struct AsyncQueue<Item> {
  capacity: usize,
  queue:    VecDeque<Item>,
  h:        JoinHandle<()>,
  rx:       Receiver<Option<Item>>,
  closed:   bool,
}

impl<Item> Iterator for AsyncQueue<Item> {
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

pub struct RoundRobinJoinIter<Inner> {
  counter:  usize,
  closed:   bool,
  iters:    Vec<Inner>,
}

pub fn round_robin_join<Inner, F>(num_rounds: usize, iter_builder: F) -> RoundRobinJoinIter<Inner> where Inner: Iterator, F: Fn(usize) -> Inner {
  let mut iters = Vec::with_capacity(num_rounds);
  for rank in 0 .. num_rounds {
    iters.push(iter_builder(rank));
  }
  RoundRobinJoinIter{
    counter:    0,
    closed:     false,
    iters:      iters,
  }
}

impl<Inner> Iterator for RoundRobinJoinIter<Inner> where Inner: Iterator {
  type Item = <Inner as Iterator>::Item;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    let num_rounds = self.iters.len();
    let item = self.iters[self.counter].next();
    self.counter += 1;
    if self.counter >= num_rounds {
      self.counter = 0;
    }
    if item.is_none() {
      self.closed = true;
    }
    item
  }
}

pub fn round_robin_async_split<Inner>(num_rounds: usize, capacity: usize, inner: Inner) -> Vec<Option<RoundRobinSplitConsumer<<Inner as Iterator>::Item>>> where Inner: 'static + Iterator + Send, <Inner as Iterator>::Item: Send {
  let mut txs = Vec::with_capacity(num_rounds);
  let mut consumers = Vec::with_capacity(num_rounds);
  for rank in 0 .. num_rounds {
    let (tx, rx) = sync_channel(capacity);
    txs.push(tx);
    let cons = RoundRobinSplitConsumer{
      rank:     rank,
      closed:   false,
      rx:       rx,
    };
    consumers.push(Some(cons));
  }
  let _ = spawn(move || {
    let mut producer = RoundRobinSplitProducer{
      counter:    0,
      inner:      inner,
      txs:        txs,
    };
    producer._run_loop();
  });
  consumers
}

pub struct RoundRobinSplitProducer<Inner> where Inner: Iterator {
  counter:  usize,
  inner:    Inner,
  txs:      Vec<SyncSender<Option<Inner::Item>>>,
}

impl<Inner> RoundRobinSplitProducer<Inner> where Inner: Iterator {
  pub fn _run_loop(&mut self) {
    loop {
      match self.inner.next() {
        None => {
          break;
        }
        Some(item) => {
          self.txs[self.counter].send(Some(item)).unwrap();
          self.counter += 1;
          if self.counter >= self.txs.len() {
            self.counter = 0;
          }
        }
      }
    }
    for _ in 0 .. self.txs.len() {
      self.txs[self.counter].send(None).unwrap();
      self.counter += 1;
      if self.counter >= self.txs.len() {
        self.counter = 0;
      }
    }
  }
}

pub struct RoundRobinSplitConsumer<Item> {
  rank:     usize,
  //capacity: usize,
  closed:   bool,
  //queue:    VecDeque<Item>,
  rx:       Receiver<Option<Item>>,
}

impl<Item> Iterator for RoundRobinSplitConsumer<Item> {
  type Item = Item;

  fn next(&mut self) -> Option<Self::Item> {
    if self.closed {
      return None;
    }
    // TODO
    match self.rx.recv() {
      Err(_) => { None }
      Ok(None) => { None }
      Ok(Some(item)) => { Some(item) }
    }
  }
}
