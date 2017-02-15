use arraydiff::prelude::*;
use arraydiff::ops::*;
use densearray::prelude::*;

use std::rc::{Rc};

pub struct ClassifierIo<In> {
  pub x:        Rc<ArraySrc<In>>,
  pub y_label:  Rc<ArraySrc<Batch<u32>>>,
  pub y:        Rc<ArrayOp<BatchArray1d<f32>>>,
}

pub struct Classifier<In> {
  pub loss:     Rc<AutodiffObjective>,
  pub io:       ClassifierIo<In>,
}

pub struct RegressorIo<In, Out> {
  pub x:        Rc<ArraySrc<In>>,
  pub y_target: Rc<ArraySrc<Out>>,
  pub y:        Rc<ArrayOp<Out>>,
}

pub struct Regressor<In, Out> {
  pub loss:     Rc<AutodiffObjective>,
  pub io:       RegressorIo<In, Out>,
}
