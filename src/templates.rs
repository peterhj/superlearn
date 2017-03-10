use arraydiff::prelude::*;
use arraydiff::ops::*;
use densearray::prelude::*;

use std::rc::{Rc};

pub struct CategoricalNLLLoss<Input, Label, Prob, Loss> {
  pub x:        Rc<ArrayOp<Input>>,
  pub y_label:  Rc<ArrayOp<Label>>,
  pub y:        Rc<ArrayOp<Prob>>,
  pub loss:     Rc<ArrayOp<Loss>>,
  pub obj:      Rc<AutodiffSink>,
}

pub struct BatchNormIo {
  pub vars:         (),
  pub stat_vars:    (),
  pub ctrl:         (),
}

pub struct ClassifierIo<In> {
  pub x:        Rc<ArraySrc<In>>,
  pub y_label:  Rc<ArraySrc<Batch<u32>>>,
  pub y:        Rc<ArrayOp<BatchArray1d<f32>>>,
}

pub struct Classifier<In> {
  pub io:       ClassifierIo<In>,
  /*pub loss:     Rc<AutodiffObjective>,*/
  pub train_vars:   (),
  pub optimizer:    (),
  pub batchnorm:    Option<BatchNormIo>,
}

impl<In> Classifier<In> {
  pub fn train_step(&self, data: ()) {
    unimplemented!();
  }

  pub fn validate(&self, data: ()) {
    unimplemented!();
  }
}

pub struct RegressorIo<In, Out> {
  pub x:        Rc<ArraySrc<In>>,
  pub y_target: Rc<ArraySrc<Out>>,
  pub y:        Rc<ArrayOp<Out>>,
}

pub struct Regressor<In, Out> {
  pub io:       RegressorIo<In, Out>,
  /*pub loss:     Rc<AutodiffObjective>,*/
}
