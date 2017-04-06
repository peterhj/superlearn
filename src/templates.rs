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

use arraydiff::prelude::*;
use arraydiff::ops::*;
use densearray::prelude::*;

use std::rc::{Rc};

pub struct CategoricalNLLLoss<Input, Label, Prob, Loss> {
  pub obj:          Rc<AutodiffSink>,
  pub input:        Rc<ArrayOp<Input>>,
  pub label:        Rc<ArrayOp<Label>>,
  pub prob:         Rc<ArrayOp<Prob>>,
  pub loss:         Rc<ArrayOp<Loss>>,
  pub input_vars:   VarSet,
  pub label_vars:   VarSet,
  pub prob_vars:    VarSet,
  pub loss_vars:    VarSet,
  pub const_vars:   VarSet,
  //pub param_dim:    usize,
  pub param_vars:   VarSet,
  //pub keep_vars:    VarSet,
  //pub keep_grads:   VarSet,
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
