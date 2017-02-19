use arraydiff::prelude::*;
//use arraydiff::ops::*;

use std::rc::{Rc};

//pub mod sgd;

pub struct CommonOptimizerConfig {
  pub io_vars:      VarSet,
  pub opt_vars:     VarSet,
  //pub extra_vars:   Option<()>,
  pub loss_fn:      Rc<Fn(&AutodiffObjective)>,
  pub loss_grad:    Option<Rc<Fn(&AutodiffObjective)>>,
  pub loss_hvp:     Option<Rc<Fn(&AutodiffObjective)>>,
  pub loss_gnvp:    Option<Rc<Fn(&AutodiffObjective)>>,
  pub loss_hdiag:   Option<Rc<Fn(&AutodiffObjective)>>,
  //pub postproc:     Option<Rc<Fn(&AutodiffObjective)>>,
}

pub struct BatchNormOptimizerConfig {
  pub batch_norm_vars:  VarSet,
  pub batch_norm_stats: VarSet,
  pub batch_normalize:  Rc<Fn(&AutodiffObjective)>,
}

pub trait Optimizer {
  fn step(&mut self);
}
