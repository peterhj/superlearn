use arraydiff::prelude::*;
//use arraydiff::ops::*;

use std::rc::{Rc};

//pub mod sgd;

pub struct CommonOptimizerConfig {
  pub io_vars:      (),
  pub opt_vars:     (),
  //pub extra_vars:   Option<()>,
  pub loss_fn:      Rc<Fn(&AutodiffObjective)>,
  pub loss_grad:    Option<Rc<Fn(&AutodiffObjective)>>,
  pub loss_hvp:     Option<Rc<Fn(&AutodiffObjective)>>,
  pub loss_gnvp:    Option<Rc<Fn(&AutodiffObjective)>>,
  pub loss_hdiag:   Option<Rc<Fn(&AutodiffObjective)>>,
  //pub postproc:     Option<Rc<Fn(&AutodiffObjective)>>,
}

pub struct BatchNormOptimizerConfig {
  pub batch_norm_vars:  (),
  pub batch_norm_stats: (),
  pub batch_normalize:  Rc<Fn(&AutodiffObjective)>,
}

pub trait Optimizer {
  fn step(&mut self);
}
