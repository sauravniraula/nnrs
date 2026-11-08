use num_traits::Float;
use rand::distr::uniform::SampleUniform;
use std::{
    fmt::{Debug, Display},
    ops::AddAssign,
};

pub trait Number: Float + Sized + Debug + Display + AddAssign + SampleUniform {}

impl<T> Number for T where T: Float + Sized + Debug + Display + AddAssign + SampleUniform {}
