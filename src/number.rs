use num_traits::Float;
use std::{
    fmt::{Debug, Display},
    ops::AddAssign,
};

pub trait Number: Float + Sized + Debug + Display + AddAssign {}

impl<T> Number for T where T: Float + Sized + Debug + Display + AddAssign {}
