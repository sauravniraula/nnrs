use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

pub trait Number: Sized {}

impl Number for f32 {}
impl Number for f64 {}
