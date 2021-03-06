use std::cmp::Ordering;

use indicatif::{ProgressBar, ProgressStyle};
use num::{ToPrimitive, Unsigned};
use serde::{Deserialize, Serialize};

use super::{WindowSize, IntegralImageTrait};

/// The smallest unsigned integer primitive that can index into the Window
pub type Window = Rectangle<WindowSize>;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Rectangle<T: Unsigned + Copy> {
    pub top_left: [T; 2],
    pub bot_right: [T; 2],
}
impl<T: Unsigned + Copy> Rectangle<T> {
    pub fn new(x: T, y: T, w: T, h: T) -> Rectangle<T> {
        Rectangle::<T> {
            top_left: [x, y],
            bot_right: [x + w, y + h],
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Feature {
    pub black: (Window, Option<Window>),
    pub white: (Window, Option<Window>),
}
impl Feature {
    /// Evaluates a feature over a window of an integral image
    pub fn evaluate(&self, img: &impl IntegralImageTrait) -> i64 {
        img.rect_sum(&self.black.0)
            + self.black.1.map_or(0, |r| img.rect_sum(&r))
            - img.rect_sum(&self.white.0)
            - self.white.1.map_or(0, |r| img.rect_sum(&r))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct OrderedF64(pub f64);
impl Eq for OrderedF64 {}
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub fn new_bar(size: impl ToPrimitive, prefix: &str) -> ProgressBar {
    let bar = ProgressBar::new(size.to_u64().unwrap());
    let template = "{prefix} {bar:40.blue/grey} \
        {pos:>7}/{len:7} [{elapsed}]";

    let style = ProgressStyle::default_bar().template(template);
    bar.set_style(style);
    bar.set_prefix(prefix);
    bar
}
