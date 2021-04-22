use serde::{Deserialize, Serialize};
use num::Unsigned;

/// The smallest unsigned integer primitive that can index into the Window
pub type WindowSize = u8;

/// The relative size of sweeping window used in image detection
pub const WS: WindowSize = 2;
pub const WL: WindowSize = WS * 7;
pub const WH: WindowSize = WS * 8;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;
pub const WL_: usize = WL as usize;
pub const WH_: usize = WH as usize;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Rectangle<T: Unsigned + Copy> {
    pub top_left: [T; 2],
    pub bot_right: [T; 2],
} impl<T: Unsigned + Copy> Rectangle<T> {
    pub fn new(
        x: T, y: T, 
        w: T, h: T
    ) -> Rectangle::<T> {
        Rectangle::<T> {
            top_left: [x, y],
            bot_right: [x+w, y+h]
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub enum Feature {
        TwoRect { black: Rectangle<WindowSize>, white: Rectangle<WindowSize> },
        ThreeRect { black: Rectangle<WindowSize>, white: [Rectangle<WindowSize>; 2] },
        FourRect { black: [Rectangle<WindowSize>; 2], white: [Rectangle<WindowSize>; 2] },
}
