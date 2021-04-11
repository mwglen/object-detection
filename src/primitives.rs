use serde::{Deserialize, Serialize};

/// The smallest unsigned integer primitive that can index into the Window
type WindowSize = u8;

/// The relative size of sweeping window used in image detection
pub const WS: WindowSize = 2;
pub const WL: WindowSize = WS * 7;
pub const WH: WindowSize = WS * 8;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;
pub const WL_: usize = WL as usize;
pub const WH_: usize = WH as usize;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Rectangle {
    pub top_left: [WindowSize; 2],
    pub bot_right: [WindowSize; 2],
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub enum Feature {
        TwoRect { black: Rectangle, white: Rectangle },
        ThreeRect { black: Rectangle, white: [Rectangle; 2] },
        FourRect { black: [Rectangle; 2], white: [Rectangle; 2] },
}
