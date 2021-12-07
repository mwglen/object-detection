pub type WindowSize = u8;

pub const WS: WindowSize = 4;
pub const WL: WindowSize = WS * 7;
pub const WH: WindowSize = WS * 8;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;

/// The number of negative training images to start with
pub const NUM_NEG: usize = 5000;

// CONSTANTS HOLDING PATHS/DIRECTORIES
/// Path to positive training images
pub const OBJECT_DIR: &str = "images/training/object";

/// Path to negative training images
pub const OTHER_DIR: &str = "images/training/other";

/// Path to images not containing object to slice
pub const SLICE_DIR: &str = "images/training/to_slice";

/// Path to cached training images
pub const CACHED_IMAGES: &str = "cache/images.json";

/// Path to output the cascade
pub const CASCADE: &str = "cache/cascade.json";

// CONSTANTS USED IN BUILDING THE CASCADE
/// The number of strong classifiers in the cascade
pub const CASCADE_SIZE: usize = 4;

/// Maximum acceptable false positive rate per layer
pub const MAX_FALSE_POS: f64 = 0.5;

/// Target false positive rate for entire cascade
pub const TARGET_FALSE_POS: f64 = 0.001;

/// Sets whether or not to use a layout when building the cascade
pub const USE_LAYOUT: bool = false;

/// Determines the layout to use when building the cascade
pub const LAYOUT: [usize; CASCADE_SIZE] = [1, 5, 15, 30];

// CONSTANTS FOR FILTERING WEAK CLASSIFIERS
/// Sets whether or not to filter out underperforming weak
/// classifiers
pub const FILTER: bool = false;

/// Sets the percentage of weak classifiers to filter out
pub const PERCENTAGE_TO_FILTER: f64 = 10.0;
