pub type WindowSize = u8;

pub const WS: WindowSize = 4;
pub const WL: WindowSize = WS * 7;
pub const WH: WindowSize = WS * 8;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;

// The number of negative training images to start with
pub const NUM_NEG: usize = 5000;

// CONSTANTS HOLDING PATHS/DIRECTORIES
/// Path to images of the object
pub const OBJECT_DIR: &str = "images/training/object";

/// Path to images that are not of object
pub const OTHER_DIR: &str = "images/training/other";

/// Path to images not containing object to slice
pub const SLICE_DIR: &str = "images/training/to_slice";

/// Path to cached training images
pub const CACHED_IMAGES: &str = "cache/images.json";

// CONSTANTS USED IN BUILDING THE CASCADE
/// Path to output the cascade
pub const CASCADE: &str = "cache/cascade.json";

/// The number of strong classifiers in the cascade (used when building from
/// layout)
pub const CASCADE_SIZE: usize = 4;

/// Maximum acceptable false positive rate per layer
pub const MAX_FALSE_POS: f64 = 0.5;

/// Target overall false positive rate
pub const TARGET_FALSE_POS: f64 = 0.5;
