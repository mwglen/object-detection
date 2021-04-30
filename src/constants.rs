pub type WindowSize = u8;

pub const WS: WindowSize = 19;
pub const WL: WindowSize = WS;
pub const WH: WindowSize = WS;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;
pub const WL_: usize = WL as usize;
pub const WH_: usize = WH as usize;

// CONSTANTS HOLDING PATHS/DIRECTORIES
/// Path to cached training images
pub const TRAIN_OBJECT_DIR: &str = "images/training/object";

/// Path to cached training images
pub const TRAIN_OTHER_DIR: &str = "images/training/other";

/// Path to cached training images
pub const TEST_OBJECT_DIR: &str = "images/testing/object";

/// Path to cached training images
pub const TEST_OTHER_DIR: &str = "images/testing/other";

/// Path to cached training images
pub const CACHED_TRAIN_IMAGES: &str = "cache/train_images.json";

/// Path to cached test images
pub const CACHED_TEST_IMAGES: &str = "cache/test_images.json";

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
