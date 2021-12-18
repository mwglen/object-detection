mod constants;
mod images;
mod primitives;
mod strong_classifier;
mod weak_classifier;
mod classifier;
mod cascade;
mod integral_image;

use std::{fs, path::Path};
use clap::{load_yaml, App, AppSettings};

pub use integral_image::{
    ImageData, IntegralImage, 
    IntegralImageTrait, 
    WindowedIntegralImage,
};
pub use constants::*;
pub use primitives::*;
pub use strong_classifier::StrongClassifier;
pub use weak_classifier::WeakClassifier;
pub use classifier::Classifier;
pub use cascade::Cascade;
pub use images::{
    ColorImage, 
    GreyscaleImage, 
    DynamicImage,
    draw_rectangle,
};

fn main() {
    // Parse the cli arguments using clap
    let yaml = load_yaml!("cli.yml");
    let app = App::from_yaml(yaml)
        .setting(AppSettings::ArgRequiredElseHelp)
        .get_matches();

    // Run the specified subcommand
    match app.subcommand() {
        ("process_images", Some(_)) => process_images(),
        ("cascade", Some(_)) => cascade(),
        ("test", Some(_)) => test(),
        ("detect", Some(m)) => detect(m),
        _ => println!("Incorrect subcommand"),
    }
}

/// Processes images for use in building the cascade
fn process_images() {
    // Find and process images
    println!("Training Image:");
    let set = ImageData::from_dirs(
        OBJECT_DIR, OTHER_DIR, SLICE_DIR, NUM_NEG,
    );
    println!("Processed {} images", set.len());

    // Save image data to cache
    let data = serde_json::to_string(&set).unwrap();
    fs::write(CACHED_IMAGES, &data).expect("Unable to cache images");
}

/// Builds the cascade
fn cascade() {
    // Get training images from cache or process raw images if cache
    // is empty
    let set: Vec<ImageData> = {
        if Path::new(CACHED_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached images")
        } else {
            println!("Training image data not found in cache");
            return;
        }
    };

    println!("{:-^30}", " Building Cascade ");
    let cascade = Cascade::build(set);

    // Output the data
    println!("Saving cascade to {}", CASCADE);
    let data = serde_json::to_string_pretty(&cascade).unwrap();
    fs::write(CASCADE, &data).expect("Unable to write to file");
}

/// Tests cached cascade over training images
fn test() {
    // Get the cached cascade
    let cascade: Cascade = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data)
                .expect("Unable to read cached cascade")
        } else {
            println!("Cascade not found in cache");
            return;
        }
    };

    // Get processed training images from cache
    let train_set: Vec<ImageData> = {
        if Path::new(CACHED_IMAGES).exists() {
            let data =
                std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data)
                .expect("Unable to read cached image data")
        } else {
            println!("Testing image data not found in cache");
            return;
        }
    };

    println!("Testing the Cascade...");
    let (fpr, dtr) = cascade.test(&train_set);
    
    // Print test results
    println!("False Positive Rate: {}", fpr);
    println!("Detection Rate: {}", dtr);
}

/// This detects objects by sending a "windowed" view into the image
/// to be evaluated by the cascade. The window moves across the image
/// and grows in size. This tests all rectangles in the images for the
/// object
fn detect(m: &clap::ArgMatches) {
    // Get the cached cascade
    let cascade: Cascade = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).unwrap()
        } else {
            println!("Cascade not found in cache");
            return;
        }
    };

    // Get the input image
    let path = m.value_of("input_image").unwrap();

    // Get the location to store the output image
    let output_img = "output/".to_owned()
        + Path::new(path).file_name().unwrap().to_str().unwrap();

    // Open the image
    let img = DynamicImage::from(path);
    let img = GreyscaleImage::from(img);
    let img_width = img.width();
    let img_height = img.height();

    // Convert image to integral image
    let ii = IntegralImage::from(&img);

    // Vector to hold detected objects
    let mut objects = Vec::<Rectangle<u32>>::new();

    let max_width = if (img_width / WL_32) < (img_height / WH_32) {
        img_width
    } else { img_height * WL_32 / WH_32 };

    let step_size = (f64::from(WL) / 5.0).round() as usize;
    for curr_width in (WL_32..=max_width).step_by(step_size) {
        let curr_height = curr_width * WH_32 / WL_32;
        let f = f64::from(curr_width) / f64::from(WL_32);
        for x in 0..(img_width - curr_width) {
            for y in 0..(img_height - curr_height) {
                let img = WindowedIntegralImage {
                    ii: &ii,
                    x_offset: x as usize,
                    y_offset: y as usize,
                    f: f as i64,
                };
                if cascade.classify(&img) {
                    objects.push(Rectangle::<u32>::new(
                        x, y, curr_width, curr_height
                    ));
                }
            }
        }
    }
    println!("Found {} instances of object", objects.len());

    // Reopen the image and conver to rgb, draw rectangles, and then
    // save image
    let mut img = ColorImage::from(DynamicImage::from(path));
    for o in objects.iter_mut() { draw_rectangle(&mut img, o); }
    // draw_rectangle(&mut img, &objects[0]);
    img.save(output_img).unwrap();

    // Output detected object
    let data = serde_json::to_string_pretty(&objects).unwrap();
    fs::write("output/object.json", &data)
        .expect("Unable to write to file");
}
