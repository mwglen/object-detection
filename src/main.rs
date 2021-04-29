mod weak_classifier;
mod strong_classifier;
mod primitives;
mod images;
pub use primitives::*;
pub use images::{IntegralImage, ImageData};
pub use weak_classifier::WeakClassifier;
pub use strong_classifier::StrongClassifier;

use clap::{load_yaml, App, AppSettings};
use std::path::Path;
use std::fs;
use image::io::Reader as ImageReader;

const FACES_DIR: &str = "images/faces";
const NOT_FACES_DIR: &str = "images/not_faces";
const CACHED_IMAGES: &str = "cache/images.json";
const CASCADE: &str = "cache/cascade.json";
const SC_SIZE: usize = 2;
const CASCADE_SIZE: usize = 2;

fn main() {
    // Parse the cli arguments using clap
    let yaml = load_yaml!("cli.yml");
    let app =  App::from_yaml(yaml)
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
    let set = ImageData::from_dirs(FACES_DIR, NOT_FACES_DIR);

    // Save image data to cache
    let data = serde_json::to_string(&set).unwrap();
    fs::write(CACHED_IMAGES, &data).expect("Unable to write to file");
}

/// Builds the cascade
fn cascade() {
    // Get training images from cache or process raw images if cache is empty
    let mut set: Vec<ImageData> = {
        if Path::new(CACHED_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached images")
        } else { println!("Training image data not found in cache"); return; }
    };
    
    // Get all possible weak classifiers
    let mut wcs = WeakClassifier::get_all();
    println!("{:-^30}", " Getting Weak Classifiers ");
    println!("Created vector of {} possible weak classifiers", wcs.len());
    println!("{:-^30}", " Building Cascade ");

    // Build the cascade using the weak classifiers
    let cascade: Vec<_> = (1..=CASCADE_SIZE).map(|i| {
        println!("Building Strong Classifier {} of {}", i, CASCADE_SIZE);
        
        // Create a strong classifier
        let sc = StrongClassifier::new(&mut wcs, &mut set);

        // Remove the true negatives from the training set
        set.retain(|data| data.is_face || sc.classify(&data.image, None));
        sc
    }).collect();

    // Output the data
    println!("Saving cascade to {}", CASCADE);
    let data = serde_json::to_string_pretty(&cascade).unwrap();
    fs::write(CASCADE, &data).expect("Unable to write to file");
}

/// Test a cascade over training images
fn test() {
    // Get the cached cascade
    let cascade: Vec<StrongClassifier> = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached cascade")
        } else { println!("Cascade not found in cache"); return; }
    };

    // Get processed training images from cache
    let set: Vec<ImageData> = {
        if Path::new(CACHED_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached image data")
        } else { println!("Training image data not found in cache"); return; }
    };

    // Test the cascade over training images
    println!("Testing the Cascade");
    let mut correct_faces: f64 = 0.0;
    let mut correct_others: f64 = 0.0;
    let mut num_faces: f64 = 0.0;
    for data in &set {
        let mut eval = true;
        for sc in &cascade {
            if !sc.classify(&data.image, None) {eval = false; break;}
        }
        if data.is_face {num_faces += 1.0;}
        if data.is_face && eval {correct_faces += 1.0;}
        if !data.is_face && !eval {correct_others += 1.0;}
    }

    // Print test results
    println!("correct_faces: {}", correct_faces);
    println!("correct_others: {}", correct_others);
    println!("num_faces: {}", num_faces);
    println!("images_len: {}", set.len());
    println!("Percent of correctly evaluated images: {:.2}%", 
        (correct_faces + correct_others) * 100.0 / (set.len() as f64));
    println!("Percent of correctly evaluated images of faces: {:.2}%",
        correct_faces * 100.0 / num_faces);
    println!("Percent of correctly evaluated images which are not faces: {:.2}%",
        correct_others * 100.0 / (set.len() as f64 - num_faces));
}

/// Detects faces in an image
fn detect(m: &clap::ArgMatches) {
    // Get the cached cascade
    let cascade: Vec<StrongClassifier> = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).unwrap()
        } else { println!("Cascade not found in cache"); return; }
    };
    
    // Get the input image
    let path = m.value_of("input_image").unwrap();
    //let output_img = "output/".to_owned() + 
    //    Path::new(path).file_name().unwrap().to_str().unwrap();
    let img = ImageReader::open(path).unwrap().decode().unwrap().to_luma8();
    let img_width = img.width(); let img_height = img.height();

    // Convert image to integral image
    let ii = IntegralImage::new(&img);

    // Vector to hold detected faces
    let mut faces = Vec::<Rectangle::<u32>>::new();
    
    // This detects faces by sending a "windowed" view into the image to
    // be evaluated by the cascade. The window moves across the image and grows
    // in size. Testing all combinations of rectangles in the images for a face
    let max_f = (img_width/WL_32).min(img_height/WH_32);
    for f in 1..max_f {
        for x in 0..(img_width - f*WL_32) {
            for y in 0..(img_height - f*WH_32) { 
                let w = Rectangle::<u32>::new(x, y, f*WL_32, f*WH_32);
                if cascade.iter().all(|sc| sc.classify(&ii, Some(w))) {
                    faces.push(w);
                }
            }
        }
    }
    println!("Found {} faces", faces.len());

    // Output detected faces
    let data = serde_json::to_string_pretty(&faces).unwrap();
    fs::write("output/faces.json", &data).expect("Unable to write to file");
}
