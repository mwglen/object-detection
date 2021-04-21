#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod classifiers;
mod primitives;
mod images;
pub use classifiers::WeakClassifier;
pub use primitives::*;
pub use images::{IntegralImage, TrainingImages};

use clap::{load_yaml, App};
use indicatif::ProgressBar;
use std::fs;
use std::time::Instant;
use std::path::Path;

const FACES_DIR: &str = "images/faces";
const NOT_FACES_DIR: &str = "images/not_faces";
const CACHED_IMAGES: &str = "cache/images.json";
const CASCADE: &str = "cache/cascade.json";

fn main() {
    let yaml = load_yaml!("cli.yml");
    match App::from_yaml(yaml).get_matches().subcommand() {
        ("cascade", Some(m)) => cascade(m),
        ("detect", Some(m)) => detect(m),
        _ => {
            println!("Argument required:");
            println!("Run with argument \"cascade\" to build and train the cascade
                used in facial detection.");
            println!("Run with argument \"detect\" to detect faces in an image");
        },

    }
}

fn process_images() -> TrainingImages {
    println!("Processing images...");
    let now = Instant::now();
    let images = TrainingImages::from_dirs(FACES_DIR, NOT_FACES_DIR);
    println!( "Finished processing images in {} seconds", now.elapsed().as_secs() );

    let data = serde_json::to_string(&images).unwrap();
    fs::write(CACHED_IMAGES, &data).expect("Unable to write to file");
    images
}

fn cascade(m: &clap::ArgMatches) {

    let mut images: TrainingImages = {
        if Path::new(CACHED_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_IMAGES).unwrap();
            serde_json::from_str(&data).unwrap()
        } else {
            process_images()
        }
    };

    println!("Creating a vector of all weak classifiers");
    let now = Instant::now();
    let mut wcs = WeakClassifier::get_all();
    println!("Created Vector in {} seconds", now.elapsed().as_secs());
    println!("Vector has size {}", wcs.len());
    use std::mem::size_of;
    println!("It takes up {} bytes in memory",
        wcs.len()*size_of::<WeakClassifier>());
    println!("It takes up {} bytes in memory",
        wcs.capacity()*size_of::<WeakClassifier>());

    println!("Building the cascade of weak classifiers");
    let now = Instant::now();
    let wcs = WeakClassifier::build_cascade(&wcs, &mut images, 5);
    println!("Built the cascade in {} seconds", now.elapsed().as_secs());

    // Output the data
    println!("Saving to {}", CASCADE);
    let data = serde_json::to_string_pretty(&wcs).unwrap();
    fs::write(CASCADE, &data).expect("Unable to write to file");

    println!("Finished");

}

fn detect(m: &clap::ArgMatches) {
    // let path = m.value_of("input_image").unwrap();
    // let img = ImageReader::open(path).unwrap().decode().unwrap();
    // let img = IntegralImage::new(img);
    unimplemented!();
}
