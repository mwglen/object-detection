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
use image::{imageops::FilterType, io::Reader as ImageReader, GenericImageView};

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
    let wcs = WeakClassifier::build_cascade(&mut wcs, &mut images, 5);
    println!("Built the cascade in {} seconds", now.elapsed().as_secs());

    // Output the data
    println!("Saving to {}", CASCADE);
    let data = serde_json::to_string_pretty(&wcs).unwrap();
    fs::write(CASCADE, &data).expect("Unable to write to file");

    println!("Finished");

}

fn detect(m: &clap::ArgMatches) {

    let cascade: Vec<WeakClassifier> = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).unwrap()
        } else {
            println!("Cascade not found in cache");
            return;
        }
    };
    
    let path = m.value_of("input_image").unwrap();
    let output_img = "output/".to_owned() + 
        Path::new(path).file_name().unwrap().to_str().unwrap();
    let img = ImageReader::open(path).unwrap().decode().unwrap().to_luma8();
    let img_width = img.width(); let img_height = img.height();
    let ii = IntegralImage::new(&img);
    let mut faces = Vec::<Rectangle::<u32>>::new();

    let max_f = (img_width/WL_32).min(img_height/WH_32);
    for f in 1..max_f {
        for x in 0..(img_width - f*WL_32) {
            for y in 0..(img_height - f*WH_32) { 
                let w = Rectangle::<u32>::new(x, y, f*WL_32, f*WH_32);
                for wc in &cascade {if !wc.evaluate_(&ii, &w) {continue;}}
                faces.push(w);
            }
        }
    }

    let data = serde_json::to_string_pretty(&faces).unwrap();
    fs::write("output/faces.json", &data).expect("Unable to write to file");

    println!("Finished");
}
