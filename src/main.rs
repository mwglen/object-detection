#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::fs;
use text_io::read;

mod images;
use images::{GreyscaleImage, RGBImage, IntegralImage};

mod features;
use features::Feature;

fn main() {

    println!("Enter the directory containing test images of faces:");
    let dir: String = read!("{}\n");

    println!("Training. This may take some time...");

    let data = find_features(&dir);
    //let data = fs::write(file, &data).expect("Unable to write to file");

    //fs::write("features.json", &data).expect("Unable to write file");
}

fn find_features(dir: &String) -> Vec::<Feature> {
    unimplemented!();
}
