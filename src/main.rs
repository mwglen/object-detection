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

    println!("Training using the images found in ./images");
    println!("This may take some time...");
    let data = find_features();
    let data = serde_json::to_string(&data).unwrap();
    fs::write("features.json", &data).expect("Unable to write to file");

}

fn find_features() -> Vec::<Feature> {
    println!("{} images found", fs::read_dir("images").unwrap().count());

    let paths = fs::read_dir("images").unwrap();
    for path in paths {
        println!("Name: {}", path.unwrap().path().display());
    }
    unimplemented!();
}

/*
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
}
*/
