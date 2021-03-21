#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

mod features;
mod recognize;
mod detect;
mod map;

extern crate clap;
use clap::{load_yaml, App};

use image::{GrayImage, DynamicImage, ImageBuffer, GenericImageView, Luma};
use image::imageops::FilterType;

use serde::{Serialize, Deserialize};

fn main() {
    let yaml = load_yaml!("cli.yml");
    let matches = App::from_yaml(yaml).get_matches();
}



