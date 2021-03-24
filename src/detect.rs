extern crate serde;
use image::io::Reader as ImageReader;

pub fn main(m: &clap::ArgMatches) {
    let path = m.value_of("input_image").unwrap();
    let img = ImageReader::open(path).unwrap().decode().unwrap();
}

/*
lazy_static! {
    pub static ref FEATURES: Vec<Feature> = {
        let features = include_str!("features.json");
        let features: Vec<Feature> = serde_json::from_str(&features).unwrap();
        serde_json::from_value(features).unwrap()
    };
}
*/
