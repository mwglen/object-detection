extern crate serde;
use image::io::Reader as ImageReader;

pub fn main(img: &String) {
    let img = ImageReader::open(img).unwrap().decode().unwrap();
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
