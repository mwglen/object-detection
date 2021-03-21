extern crate serde;
use image::io::Reader as ImageReader;

struct Face {
    x: u16,
    y: u16,
    length: u16,
} impl {
    pub fn detect(img: &String) -> Result<Vec<Faces>, ImageResult<DynamicImage>> {
        let img = ImageReader::open(img)?.decode()?.greyscale().resize();

    }
}

lazy_static! {
    pub static ref FEATURES: Vec<Feature> = {
        let features = include_str!("features.json");
        let features: Vec<Feature> = serde_json::from_str(&features).unwrap();
        serde_json::from_value(features).unwrap()
    };
}
