extern crate serde;

struct Face {
    x: u16,
    y: u16,
    length: u16,
} impl {
    pub fn detect(img: &String) -> Vec<Faces> {
        unimplemented!();
    }
}

lazy_static! {
    pub static ref FEATURES: Vec<Feature> = {
        let features = include_str!("features.json");
        let features: Vec<Feature> = serde_json::from_str(&features).unwrap();
        serde_json::from_value(features).unwrap()
    };
}
