use image::io::Reader as ImageReader;
// use super::{IntegralImage, Rectangle};

pub fn main(m: &clap::ArgMatches) {
    // let path = m.value_of("input_image").unwrap();
    // let img = ImageReader::open(path).unwrap().decode().unwrap();
    // let img = IntegralImage::new(img);
    // detect_faces(&img);
    unimplemented!();
}

// fn detect_faces(img: &IntegralImage) -> Vec<Rectangle> {
//     unimplemented!();
// }

// struct Window {
//     rect: Rectangle
// } impl Window {
//     pub fn detect_face(&self) -> Vec<Rectangle> {
//         // If the cascade accepts the window as containing a face
//         if CASCADE.iter().all(|wc| wc.evaluate() >= wc.threshold) {
            
//         }
//     }
// }

// lazy_static! {
//     pub static ref CASCADE: Vec<WeakClassifier> = {
//         let classifiers = include_str!("classifiers.json");
//         let classifiers: Vec<WeakClassifier> =
//             serde_json::from_str(&features).unwrap();
//         serde_json::from_value(classifiers).unwrap()
//     };
// }
