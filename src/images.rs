use super::{WH_, WH_32, WL_, WL_32, WeakClassifier, Rectangle};
use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::fs;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct IntegralImage {
    pixels: Vec<u64>,
} 
impl IntegralImage {
    pub fn new(img: DynamicImage) -> IntegralImage {
        // Resize the image and turn it to grayscale
        let img = img.resize(WL_32, WH_32, FilterType::Triangle).into_luma8();

        // Calculate each pixel of the integral image
        let mut pixels = Vec::<u64>::with_capacity(WL_ * WH_);
        for y in 0..WH_ {
            for x in 0..WL_ {
                let mut pixel = u64::from(img.get_pixel(x as u32, y as u32)[0]);
                if y != 0 { pixel += pixels[x + WL_*(y-1)]; }
                if x != 0 { pixel += pixels[(x-1) + WL_*y]; }
                if x != 0 && y != 0 {
                    pixel -= pixels[(x-1) + WL_*(y-1)];
                }
                pixels.push(pixel);
            }
        }
        IntegralImage { pixels }
    }
    pub fn rect_sum(&self, r: &Rectangle) -> i64 {
        let xtl = usize::from(r.top_left[0]);
        let ytl = usize::from(r.top_left[1]);
        let xbr = usize::from(r.bot_right[0]);
        let ybr = usize::from(r.bot_right[1]);
        
        self.pixels[xbr + WL_*ybr] as i64
            - self.pixels[xbr + WL_*ytl] as i64
            - self.pixels[xtl + WL_*ybr] as i64
            + self.pixels[xtl + WL_*ytl] as i64
    }
}

/// A struct-of-arrays representing all of the training images 
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TrainingImages {
    images: Vec<IntegralImage>,
    weights: Vec<f64>,
    is_face: Vec<bool>,
}
impl TrainingImages {
    pub fn iter(&self) -> impl Iterator<Item = (&IntegralImage, &f64, &bool)> {
        use itertools::izip;
        izip!(self.images.iter(), self.weights.iter(), self.is_face.iter())
    }
    pub fn iter_mut(&mut self)
        -> impl Iterator<Item = (&IntegralImage, &mut f64, &bool)> {
        use itertools::izip;
        izip!(self.images.iter(), self.weights.iter_mut(), self.is_face.iter())
    }
    pub fn len(&self) -> usize { self.images.len() }
    
    pub fn normalize_weights(set: &mut TrainingImages) {
        // Sum over the weights of all the images
        let sum = set.weights.iter().sum::<f64>();
        
        // Divide each image's original weight by the sum
        for weight in &mut set.weights { *weight /= sum; }
    }

    pub fn from_dirs(faces_dir: &str, not_faces_dir: &str) -> TrainingImages {
        // Open all of the images of faces
        let faces = fs::read_dir(faces_dir).unwrap().map(|img| {
            // Open the image
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();
    
            // Convert image to Integral Image
            (true, IntegralImage::new(img))
        });

        let not_faces = fs::read_dir(not_faces_dir).unwrap().map(|img| {
            // Open the image
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();
    
            // Convert image to Integral Image
            (false, IntegralImage::new(img))
        });

        let (is_face, images): (Vec<_>, Vec<_>) = faces.chain(not_faces).unzip();
        
        // Initialize a weights vector
        let weights = vec![1.0/(2.0 * images.len() as f64); images.len()];

        // Return the created set
        TrainingImages { images, weights, is_face}
    }
    pub fn sort(&mut self, wc: &WeakClassifier) {
        use permutation::permutation::sort_by;
        let evals: Vec<_> = self.images.iter().map(|img| wc.evaluate(img)).collect();
        let perm = sort_by(&evals[..], |a, b| {
            a.partial_cmp(&b).unwrap()
        });
        self.weights = perm.apply_slice(&self.weights[..]);
        self.images = perm.apply_slice(&self.images[..]);
        self.is_face = perm.apply_slice(&self.is_face[..]);
    }
}
