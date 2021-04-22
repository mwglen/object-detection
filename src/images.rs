use super::{WH_, WH_32, WL_, WL_32, WeakClassifier, Rectangle, WindowSize};
use image::{imageops::FilterType, io::Reader as ImageReader,ImageBuffer, Luma};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::fs;
use indicatif::ProgressBar;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct IntegralImage {
    pixels: Vec<u64>,
    width: usize,
    height: usize,
} 
impl IntegralImage {
    pub fn new(img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> IntegralImage {
        // Calculate each pixel of the integral image
        let w = img.width() as usize;
        let h = img.height() as usize;
        let mut pixels = Vec::<u64>::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let mut pixel = u64::from(img.get_pixel(x as u32, y as u32)[0]);
                if y != 0 { pixel += pixels[x + w*(y-1)]; }
                if x != 0 { pixel += pixels[(x-1) + w*y]; }
                if x != 0 && y != 0 {
                    pixel -= pixels[(x-1) + w*(y-1)];
                }
                pixels.push(pixel);
            }
        }
        IntegralImage { pixels, width: w, height: h}
    }
    pub fn rect_sum(&self, r: &Rectangle<WindowSize>) -> i64 {
        let xtl = usize::from(r.top_left[0]);
        let ytl = usize::from(r.top_left[1]);
        let xbr = usize::from(r.bot_right[0]);
        let ybr = usize::from(r.bot_right[1]);
        
        self.pixels[xbr + self.width*ybr] as i64
            - self.pixels[xbr + self.width*ytl] as i64
            - self.pixels[xtl + self.width*ybr] as i64
            + self.pixels[xtl + self.width*ytl] as i64
    }
    
    pub fn rect_sum_(
        &self, 
        r: &Rectangle<WindowSize>, 
        w: &Rectangle<u32>,
    ) -> i64 {
        let xtl = usize::from(r.top_left[0]) + (w.top_left[0] as usize);
        let xbr = usize::from(r.bot_right[0]) + (w.top_left[0] as usize);

        let ytl = usize::from(r.top_left[1]) + (w.top_left[1] as usize);
        let ybr = usize::from(r.bot_right[1]) + (w.top_left[1] as usize);
        
        self.pixels[xbr + self.width*ybr] as i64
            - self.pixels[xbr + self.width*ytl] as i64
            - self.pixels[xtl + self.width*ybr] as i64
            + self.pixels[xtl + self.width*ytl] as i64
    }
}


/// A struct-of-arrays representing all of the training images 
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TrainingImages {
    pub images: Vec<IntegralImage>,
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

    pub fn from_dirs(
        faces_dir: &str, 
        not_faces_dir: &str, 
    ) -> TrainingImages {
        let num_faces = fs::read_dir(faces_dir).unwrap().count();
        let num_not_faces = fs::read_dir(not_faces_dir).unwrap().count();
        let bar = ProgressBar::new((num_faces + num_not_faces) as u64);

        // Open all of the images of faces
        let faces = fs::read_dir(faces_dir).unwrap().map(|img| {
            // Open the image
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();
            bar.inc(1);
            
            // Resize the image and turn it to grayscale
            let img = img.resize_to_fill(WL_32, WH_32, FilterType::Triangle).into_luma8();

            // Convert image to Integral Image
            (true, IntegralImage::new(&img))
        });

        let not_faces = fs::read_dir(not_faces_dir).unwrap().map(|img| {
            // Open the image
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();
            bar.inc(1);
            
            // Resize the image and turn it to grayscale
            let img = img.resize_to_fill(WL_32, WH_32, FilterType::Triangle).into_luma8();
    
            // Convert image to Integral Image
            (false, IntegralImage::new(&img))
        });

        // Initialize a weights vector
        let mut weights = vec![1.0/(2.0 * num_faces as f64); num_faces];
        weights.append(&mut vec![1.0/(2.0 * num_not_faces as f64); num_not_faces]);
        
        // Chain all of the images
        let (is_face, images): (Vec<_>, Vec<_>) = faces.chain(not_faces).unzip();
        bar.finish();

        // Return the created set
        TrainingImages { images, weights, is_face}
    }
    pub fn sort(&mut self, wc: &WeakClassifier) {
        use permutation::permutation::sort_by;
        let evals: Vec<_> = self.images.iter()
            .map(|img| wc.evaluate_num(img)).collect();
        let perm = sort_by(&evals[..], |a, b| {
            a.partial_cmp(&b).unwrap()
        });
        self.weights = perm.apply_slice(&self.weights[..]);
        self.images = perm.apply_slice(&self.images[..]);
        self.is_face = perm.apply_slice(&self.is_face[..]);
    }
}
