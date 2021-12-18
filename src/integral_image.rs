use std::fs;
use rand::seq::SliceRandom;
use image::imageops::{crop_imm, FilterType};
use serde::{Deserialize, Serialize};
use super::{
    new_bar, Window, 
    WH_32, WL_32, DynamicImage, 
    GreyscaleImage,
};

/// A trait that allows both windowed and non-windowed 
/// integral images to be easily classified
pub trait IntegralImageTrait {
    /// Gets the sum of pixels within in a rectangular 
    /// region of an image
    fn rect_sum(&self, r: &Window) -> i64;
    fn width(&self) -> usize;
    fn height(&self) -> usize;
} 

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct IntegralImage {
    pixels: Vec<u64>,
    width: usize,
    height: usize,
} impl From<&GreyscaleImage> for IntegralImage {
    fn from(img: &GreyscaleImage) -> Self {
        // Calculate each pixel of the integral image
        let w = img.width() as usize;
        let h = img.height() as usize;
        let mut pixels = Vec::<u64>::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                
                let mut pixel = u64::from(
                    img.get_pixel(x as u32, y as u32)[0]);

                if y != 0 {
                    pixel += pixels[x + w * (y - 1)];
                }
                if x != 0 {
                    pixel += pixels[(x - 1) + w * y];
                }
                if x != 0 && y != 0 {
                    pixel -= pixels[(x - 1) + w * (y - 1)];
                }
                pixels.push(pixel);
            }
        }
        IntegralImage { pixels, width: w, height: h }
    }
} impl IntegralImageTrait for IntegralImage {
    fn rect_sum(&self, r: &Window) -> i64 {
        let xtl = usize::from(r.top_left[0]);
        let ytl = usize::from(r.top_left[1]);
        let xbr = usize::from(r.bot_right[0]);
        let ybr = usize::from(r.bot_right[1]);

        self.pixels[xbr + self.width * ybr] as i64
            - self.pixels[xbr + self.width * ytl] as i64
            - self.pixels[xtl + self.width * ybr] as i64
            + self.pixels[xtl + self.width * ytl] as i64
    }
    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }
}

pub struct WindowedIntegralImage<'a> {
    pub ii: &'a IntegralImage,
    pub x_offset: usize,
    pub y_offset: usize,
    pub f: i64,
} impl<'a> IntegralImageTrait for WindowedIntegralImage<'_> {
    fn rect_sum(&self, r: &Window) -> i64 {
        let xtl = usize::from(r.top_left[0]) + self.x_offset;
        let ytl = usize::from(r.top_left[1]) + self.y_offset;
        let xbr = usize::from(r.bot_right[0]) + self.x_offset;
        let ybr = usize::from(r.bot_right[1]) + self.y_offset;

        (self.ii.pixels[xbr + self.ii.width * ybr] as i64
            - self.ii.pixels[xbr + self.ii.width * ytl] as i64
            - self.ii.pixels[xtl + self.ii.width * ybr] as i64
            + self.ii.pixels[xtl + self.ii.width * ytl] as i64)
            / (self.f * self.f)
    }
    fn width(&self) -> usize { self.ii.width }
    fn height(&self) -> usize { self.ii.height }
}

/// A wrapper over an integral image that holds training data
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ImageData {
    pub image: IntegralImage,
    pub weight: f64,
    pub is_object: bool,
} impl ImageData {
    pub fn from_slice_dir(slice_dir: &str) -> Vec<IntegralImage> {
        let mut sliced = Vec::<IntegralImage>::new();
        for img in fs::read_dir(slice_dir).unwrap() {
            let img = DynamicImage::from(img.unwrap().path());
            let img = GreyscaleImage::from(img);
            let w = img.width();
            let h = img.height();

            for x in 0..(w / WL_32) {
                for y in 0..(h / WH_32) {
                    let img = crop_imm(&img, x * WL_32, y * WL_32, WL_32, WH_32)
                        .to_image();
                    let image = IntegralImage::from(&img);
                    sliced.push(image);
                }
            }
        }
        sliced
    }

    /// Create image data from directories
    pub fn from_dirs(
        object_dir: &str, 
        other_dir: &str, 
        slice_dir: &str,
        num_neg: usize,
    ) -> Vec<Self> {
        // Slice images
        let sliced = Self::from_slice_dir(slice_dir);
        let sliced_size = num_neg - fs::read_dir(other_dir).unwrap().count(); 
        let sliced = sliced.choose_multiple(&mut rand::thread_rng(), sliced_size);

        // Find the number of objects and others
        let num_objects = fs::read_dir(object_dir).unwrap().count();

        // Create a vector to hold the image data
        let mut set = Vec::<Self>::with_capacity(num_objects + num_neg);
        let bar = new_bar(num_objects + num_neg, "Processing Images...");

        // Calculate the weight of each object image
        let weight = 1.0 / (2 * num_objects) as f64;

        // Add each image from the objects directory to the vector
        for path in fs::read_dir(object_dir).unwrap() {
            // Open the image
            let img = DynamicImage::from(path.unwrap().path());

            // Resize the image and turn it to grayscale
            let img = img.resize(WL_32, WH_32, FilterType::Triangle);
            let img = GreyscaleImage::from(img);

            // Convert image to Integral Image
            let image = IntegralImage::from(&img);

            // Push to vector
            set.push(Self{image, weight, is_object: true});
            bar.inc(1);
        }

        // Calculate the weight of each object image
        let weight = 1.0 / (2 * num_neg) as f64;

        // Add each image from the objects directory to the vector
        for path in fs::read_dir(other_dir).unwrap() {
            // Open the image
            let img = DynamicImage::from(path.unwrap().path());
            
            // Resize the image and turn it to grayscale
            let img = img.resize(WL_32, WH_32, FilterType::Triangle);
            let img = GreyscaleImage::from(img);
            
            // Convert image to Integral Image
            let image = IntegralImage::from(&img);

            // Push to vector
            set.push(Self {
                image,
                weight,
                is_object: false,
            });
            bar.inc(1);
        }
        for image in sliced.cloned() {
            set.push(ImageData {
                image,
                weight,
                is_object: false
            });
            bar.inc(1);
        }
        bar.finish();
        set
    }

    /// Normalize the weights of a set of image data
    pub fn normalize_weights(set: &mut [ImageData]) {
        // Sum over the weights of all the images
        let sum: f64 = set.iter().map(|d| d.weight).sum();

        // Divide each image's original weight by the sum
        for data in set.iter_mut() {
            data.weight /= sum;
        }
    }
}
