use super::{WH_32, WL_32, Rectangle, Window, new_bar};
use image::{imageops::FilterType, io::Reader as ImageReader,ImageBuffer, Luma};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct IntegralImage {
    pixels: Vec<u64>,
    width: usize,
    height: usize,
} impl IntegralImage {
    /// Creates an integral image from an image
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

    /// Gets the sum of pixels in a rectangular region of the original image
    /// using the images corresponding integral image
    pub fn rect_sum(&self, r: &Window, w: Option<Rectangle<u32>>) -> i64 {
        let mut xtl = usize::from(r.top_left[0]);
        let mut ytl = usize::from(r.top_left[1]);
        let mut xbr = usize::from(r.bot_right[0]);
        let mut ybr = usize::from(r.bot_right[1]);
       
        if let Some(w) = w {
            xtl += w.top_left[0] as usize;
            ytl += w.top_left[1] as usize;
            xbr += w.top_left[0] as usize;
            ybr += w.top_left[1] as usize;
        }

        self.pixels[xbr + self.width*ybr] as i64
            - self.pixels[xbr + self.width*ytl] as i64
            - self.pixels[xtl + self.width*ybr] as i64
            + self.pixels[xtl + self.width*ytl] as i64
    }
}

/// A struct-of-arrays representing all of the training images 
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ImageData {
    pub image: IntegralImage,
    pub weight: f64,
    pub is_face: bool,
} impl ImageData {

    /// Create image data from a directories
    pub fn from_dirs(faces_dir: &str, others_dir: &str) -> Vec<ImageData> {

        // Find the number of faces and others
        let num_faces = fs::read_dir(faces_dir).unwrap().count();
        let num_others = fs::read_dir(others_dir).unwrap().count();

        // Create a vector to hold the image data
        let mut set = Vec::<ImageData>::with_capacity(num_faces + num_others);
        let bar = new_bar(num_faces + num_others, "Processing Images...");

        // Calculate the weight of each face image
        let weight = 1.0/(2*num_faces) as f64;

        // Add each image from the faces directory to the vector
        for img in fs::read_dir(faces_dir).unwrap() {
            // Open the image
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();
            
            // Resize the image and turn it to grayscale
            let img = img.resize_to_fill(WL_32, WH_32, FilterType::Triangle)
                .into_luma8();

            // Convert image to Integral Image
            let image = IntegralImage::new(&img);
           
            // Push to vector
            set.push(ImageData {image, weight, is_face: true});
            bar.inc(1);
        }

        // Calculate the initial weight of each not_face image
        let weight = 1.0/(2*num_others) as f64;

        // Add each image from the others directory to the vector
        for img in fs::read_dir(others_dir).unwrap() {

            // Open the image
            let img = ImageReader::open(img.unwrap().path())
                .unwrap()
                .decode()
                .unwrap();
            
            // Resize the image and turn it to grayscale
            let img = img.resize_to_fill(WL_32, WH_32, FilterType::Triangle)
                .into_luma8();

            // Convert image to Integral Image
            let image = IntegralImage::new(&img);
           
            // Push to vector
            set.push(ImageData {image, weight, is_face: false});
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
        for data in set.iter_mut() { data.weight /= sum; }
    }

}
