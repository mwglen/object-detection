use std::fs;
use serde::{Deserialize, Serialize};
use image::{DynamicImage, imageops::FilterType, io::Reader as ImageReader};

pub fn main() {
    let data = find_features();

    // Output the data
    let data = serde_json::to_string(&data).unwrap();
    fs::write("features.json", &data).expect("Unable to write to file");
}

fn find_features() -> Vec::<Feature> {
    
    // Store the training images into a vector after resizing them
    // to 24x24 and changing them to greyscale Integral Images
    let count = fs::read_dir("images").unwrap().count();
    println!("{} images found in ./images", count);
    let mut images = Vec::<IntegralImage>::with_capacity(count);
    for path in fs::read_dir("images").unwrap() {
        // Open the image
        let img = ImageReader::open(path.unwrap().path()).unwrap().decode().unwrap();
        
        // Turn the image into an Integral Image
        let img = IntegralImage::new(img);

        // Add the image to images
        images.push(img);
    }
    println!("Finished gathering images");
    
    
    println!("Creating a vector of all possible features");
    let features = Feature::get_all();


    println!("Finished");
    unimplemented!();
}

pub struct Rectangle {
    top_left: [usize; 2],
    bot_right: [usize; 2],
}

// Two rectangle, three rectangle, and four rectangle features can all be 
// represented using only two rectangles
#[derive(Serialize, Deserialize, Debug)]
struct Feature {
    top_left: [usize; 2],
    mid_point: [usize; 2],
    bot_right: [usize; 2],
    false_negatives: usize,
    false_positives: usize,
} impl Feature {
    /// Gets a vector of all possible features
    pub fn get_all() -> Vec<Feature>{
        let features = Vec::<Feature>::with_capacity(200_000);
        unimplemented!();
    }

    /// Evaluates whether or not a feature is in an integral image
    pub fn evaluate(&self, ii: IntegralImage) -> isize {
        let mut sum: isize = 0;
        for rect in &self.get_white_rects() { sum += ii.rect_sum(rect); }
        for rect in &self.get_black_rects() { sum -= ii.rect_sum(rect); }
        sum
    }

    /// Gets an array of all of the white rectangles associated with the feature
    fn get_white_rects(&self) -> [Rectangle; 2] {
    
        let tl = self.top_left; let br = self.bot_right; let mp = self.mid_point;

        // The rectangle in the top right of the feature
        let rect1 = Rectangle {
            top_left: [mp[0], tl[1]],
            bot_right: [br[0], mp[1]],
        };

        // The rectangle in the bottom left of the feature
        let rect2 = Rectangle {
            top_left: [tl[0], mp[1]],
            bot_right: [mp[0], br[1]],
        };
        [rect1, rect2]
    }

    /// Gets an array of all of the black rectangles associated with the feature
    fn get_black_rects(&self) -> [Rectangle; 2] {
        
        let tl = self.top_left; let br = self.bot_right; let mp = self.mid_point;

        // The rectangle in the top left of the feature
        let rect1 = Rectangle {
            top_left: tl,
            bot_right: mp,
        };

        // The rectangle in the bottom right of the feature
        let rect2 = Rectangle {
            top_left: mp,
            bot_right: br,
        };
        [rect1, rect2]
    }
}

/// The size of sweeping window used in image detection
pub const WS: usize = 24;

pub struct IntegralImage {
    pixels: Vec<usize>,
} impl IntegralImage {
    pub fn new(img: DynamicImage) -> IntegralImage {
        // Resize the image and turn it to grayscale
        let mut img = img.resize(WS as u32, WS as u32, FilterType::Triangle)
            .into_luma8();

        // Calculate each pixel of the integral image
        let mut pixels = Vec::<usize>::with_capacity(WS*WS);
        for y in 0..WS {
            for x in 0..WS {
                let pixel = img.get_pixel_mut(x as u32, y as u32)[0] as usize;
                let x = x as usize; let y = y as usize;
                pixels.push({
                    if (x == 0) && (y == 0) { pixel }
                    else if x == 0 { pixels[x + WS*(y-1)] as usize + pixel }
                    else if y == 0 { pixels[(x-1) + WS*y] as usize + pixel }
                    else { 
                        pixels[((x-1) + WS*y)] as usize
                            + (pixels[x + WS*(y-1)] as usize)
                            + pixel 
                            - (pixels[(x-1) + WS*(y-1)] as usize)
                    }
                })
            }
        }
        IntegralImage {
            pixels,
        }

    }
    pub fn rect_sum(&self, r: &Rectangle) -> isize {
        let tl = r.top_left; let br = r.bot_right;
        (self.pixels[br[0] + WS*br[1]] 
            - self.pixels[br[0] + WS*tl[1]] 
            - self.pixels[tl[0] + WS*br[1]] 
            + self.pixels[tl[0] + WS*tl[1]]) as isize
    }
}
