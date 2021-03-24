use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

pub fn main(m: &clap::ArgMatches) {
    let out_path = m.value_of("output").unwrap();
    let in_path = m.value_of("input").unwrap();

    println!("Gathering images");
    let now = Instant::now();
    let images = get_integral_images(in_path);
    println!(
        "{} images found in {} taking {} seconds",
        images.len(),
        in_path,
        now.elapsed().as_secs()
    );

    println!("Creating a vector of all possible features");
    let now = Instant::now();
    let features = Feature::get_all();
    println!("Created Vector in {} seconds", now.elapsed().as_secs());

    println!("Filtering out the most important features");
    let features = Feature::filter(&features);

    // Output the data
    println!("Saving to {}", out_path);
    let data = serde_json::to_string(&features).unwrap();
    fs::write(out_path, &data).expect("Unable to write to file");

    println!("Finished");
    unimplemented!();
}

fn get_integral_images(in_path: &str) -> Vec<IntegralImage> {
    let count = fs::read_dir(in_path).unwrap().count();
    let mut images = Vec::<IntegralImage>::with_capacity(count);
    for path in fs::read_dir("images").unwrap() {
        // Open the image
        let img = ImageReader::open(path.unwrap().path())
            .unwrap()
            .decode()
            .unwrap();

        // Convert image to Integral Image
        let img = IntegralImage::new(img);

        // Add the image to images
        images.push(img);
    }
    images
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
    false_positives: usize,
    false_negatives: usize,
}
impl Feature {
    /// Gets a vector of all possible features
    pub fn get_all() -> Vec<Feature> {
        let mut features = Vec::<Feature>::with_capacity(200_000);
        for xtl in 0..WL {
            for xmp in xtl..WL {
                for xbr in xmp..WL {
                    for ytl in 0..WL {
                        for ymp in ytl..WL {
                            for ybr in ymp..WL {
                                let f = Feature {
                                    top_left: [xtl, ytl],
                                    mid_point: [xmp, ymp],
                                    bot_right: [xbr, ybr],
                                    false_negatives: 0,
                                    false_positives: 0,
                                };
                                features.push(f);
                            }
                        }
                    }
                }
            }
        }
        features
    }

    pub fn filter(_features: &[Feature]) -> Vec<Feature> {
        unimplemented!();
    }

    /// Evaluates whether or not a feature is in an integral image
    pub fn evaluate(&self, ii: IntegralImage) -> isize {
        let mut sum: isize = 0;
        for rect in &self.get_white_rects() {
            sum += ii.rect_sum(rect);
        }
        for rect in &self.get_black_rects() {
            sum -= ii.rect_sum(rect);
        }
        sum
    }

    /// Gets an array of all of the white rectangles associated with the feature
    fn get_white_rects(&self) -> [Rectangle; 2] {
        let tl = self.top_left;
        let br = self.bot_right;
        let mp = self.mid_point;

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
        let tl = self.top_left;
        let br = self.bot_right;
        let mp = self.mid_point;

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

/// The relative size of sweeping window used in image detection
pub const WS: usize = 4;
pub const WL: usize = WS * 7;
pub const WH: usize = WS * 8;

pub struct IntegralImage {
    pixels: Vec<usize>,
}
impl IntegralImage {
    pub fn new(img: DynamicImage) -> IntegralImage {
        // Resize the image and turn it to grayscale
        let img = img
            .resize(WL as u32, WH as u32, FilterType::Triangle)
            .into_luma8();

        // Calculate each pixel of the integral image
        let mut pixels = Vec::<usize>::with_capacity(WL * WH);
        for y in 0..WH {
            for x in 0..WL {
                let pixel = img.get_pixel(x as u32, y as u32)[0] as usize;
                pixels.push({
                    if (x == 0) && (y == 0) {
                        pixel
                    } else if x == 0 {
                        pixels[x + WL * (y - 1)] as usize + pixel
                    } else if y == 0 {
                        pixels[(x - 1) + WL * y] as usize + pixel
                    } else {
                        pixels[((x - 1) + WL * y)] as usize
                            + (pixels[x + WL * (y - 1)] as usize)
                            + pixel
                            - (pixels[(x - 1) + WL * (y - 1)] as usize)
                    }
                })
            }
        }
        IntegralImage { pixels }
    }
    pub fn rect_sum(&self, r: &Rectangle) -> isize {
        let tl = r.top_left;
        let br = r.bot_right;
        (self.pixels[br[0] + WL * br[1]]
            - self.pixels[br[0] + WL * tl[1]]
            - self.pixels[tl[0] + WL * br[1]]
            + self.pixels[tl[0] + WL * tl[1]]) as isize
    }
}
