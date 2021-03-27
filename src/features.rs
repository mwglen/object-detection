use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

/// The smallest unsigned integer primitive that can index into the Window
type uint_ws = u8;

/// The relative size of sweeping window used in image detection
pub const WS: uint_ws = 4;
pub const WL: uint_ws = WS * 7;
pub const WH: uint_ws = WS * 8;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WL as u32;
pub const WL_: usize = WL as usize;
pub const WH_: usize = WL as usize;


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
    println!("Vector has size {}", features.len());
    use std::mem::size_of;
    println!("It takes up {} bytes in memory",
        features.len()*size_of::<FeatureData>());
    println!("It takes up {} bytes in memory",
        features.capacity()*size_of::<FeatureData>());

    println!("Filtering out the most important features");
    let features = Feature::filter(features.as_slice());

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

#[derive(Serialize, Deserialize, Debug)]
pub struct Rectangle {
    top_left: [uint_ws; 2],
    bot_right: [uint_ws; 2],
}

#[derive(Serialize, Deserialize, Debug)]
struct FeatureData {
    feature: Feature,
    false_positives: u16,
    false_negatives: u16,
}


// Two rectangle, three rectangle, and four rectangle features can all be
// represented using only two rectangles
#[derive(Serialize, Deserialize, Debug)]
enum Feature {
        TwoRect { black: Rectangle, white: Rectangle },
        ThreeRect { black: Rectangle, white: [Rectangle; 2] },
        FourRect { black: [Rectangle; 2], white: [Rectangle; 2] },
}
impl Feature {
    /// Gets a vector of all possible features
    pub fn get_all() -> Vec<FeatureData> {
        let mut features = Vec::<FeatureData>::with_capacity(200_000);
        for xtl in 0..WL {
            for xmp in xtl..WL {
                for xbr in xmp..WL {
                    for ytl in 0..WL {
                        for ymp in ytl..WL {
                            for ybr in ymp..WL {
                                // If the rectangle has no area continue
                                if (xtl == xbr) || (ytl == ybr) {continue;}

                                // If it is a vertical two rectangle feature
                                if xtl == xmp {
                                    // If it is a one rectangle feature skip it
                                    if (ymp == ytl) || (ymp == xbr) {continue;}
                                    
                                    let white = Rectangle {
                                        top_left: [xtl, ytl],
                                        bot_right: [xbr, ymp],
                                    };
                                    let black = Rectangle {
                                        top_left: [xtl, ymp],
                                        bot_right: [xbr, ybr],
                                    };
                                    let f = FeatureData {
                                        feature: Feature::TwoRect { white, black },
                                        false_positives: 0,
                                        false_negatives: 0,
                                    };
                                    features.push(f);
                                    continue;
                                }
                                
                                // If it is a horizontal two rectangle feature
                                if ytl == ymp {
                                    // If it is a one rectangle feature skip it
                                    if (xmp == xtl) || (xmp == xbr) {continue;}
                                    
                                    let white = Rectangle {
                                        top_left: [xtl, ytl],
                                        bot_right: [xmp, ybr],
                                    };
                                    let black = Rectangle {
                                        top_left: [xmp, ytl],
                                        bot_right: [xbr, ybr],
                                    };
                                    let f = FeatureData {
                                        feature: Feature::TwoRect { white, black },
                                        false_positives: 0,
                                        false_negatives: 0,
                                    };
                                    features.push(f);
                                    continue;
                                }

                                // If it is a mirror image of an already added
                                // two rectangle feature, continue
                                if (xbr == xmp) || (ybr == ymp) {continue;}

                                // If it is a four rectangle Feature
                                let white1 = Rectangle {
                                    top_left: [xmp, ytl],
                                    bot_right: [xbr, ymp],
                                };
                                let white2 = Rectangle {
                                    top_left: [xtl, ymp],
                                    bot_right: [xmp, ybr],
                                };
                                let black1 = Rectangle {
                                    top_left: [xtl, ytl],
                                    bot_right: [xmp, ymp],

                                };
                                let black2 = Rectangle {
                                    top_left: [xmp, ymp],
                                    bot_right: [xbr, ybr],
                                };
                                let white = [white1, white2];
                                let black = [black1, black2];
                                let f = FeatureData {
                                    feature: Feature::FourRect { white, black },
                                    false_positives: 0,
                                    false_negatives: 0,
                                };
                                features.push(f);
                            }
                        }
                    }
                }
            }
        }

        for xtl in 0..(WL-3) {
            for xm1 in (xtl+1)..(WL-2) {
                for xm2 in (xm1+1)..(WL-1) {
                    for xbr in (xm2+1)..WL {
                        for ytl in 0..(WL-1) {
                            for ybr in (ytl+1)..WL {
                                // Horizontal Feature
                                let white1 = Rectangle {
                                    top_left: [xtl, ytl],
                                    bot_right: [xm1, ybr],
                                };
                                let white2 = Rectangle {
                                    top_left: [xm2, ytl],
                                    bot_right: [xbr, ybr],
                                };
                                let black = Rectangle {
                                    top_left: [xm1, ytl],
                                    bot_right: [xm2, ybr],
                                };
                                let white = [white1, white2];
                                let f = FeatureData {
                                    feature: Feature::ThreeRect { white, black },
                                    false_positives: 0,
                                    false_negatives: 0,
                                };
                                features.push(f);

                                let xtl = ytl; let ytl = xbr; let xbr = ybr;
                                let ybr = xtl; let ym1 = xm2; let ym2 = xm1;

                                // Vertical Feature
                                let white1 = Rectangle {
                                    top_left: [xtl, ytl],
                                    bot_right: [xbr, ym1],
                                };
                                let white2 = Rectangle {
                                    top_left: [xtl, ym2],
                                    bot_right: [xbr, ybr],
                                };
                                let black = Rectangle {
                                    top_left: [xtl, ym1],
                                    bot_right: [xbr, ym2],
                                };
                                let white = [white1, white2];
                                let f = FeatureData {
                                    feature: Feature::ThreeRect { white, black },
                                    false_positives: 0,
                                    false_negatives: 0,
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

    pub fn filter(_features: &[FeatureData]) -> Vec<Feature> {
        unimplemented!();
    }

    /// Evaluates whether or not a feature is in an integral image
    pub fn evaluate(&self, ii: IntegralImage) -> isize {
        match self {
            Feature::TwoRect{black, white} => {
                ii.rect_sum(&black) - ii.rect_sum(&white)
            }
            Feature::ThreeRect{black, white} => {
                ii.rect_sum(&black) - ii.rect_sum(&white[0]) - ii.rect_sum(&white[1])
            }
            Feature::FourRect{black, white} => {
                ii.rect_sum(&black[0]) + ii.rect_sum(&black[1])
                    - ii.rect_sum(&white[0]) - ii.rect_sum(&white[1])
            }
        }
    }
}

pub struct IntegralImage {
    pixels: Vec<usize>,
}
impl IntegralImage {
    pub fn new(img: DynamicImage) -> IntegralImage {
        // Resize the image and turn it to grayscale
        let img = img.resize(WL_32, WH_32, FilterType::Triangle).into_luma8();

        // Calculate each pixel of the integral image
        let mut pixels = Vec::<usize>::with_capacity(WL_ * WH_);
        for y in 0..WH_ {
            for x in 0..WL_ {
                let pixel = usize::from(img.get_pixel(x as u32, y as u32)[0]);
                pixels.push({
                    if (x == 0) && (y == 0) {pixel} 
                    else if x == 0 {usize::from(pixels[WL_*(y-1)]) + pixel}
                    else if y == 0 {usize::from(pixels[x-1]) + pixel}
                    else {
                        usize::from(pixels[((x-1) + WL_*y)])
                            + usize::from(pixels[x + WL_*(y-1)])
                            - usize::from(pixels[(x-1) + WL_*(y-1)])
                            + pixel
                    }
                })
            }
        }
        IntegralImage { pixels }
    }
    pub fn rect_sum(&self, r: &Rectangle) -> isize {
        let xtl = usize::from(r.top_left[0]);
        let ytl = usize::from(r.top_left[1]);
        let xbr = usize::from(r.bot_right[0]);
        let ybr = usize::from(r.bot_right[1]);
        
        (self.pixels[xbr + WL_*ybr]
            - self.pixels[xbr + WL_*ytl]
            - self.pixels[xtl + WL_*ybr]
            + self.pixels[xtl + WL_*ytl]) as isize
    }
}
