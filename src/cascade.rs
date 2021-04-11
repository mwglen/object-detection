use image::{imageops::FilterType, io::Reader as ImageReader, DynamicImage};
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use indicatif::ProgressBar;

/// The smallest unsigned integer primitive that can index into the Window
type WindowSize = u8;

/// The relative size of sweeping window used in image detection
pub const WS: WindowSize = 2;
pub const WL: WindowSize = WS * 7;
pub const WH: WindowSize = WS * 8;
pub const WL_32: u32 = WL as u32;
pub const WH_32: u32 = WH as u32;
pub const WL_: usize = WL as usize;
pub const WH_: usize = WH as usize;

pub fn main(m: &clap::ArgMatches) {
    let out_path = m.value_of("output").unwrap();
    let faces_dir = m.value_of("faces_dir").unwrap();
    // let bg_dir = m.value_of("bg_dir").unwrap();
    let bg_dir = faces_dir;

    println!("Gathering images");
    let now = Instant::now();
    let mut images = TrainingImages::from_dirs(faces_dir, bg_dir);
    println!(
        "{} images found in {} taking {} seconds",
        images.len(),
        faces_dir,
        now.elapsed().as_secs()
    );

    println!("Creating a vector of all weak classifiers");
    let now = Instant::now();
    let mut wcs = WeakClassifier::get_all();
    println!("Created Vector in {} seconds", now.elapsed().as_secs());
    println!("Vector has size {}", wcs.len());
    use std::mem::size_of;
    println!("It takes up {} bytes in memory",
        wcs.len()*size_of::<WeakClassifier>());
    println!("It takes up {} bytes in memory",
        wcs.capacity()*size_of::<WeakClassifier>());

    println!("Calculating thresholds of weak classifiers");
    let now = Instant::now();
    let bar = ProgressBar::new(wcs.len() as u64);
    for wc in &mut wcs {
        wc.calculate_threshold(&mut images);
        bar.inc(1);
    }
    bar.finish();
    println!("Calculated thresholds in {} seconds", now.elapsed().as_secs());

    println!("Building the cascade of weak classifiers");
    let now = Instant::now();
    let wcs = WeakClassifier::build_cascade(&wcs, &mut images, 5);
    println!("Built the cascade in {} seconds", now.elapsed().as_secs());

    // Output the data
    println!("Saving to {}", out_path);
    let data = serde_json::to_string_pretty(&wcs).unwrap();
    fs::write(out_path, &data).expect("Unable to write to file");

    println!("Finished");
}


#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct Rectangle {
    top_left: [WindowSize; 2],
    bot_right: [WindowSize; 2],
}

// Two rectangle, three rectangle, and four rectangle features can all be
// represented using only two rectangles
#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub enum Feature {
        TwoRect { black: Rectangle, white: Rectangle },
        ThreeRect { black: Rectangle, white: [Rectangle; 2] },
        FourRect { black: [Rectangle; 2], white: [Rectangle; 2] },
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct WeakClassifier {
    feature: Feature,
    threshold: i64,
}
impl WeakClassifier {
    fn calculate_threshold(&mut self, set: &mut TrainingImages) {
        // Sort the training images based on it's evaluation
        set.sort(self);

        // Sum of the weights of the face samples
        let afs: f64 = set.iter()
            .filter(|(_, _, &is_face)| is_face)
            .map(|(_, weight, _)| weight).sum();

        // Sum of the weights of the face samples
        let abg: f64 = set.iter()
            .filter(|(_, _, &is_face)| !is_face)
            .map(|(_, weight, _)| weight).sum();

        let mut fs: f64 = 0.0; // Sum of the weights of the face samples so far
        let mut bg: f64 = 0.0; // Sum of the weights of background samples so far

        let errors: Vec<_> = set.iter().map(|(_, weight, is_face)| {
            // Add the weight to fs/bg
            if *is_face {fs += *weight} else {bg += *weight};
            
            // Comput the error function
            f64::min(bg+(afs-fs), fs+(abg-bg))
        }).collect();
    }
    pub fn get_all() -> Vec<WeakClassifier> {
        let mut wcs = Vec::<WeakClassifier>::with_capacity(200_000);
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
                                    wcs.push(WeakClassifier {
                                        feature: Feature::TwoRect { white, black },
                                        threshold: 0,
                                    });
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
                                    wcs.push(WeakClassifier {
                                        feature: Feature::TwoRect { white, black },
                                        threshold: 0,
                                    });
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
                                wcs.push(WeakClassifier {
                                    feature: Feature::FourRect { white, black },
                                    threshold: 0,
                                });
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
                                wcs.push(WeakClassifier {
                                    feature: Feature::ThreeRect { white, black },
                                    threshold: 0,
                                });

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
                                wcs.push(WeakClassifier {
                                    feature: Feature::ThreeRect { white, black },
                                    threshold: 0,
                                });
                            }
                        }
                    }
                }
            }
        }
        wcs
    }

    pub fn build_cascade(
        wcs: &Vec<WeakClassifier>, 
        set: &mut TrainingImages,
        cascade_size: usize
    ) -> Vec<WeakClassifier> {
        (0..cascade_size).map(|i| {
            println!("#########################");
            println!("Finding Classifier {} of {}", i + 1, cascade_size);
            println!("#########################");

            // Normalize the weights of the training images
            println!("Normalizing Image Weights");
            TrainingImages::normalize_weights(set);

            // Calculate the error for each weak classifier
            println!("Calculating Errors for Weak Classifiers");
            let bar = ProgressBar::new(wcs.len() as u64);
            let errors: Vec<_> = wcs.iter().map(|wc| {
                bar.inc(1);
                set.iter().filter(|(image, _, &is_face)| {
                    is_face ^ !wc.evaluate(image)
                }).map(|(_, w, _)| *w).sum::<f64>()
            }).collect();
            bar.finish();
            
            // Find the index of the classifier with the smallest error value
            println!("Finding Classifier with the Smallest Error");
            let (index, err) = errors
                .iter()
                .enumerate()
                .min_by(|v1, v2| {
                v1.1.partial_cmp(v2.1).unwrap()
            }).expect("Errors was empty!?");

            // Update the weights:
            // If the chosen classifier incorrectly classified the image
            // add more weight, else decrease weight
            println!("Updating the Image Weights");
            let beta_t = err / (1.0 - err);

            set.iter_mut().filter(|(image, _, &is_face)| {
                wcs[index].evaluate(image) == is_face
            }).for_each(|(_, weight, _)| {
                *weight *= beta_t;
            });
            
            wcs[index]
        }).collect()
    }
    
    pub fn evaluate(&self, ii: &IntegralImage) -> bool {
        let sum = match self.feature {
            Feature::TwoRect{black, white} => {
                ii.rect_sum(&black) - ii.rect_sum(&white)
            }
            Feature::ThreeRect{black, white} => {
                ii.rect_sum(&black) - ii.rect_sum(&white[0]) - ii.rect_sum(&white[1])
            }
            Feature::FourRect{black, white} => {
                black.iter().map(|rect| ii.rect_sum(rect)).sum::<i64>()
                    - white.iter().map(|rect| ii.rect_sum(rect)).sum::<i64>()
            }
        };
        sum >= self.threshold
    }
}

#[derive(Debug, Clone)]
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
pub struct TrainingImages {
    images: Vec<IntegralImage>,
    weights: Vec<f64>,
    is_face: Vec<bool>,
}

pub struct TrainingImage {
    images: IntegralImage,
    weights: f64,
    is_face: bool,
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


