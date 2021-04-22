use super::{
    TrainingImages, WL, WH, Feature, 
    Rectangle, IntegralImage, WindowSize
};
use serde::{Deserialize, Serialize};
use indicatif::ProgressBar;
use std::time::Instant;
use std::cmp::Ordering;

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
struct OrderedF64(f64);
impl Eq for OrderedF64 { }
impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct WeakClassifier {
    feature: Feature,
    threshold: i64,
    pos_polarity: bool,
}
impl WeakClassifier {
    pub fn calculate_threshold(
        &mut self, 
        set: &mut TrainingImages, 
        afs: f64, 
        abg: f64
    ) {
        // Sort the training images based on it's evaluation
        set.sort(self);

        let mut fs: f64 = 0.0; // Sum of the weights of the face samples so far
        let mut bg: f64 = 0.0; // Sum of the weights of background samples so far

        let image = set.iter().min_by_key(|(_, &weight, &is_face)| {
            // Add the weight to fs/bg
            if is_face {fs += weight} else {bg += weight};
            
            // Compute the error function
            OrderedF64(f64::min(bg+(afs-fs), fs+(abg-bg)))
        }).unwrap().0;

        self.threshold = self.evaluate_num(image);
    }
    pub fn new(feature: Feature) -> WeakClassifier {
        WeakClassifier {
            feature,
            threshold: 0,
            pos_polarity: false,
        }
    }
    pub fn get_all() ->Vec<WeakClassifier> {
        use Feature::*;
        let mut wcs = Vec::<WeakClassifier>::with_capacity(200_000);
        for w in 1..(WL+1) {
            for h in 1..(WH+1) {
                let mut i = 0;
                while (i + w) < WL {
                    let mut j = 0;
                    while (j + h) < WH {
                        // Horizontal Two Rectangle Features
                        if (i + 2 * w) < WL { 
                            let white = Rectangle::<WindowSize>::new(i, j, w, h);
                            let black = Rectangle::<WindowSize>::new(i+w, j, w, h);
                            let wc = WeakClassifier::new(TwoRect{white, black});
                            wcs.push(wc);
                        }

                        // Vertically Two Rectangle Feature
                        if (j + 2 * h) < WH { 
                            let white = Rectangle::<WindowSize>::new(i, j, w, h);
                            let black = Rectangle::<WindowSize>::new(i, j+h, w, h);
                            let wc = WeakClassifier::new(TwoRect{white, black});
                            wcs.push(wc);
                        }

                        // Horizontal Three Rectangle Feature
                        if (i + 3 * w) < WL {
                            let left = Rectangle::<WindowSize>::new(i, j, w, h);
                            let mid = Rectangle::<WindowSize>::new(i+w, j, w, h);
                            let right = Rectangle::<WindowSize>::new(i+2*w, j, w, h);

                            let white = [left, right];
                            let black = mid; 
                            let wc = WeakClassifier::new(ThreeRect{white, black});
                            wcs.push(wc);
                        }
                        
                        // Vertical Three Rectangle Feature
                        if (j + 3 * h) < WH {
                            let top = Rectangle::<WindowSize>::new(i, j, w, h);
                            let mid = Rectangle::<WindowSize>::new(i, j+h, w, h);
                            let bot = Rectangle::<WindowSize>::new(i, j+2*h, w, h);

                            let white = [top, bot];
                            let black = mid; 
                            let wc = WeakClassifier::new(ThreeRect{white, black});
                            wcs.push(wc);
                        }
                        
                        // Four rectangle features
                        if (i + 2 * w) < WL && (j + 2 * h) < WH {
                            let top_left = Rectangle::<WindowSize>::new(i, j, w, h);
                            let top_right = Rectangle::<WindowSize>::new(i+w, j, w, h);
                            let bot_left = Rectangle::<WindowSize>::new(i, j+h, w, h);
                            let bot_right = Rectangle::<WindowSize>::new(i+w, j+h, w, h);

                            let white = [top_right, bot_left];
                            let black = [top_left, bot_right]; 
                            let wc = WeakClassifier::new(FourRect{white, black});
                            wcs.push(wc);
                        }
                        j += 1;
                    }
                    i += 1;
                }
            }
        }
        wcs
    }
    
    pub fn build_cascade(
        wcs: &mut Vec<WeakClassifier>, 
        set: &mut TrainingImages,
        cascade_size: usize,
    ) -> Vec<WeakClassifier> {
        (0..cascade_size).map(|i| {
            println!("#########################");
            println!("Finding Classifier {} of {}", i + 1, cascade_size);
            println!("#########################");

            // Normalize the weights of the training images
            println!("Normalizing Image Weights");
            TrainingImages::normalize_weights(set);
            
            // Calculate thresholds using the updated weights
            println!("Calculating thresholds of weak classifiers");
            let now = Instant::now();
            let bar = ProgressBar::new(wcs.len() as u64);

            // Sum of the weights of the face samples
            let afs: f64 = set.iter()
                .filter(|(_, _, &is_face)| is_face)
                .map(|(_, weight, _)| weight).sum();
            // Sum of the weights of the non-face samples
            let abg: f64 = set.iter()
                .filter(|(_, _, &is_face)| !is_face)
                .map(|(_, weight, _)| weight).sum();
            
            for wc in wcs.iter_mut() {
                wc.calculate_threshold(set, afs, abg);
                bar.inc(1);
            }
            bar.finish();
            println!("Calculated thresholds in {} seconds", now.elapsed().as_secs());

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
        self.pos_polarity ^ (self.evaluate_num(ii) > self.threshold)
    }
    
    pub fn evaluate_num(&self, ii: &IntegralImage) -> i64 {
        match self.feature {
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
        }
    }
    
    pub fn evaluate_(
        &self, 
        ii: &IntegralImage, 
        r: &Rectangle::<u32>,
    ) -> bool {
        let value = match self.feature {
             Feature::TwoRect{black, white} => {
                 ii.rect_sum_(&black, r) - ii.rect_sum_(&white, r)
             }
             Feature::ThreeRect{black, white} => {
                 ii.rect_sum_(&black, r) - ii.rect_sum_(&white[0], r) 
                     - ii.rect_sum_(&white[1], r)
             }
             Feature::FourRect{black, white} => {
                 black.iter().map(|rect| ii.rect_sum_(rect, r)).sum::<i64>()
                     - white.iter().map(|rect| ii.rect_sum_(rect, r)).sum::<i64>()
             }
         };
         self.pos_polarity ^ (value > self.threshold)
     }
}
