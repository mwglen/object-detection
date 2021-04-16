use super::{TrainingImages, WL, Feature, Rectangle, IntegralImage};
use serde::{Deserialize, Serialize};
use indicatif::ProgressBar;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct WeakClassifier {
    feature: Feature,
    threshold: i64,
}
impl WeakClassifier {
    pub fn calculate_threshold(&mut self, set: &mut TrainingImages) {
        // Sort the training images based on it's evaluation
        set.sort(self);

        // Sum of the weights of the face samples
        let afs: f64 = set.iter()
            .filter(|(_, _, &is_face)| is_face)
            .map(|(_, weight, _)| weight).sum();

        // Sum of the weights of the non-face samples
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
                    for ytl in 0..WH {
                        for ymp in ytl..WH {
                            for ybr in ymp..WH {
                                // If the rectangle has no area continue
                                if (xtl == xbr) || (ytl == ybr) {continue;}

                                // If it is a vertical two rectangle feature
                                if xtl == xmp {
                                    // If it is a one rectangle feature skip it
                                    if (ymp == ytl) || (ymp == ybr) {continue;}
                                    
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
                        for ytl in 0..(WH-1) {
                            for ybr in (ytl+1)..WH {
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
                            }
                        }
                    }
                }
            }
        }
        for ytl in 0..(WH-3) {
            for ym1 in (ytl+1)..(WH-2) {
                for ym2 in (ym1+1)..(WH-1) {
                    for ybr in (ym2+1)..WH {
                        for xtl in 0..(WL-1) {
                            for xbr in (xtl+1)..WL {
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
            
            // Calculate thresholds using the updated weights
            println!("Calculating thresholds of weak classifiers");
            let now = Instant::now();
            let bar = ProgressBar::new(wcs.len() as u64);
            for wc in &mut wcs {
                wc.calculate_threshold(&mut images);
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

