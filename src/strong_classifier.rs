use super::{WeakClassifier, ImageData, SC_SIZE, WC_PATH, IntegralImage};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use std::fs;

/// A strong classifier (made up of weighted weak classifiers)
#[derive(Serialize, Deserialize)]
pub struct StrongClassifier {
    wcs: Vec<WeakClassifier>,
    weights: Vec<f64>,
} impl StrongClassifier {
    
    /// Builds a strong classifier out of weak classifiers
    pub fn new(
        wcs: &[WeakClassifier],
        set: &mut [ImageData],
        tag: usize,
    ) -> StrongClassifier {
        let now = Instant::now();
        let wcs: Vec<_> = (1..=SC_SIZE).map(|i| {
            // Get the best weak classifier
            println!("Choosing weak classifier {} of {}", i, SC_SIZE);
            let wc = WeakClassifier::get_best(&wcs, set);
            
            // Save weak classifier to cache
            let out_path = WC_PATH.to_owned() + &tag.to_string() + "." 
                + &i.to_string() + ".json"; 
            let data = serde_json::to_string(&wc).unwrap();
            fs::write(out_path, &data).expect("Unable to cache weak classifier");
            wc
        }).collect();
        println!("Built strong classifier in {} seconds", now.elapsed().as_secs());

        // Calculate the weights of each weak classifier in the strong classifier
        let weights = wcs.iter().map(|wc| wc.error(set))
            .map(|err| err / (1.0 - err))
            .map(|beta| f64::ln(1.0/beta)).collect();

        // Build and return the strong classifier
        StrongClassifier {
            wcs,
            weights,
        }
    }
    pub fn classify(&self, ii: &IntegralImage) -> bool {
        self.wcs.iter().zip(self.weights.iter())
            .filter(|(wc, _)| wc.classify(ii, None))
            .map(|(_, weight)| weight)
            .sum::<f64>() >= self.weights.iter().sum::<f64>()/2.0
    }
}
