use super::{WeakClassifier, ImageData, IntegralImage, Rectangle};
use serde::{Deserialize, Serialize};

/// A strong classifier (made up of weighted weak classifiers)
#[derive(Debug, Serialize, Deserialize)]
pub struct StrongClassifier {
    wcs: Vec<WeakClassifier>,
    weights: Vec<f64>,
} impl StrongClassifier {
    
    /// Builds a strong classifier out of weak classifiers
    pub fn new(
        wcs: &mut [WeakClassifier],
        set: &mut [ImageData],
        size: usize,
    ) -> StrongClassifier {
        let (wcs, weights) = (1..=2u64.pow(size as u32)).map(|i| {
            // Calculate Thresholds
            WeakClassifier::calculate_thresholds(wcs, set);
            
            // Get the best weak classifier
            println!("Choosing Weak Classifier {} of {}", i, 2u64.pow(size as u32));
            let wc = WeakClassifier::get_best(&wcs, set);
            
            // Update the weights
            let weight = wc.update_weights(set);
            (wc, weight)
        }).unzip();

        // Build and return the strong classifier
        StrongClassifier {
            wcs,
            weights,
        }
    }
    pub fn classify(
        &self, ii: &IntegralImage, 
        w: Option<(Rectangle<u32>, u32)>
    ) -> bool {
        self.wcs.iter().zip(self.weights.iter())
            .filter(|(wc, _)| wc.classify(ii, w))
            .map(|(_, weight)| weight)
            .sum::<f64>() >= self.weights.iter().sum::<f64>()/2.0
    }
}
