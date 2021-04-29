use super::{WeakClassifier, ImageData, 
    SC_SIZE, IntegralImage, Rectangle};
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
    ) -> StrongClassifier {
        let wcs: Vec<_> = (1..=SC_SIZE).map(|i| {
            // Calculate Thresholds
            WeakClassifier::calculate_thresholds(wcs, set);
            
            // Get the best weak classifier
            println!("Choosing Weak Classifier {} of {}", i, SC_SIZE);
            let wc = WeakClassifier::get_best(&wcs, set);
            
            // Update the weights
            wc.update_weights(set);
            
            wc
        }).collect();

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
