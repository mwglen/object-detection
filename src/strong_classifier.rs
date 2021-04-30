use serde::{Deserialize, Serialize};

use super::{ImageData, IntegralImage, Rectangle, WeakClassifier};

/// A strong classifier (made up of weighted weak classifiers)
#[derive(Debug, Serialize, Deserialize)]
pub struct StrongClassifier {
    wcs: Vec<WeakClassifier>,
    weights: Vec<f64>,
}
impl StrongClassifier {
    /// Builds a strong classifier out of weak classifiers
    pub fn new(
        wcs: &mut [WeakClassifier],
        set: &mut [ImageData],
        size: usize,
    ) -> StrongClassifier {
        let (wcs, weights) = (1..=size)
            .map(|i| {
                // Normalize weights
                ImageData::normalize_weights(set);

                // Calculate Thresholds
                WeakClassifier::calculate_thresholds(wcs, set);

                // Get the best weak classifier
                println!(
                    "Choosing Weak Classifier {} of {}",
                    i,
                    size,
                );
                let wc = WeakClassifier::get_best(&wcs, set);

                // Update the weights
                let weight = wc.update_weights(set);
                (wc, weight)
            })
            .unzip();

        // Build and return the strong classifier
        StrongClassifier { wcs, weights }
    }

    pub fn classify(
        &self,
        ii: &IntegralImage,
        w: Option<(Rectangle<u32>, f64)>,
    ) -> bool {
        self.wcs
            .iter()
            .zip(self.weights.iter())
            .filter(|(wc, _)| wc.classify(ii, w))
            .map(|(_, weight)| weight)
            .sum::<f64>()
            >= self.weights.iter().sum::<f64>() / 2.0
    }
}
