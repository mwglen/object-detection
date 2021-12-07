use serde::{Deserialize, Serialize};
use super::{
    ImageData, 
    WeakClassifier,
    Classifier,
    MAX_FALSE_POS,
    ImageTrait,
};

/// A strong classifier (made up of weighted weak classifiers)
#[derive(Debug, Serialize, Deserialize)]
pub struct StrongClassifier {
    wcs: Vec<WeakClassifier>,
    weights: Vec<f64>,
} impl StrongClassifier {

    /// Builds a strong classifier out of weak classifiers
    /// from a list of potential weak classifiers
    pub fn build(
        all_wcs: &mut [WeakClassifier],
        set: &mut [ImageData],
        num_wcs: Option<usize>,
    ) -> StrongClassifier {

        let mut wcs = Vec::<WeakClassifier>::new();
        let mut weights = Vec::<f64>::new();

        let mut false_pos = 0.0;
        let mut i = 1;
        loop {
            // Normalize weights
            ImageData::normalize_weights(set);

            // Calculate Thresholds
            WeakClassifier::calculate_thresholds(all_wcs, set);

            // Tell user that we are finding new weak classifier
            println!(
                "Choosing Weak Classifier {}{}{}", 
                i.to_string(),
                num_wcs.map_or("", |_| " of "),
                num_wcs.map_or("".to_owned(), |n| n.to_string()),
            );
            
            // Get the best weak classifier
            let wc = WeakClassifier::get_best(&all_wcs, set);

            // Update the weights
            weights.push(wc.update_weights(set));
            wcs.push(wc);
            
            // Print informattion about current cascade
            println!("Current False Positive Rate: {}", 
                false_pos);

            // Determine whether or not to break
            let should_break = num_wcs.map_or_else(
                || false_pos <= MAX_FALSE_POS, |n| i == n);
            if should_break { break }

            i+=1;
        }

        // Build and return the strong classifier
        StrongClassifier { wcs, weights }
    }
} impl Classifier for StrongClassifier {
    fn classify(&self, img: &impl ImageTrait) -> bool {
        self.wcs
            .iter()
            .zip(self.weights.iter())
            .filter(|(wc, _)| wc.classify(img))
            .map(|(_, weight)| weight)
            .sum::<f64>()
            >= self.weights.iter().sum::<f64>() / 2.0
    }
}
