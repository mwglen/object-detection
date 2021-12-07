use super::{
    Classifier,
    StrongClassifier,
    ImageTrait,
    ImageData,
    WeakClassifier,
    TARGET_FALSE_POS,
    FILTER, USE_LAYOUT, LAYOUT
};
use serde::{Deserialize, Serialize};

/// A cascade of strong classifiers
#[derive(Deserialize, Serialize, Debug)]
pub struct Cascade {
    /// The strong classifiers contained in the cascade
    scs: Vec<StrongClassifier>,
} impl Cascade {

    /// Builds a cascade
    pub fn build(mut set: Vec<ImageData>) -> Cascade {
        // Get weak classifiers
        println!("{:-^30}", " Getting Weak Classifiers ");
        let wcs = WeakClassifier::get_all();
        println!("Found {} possible weak classifiers", wcs.len());
        
        // Filter out underperforming weak classifiers if specified
        let mut wcs = if FILTER {
            println!("Filtering out underperforming weak classifiers");
            WeakClassifier::filter(wcs, &mut set)
        } else {wcs};

        let mut scs = Vec::<StrongClassifier>::new();
        let num_scs = if USE_LAYOUT {Some(LAYOUT.len())} else {None};
        
        let mut false_pos = 0.0;
        let mut i = 1;
        loop {

            // Tell user that we are building a new strong classifier
            println!(
                "Building Strong Classifier {}{}{}", 
                i.to_string(),
                num_scs.map_or("", |_| " of "),
                num_scs.map_or("".to_owned(), |n| n.to_string()),
            );
            
            // Get the best weak classifier
            let num_wcs = if USE_LAYOUT {Some(LAYOUT[i])} else {None};
            let sc = StrongClassifier::build(&mut wcs, &mut set, num_wcs);

            // Remove the true negatives from the training set
            set.retain(|id| id.is_object || sc.classify(&id.image));
            scs.push(sc);
            
            // Print informattion about current cascade
            println!("Current False Positive Rate: {}", 
                false_pos);
            
            // Determine whether or not to break
            let should_break = num_scs.map_or_else(
                || false_pos <= TARGET_FALSE_POS, |n| i == n);
            if should_break { break }

            i += 1;
        }

        Cascade{scs}
    }
} impl Classifier for Cascade {
    fn classify(&self, img: &impl ImageTrait) -> bool {
        self.scs.iter().all(|sc| sc.classify(img))
    }
}
