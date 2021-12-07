use super::{
    ImageData,
    ImageTrait,
};

pub trait Classifier {
    /// Classifies an image
    fn classify(&self, img: &impl ImageTrait) -> bool;

    /// Tests the classifier over a set of images and returns a tuple
    /// containing the false positive rate and the detection rate.
    fn test(&self, set: &Vec<ImageData>) -> (f64, f64) {
        // Test the cascade over training images
        let mut correct_objects: f64 = 0.0;
        let mut correct_others: f64 = 0.0;
        let mut num_objects: f64 = 0.0;
        let mut num_others: f64 = 0.0;

        for data in set {
            let eval = self.classify(&data.image);
            if data.is_object {num_objects += 1.0;}
            if !data.is_object {num_others += 1.0;}
            if data.is_object && eval {correct_objects += 1.0;}
            if !data.is_object && !eval {correct_others += 1.0;}
        }

        // Return the false positive rate and the detection rate
        let fpr = (num_others - correct_others) / num_others;
        let dtr = correct_objects / num_objects;
        return (fpr, dtr);
    }

    /// Validates that a classifier has a false positive rate below
    /// a target rate and a detection rate above a target rate.
    fn validate(&self, target_fpr: f64, target_dtr: f64, set: &Vec<ImageData>)
    -> bool{
        let (fpr, dtr) = self.test(set);
        return (fpr < target_fpr) && (dtr > target_dtr);
    }
}

