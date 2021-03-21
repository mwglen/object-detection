use super::IntegralImage;
use std::fs;
use serde::{Deserialize, Serialize};
use image::io::Reader as ImageReader;

pub fn main() {
    let data = find_features();

    // Output the data
    let data = serde_json::to_string(&data).unwrap();
    fs::write("features.json", &data).expect("Unable to write to file");
}

fn find_features() -> Vec::<Feature> {
    
    // Store the training images into a vector after resizing them
    // to 24x24 and changing them to greyscale Integral Images
    let count = fs::read_dir("images").unwrap().count();
    println!("{} images found in ./images", count);
    let mut images = Vec::<IntegralImage>::with_capacity(count);
    for path in fs::read_dir("images").unwrap() {
        // Open the image
        let img = ImageReader::open(path.unwrap().path()).unwrap().decode().unwrap();
        
        // Turn the image into an Integral Image
        let img = IntegralImage::new(img);

        // Add the image to images
        images.push(img);
    }
    println!("Finished gathering images");
    
    
    println!("Getting a list of features");
    let features = Vec::<TwoRectFeature>::with_capacity(100_000);
    /*
    // Calculate two rectangle features
    for start_y in 0..23 {
        for start_x in 0..22 {
            for end_y in (start_y+1)..24 {
                for end_x in (start_x+2)..24 {
                    for mid_x in (start_x+1)..(24-end_x) {
                        // Calculate horizontal features
                        let area = img.rectangle_sum(start_x, start_y, mid_x, end_y) as isize 
                            - img.rectangle_sum(mid_x, start_y, end_x, end_y) as isize;

                        // If the area of the rectangle is greater than the threshold
                        if (isFace==false) features.fale_positive++;
                        if (isFace==true) features.false_negatives++;

                        // Calculate vertical features
                        let area = img.rectangle_sum(start_y, start_x, mid_x, end_x) as isize
                                - img.rectangle_sum(mid_x, start_x, end_y, end_x)
                        if (isFace==false) features.fale_positive++;
                        if (isFace==true && area > 0) features.false_negatives++;
                    }
                }
            }
        }
    }
        
    // Get a vector for three rectangle features

    // Get a vector for four rectangle features
    */
    unimplemented!();
}

struct TwoRectFeature {
    start_x: u8,
    start_y: u8,
    mid_x: u8,
    end_x: u8,
    end_y: u8,
    false_negatives: u32,
    false_positives: u32,
    horizontal: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct Feature {
}
