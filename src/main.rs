mod weak_classifier;
mod strong_classifier;
mod primitives;
mod images;
mod constants;
pub use primitives::*;
pub use constants::*;
pub use images::{IntegralImage, ImageData, draw_rectangle};
pub use weak_classifier::WeakClassifier;
pub use strong_classifier::StrongClassifier;

use clap::{load_yaml, App, AppSettings};
use std::path::Path;
use std::fs;
use image::io::Reader as ImageReader;

fn main() {
    // Parse the cli arguments using clap
    let yaml = load_yaml!("cli.yml");
    let app =  App::from_yaml(yaml)
        .setting(AppSettings::ArgRequiredElseHelp)
        .get_matches();

    // Run the specified subcommand
    match app.subcommand() {
        ("process_images", Some(_)) => process_images(),
        ("cascade", Some(_)) => cascade(),
        ("test", Some(_)) => test(),
        ("detect", Some(m)) => detect(m),
        _ => println!("Incorrect subcommand"),
    }
}

/// Processes images for use in building the cascade
fn process_images() {
    // Find and process images
    println!("Training Image:");
    let train_set = ImageData::from_dirs(TRAIN_OBJECT_DIR, TRAIN_OTHER_DIR);
    
    println!("Testing Image:");
    let test_set = ImageData::from_dirs(TEST_OBJECT_DIR, TEST_OTHER_DIR);

    // Save image data to cache
    let train_data = serde_json::to_string(&train_set).unwrap();
    let test_data = serde_json::to_string(&test_set).unwrap();
    fs::write(CACHED_TRAIN_IMAGES, &train_data).expect("Unable to write to file");
    fs::write(CACHED_TEST_IMAGES, &test_data).expect("Unable to write to file");
}

/// Builds the cascade
fn cascade() {
    // Get training images from cache or process raw images if cache is empty
    let mut set: Vec<ImageData> = {
        if Path::new(CACHED_TRAIN_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_TRAIN_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached images")
        } else { println!("Training image data not found in cache"); return; }
    };
    
    // Get all possible weak classifiers
    let mut wcs = WeakClassifier::get_all();
    println!("{:-^30}", " Getting Weak Classifiers ");
    println!("Created vector of {} possible weak classifiers", wcs.len());
    //println!("Obatining the top 10% of weak classifiers");
    //let mut wcs = WeakClassifier::filter(wcs, &mut set);

    println!("{:-^30}", " Building Cascade ");

    // Build the cascade using the weak classifiers
    let layout: Vec<usize> = vec![1, 5, 10];
    let cascade = cascade_from_layout(&layout, &mut wcs, &mut set);

    // Output the data
    println!("Saving cascade to {}", CASCADE);
    let data = serde_json::to_string_pretty(&cascade).unwrap();
    fs::write(CASCADE, &data).expect("Unable to write to file");
}

fn cascade_from_layout(
    layout: &Vec<usize>, wcs: &mut Vec<WeakClassifier>, set: &mut Vec<ImageData>
) -> Vec<StrongClassifier> {
    // Build the cascade using the weak classifiers
    let cascade_size = layout.len();
    let mut cascade = Vec::<StrongClassifier>::with_capacity(cascade_size);
    for (i, &size) in layout.iter().enumerate() {
        println!("Building Strong Classifier {} of {}", i, cascade_size);
        
        // Create a strong classifier
        let sc = StrongClassifier::new(wcs, set, size);

        // Remove the true negatives from the training set
        set.retain(|data| data.is_object || sc.classify(&data.image, None));
        cascade.push(sc);
    }
    cascade
}

// fn cascade_from_false_pos(
//     target_rate: usize, 
//     wcs: &Vec<WeakClassifier>, 
//     set: &Vec<ImageData>,
//     max_rate: usize,
// ) -> Vec<StrongClassifier> {
//     let f: f64 = 1.0;
//     let mut i = 0;
//     while f > TARGET_FALSE_POS {
//         i++;
//         let mut n_i = 0;
//         let prev_f = f
//         while f > (MAX_FALSE_POS * prev_f) {
//             n_i++;
//             // Create a strong classifier
//             let sc = StrongClassifier::new(&mut wcs, &mut set, n_i);
//             let f_i =;
//             let d_i =;

//         }
//         // Remove the true negatives
//         set.retain(|data| data.is_object || sc.classify(&data.image, None));
//     }
// }

/// Test a cascade over training images
fn test() {
    // Get the cached cascade
    let cascade: Vec<StrongClassifier> = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached cascade")
        } else { println!("Cascade not found in cache"); return; }
    };

    // Get processed training images from cache
    let train_set: Vec<ImageData> = {
        if Path::new(CACHED_TRAIN_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_TRAIN_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached image data")
        } else { println!("Testing image data not found in cache"); return; }
    };
    

    println!("Training_Set");
    test_images(&train_set, &cascade);
    
    // Get processed training images from cache
    let test_set: Vec<ImageData> = {
        if Path::new(CACHED_TEST_IMAGES).exists() {
            let data = std::fs::read_to_string(CACHED_TEST_IMAGES).unwrap();
            serde_json::from_str(&data).expect("Unable to read cached image data")
        } else { println!("Testing image data not found in cache"); return; }
    };
    
    println!("Testing_Set");
    test_images(&test_set, &cascade);
}

fn test_images(set: &Vec<ImageData>, cascade: &Vec<StrongClassifier>) {
    // Test the cascade over training images
    println!("Testing the Cascade");
    let mut correct_objects: f64 = 0.0;
    let mut correct_others: f64 = 0.0;
    let mut num_objects: f64 = 0.0;
    for data in set {
        let mut eval = true;
        for sc in cascade {
            if !sc.classify(&data.image, None) {eval = false; break;}
        }
        if data.is_object {num_objects += 1.0;}
        if data.is_object && eval {correct_objects += 1.0;}
        if !data.is_object && !eval {correct_others += 1.0;}
    }

    // Print test results
    println!("correct_objects: {}", correct_objects);
    println!("correct_others: {}", correct_others);
    println!("num_objects: {}", num_objects);
    println!("images_len: {}", set.len());
    println!("Percent of correctly evaluated images: {:.2}%", 
        (correct_objects + correct_others) * 100.0 / (set.len() as f64));
    println!("Percent of correctly evaluated images of the object: {:.2}%",
        correct_objects * 100.0 / num_objects);
    println!("Percent of correctly evaluated images which aren't the object: {:.2}%",
        correct_others * 100.0 / (set.len() as f64 - num_objects));
}

/// Detects the object in an image
fn detect(m: &clap::ArgMatches) {
    // Get the cached cascade
    let cascade: Vec<StrongClassifier> = {
        if Path::new(CASCADE).exists() {
            let data = std::fs::read_to_string(CASCADE).unwrap();
            serde_json::from_str(&data).unwrap()
        } else { println!("Cascade not found in cache"); return; }
    };
    
    // Get the input image
    let path = m.value_of("input_image").unwrap();

    // Get the location to store the output image
    let output_img = "output/".to_owned() + 
        Path::new(path).file_name().unwrap().to_str().unwrap();

    // Open the image
    let img = ImageReader::open(path).unwrap().decode().unwrap().to_luma8();
    let img_width = img.width(); let img_height = img.height();

    // Convert image to integral image
    let ii = IntegralImage::new(&img);

    // Vector to hold detected objects
    let mut objects = Vec::<Rectangle::<u32>>::new();
    
    // This detects objects by sending a "windowed" view into the image to
    // be evaluated by the cascade. The window moves across the image and grows
    // in size. Testing all combinations of rectangles in the images for the object
    let max_f = (img_width/WL_32).min(img_height/WH_32);
    println!("{}", max_f);
    for f in 1..max_f {
        for x in 0..(img_width - f*WL_32) {
            for y in 0..(img_height - f*WH_32) { 
                let w = Rectangle::<u32>::new(x, y, f*WL_32, f*WH_32);
                if cascade.iter().all(|sc| sc.classify(&ii, Some((w, f)))) {
                    objects.push(w);
                }
            }
        }
    }
    println!("Found {} instances of object", objects.len());

    // Reopen the image and conver to rgb, draw rectangles, and then save image
    let mut img = ImageReader::open(path).unwrap().decode().unwrap().to_rgb8();
    for o in objects.iter_mut() { draw_rectangle(&mut img, o); }
    // draw_rectangle(&mut img, &objects[0]);
    img.save(output_img).unwrap();


    // Output detected object
    let data = serde_json::to_string_pretty(&objects).unwrap();
    fs::write("output/object.json", &data).expect("Unable to write to file");
}
