use std::path::PathBuf;
use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::{ImageBuffer, Luma, Rgb};
use super::Rectangle;

/// These are images without a set form. They can be converted to either
/// greyscale or color images. They can also be resized
pub struct DynamicImage(image::DynamicImage);
impl DynamicImage {
    pub fn resize(&self, w: u32, h: u32, f: FilterType) -> Self {
        Self(self.0.resize_to_fill(w, h, f))
    }
} impl From<PathBuf> for DynamicImage {
    fn from(path: PathBuf) -> Self {
        let img = ImageReader::open(path).unwrap().decode().unwrap();
        DynamicImage(img)
    }
}

/// A wrapper over a buffer representing a color image
pub type ColorImage = ImageBuffer<Rgb<u8>, Vec<u8>>;
impl From<DynamicImage> for ColorImage {
    fn from(img: DynamicImage) -> Self { img.0.to_rgb8() }
}

/// A wrapper over a buffer representing a greyscale image
pub type GreyscaleImage = ImageBuffer<Luma<u8>, Vec<u8>>;
impl From<DynamicImage> for GreyscaleImage {
    fn from(img: DynamicImage) -> Self { img.0.to_luma8() }
}

/// Draws a rectangle over an image
pub fn draw_rectangle(img: &mut ColorImage, r: &Rectangle<u32>) {
    let pixel: Rgb<u8> = Rgb::from([0x88, 0x95, 0x8D]);
    for x in r.top_left[0]..r.bot_right[0] {
        img.put_pixel(x, r.top_left[1], pixel);
        img.put_pixel(x, r.bot_right[1], pixel);
    }
    for y in r.top_left[1]..r.bot_right[1] {
        img.put_pixel(r.top_left[0], y, pixel);
        img.put_pixel(r.bot_right[0], y, pixel);
    }
}

