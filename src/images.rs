use image::{GrayImage, DynamicImage, ImageBuffer, GenericImageView, Luma};
use image::imageops::FilterType;



pub struct IntegralImage {
    pixels: Vec<usize>,
} impl IntegralImage {
    pub fn new(img: DynamicImage) -> IntegralImage {
        // Resize the image and turn it to grayscale
        let mut img = img.resize(24, 24, FilterType::Triangle).into_luma8();

        // Calculate each pixel of the integral image
        let mut pixels = Vec::<usize>::with_capacity(24*24);
        for y in 0..24 {
            for x in 0..24 {
                let pixel = img.get_pixel_mut(x, y)[0] as usize;
                let x = x as usize; let y = y as usize;
                pixels.push({
                    if (x == 0) && (y == 0) { pixel }
                    else if x == 0 { pixels[x + 24*(y-1)] as usize + pixel }
                    else if y == 0 { pixels[(x-1) + 24*y] as usize + pixel } 
                    else { pixels[(x-1) + 24*y] as usize
                            + (pixels[x + 24*(y-1)] as usize)
                            + pixel - (pixels[(x-1) + 24*(y-1)] as usize)
                    }
                })
            }
        }
        IntegralImage {
            pixels,
        }

    }
    pub fn rectangle_sum(&self, x1: u8, y1: u8, x2: u8, y2: u8) -> usize {
        (self.pixels[(x2 as usize) + 24*(y2 as usize)] 
            - self.pixels[(x2 as usize) + 24*(y1 as usize)] 
            - self.pixels[(x1 as usize) + 24*(y2 as usize)] 
            + self.pixels[(x1 as usize) + 24*(y1 as usize)]) as usize
    }
}
