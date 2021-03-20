use std::convert::From;

#[derive(Debug)]
pub struct IntegralImage {
    pub pixels: Vec<usize>,
    pub size: Size,
} impl IntegralImage {
    fn rectangle_sum(&self, x1: u16, y1: u16,
    x2: u16, y2: u16) -> usize {
        let f = |y: u16, x: u16| -> usize {
            self.pixels[(y as usize) * (self.size.width as usize) + (x as usize)]
        };
        let w = self.size.width;
        self.pixels[ f(x2, y2) - f(x2, y1) - f(x1, y2) + f(x1, y1) ]
    }
} impl From<GreyscaleImage> for IntegralImage {
    fn from(img: GreyscaleImage) -> Self {
        let mut pixels = Vec::<usize>::new();
        let size = img.size;
        for (i, pixel) in img.pixels.iter().enumerate() {
            pixels[i] = {
                if i == 0 { img.pixels[i] as usize}
                else if i % (size.width as usize) == 0 { 
                    pixels[i - size.width as usize] + (img.pixels[i] as usize) 
                }
                else { pixels[i - 1] + (img.pixels[i] as usize) }
            }
        }
        IntegralImage {
            pixels, 
            size,
        }
    }
}

#[derive(Debug)]
pub struct GreyscaleImage {
    pub pixels: Vec<u8>,
    pub size: Size,
} impl From<RGBImage> for GreyscaleImage {
    fn from(img: RGBImage) -> Self {
        let mut pixels = Vec::<u8>::with_capacity(img.pixels.len());
        for (i, pixel) in img.pixels.iter().enumerate() {
            pixels[i] = ((pixel.red as u16 + pixel.green as u16 
                + pixel.blue as u16)/3) as u8;
        }
        GreyscaleImage {
            pixels,
            size: img.size,
        }
    }
}

#[derive(Debug)]
pub struct RGBImage {
    pub pixels: Vec<Pixel>,
    pub size: Size,
}

#[derive(Clone, Copy, Debug)]
pub struct Pixel {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Size {
    pub width: u16,
    pub length: u16,
}
