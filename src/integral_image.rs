use std::convert::From;

#![derive(Debug)]
struct IntegralImage {
    pub const pixels: vec<usize>,
    pub const size: Size,
} impl IntegralImage {
    fn rectangle_sum(&self, start_x: usize, start_y: usize,
    end_x: usize, end_y: usize ) -> usize {
        self.pixels[end_y * self.size.width + end_x]
            - self.pixels[start_y * self.size.width + end_x]
            - self.pixels[end_y * self.size.width + end_x]
            + self.pixels[start_y * self.size.width + start_x]
    }
} impl From<GreyscaleImage> for IntegralImage {
    fn from(img: GreyscaleImage) -> Self {
        let mut pixels = Vec::<usize>::new();
        let size = self.size;
        for (i, pixel) in self.pixels.enumerate() {
            pixels[i] = {
                if i == 0 { self.pixels[i] }
                else if i % size.width == 0 { pixels[i-size.width] + self.pixels[i] }
                else { pixels[i-1] + self.pixels[i] }
            }
        }
        IntegralImage {
            pixels, 
            size,
        }
    }
}

#![derive(Debug)]
struct GreyscaleImage {
    pub const pixels: vec<usize>,
    pub const size: Size,
} impl From<RGBImage> for GreyscaleImage {
    fn from(img: RGBImage) -> Self {
        let mut pixels = vec<usize>;
        for (i, pixel) in img.pixels.enumerate() {
            pixels[i] = (pixel.red + pixel.green + pixel.blue) / 3;
        }
        GreyScaleImage {
            pixels,
            size: img.size,
        }
    }
}

#![derive(Debug)]
struct RGBImage {
    pub const pixels: vec<Pixel>,
    pub const size: Size,
}

#![derive(Clone, Copy, Debug)]
struct Pixel {
    pub const u8: red,
    pub const u8: green
    pub const u8: blue,
}

#![derive(Clone, Copy, Debug)]
struct Size {
    length: u16,
    height: u16,
}
