use io::codecs::*;

use densearray::prelude::*;
use sharedmem::*;
use stb_image::image::{Image, LoadResult, load_from_memory};
use turbojpeg::{TurbojpegDecoder};

pub struct JpegDecoder {
  turbo:    TurbojpegDecoder,
}

impl Default for JpegDecoder {
  fn default() -> Self {
    JpegDecoder{turbo: TurbojpegDecoder::create().unwrap()}
  }
}

impl Decoder for JpegDecoder {
  type Src = SharedMem<u8>;
  type Dst = Result<Array3d<u8, SharedMem<u8>>, ()>;

  fn decode(&mut self, buf: SharedMem<u8>) -> Result<Array3d<u8, SharedMem<u8>>, ()> {
    let (pixels, width, height) = match self.turbo.decode_rgb8(&*buf) {
      Ok((head, pixels)) => {
        (pixels, head.width, head.height)
      }
      Err(_) => {
        match load_from_memory(&*buf) {
          LoadResult::ImageU8(mut im) => {
            match im.depth {
              3 => {}
              1 => {
                // Gray.
                let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
                assert_eq!(im.width * im.height, im.data.len());
                for i in 0 .. im.width * im.height {
                  rgb_data.push(im.data[i]);
                  rgb_data.push(im.data[i]);
                  rgb_data.push(im.data[i]);
                }
                assert_eq!(3 * im.width * im.height, rgb_data.len());
                im = Image::new(im.width, im.height, 3, rgb_data);
              }
              2 => {
                // Gray/alpha.
                let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
                assert_eq!(2 * im.width * im.height, im.data.len());
                for i in 0 .. im.width * im.height {
                  rgb_data.push(im.data[2 * i]);
                  rgb_data.push(im.data[2 * i]);
                  rgb_data.push(im.data[2 * i]);
                }
                assert_eq!(3 * im.width * im.height, rgb_data.len());
                im = Image::new(im.width, im.height, 3, rgb_data);
              }
              4 => {
                // RGB/alpha.
                let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
                assert_eq!(4 * im.width * im.height, im.data.len());
                for i in 0 .. im.width * im.height {
                  rgb_data.push(im.data[    4 * i]);
                  rgb_data.push(im.data[1 + 4 * i]);
                  rgb_data.push(im.data[2 + 4 * i]);
                }
                assert_eq!(3 * im.width * im.height, rgb_data.len());
                im = Image::new(im.width, im.height, 3, rgb_data);
              }
              _ => {
                println!("jpeg decoder: stb_image: unsupported depth: {}", im.depth);
                return Err(());
              }
            }
            assert_eq!(im.depth * im.width * im.height, im.data.len());
            assert_eq!(3, im.depth);
            (im.data, im.width, im.height)
          }
          LoadResult::ImageF32(_) => {
            println!("jpeg decoder: stb_image: f32 image unsupported");
            return Err(());
          }
          LoadResult::Error(_) => {
            println!("jpeg decoder: stb_image: backup decoder failed");
            return Err(());
          }
        }
      }
    };
    Ok(Array3d::from_storage((3, width, height), SharedMem::new(pixels)))
  }
}
