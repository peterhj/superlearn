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
        //println!("DEBUG: JpegDecoderData: decoded jpeg");
        (pixels, head.width, head.height)
      }
      Err(_) => {
        match load_from_memory(&*buf) {
          LoadResult::ImageU8(mut im) => {
            if im.depth != 3 && im.depth != 1 {
              panic!("jpeg decoder: stb_image: unsupported depth: {}", im.depth);
            }
            assert_eq!(im.depth * im.width * im.height, im.data.len());

            if im.depth == 1 {
              let mut rgb_data = Vec::with_capacity(3 * im.width * im.height);
              assert_eq!(im.width * im.height, im.data.len());
              for i in 0 .. im.data.len() {
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
                rgb_data.push(im.data[i]);
              }
              assert_eq!(3 * im.width * im.height, rgb_data.len());
              im = Image::new(im.width, im.height, 3, rgb_data);
            }
            assert_eq!(3, im.depth);

            (im.data, im.width, im.height)
          }
          LoadResult::Error(_) |
          LoadResult::ImageF32(_) => {
            println!("jpeg decoder: stb_image: backup decoder failed");
            return Err(());
          }
        }
      }
    };
    Ok(Array3d::from_storage((3, width, height), SharedMem::new(pixels)))
  }
}
