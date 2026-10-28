# catalog-image-upscaler

Batch upscaler for e-commerce catalog images. It uses Real-ESRGAN to upscale old, small product photos by 4x, then it resizes them to one standard size.

## Why

A flower e-commerce store had about 16,700 catalog images at 200x200 or 300x300 pixels. The original files were lost, and many of the remaining files were already cropped. The store sells through its product photos, so it needed high resolution images.

The catalog photos have two useful constraints: they are square, and the flowers are in the center. A general purpose GAN upscaler works well on this kind of image. The result looked natural, not generated.

## How it works

1. `main.py` reads the input folder and skips images that are already larger than 1024 px.
2. `upscaler.py` runs Real-ESRGAN x4plus (RRDBNet) on batches of 5,000 images.
3. `resizer.py` writes each image as a 1024x1024 WebP file (quality 90) and a 250x250 thumbnail.

## Usage

```bash
pip install -r requirements.txt
mkdir models
curl -L -o models/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
python main.py --input input --upscaled upscaled
python main.py --resize --upscaled upscaled --output output --thumbnails thumbnails
```

Use `--tile 256` (or smaller) if the GPU runs out of memory.

## Credits

The upscaling model is [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al., under the BSD 3-Clause license. This repository uses it as a dependency and does not include its code or weights.
