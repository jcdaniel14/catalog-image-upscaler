import resizer
import upscaler
import argparse
import os
import glob
import cv2
import logging
import time
import shutil

BATCH_SIZE = 5000
MAX_SIDE = 1024

logging.basicConfig(level='INFO', format='%(asctime)s [%(name)16s] %(levelname)-8s %(message)s')
logger = logging.getLogger("image-optimizer")


def collect(args):
    images = []
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    logger.info(f"To be processed {len(paths)}")
    for imgpath in paths:
        imgname, extension = os.path.splitext(os.path.basename(imgpath))
        fullname = f"{imgname}{extension}"
        img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"image {fullname} has no shape, gif?")
            continue
        if img.shape[0] > MAX_SIDE:
            shutil.copy(imgpath, os.path.join(args.done, fullname))
            shutil.move(imgpath, os.path.join(args.upscaled, fullname))
            continue
        images.append((img, imgname, extension))
        if len(images) % BATCH_SIZE == 0:
            break
    return images


def main(args):
    for folder in (args.upscaled, args.done, args.output, args.thumbnails):
        os.makedirs(folder, exist_ok=True)
    if args.resize:
        resizer.resize(args.upscaled, args.output, args.thumbnails)
        return
    upscaler.upscale(collect(args), args.input, args.upscaled, args.done, args)


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input', help='Folder (or single file) with the original images')
    parser.add_argument('--upscaled', type=str, default='upscaled', help='Folder for the x4 upscaled images')
    parser.add_argument('--done', type=str, default='done', help='Folder where processed originals are moved')
    parser.add_argument('--output', type=str, default='output', help='Folder for the final 1024x1024 images')
    parser.add_argument('--thumbnails', type=str, default='thumbnails', help='Folder for the 250x250 thumbnails')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--ext', type=str, default='webp', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument("--resize", action="store_true", help="Runs only resizing.")

    main(parser.parse_args())
    logger.info(f"Finished in {round(time.time() - start, 3)} seconds")
