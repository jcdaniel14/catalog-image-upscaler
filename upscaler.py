from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os
import cv2
import logging
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{ROOT_DIR}/models/RealESRGAN_x4plus.pth"
UPSCALE_FACTOR = 4
logger = logging.getLogger(__name__)


def upscale(images, input_folder, output_folder, dump_folder, args):
    # Prepare output folders
    os.makedirs(output_folder, exist_ok=True)

    # Prepare Model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=UPSCALE_FACTOR)
    upsampler = RealESRGANer(scale=UPSCALE_FACTOR, model_path=MODEL_PATH, dni_weight=None, model=model, tile=args.tile, tile_pad=args.tile_pad,
                             pre_pad=args.pre_pad, half=False)

    count = 0
    for img, imgname, extension in images:
        if img is None:
            continue
        fullname = f"{imgname}{extension}"
        # Determine image extension
        extension = args.ext
        if args.ext == 'auto':
            extension = extension[1:]
        if len(img.shape) == 3 and img.shape[2] == 4:
            extension = 'png'  # RGBA images should be saved in png format

        # Inference & saving
        try:
            output, _ = upsampler.enhance(img, outscale=UPSCALE_FACTOR)
            save_path = os.path.join(output_folder, f"{imgname}.{extension}")
            cv2.imwrite(save_path, output)

            count += 1
            shutil.move(f"{input_folder}/{fullname}", f"{dump_folder}/{fullname}")
            if count % 10 == 0:
                logger.info(f"Upscaled {imgname}.{extension} ({count}/{len(images)})")
        except RuntimeError as error:
            logger.error('Error', error)
            logger.error('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
