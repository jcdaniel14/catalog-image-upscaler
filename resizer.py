import PIL
from PIL import Image
import os
import glob
import logging
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
QUALITY = 90
W, H = 1024, 1024
WIDTH, HEIGHT = 250, 250
logger = logging.getLogger(__name__)


# def resize(INPUT_FOLDER, OUTPUT_FOLDER, DUMP_FOLDER):
def resize(INPUT_FOLDER, OUTPUT_FOLDER, THUMB_FOLDER):
    images = sorted(glob.glob(os.path.join(INPUT_FOLDER, '*')))
    count = 0
    fullname = ""
    for image in images:
        if "\\" in image:
            fullname = image.split("\\")[-1]
        elif "/" in image:
            fullname = image.split("/")[-1]
        image_name = fullname.split(".")[:-1]
        image_name = ".".join(image_name)
        image = Image.open(image)
        image = image.resize((W, H), PIL.Image.NEAREST)
        image.save(f"{OUTPUT_FOLDER}/{image_name}.webp", "webp", optimize=True, quality=QUALITY)
        image = image.resize((WIDTH, HEIGHT), PIL.Image.NEAREST)
        image.save(f"{THUMB_FOLDER}/{image_name}_thumb.webp", "webp", optimize=True, quality=QUALITY)
        # shutil.move(f"{INPUT_FOLDER}/{fullname}", f"{DUMP_FOLDER}/{fullname}")

        count += 1
        if count %100 == 0:
            logger.info(f"Resized {fullname} {count}/{len(images)}")