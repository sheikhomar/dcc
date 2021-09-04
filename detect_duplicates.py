import os, shutil

from pathlib import Path
from pprint import pprint

import click
import cv2
import numpy as np
import tensorflow as tf
import imagehash
from PIL import Image

import utils
from tqdm import tqdm

def detect_duplicates(experiment_dir: str):
    source_dir = Path(experiment_dir)
    data_dir = source_dir / "data" 
    image_paths = list((data_dir).glob("**/*.png"))
    print(f"Num: {len(image_paths)}")

    hashes = dict()

    hash_funcs = [
        ('ahash', imagehash.average_hash),
        ('phash', imagehash.phash),
        ('dhash', imagehash.dhash),
        ('whash-haar', imagehash.whash),
        ('whash-db4', lambda img: imagehash.whash(img, mode='db4')),
        ('cropresistant', imagehash.crop_resistant_hash),
    ]

    for image_path in tqdm(image_paths):
        with Image.open(str(image_path)) as img:
            computed_hash = imagehash.average_hash(img, 32)
            if hash in hashes:
                hashes[computed_hash].append(image_path)
                print(f"Duplicate found")
            else:
                hashes[computed_hash] = [image_path]
   



@click.command(help="Detect duplicates.")
@click.option(
    "-e",
    "--experiment-dir",
    type=click.STRING,
    required=True,
    help="Path to experiment directory."
)
def main(experiment_dir: str):
    detect_duplicates(
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
