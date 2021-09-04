import os, shutil

from pathlib import Path
from pprint import pprint

import click
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

import utils
from utils import CLASS_NAMES, MetaData, MetaDataItem


def generate_splits(data_dir: Path, destination_dir: Path):
    print("Generating splits")
    for class_name in CLASS_NAMES:
        print(f"- Processing {class_name}")
        image_paths = list((data_dir / class_name).glob("*.png"))
        train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=492)
        set_paths = {"train": train_paths, "val": val_paths}
        for set_name, paths in set_paths.items():
            for src_path in paths:
                dest_path = destination_dir / set_name / src_path.parent.name / src_path.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src=src_path, dst=dest_path)


def copy_cleaned(source_experiment_dir: str, target_experiment_dir: str):
    source_dir = Path(source_experiment_dir)
    target_dir = Path(target_experiment_dir)

    if target_dir.exists():
        raise Exception(f"Target directory {target_dir} exists. Please remove it.")

    print("Copying files...")
    for class_name in CLASS_NAMES:
        print(f"Processing {class_name}...")
        for set_name in ["train", "val"]:
            data_dir = os.path.join(source_dir, 'data', set_name, class_name)
            meta_file_path = os.path.join(data_dir, "meta-data.json")
            meta_data = MetaData.from_csv(meta_file_path)
            items = meta_data.get_all()
            for item in items:
                target_path = target_dir / "data-raw" / item.new_label / item.full_path.name
                target_path.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy(item.full_path, target_path)
    
    generate_splits(
        data_dir=target_dir / "data-raw", 
        destination_dir=target_dir / "data"
    )
    



@click.command(help="Augment dataset.")
@click.option(
    "-s",
    "--source-experiment-dir",
    type=click.STRING,
    required=True,
    help="Path to source experiment directory."
)
@click.option(
    "-t",
    "--target-experiment-dir",
    type=click.STRING,
    required=True,
    help="Path to target experiment directory."
)
def main(source_experiment_dir: str, target_experiment_dir: str):
    copy_cleaned(
        source_experiment_dir=source_experiment_dir,
        target_experiment_dir=target_experiment_dir,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
