import os, shutil

from pathlib import Path
from pprint import pprint

import click
import cv2
import numpy as np
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from predict import CLASS_NAMES, load_images

tf.random.set_seed(123)

def check_source_dirs(source_root_dir: Path):
    paths = [
        source_root_dir,
        source_root_dir / "data",
        source_root_dir / "data" / "train",
        source_root_dir / "data" / "val",
        source_root_dir / "data" / "test",
    ]
    for p in paths:
        if not p.exists():
            raise Exception("Source path does not exist")


def generate_augmented_images(source_dir: Path, target_dir: Path, class_name: str, n_target: int, quantize: bool):
    image_paths = list((source_dir / class_name).glob("*.png"))
    n_augmentations = int(n_target / len(image_paths))
    
    datagen = ImageDataGenerator(
        shear_range=0.2, 
        zoom_range=0.3,
        rotation_range=20,
    )

    print(f"Generating augmentations for {class_name} in {source_dir} (quantize? {'yes' if quantize else 'no'})")
    with tqdm(total=n_target) as pbar:
        for image_path in image_paths:
            # Build destination file path
            file_name = image_path.name
            destination_path = target_dir / class_name / file_name
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(image_path, destination_path)
            pbar.update(1)

            # Load image file
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            img = np.expand_dims(img, axis=0) # Samples  => (1, ?, ?)
            img = np.expand_dims(img, axis=3) # Channels => (1, ?, ?, 1)

            # Generate augmentations
            for i, batch in enumerate(datagen.flow(img, batch_size=1, save_prefix="aug")):
                aug_img_name = f"{destination_path.stem}-aug-{i}.png"
                aug_img_path = destination_path.parent / aug_img_name

                aug_img = np.squeeze(batch, axis=0)

                if quantize:
                    # Quantize image
                    aug_img = (aug_img // 43) * 43
                    aug_img[aug_img > 43] = 255

                # Save augmented image
                cv2.imwrite(str(aug_img_path), aug_img)

                pbar.update(1)

                if i >= n_augmentations-2:
                    break


def augment(source_experiment_dir: str, target_experiment_dir: str, train_fraction: float, quantize: bool):
    source_dir = Path(source_experiment_dir)
    target_dir = Path(target_experiment_dir)

    train_set_size = 10000 * train_fraction
    val_set_size = 10000 - train_set_size

    check_source_dirs(source_dir)
    
    for class_name in CLASS_NAMES:
        generate_augmented_images(
            source_dir=source_dir / "data" / "train",
            target_dir=target_dir / "data" / "train",
            class_name=class_name,
            n_target=int(train_set_size / len(CLASS_NAMES)),
            quantize=quantize,
        )
        generate_augmented_images(
            source_dir=source_dir / "data" / "val",
            target_dir=target_dir / "data" / "val",
            class_name=class_name,
            n_target=int(val_set_size / len(CLASS_NAMES)),
            quantize=quantize,
        )
        generate_augmented_images(
            source_dir=source_dir / "data" / "test",
            target_dir=target_dir / "data" / "test",
            class_name=class_name,
            n_target=int(50),
            quantize=quantize,
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
@click.option(
    "-n",
    "--train-fraction",
    type=click.FLOAT,
    default=0.9,
    show_default=True,
    help="The size of the training set."
)
@click.option(
    "-q",
    "--quantize/--no-quantize",
    required=True,
    help="Whether to quantize augmented images."
)
def main(source_experiment_dir: str, target_experiment_dir: str, train_fraction: float, quantize: bool):
    augment(
        source_experiment_dir=source_experiment_dir,
        target_experiment_dir=target_experiment_dir,
        train_fraction=train_fraction,
        quantize=quantize,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
