import os, shutil, math

from pathlib import Path
from pprint import pprint

import click
import cv2
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa



from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

from predict import CLASS_NAMES, load_images

tf.random.set_seed(123)

def draw_random_lines(img):
    pxstep = int(img.shape[0] / 3)
    thickness = int(np.random.randint(low=1, high=3))
    x = pxstep
    y = pxstep
    line_colour = int(np.min(img) + np.std(img))
    while x < img.shape[1]:
        if np.random.choice([True, False]):
            start_x = x + np.random.randint(0, 10)
            end_x = x
            start_y = np.random.randint(0, 10)
            end_y = img.shape[0]
            cv2.line(img, (start_x, start_y), (end_x, end_y), color=line_colour, lineType=cv2.LINE_AA, thickness=thickness)
        x += pxstep + np.random.randint(0, 100)

    while y < img.shape[0]:
        if np.random.choice([True, False]):
            start_x = np.random.randint(0, 10)
            end_x = img.shape[1]
            start_y = y+ np.random.randint(0, 10)
            end_y = y
            cv2.line(img, (start_x, start_y), (end_x, end_y), color=line_colour, lineType=cv2.LINE_AA, thickness=thickness)
        y += pxstep + np.random.randint(0, 100)


def draw_random_dots(img):
    pxstep = 10
    radius = int(np.random.randint(low=1, high=3))
    x = pxstep
    y = pxstep
    line_colour = int(np.min(img) + np.std(img))
    while x < img.shape[1]:
        start_x = x + np.random.randint(0, int(img.shape[0]/6))
        end_x = x + np.random.randint(0, int(img.shape[0]/6))
        start_y = np.random.randint(0, int(img.shape[0]))
        end_y = np.random.randint(0, int(img.shape[0]))
        cv2.circle(img, (start_x, start_y), radius, color=line_colour, thickness=-1)
        x += pxstep + np.random.randint(0, 10)


def custom_augmentations(images, random_state, parents, hooks):
    for img in images:
        draw_random_dots(img)
        draw_random_lines(img)
    return images


def get_augmenter():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
        [
            iaa.Affine(
                rotate=(-15, 15),
                # translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                shear=(-5, 5),
                cval=(0, 255),
                mode="edge"
            ),
            sometimes(iaa.Pepper(0.001)),

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1))),

            iaa.Lambda(custom_augmentations),
        ],
        
        # do all of the above augmentations in random order
        random_order=True
    )

    return seq


def check_source_dirs(source_root_dir: Path):
    paths = [
        source_root_dir,
        source_root_dir / "data",
        source_root_dir / "data" / "train",
        source_root_dir / "data" / "val",
    ]
    for p in paths:
        if not p.exists():
            raise Exception("Source path does not exist")


def generate_augmented_images(source_dir: Path, target_dir: Path, class_name: str, n_target: int, quantize: bool):
    image_paths = list((source_dir / class_name).glob("*.png"))

    perform_augmentation = get_augmenter()
    
    print(f"Generating augmentations for {class_name} in {source_dir/class_name} (quantize? {'yes' if quantize else 'no'})")
    with tqdm(total=n_target) as pbar:
        for image_path in image_paths:
            # Build destination file path
            file_name = image_path.name
            destination_path = target_dir / class_name / file_name
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(image_path, destination_path)
            pbar.update(1)

        n_missing = n_target - len(image_paths)

        i = 0
        
        while i + len(image_paths) < n_target:
            image_path = np.random.choice(image_paths)

            # Build destination file path
            file_name = image_path.name
            destination_path = target_dir / class_name / file_name
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            # Load image file
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            # Generate augmentations
            aug_img_name = f"{destination_path.stem}-aug-{i}.png"
            aug_img_path = destination_path.parent / aug_img_name

            aug_img = perform_augmentation(image=img)

            if quantize:
                # Quantize image
                aug_img = (aug_img // 43) * 43
                aug_img[aug_img > 43] = 255

            # Save augmented image
            cv2.imwrite(str(aug_img_path), aug_img)

            pbar.update(1)
            i += 1


def augment(source_experiment_dir: str, target_experiment_dir: str, quantize: bool):
    source_dir = Path(source_experiment_dir)
    target_dir = Path(target_experiment_dir)

    val_paths = list((source_dir / "data" / "val").glob("**/*.png"))

    val_set_size = len(val_paths)
    train_set_size = 10000 - val_set_size
    check_source_dirs(source_dir)
    
    total_size = val_set_size

    for class_name in CLASS_NAMES:
        n_target = int(math.floor(train_set_size / len(CLASS_NAMES)))
        total_size += n_target
        generate_augmented_images(
            source_dir=source_dir / "data" / "train",
            target_dir=target_dir / "data" / "train",
            class_name=class_name,
            n_target=n_target,
            quantize=quantize,
        )

        shutil.copytree(
            src=source_dir / "data" / "val" / class_name,
            dst=target_dir / "data" / "val" / class_name
        )

    print(f"Total size: {total_size}")


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
    "-q",
    "--quantize/--no-quantize",
    required=True,
    help="Whether to quantize augmented images."
)
def main(source_experiment_dir: str, target_experiment_dir: str, quantize: bool):
    augment(
        source_experiment_dir=source_experiment_dir,
        target_experiment_dir=target_experiment_dir,
        quantize=quantize,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
