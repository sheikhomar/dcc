import os, sys

from typing import List, Tuple
from pathlib import Path

import click
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image_dataset import load_image
from tqdm import tqdm

from predict import create_prediction_files

tf.random.set_seed(123)


def get_feature_extractor(model_path: str):
    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, x)
    model.load_weights(model_path)
    return model


def get_feature_map(model: tf.keras.Model, data_dir: str) -> Tuple[List[Path], np.ndarray]:
    image_paths = list(sorted(Path(data_dir).glob("**/*.png")))
    n_samples = len(image_paths)
    feature_map = np.zeros((n_samples, 256))
    with tqdm(total=n_samples) as pbar:
        for i, image_path in enumerate(image_paths):
            img = load_image(str(image_path), image_size=(32,32), num_channels=3, interpolation="bilinear")
            img = tf.expand_dims(img, axis=0) # TensorShape: [1, 32, 32, 3]
            img_features = model(img) # TensorShape: [1, 256]
            img_features = img_features.numpy().squeeze() # Shape: (256,)
            feature_map[i, :] = img_features
            pbar.update(1)
    return image_paths, feature_map


def extract_features(experiment_dir: str):
    model_path = os.path.join(experiment_dir, "checkpoints", "best_model")
    model = get_feature_extractor(model_path)


@click.command(help="Extract features.")
@click.option(
    "-e",
    "--experiment-dir",
    type=click.STRING,
    required=True,
    help="Path to the experiment."
)
def main(experiment_dir: str):
    extract_features(
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
