import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import sys
import os
from pathlib import Path

from predict import create_prediction_files

import click

batch_size = 8
tf.random.set_seed(123)


def train(experiment_dir: str):
    train_set = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(experiment_dir, 'data', 'train'),
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    valid = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(experiment_dir, 'data', 'val'),
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    total_length = ((train_set.cardinality() + valid.cardinality()) * batch_size).numpy()
    if total_length > 10_000:
        print(f"Dataset size larger than 10,000. Got {total_length} examples")
        sys.exit()

    test_set_path = os.path.join(experiment_dir, 'data', 'test')
    test = tf.keras.preprocessing.image_dataset_from_directory(
        test_set_path,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    model.compile(
        # optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        optimizer=tf.keras.optimizers.SGD(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    loss_0, acc_0 = model.evaluate(valid)
    print(f"loss {loss_0}, acc {acc_0}")

    best_model_path = Path(experiment_dir) / 'checkpoints' / 'imagenet-finetuned'
    best_model_path.parent.mkdir(exist_ok=True, parents=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_set,
        validation_data=valid,
        epochs=100,
        callbacks=[checkpoint],
    )

    model.load_weights(best_model_path)

    loss, acc = model.evaluate(valid)
    print(f"final loss {loss}, final acc {acc}")

    test_loss, test_acc = model.evaluate(test)
    print(f"test loss {test_loss}, test acc {test_acc}")

    predictions_path = Path(experiment_dir) / 'predictions' / 'predictions-finetuned'
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    create_prediction_files(
        model=model,
        dataset_path=test_set_path,
        output_path=str(predictions_path),
    )


@click.command(help="Fine-tune a model pre-trained on ImageNet.")
@click.option(
    "-e",
    "--experiment-dir",
    type=click.STRING,
    required=True,
    help="Path to the experiment."
)
def main(experiment_dir: str):
    train(
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
