import click
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow.python.keras.preprocessing import dataset_utils

tf.random.set_seed(123)

CLASS_NAMES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]


def create_model():
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
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def create_prediction_files(model, dataset_path: str, output_path: str):
    image_paths, image_labels, _ = dataset_utils.index_directory(
        directory=dataset_path,
        labels="inferred",
        formats=('.bmp', '.gif', '.jpeg', '.jpg', '.png'),
        class_names=CLASS_NAMES,
        shuffle=False,
        seed=123,
        follow_links=False
    )

    n_classes = len(CLASS_NAMES)
    n_samples = len(image_paths)
    output_data = np.zeros(shape=(n_samples, n_classes + 2))

    print("Generating predictions...")
    with tqdm(total=n_samples) as pbar:
        for i, image_path in enumerate(image_paths):
            # Load image
            img = cv2.imread(image_path)

            # Resize image
            img = np.expand_dims(img, axis=0)
            x = tf.image.resize(img, size=(32, 32))
            y_actual = image_labels[i]
            
            # Generate predictions
            y_preds_logits = model.predict(x)

            # Convert logits to softmax
            y_preds_exp = np.exp(y_preds_logits.squeeze())
            y_proba = y_preds_exp / np.sum(y_preds_exp, axis=0) 
            y_pred = y_proba.argmax(axis=0)

            output_data[i,0] = y_actual
            output_data[i,1] = y_pred
            output_data[i,2:] = y_proba
            pbar.update(1)

    if not output_path.endswith(".npz"):
        output_path += ".npz"

    # Save predictions
    print(f"Saving predictions to {output_path}")
    np.savez_compressed(output_path, output=output_data)

    names_path = output_path + "-image-paths.txt"
    print(f"Writing image paths to {names_path}")
    with open(names_path, "w") as fp:
        for image_path in image_paths:
            fp.write(f"{image_path}\n")


def predict(model_path: str, dataset_path: str, output_path: str):
    # Load model
    model = create_model()
    model.summary()
    model.load_weights(model_path)

    # Load dataset
    dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=False,
        seed=123,
        batch_size=1,
        image_size=(32, 32),
    )
    loss, acc = model.evaluate(dataset)
    print(f"final loss {loss}, final acc {acc}")

    create_prediction_files(model=model, dataset_path=dataset_path, output_path=output_path)


@click.command(help="Generate predictions for a test set.")
@click.option(
    "-m",
    "--model-path",
    type=click.STRING,
    required=True,
    help="Path to a stored model"
)
@click.option(
    "-d",
    "--dataset-path",
    type=click.STRING,
    required=True,
    help="Path to the dataset to score the model on."
)
@click.option(
    "-o",
    "--output-path",
    type=click.STRING,
    required=True,
    help=""
)
def main(model_path: str, dataset_path: str, output_path: str):
    predict(
        model_path=model_path,
        dataset_path=dataset_path,
        output_path=output_path,
    )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
