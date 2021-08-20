import tensorflow as tf

import click
import numpy as np
from tqdm import tqdm


tf.random.set_seed(123)


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


def generate_predictions(model_path: str, dataset_path: str, output_path: str):
    # Load model
    model = create_model()
    model.summary()
    model.load_weights(model_path)

    # Load dataset
    dataset: tf.data.Dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=1,
        image_size=(32, 32),
    )
    loss, acc = model.evaluate(dataset)
    print(f"final loss {loss}, final acc {acc}")

    n_classes = len(dataset.class_names)
    n_samples = len(dataset.file_paths)
    output_data = np.zeros(shape=(n_samples, n_classes + 2))

    print("Generating predictions...")
    with tqdm(total=n_samples) as pbar:
        for i, item in enumerate(dataset.as_numpy_iterator()):
            x, y = item
            y_actual = y.argmax(axis=1)[0]
            
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

    # Save predictions
    print(f"Saving predictions to {output_path}")
    np.savez_compressed(output_path, output=output_data)


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
    generate_predictions(
        model_path=model_path,
        dataset_path=dataset_path,
        output_path=output_path
    )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
