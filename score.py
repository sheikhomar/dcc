import tensorflow as tf

import click


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


def score_model(model_path: str, dataset_path: str):
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
        shuffle=True,
        seed=123,
        batch_size=8,
        image_size=(32, 32),
    )
    loss, acc = model.evaluate(dataset)
    print(f"final loss {loss}, final acc {acc}")


@click.command(help="Score model.")
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
    help="Path to the dataset to score the model on "
)
def main(model_path: str, dataset_path: str):
    score_model(
        model_path=model_path,
        dataset_path=dataset_path,
    )

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
