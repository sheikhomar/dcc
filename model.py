from pathlib import Path
from typing import List, Optional
from pprint import pprint

import numpy as np
from numpy import random
import tensorflow as tf
from sklearn.base import BaseEstimator

import cv2

from tqdm import tqdm
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.preprocessing.image_dataset import paths_and_labels_to_dataset

import utils


class DataCentricClassifier(BaseEstimator):
    def __init__(self, experiment_dir: str, epoch: int = 100, batch_size: int=8, random_seed: Optional[int]=123) -> None:
        super().__init__()
        self._experiment_dir = Path(experiment_dir)
        self._batch_size = batch_size
        self._random_seed = random_seed
        self._epoch = epoch
        self._model = self._create_model()
        self._dataset_paths = self._load_dataset_paths(["train", "val", "test"])
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)


    def load_weights(self):
        best_model_path = self._experiment_dir / 'checkpoints' / 'best_model'
        print(f"Loading weights from {best_model_path}")
        self._model.load_weights(str(best_model_path))

    def fit(self, train_idx, train_labels=None, sample_weight=None, loader='train'):
        train_set = self._get_dataset(dir_name=loader, shuffle=True, selected_indices=train_idx)
        # valid_set = self._get_dataset(dir_name="val", shuffle=True)
        self._model.fit(
            train_set,
            # validation_data=valid_set,
            epochs=self._epoch,
        )

    def predict(self, idx=None, loader=None):
        """Get predicted labels from trained model."""
        probs = self.predict_proba(idx, loader)
        return probs.argmax(axis=1)

    def predict_proba(self, idx=None, loader=None):
        if loader is None:
            loader = "test"

        print(f"Generating predictions on {loader}")

        img_paths, img_labels, classes = self._dataset_paths[loader]
        if idx is not None:
            img_paths = np.array(img_paths)[idx].tolist()
            img_labels = np.array(img_labels)[idx].tolist()
            print(f"Filtered dataset size: {len(img_paths)}")

        n_classes = len(classes)
        n_samples = len(img_paths)
        preds = np.zeros(shape=(n_samples, n_classes))
        with tqdm(total=n_samples) as pbar:
            for i, image_path in enumerate(img_paths):
                # Load image
                img = cv2.imread(image_path)

                # Resize image
                img = np.expand_dims(img, axis=0)
                x = tf.image.resize(img, size=(32, 32))

                # Generate predictions
                y_preds_logits = self._model.predict(x)

                # Convert logits to softmax
                y_preds_exp = np.exp(y_preds_logits.squeeze())
                y_proba = y_preds_exp / np.sum(y_preds_exp, axis=0) 

                preds[i] = y_proba
                pbar.update(1)
        return preds
    
    def get_dataset_paths_and_labels(self, dir_name: str):
        img_paths, img_labels, classes = self._dataset_paths[dir_name]
        return img_paths, img_labels

    def _load_dataset_paths(self, dir_names: List[str]):
        paths_map = dict()
        for dir_name in dir_names:
            dataset_dir = self._experiment_dir / "data" / dir_name
            if not dataset_dir.exists():
                raise Exception(f"Dataset directory {dataset_dir} does not exist.")
            print(f"Loading paths from {dataset_dir}")

            should_shuffle = dir_name in ["train"]
            img_paths, img_labels, classes = utils.get_image_paths_and_labels(dataset_dir, should_shuffle, self._random_seed)
            paths_map[dir_name] = (img_paths, img_labels, classes)
        return paths_map

    def _create_checkpoint_callback(self):
        best_model_path = Path(self._experiment_dir) / 'checkpoints' / 'best_model'
        best_model_path.parent.mkdir(exist_ok=True, parents=True)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )
        return checkpoint

    def _get_dataset(self, dir_name: str, shuffle: bool, selected_indices: Optional[np.ndarray]=None):
        img_paths, img_labels, classes = self._dataset_paths[dir_name]

        if selected_indices is not None:
            img_paths = np.array(img_paths)[selected_indices].tolist()
            img_labels = np.array(img_labels)[selected_indices].tolist()
            print(f"Filtered dataset size: {len(img_paths)}")
            print("Before shuffling: ")
            pprint(img_paths[:5])
            pprint(img_labels[:5])
        
        if shuffle:
            shuffle_indices = np.arange(len(img_paths))
            np.random.shuffle(shuffle_indices)
            img_paths = np.array(img_paths)[shuffle_indices].tolist()
            img_labels = np.array(img_labels)[shuffle_indices].tolist()

            print("After shuffling: ")
            pprint(img_paths[:5])
            pprint(img_labels[:5])


        dataset = paths_and_labels_to_dataset(
            image_paths=img_paths,
            image_size=(32,32),
            num_channels=3,
            labels=img_labels,
            label_mode="categorical",
            num_classes=len(classes),
            interpolation="bilinear",
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self._batch_size * 8, seed=self._random_seed)
        dataset = dataset.prefetch(tf.data.AUTOTUNE).batch(self._batch_size)
        dataset.class_names = classes
        dataset.file_paths = img_paths
        return dataset

    def _create_model(self):
        print("Creating ResNet50 model...")
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
