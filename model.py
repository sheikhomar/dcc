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
    def __init__(self, model_path: str, image_paths: List[str], image_labels: np.ndarray, class_names: List[str], epoch: int = 100, batch_size: int=8, random_seed: Optional[int]=123) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._random_seed = random_seed
        self._epoch = epoch
        self._model = self._create_model()
        self._model_path = model_path
        self._image_paths = image_paths
        self._image_labels = image_labels
        self._class_names = class_names
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

    def load_weights(self):
        print(f"Loading weights from {self._model_path}")
        self._model.load_weights(str(self._model_path))

    def fit(self, train_idx, train_labels=None):
        train_set = self._get_dataset(shuffle=True, selected_indices=train_idx)
        self._model.fit(train_set, epochs=self._epoch)

    def predict(self, idx=None):
        """Get predicted labels from trained model."""
        probs = self.predict_proba(idx)
        return probs.argmax(axis=1)

    def predict_proba(self, idx=None):
        img_paths, img_labels, classes = self._image_paths, self._image_labels, self._class_names
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

    def _get_dataset(self, shuffle: bool, selected_indices: Optional[np.ndarray]=None):
        img_paths, img_labels, classes = self._image_paths, self._image_labels, self._class_names

        if selected_indices is not None:
            img_paths = np.array(img_paths)[selected_indices].tolist()
            img_labels = np.array(img_labels)[selected_indices].tolist()
            print(f"Filtered dataset size: {len(img_paths)}")
        
        if shuffle:
            shuffle_indices = np.arange(len(img_paths))
            np.random.shuffle(shuffle_indices)
            img_paths = np.array(img_paths)[shuffle_indices].tolist()
            img_labels = np.array(img_labels)[shuffle_indices].tolist()

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
