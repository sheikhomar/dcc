import dataclasses
import sys, os, json

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras.preprocessing.image_dataset import load_image
from tqdm import tqdm
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from streamlit_tags import st_tags, st_tags_sidebar

from tensorflow.python.keras.preprocessing import dataset_utils


CLASS_NAMES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]


def get_image_paths_and_labels(dataset_dir: str, shuffle: bool, random_seed: int):
    image_paths, image_labels, class_names = dataset_utils.index_directory(
        directory=dataset_dir,
        labels="inferred",
        formats=('.bmp', '.gif', '.jpeg', '.jpg', '.png'),
        class_names=CLASS_NAMES,
        shuffle=shuffle,
        seed=random_seed,
        follow_links=False
    )
    return image_paths, image_labels, class_names


def get_feature_extractor(model_path: str):
    weights = "imagenet" if model_path == "imagenet" else None
    print(f"Using weights: {weights}")
    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=weights,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, x)
    if model_path is not None:
        print(f"Loading weights: {model_path}")
        model.load_weights(model_path)
    return model


def get_feature_map(model: tf.keras.Model, image_paths: List[Path]) -> Tuple[List[Path], np.ndarray]:
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
    return feature_map


@dataclass
class MetaDataItem:
    full_path: Path
    given_label: str
    new_label: str
    tags: List[str]

    def set_new_label(self, new_val: str) -> None:
        self.new_label = new_val

    def to_json(self) -> Dict[str, object]:
        return {
            "full_path": str(self.full_path),
            "given_label": self.given_label,
            "new_label": self.new_label,
            "tags": self.tags
        }
    
    @classmethod
    def from_json(cls, input_dict: Dict[str, object]):
        return cls(
            full_path=Path(input_dict["full_path"]),
            given_label=input_dict["given_label"],
            new_label=input_dict["new_label"],
            tags=input_dict["tags"],
        )


class MetaData:
    def __init__(self) -> None:
        self._items : Dict[str, MetaDataItem] = dict()

    def add(self, item: MetaDataItem) -> None:
        self._items[item.full_path.name] = item

    def get(self, image_path: Path) -> MetaDataItem:
        return self._items[image_path.name]

    @classmethod
    def from_csv(cls, file_path: Path):
        if not os.path.exists(str(file_path)):
            raise Exception(f"File not found: {file_path}")
        with open(str(file_path), "r") as file:
            meta_data_json = json.load(file)

        meta_data = MetaData()
        for v in meta_data_json["images"]:
            meta_data.add(MetaDataItem.from_json(v))
        return meta_data

    def to_csv(self, file_path: Path):
        image_list = []
        for k, v in self._items.items():
            image_list.append(v.to_json())
        json_obj = {
            "images": image_list
        }
        with open(file_path, "w") as file:
            json.dump(json_obj, file, indent=2)
