import sys, os, re, base64

sys.path.append(os.path.abspath("../"))

from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import umap

from tensorflow.python.keras.preprocessing.image_dataset import load_image
from tqdm import tqdm
import streamlit as st
from matplotlib import pyplot as plt
from IPython.display import display, HTML
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import utils

CLASS_NAMES = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]


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


class StreamlitApp:
    def __init__(self) -> None:
        self._experiment_dir = "experiments"
        self._outlier_detectors = {
            "IsolationForest": ("Isolation Forest", IsolationForest),
            "EllipticEnvelope": ("Elliptic Envelope", EllipticEnvelope),
            "OneClassSVM": ("SVM", OneClassSVM),
        }

    def run(self):
        
        self._build_source_experiment_dropbox()
        self._build_outlier_detector_dropbox()
        self._build_models_dropbox()
        self._build_dataset_dropbox()
        self._build_class_name_dropbox()
        self._build_find_outliers_button()
        self._build_main_container()

    def _build_main_container(self) -> None:
        st.write("# Bad Label Detector")
        if "outlier_labels" not in st.session_state:
            st.write("Please run outlier detection")
        else:
            st.write(f"Found {self.n_outliers} outlier images")
            
            self._render_image_selectors()
            self._render_selected_image()

    @property
    def n_outliers(self) -> int:
        return int((self.outlier_labels == -1).sum())
    
    @property
    def outlier_labels(self) -> np.ndarray:
        return st.session_state["outlier_labels"]

    @property
    def class_name(self) -> np.ndarray:
        return st.session_state["class_name"]

    @property
    def current_image_index(self) -> int:
        return st.session_state["current_image_index"]

    @property
    def current_image_path(self) -> Path:
        index = self.current_image_index - 1
        image_path = self.bad_image_paths[index]
        return image_path

    @current_image_index.setter
    def current_image_index(self, new_val: int) -> None:
        st.session_state["current_image_index"] = new_val

    @property
    def bad_image_paths(self) -> np.ndarray:
        return st.session_state["bad_image_paths"]

    @bad_image_paths.setter
    def bad_image_paths(self, new_val: np.ndarray) -> None:
        st.session_state["bad_image_paths"] = new_val

    def _go_next_image(self) -> None:
        new_val = self.current_image_index + 1
        max_val = self.n_outliers
        if new_val >= max_val:
            new_val = max_val
        self.current_image_index = new_val

    def _go_prev_image(self) -> None:
        new_val = self.current_image_index - 1
        if new_val < 1:
            new_val = 1
        self.current_image_index = new_val

    def _render_image_selectors(self) -> None:
        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
        col1.button("Prev", on_click=lambda: self._go_prev_image())
        col2.slider('Current Image Index', min_value=1, max_value=self.n_outliers, key="current_image_index")
        col3.button("Next", on_click=lambda: self._go_next_image())

    def _render_selected_image(self) -> None:
        col1, col2 = st.columns([0.7, 0.3])
        col1.header("Image")
        
        image_path = self.current_image_path
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        col1.image(img, use_column_width=False)

        col1.text(image_path.name)

        col2.header("Label")
        radio_options = ["invalid", "unsure"] + CLASS_NAMES
        st.session_state["radio_val_selected_image_label"] = self._get_current_image_label()
        # print(self._get_current_image_label())
        col2.radio(
            label='',
            options=radio_options,
            key="radio_val_selected_image_label",
            on_change=lambda: self._on_label_change(),
        )

    def _on_label_change(self) -> None:
        new_label = st.session_state["radio_val_selected_image_label"]
        self._set_current_image_label(new_label)

    def _get_current_image_label(self) -> str:
        if "new_labels" not in st.session_state:
            return self.class_name
        key = self.current_image_path
        if key in st.session_state["new_labels"]:
            return st.session_state["new_labels"][key]
        return self.class_name
    
    def _set_current_image_label(self, new_label: str) -> None:
        old_label = self.class_name
        if "new_labels" not in st.session_state:
            new_labels_dict = dict()
        else:
            new_labels_dict = st.session_state["new_labels"]
        
        key = self.current_image_path
        if old_label == new_label and key in new_labels_dict:
            print(f"Removing item for {key}")
            new_labels_dict.pop(key)
        else:
            new_labels_dict[key] = new_label
        st.session_state["new_labels"] = new_labels_dict


    def _build_source_experiment_dropbox(self) -> None:
        source_experiments = os.listdir(self._experiment_dir)
        vals = [exp for exp in source_experiments]

        st.sidebar.selectbox(
            label='Source experiment',
            options=vals,
            index=vals.index("original"),
            key="source-experiment"
        )

    def _build_outlier_detector_dropbox(self) -> None:
        st.sidebar.selectbox(
            label='Outlier detector',
            options=list(self._outlier_detectors.keys()),
            index=0,
            key="outlier_detector"
        )

    def _build_models_dropbox(self) -> None:
        model_names = {
            "best_model": "RomanNumerals-trained model",
            "imagenet-finetuned": "Fine-tuned model",
            "imagenet": "ImageNet-trained model",
        }

        st.sidebar.selectbox(
            label='Source experiment',
            options=list(model_names.keys()),
            index=1,
            key="model_name"
        )

    def _build_dataset_dropbox(self) -> None:
        st.sidebar.selectbox(
            label='Dataset',
            options=["train", "val", "test"],
            index=0,
            key="dataset"
        )

    def _build_class_name_dropbox(self) -> None:
        st.sidebar.selectbox(
            label='Class name',
            options=CLASS_NAMES,
            index=0,
            key="class_name"
        )

    def _build_find_outliers_button(self) -> None:
        st.sidebar.button("Find outliers", on_click=lambda: self._on_find_outliers_button_click())

    def _on_find_outliers_button_click(self) -> None:
        print("Button clicked!")

        source_experiment = st.session_state["source-experiment"]
        model_name = st.session_state["model_name"]
        class_name = st.session_state["class_name"]
        outlier_detector = st.session_state["outlier_detector"]

        if model_name == "imagenet":
            model_path = "imagenet"
        else: 
            model_path = os.path.join(self._experiment_dir, source_experiment, "checkpoints", model_name)

        print(f"Model path: {model_path}")

        print(f"Loading model: {model_path}")
        model = get_feature_extractor(model_path)

        data_dir = os.path.join(self._experiment_dir, source_experiment, "data", "train", class_name)
        image_paths = list(sorted(Path(data_dir).glob("**/*.png")))

        st.session_state["image_paths"] = image_paths

        print("Computing feature map")
        feature_map = get_feature_map(model, image_paths)

        print("Computing PCA features")
        pca = PCA(n_components=0.9, svd_solver="full", whiten=True)
        feature_map_pca = pca.fit_transform(feature_map)

        od_algo = self._outlier_detectors[outlier_detector][1]()
        outlier_labels = od_algo.fit_predict(feature_map_pca)

        st.session_state["outlier_labels"] = outlier_labels

        self.bad_image_paths = np.array(image_paths)[(outlier_labels == -1)]
        self.current_image_index = 1



StreamlitApp().run()
