import dataclasses
import sys, os, json

sys.path.append(os.path.abspath("../"))

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import umap

from tensorflow.python.keras.preprocessing.image_dataset import load_image
from tqdm import tqdm
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from streamlit_tags import st_tags, st_tags_sidebar

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


class StreamlitApp:
    def __init__(self) -> None:
        self._experiment_dir = "experiments"
        self._outlier_detectors = {
            "IsolationForest": ("Isolation Forest", IsolationForest),
            "EllipticEnvelope": ("Elliptic Envelope", EllipticEnvelope),
            "OneClassSVM": ("SVM", OneClassSVM),
        }

    def run(self):
        self._build_sidebar()
        self._build_main_container()

    def _build_sidebar(self) -> None:
        st.sidebar.markdown("## Data Settings")
        self._build_source_experiment_dropbox()
        self._build_dataset_dropbox()
        self._build_class_name_dropbox()
        self._build_show_images_button()
        
        st.sidebar.markdown("---\n## Outlier Detection Settings")
        self._build_models_dropbox()
        self._build_outlier_detector_dropbox()
        self._build_find_outliers_button()

        st.sidebar.markdown("---\n## Other")
        st.sidebar.button("Find blank images", on_click=lambda: self._find_blank_images())

    def _build_main_container(self) -> None:
        st.write("# Labeller")
        if "filtered_image_paths" not in st.session_state:
            st.write("Please select the images to label in the side bar.")
        else:
            st.write(f"Found {self.n_images} images")
            self._render_image_selectors()
            self._render_selected_image()
            self._render_tags_container()

    @property
    def n_images(self) -> int:
        return len(self.filtered_image_paths)
    
    @property
    def class_name(self) -> np.ndarray:
        return st.session_state["class_name"]

    @property
    def dataset(self) -> np.ndarray:
        return st.session_state["dataset"]

    @property
    def current_image_index(self) -> int:
        return st.session_state["current_image_index"]

    @property
    def possible_tags(self) -> List[str]:
        if "possible_tags" not in st.session_state:
            file_path = os.path.join(self._experiment_dir, "possible-tags.txt")
            with open(file_path, "r") as file:
                possible_tags = [s.strip() for s in file.readlines() if len(s.strip()) > 1]
                st.session_state["possible_tags"] = possible_tags
        return st.session_state["possible_tags"]

    @property
    def current_image_path(self) -> Path:
        index = self.current_image_index - 1
        image_path = self.filtered_image_paths[index]
        return image_path

    @current_image_index.setter
    def current_image_index(self, new_val: int) -> None:
        st.session_state["current_image_index"] = new_val

    @property
    def filtered_image_paths(self) -> np.ndarray:
        return st.session_state["filtered_image_paths"]

    @filtered_image_paths.setter
    def filtered_image_paths(self, new_val: np.ndarray) -> None:
        st.session_state["filtered_image_paths"] = new_val

    @property
    def meta_data_file_path(self) -> str:
        source_experiment = st.session_state["source-experiment"]
        data_dir = os.path.join(self._experiment_dir, source_experiment, "data", self.dataset, self.class_name)
        return os.path.join(data_dir, "meta-data.json")

    @property
    def meta_data(self) -> MetaData:
        if "meta_data" not in st.session_state:
            raise Exception("Meta data is not loaded.")
        meta_data: MetaData = st.session_state["meta_data"]
        return meta_data

    @property
    def current_image_meta(self) -> MetaDataItem:
        return self.meta_data.get(self.current_image_path)

    def _go_next_image(self) -> None:
        new_val = self.current_image_index + 1
        max_val = self.n_images
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
        col2.slider('Current Image Index', min_value=1, max_value=self.n_images, key="current_image_index")
        col3.button("Next", on_click=lambda: self._go_next_image())

    def _render_selected_image(self) -> None:
        col1, col2 = st.columns([0.7, 0.3])
        col1.header("Image")
        
        image_path = self.current_image_path
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        col1.image(img, use_column_width=False)
        print(f"original mean: {np.mean(img):0.2f}, std: {np.std(img):0.2f}")

        new_size = (32,32)
        resized_img = load_image(str(image_path), image_size=new_size, num_channels=3, interpolation="bilinear")
        resized_img = cv2.cvtColor(resized_img.numpy(), cv2.COLOR_BGR2GRAY)
        print(f"resized mean: {np.mean(resized_img):0.2f}, std: {np.std(resized_img):0.2f}")
        col1.image(resized_img, use_column_width=False, clamp=True, output_format="png")

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

    def _render_tags_container(self) -> None:
        possible_tags = self.possible_tags
        image_meta = self.current_image_meta
        st.session_state["selected_tags"] = image_meta.tags
        st.markdown("## Tags")
        st.multiselect(
            label="Assignments for current image",
            options=possible_tags,
            default=[],
            key="selected_tags",
            on_change=lambda: self._update_current_image_tags()
        )
        st_tags(
            label='### All possible tags',
            text='Press enter to add more',
            value=possible_tags,
            suggestions=[],
            maxtags=-1
        )

    def _update_current_image_tags(self) -> None:
        self.current_image_meta.tags = st.session_state["selected_tags"]
        self.meta_data.to_csv(self.meta_data_file_path)

    def _on_label_change(self) -> None:
        new_label = st.session_state["radio_val_selected_image_label"]
        self._set_current_image_label(new_label)

    def _load_or_create_meta_data_into_session(self, image_paths: List[Path]) -> None:
        file_path = self.meta_data_file_path
        if os.path.exists(file_path):
            meta_data: MetaData = MetaData.from_csv(file_path)
        else:
            meta_data = MetaData()
            for path in image_paths:
                item = MetaDataItem(
                    full_path=path,
                    given_label=path.parent.name,
                    new_label=path.parent.name,
                    tags=[]
                )
                meta_data.add(item)
            meta_data.to_csv(file_path)
        st.session_state["meta_data"] = meta_data

    def _get_current_image_label(self) -> str:
        return self.meta_data.get(self.current_image_path).new_label
    
    def _set_current_image_label(self, new_label: str) -> None:
        meta_data = self.meta_data
        item = meta_data.get(self.current_image_path)
        item.set_new_label(new_label)
        meta_data.to_csv(self.meta_data_file_path)

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
            label='Model',
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
        source_experiment = st.session_state["source-experiment"]
        model_name = st.session_state["model_name"]
        class_name = st.session_state["class_name"]
        outlier_detector = st.session_state["outlier_detector"]
        dataset = self.dataset

        if model_name == "imagenet":
            model_path = "imagenet"
        else: 
            model_path = os.path.join(self._experiment_dir, source_experiment, "checkpoints", model_name)

        print(f"Model path: {model_path}")

        print(f"Loading model: {model_path}")
        model = get_feature_extractor(model_path)

        data_dir = os.path.join(self._experiment_dir, source_experiment, "data", dataset, class_name)
        image_paths = list(sorted(Path(data_dir).glob("**/*.png")))

        self._load_or_create_meta_data_into_session(image_paths=image_paths)

        print("Computing feature map")
        feature_map = get_feature_map(model, image_paths)

        print("Computing PCA features")
        pca = PCA(n_components=0.9, svd_solver="full", whiten=True)
        feature_map_pca = pca.fit_transform(feature_map)

        od_algo = self._outlier_detectors[outlier_detector][1]()
        outlier_labels = od_algo.fit_predict(feature_map_pca)

        self.filtered_image_paths = np.array(image_paths)[(outlier_labels == -1)]
        self.current_image_index = 1

    def _build_show_images_button(self) -> None:
        st.sidebar.button("Show images", on_click=lambda: self._show_images())
    
    def _show_images(self) -> None:
        source_experiment = st.session_state["source-experiment"]
        class_name = st.session_state["class_name"]
        dataset = self.dataset

        data_dir = os.path.join(self._experiment_dir, source_experiment, "data", dataset, class_name)
        image_paths = list(sorted(Path(data_dir).glob("**/*.png")))

        self._load_or_create_meta_data_into_session(image_paths=image_paths)
        self.filtered_image_paths = image_paths
        self.current_image_index = 1

    def _find_blank_images(self) -> None:
        source_experiment = st.session_state["source-experiment"]
        class_name = st.session_state["class_name"]
        dataset = self.dataset

        data_dir = os.path.join(self._experiment_dir, source_experiment, "data", dataset, class_name)
        image_paths = list(sorted(Path(data_dir).glob("**/*.png")))

        self._load_or_create_meta_data_into_session(image_paths=image_paths)

        filtered_image_paths = []
        for image_path in tqdm(image_paths):
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if np.mean(img) > 250 and np.std(img) < 30:
                filtered_image_paths.append(image_path)
        self.filtered_image_paths = filtered_image_paths
        self.current_image_index = 1


StreamlitApp().run()
