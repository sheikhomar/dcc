import sys, os, re, base64


sys.path.append(os.path.abspath("../"))

from glob import glob
from pathlib import Path

import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import umap

from typing import List, Tuple

from tensorflow.python.keras.preprocessing.image_dataset import load_image
from tqdm import tqdm
from dash.dash import Dash


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

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import utils


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



app = Dash(__name__)

experiment_dir = "experiments"

source_experiments = os.listdir(experiment_dir)

outlier_detectors = {
    "IsolationForest": "Isolation Forest",
    "EllipticEnvelope": "Elliptic Envelope",
    "OneClassSVM": "SVM",
}

model_names = {
    "best_model": "RomanNumerals-trained model",
    "imagenet-finetuned": "Fine-tuned model",
    "imagenet": "ImageNet-trained model",
}

app.layout = html.Div([
    html.Div([

        html.Div([
            html.Label([
                "Select source experiment directory", 
                dcc.Dropdown(
                    id="select-source-experiment",
                    options=[{'label': exp, 'value': exp} for exp in source_experiments],
                    value="original",
                    placeholder="Select source experiment directory.",
                ),
            ]),
            
            html.Label([
                "Select outlier detection algorithm", 
                dcc.Dropdown(
                    id='select-outlier-detector',
                    options=[{'label': v, 'value': k} for k, v in outlier_detectors.items()],
                    value=list(outlier_detectors.keys())[0],
                ),
            ]),
            
            html.Label([
                "Select model", 
                dcc.Dropdown(
                    id='select-model',
                    options=[{'label': v, 'value': k} for k, v in model_names.items()],
                    value="imagenet-finetuned",
                ),
            ]),
            
            
            html.Div([
                html.Label([
                    "Pick dataset", 
                    dcc.Dropdown(
                        id='select-dataset',
                        options=[{'label': v, 'value': v} for v in ["train", "valid", "test"]],
                        value="train",
                    ),
                ]),
            ]),

            html.Div([
                html.Label([
                    "Class to focus on", 
                    dcc.Input(
                        id="field-class-name",
                        value="i",
                        style={"width": "92%", "padding": "10px"}
                    ),
                ]),
            ]),

            html.Div([
                html.Label([
                    "Target experiment directory", 
                    dcc.Input(
                        id="field-target-experiment",
                        value="dash",
                        style={"width": "92%", "padding": "10px"}
                    ),
                ]),
            ]),

            
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            html.Button(
                'Find outliers',
                id='button-find-outliers',
                n_clicks=0
            ),
            html.Div(
                id='container-messages',
                 children='Enter a value and press submit'
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),


    html.Div([
        html.Div([

            html.Div([
                
                html.Button(
                    'Prev',
                    id='button-prev-image',
                    n_clicks=0,
                    style={"float": "left", "width": "10%"}
                ),

                html.Div([
                    dcc.Slider(
                        id="slider-image-selector",
                        min=0,
                        max=0,  
                    ),
                ], style={"float": "left", "width": "80%"}),

                html.Button(
                    'Next',
                    id='button-next-image',
                    n_clicks=0,
                    style={"float": "left", "width": "10%"}
                ),
            ]),
            
            html.Div([
                html.Img(id="selected-image", src=""),
            ], style={"padding": "10px"}),

            

        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            "Right"
        ],style={'width': '49%', 'float': 'right', 'display': 'inline-block', "border": "solid 1px Blue"})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'gray',
        'padding': '10px 5px'
    })


])


image_paths = None
outlier_labels = None


@app.callback(
    dash.dependencies.Output('container-messages', 'children'),
    [dash.dependencies.Input('button-find-outliers', 'n_clicks')],
    [
        dash.dependencies.State('select-source-experiment', 'value'),
        dash.dependencies.State('select-model', 'value'),
        dash.dependencies.State('select-outlier-detector', 'value'),
        dash.dependencies.State('field-class-name', 'value'),
        dash.dependencies.State('field-target-experiment', 'value'),
        
    ]
)
def update_container_messages(n_clicks, source_experiment, model_name, outlier_detector, class_name, target_experiment):
    if n_clicks == 0:
        return "first time"

    global image_paths, outlier_labels

    if model_name == "imagenet":
        model_path = "imagenet"
    else: 
        model_path = os.path.join(experiment_dir, source_experiment, "checkpoints", model_name)
    
    print(f"Loading model: {model_path}")
    model = get_feature_extractor(model_path)

    data_dir = os.path.join(experiment_dir, source_experiment, "data", "train", class_name)
    image_paths = list(sorted(Path(data_dir).glob("**/*.png")))

    print("Computing feature map")
    feature_map = get_feature_map(model, image_paths)

    print("Computing PCA features")
    pca = PCA(n_components=0.9, svd_solver="full", whiten=True)
    feature_map_pca = pca.fit_transform(feature_map)

    algos = {
        "IsolationForest": IsolationForest,
        "OneClassSVM": OneClassSVM,
        "EllipticEnvelope": EllipticEnvelope,
    }
    od_algo = algos[outlier_detector]()

    outlier_labels = od_algo.fit_predict(feature_map_pca)

    return f"{outlier_labels}"


@app.callback(
    [
        dash.dependencies.Output('slider-image-selector', 'max'),
        dash.dependencies.Output('slider-image-selector', 'value'),
    ],
    [dash.dependencies.Input('container-messages', 'children')],
    []
)
def update_slider(message):
    if outlier_labels is None:
        return [0, 0]
    n_outliers = (outlier_labels == -1).sum()
    print(f"Number of outliers: {n_outliers}")
    return [n_outliers - 1, 0]


@app.callback(
    dash.dependencies.Output('selected-image', 'src'),
    [dash.dependencies.Input('slider-image-selector', 'value')],
    []
)
def update_image(selection_index):
    outlier_paths = np.array(image_paths)[(outlier_labels == -1)]
    if selection_index is None or selection_index >= len(outlier_paths):
        return f"/assets/image-not-found.png"
    outlier_path = outlier_paths[selection_index]
    encoded_image = base64.b64encode(open(outlier_path, 'rb').read())
    return f"data:image/png;base64,{encoded_image.decode('utf-8')}"


@app.callback(
    dash.dependencies.Output('slider-image-selector', 'min'),
    [dash.dependencies.Input('button-prev-image', 'n_clicks')],
    [
        dash.dependencies.State('slider-image-selector', 'value'),
    ]
)
def update_image_on_prev(n_clicks, current_val):
    print(f"update_image_on_prev called: {current_val}")
    if current_val is None:
        return 0
    new_val = current_val - 1
    if new_val < 0:
        new_val = 0
    return new_val



# @app.callback(
#     dash.dependencies.Output('slider-image-selector', 'value'),
#     [dash.dependencies.Input('button-next-image', 'n_clicks')],
#     [
#         dash.dependencies.State('slider-image-selector', 'value'),
#         dash.dependencies.State('slider-image-selector', 'max'),
#     ]
# )
# def update_image_on_next(n_clicks, current_val, max_val):
#     if current_val is None:
#         return 0
#     new_val = current_val + 1
#     if new_val >= max_val:
#         new_val = max_val
#     return new_val
    

app.run_server()