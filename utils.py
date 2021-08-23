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
