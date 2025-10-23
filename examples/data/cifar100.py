import os
import pickle
import tarfile
import urllib.request
from typing import Tuple

import numpy as np

DATA_FILENAME = "cifar-100-python.tar.gz"
DATA_URL = f"https://www.cs.toronto.edu/~kriz/{DATA_FILENAME}"
EXTRACTED_DIRNAME = "cifar-100-python"


def _prepare_data_dir() -> str:
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "cifar100_files")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def _download_and_extract(target_dir: str) -> str:
    archive_path = os.path.join(target_dir, DATA_FILENAME)
    extracted_path = os.path.join(target_dir, EXTRACTED_DIRNAME)

    if not os.path.exists(extracted_path):
        if not os.path.isfile(archive_path):
            print(f"Downloading {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(target_dir)
    return extracted_path


def _load_split(root: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    filename = os.path.join(root, split)
    with open(filename, "rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    images = payload[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(payload[b"fine_labels"])
    return images, labels


def load_cifar100(normalize: bool = True):
    """
    Downloads (if needed) and loads the CIFAR-100 dataset.
    Returns: (train_images, train_labels, test_images, test_labels)
    """
    data_dir = _prepare_data_dir()
    extracted_path = _download_and_extract(data_dir)

    train_images, train_labels = _load_split(extracted_path, "train")
    test_images, test_labels = _load_split(extracted_path, "test")

    if normalize:
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0

    return train_images, train_labels, test_images, test_labels
