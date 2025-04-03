import json
import os
import pickle
import numpy as np


def load_bio_labels(
	file_path: str = os.path.join("data", "metadata", "bio_labels.json"),
) -> tuple[dict, dict]:
	"""
	Load the BIO labels from the specified path.

	Args:
		bio_labels_path (str, optional): Path to BIO labels. Defaults to "data/metadata/bio_labels.json".

	Returns:
		tuple[dict, dict]: Tuple containing the BIO labels, the label2id dictionary and id2label dictionary.
	"""
	with open(file_path) as f:
		data = json.load(f)
		bio_labels = data["bio_labels"]
		label2id = {label: i for i, label in enumerate(bio_labels)}
		id2label = {i: label for i, label in enumerate(bio_labels)}

	return bio_labels, label2id, id2label


def load_entity_labels(file_path: str = os.path.join("data", "metadata", "entity_labels.json")) -> list[str]:
	"""
	Load the labels from the specified path.

	Args:
		file_path (str, optional): Path to labels. Defaults to "data/metadata/labels.json".

	Returns:
		list[str]: List of labels.
	"""
	with open(file_path) as f:
		data = json.load(f)
		labels = data["labels"]
	return labels


def load_label_distribution(file_path: str = os.path.join("data", "metadata", "entity_label_distribution.json")) -> dict:
	"""
	Load the label distribution from the specified path.

	Args:
		file_path (str, optional): Path to label distribution. Defaults to "data/metadata/label_distribution.json").

	Returns:
		dict: Dictionary of label distribution.
	"""
	with open(file_path) as f:
		data = json.load(f)
		label_distribution = data["label_distribution"]
	return label_distribution


def load_pkl_data(file_path: str) -> np.array:
	"""
	Load the data from the specified file path.

	Args:
		file_path (str): Path to the pkl file.

	Returns:
		np.array: Numpy array of the data.
	"""
	with open(file_path, "rb") as f:
		data = pickle.load(f)
	return np.array(data)


def load_json_data(file_path: str) -> dict:
	"""
	Load the data from the specified file path.

	Args:
		file_path (str): Path to the JSON file.

	Returns:
		dict: Dictionary of the data.
	"""
	with open(file_path, "r") as f:
		data = json.load(f)
	return data
