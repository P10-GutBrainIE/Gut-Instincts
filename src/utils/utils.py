import json
import os
import pickle
import numpy as np


def load_bio_labels(
	file_path: str = os.path.join("src", "preprocessing", "bio_labels.json"),
) -> tuple[dict, dict]:
	"""
	Load the BIO labels from the specified path.

	Args:
		bio_labels_path (str, optional): Path to BIO labels. Defaults to "data/bio_labels.json".

	Returns:
		tuple[dict, dict]: Tuple containing the BIO labels, the label2id dictionary and id2label dictionary.
	"""
	with open(file_path) as f:
		data = json.load(f)
		bio_labels = data["bio_labels"]
		label2id = {label: i for i, label in enumerate(bio_labels)}
		id2label = {i: label for i, label in enumerate(bio_labels)}

	return bio_labels, label2id, id2label


def label_distribution(file_path: str = os.path.join("src", "exploratory_analysis", "label_distribution.json")):
	with open(file_path) as f:
		data = json.load(f)
		label_distribution = data["label_distribution"]
	return label_distribution


def load_pkl_data(file_path: str):
	"""
	Load the data from the specified file path.

	Args:
		file_path (str): Path to the file.

	Returns:
		np.array: Numpy array of the data.
	"""
	with open(file_path, "rb") as f:
		data = pickle.load(f)
	return np.array(data)


def load_json_data(file_path: str):
	"""
	Load the data from the specified file path.

	Args:
		file_path (str): Path to the file.

	Returns:
		dict: Dictionary of the data.
	"""
	with open(file_path, "r") as f:
		data = json.load(f)
	return data


def custom_colors() -> dict:
	custom_palette = {"Platinum": "#9fdfbf", "Gold": "#ffdf80", "Silver": "#bfbfbf", "Bronze": "#ffbf80"}
	return custom_palette
