import datetime as dt
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


def load_label_distribution(
	file_path: str = os.path.join("data", "metadata", "entity_label_distribution.json"),
) -> dict:
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


def save_json_data(data: dict, output_path: str):
	"""
	Save data to a JSON file.

	Args:
	    data (dict): The data to be saved.
	    output_path (str): The path where the data will be saved. If the directory does not exist, it will be created.
	"""
	output_dir = os.path.dirname(output_path)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=4, ensure_ascii=False)


def set_experiment_id(experiment_name):
	timestamp = dt.datetime.now().strftime("%m%d_%H%M%S")
	return experiment_name + "_" + timestamp


def print_metrics(metrics):
	print("Validation metrics:")
	print(f"{'  Metric':<25} {'All':>10} {'No_O':>10}")

	for metric, all_value, no_o_value in zip(metrics["all"].keys(), metrics["all"].values(), metrics["no_o"].values()):
		print(f"  {metric:<25} {all_value:>10.4f} {no_o_value:>10.4f}")


def print_evaluation_metrics(metrics):
	for metric, value in zip(metrics.keys(), metrics.values()):
		print(f"  {metric:<25} {value:>10.4f}")
