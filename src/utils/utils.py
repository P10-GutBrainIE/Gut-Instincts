import json
import os
import pickle
import yaml
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


def load_relation_labels(
	file_path: str = os.path.join("data", "metadata", "predicate_labels.json"),
) -> tuple[dict, dict]:
	with open(file_path) as f:
		data = json.load(f)
		relation_labels = data["relation_labels"]
		label2id = {label: i for i, label in enumerate(relation_labels)}
		id2label = {i: label for i, label in enumerate(relation_labels)}

	return relation_labels, label2id, id2label


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


def make_dataset_dir_name(config):
	"""
	Create a directory name string based on the dataset configuration.

	Args:
	    config (dict): Configuration dictionary.

	Returns:
	    str: Directory name representing the dataset configuration.
	"""
	dataset_dir_name = ""
	for i, quality in enumerate(config["dataset_qualities"]):
		dataset_dir_name += quality[0]
		if config["weighted_training"] and config["dataset_weights"]:
			dataset_dir_name += str(config["dataset_weights"][i])

	if config["remove_html"]:
		dataset_dir_name += "_no_html"

	return dataset_dir_name


def make_task_name(config):
	if config["model_type"] == "re":
		if config["subtask"] == "6.2.1":
			return "re_binary"
		elif config["subtask"] in ["6.2.2", "6.2.3"]:
			return "re_multiclass"
	else:
		return "ner"

def subtask_string(subtask):
		if subtask == "6.2.1":
			return "binary_tag_based_relations"
		elif subtask == "6.2.2":
			return "ternary_tag_based_relations"
		elif subtask == "6.2.3":
			return "ternary_mention_based_relations"
		else:
			raise ValueError(f"Unknown subtask type: {subtask}")

def print_metrics(metrics):
	"""
	Print formatted metric names and values.

	Args:
	    metrics (dict): Dictionary of metric names and their values.
	"""
	for metric, value in zip(metrics.keys(), metrics.values()):
		print(f"  {metric:<25} {value:>10.4f}")


def load_config(path):
	with open(path, "r") as file:
		return yaml.safe_load(file)
