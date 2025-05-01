import argparse
import os
import yaml
from transformers import AutoTokenizer, AlbertTokenizerFast
from preprocessing.ner_tokenizer import BIOTokenizer
from preprocessing.re_tokenizer import RelationTokenizer
from preprocessing.remove_html import remove_html_tags
from preprocessing.data_cleanup import clean_incorrect_text_spans, remove_incorrect_text_spans
from utils.utils import load_json_data, make_dataset_dir_name


def create_training_dataset(
	experiment_name: str,
	model_name: str,
	model_type: str,
	dataset_qualities: list[str],
	dataset_dir_name: str,
	dataset_weights: list[float],
	remove_html: bool,
	subtask: str = None,
):
	"""
	Create and preprocess the training dataset, then tokenize and save it.

	Loads the training datasets for the specified qualities, applies corrections and cleaning,
	optionally removes HTML, and tokenizes the data using the appropriate tokenizer for the model.
	The processed data is saved as a pickle file.

	Args:
	    experiment_name (str): Name of the experiment for output directory structure.
	    model_name (str): Name of the pretrained model (used to select tokenizer).
	    dataset_qualities (list[str]): List of dataset quality labels (e.g., ["gold", "silver"]).
	    dataset_dir_name (str): Name of the directory where the dataset will be saved.
	    dataset_weights (list[float]): List of weights for each dataset quality (can be None).
	    remove_html (bool): Whether to remove HTML tags from the datasets.
	"""
	datasets = {quality: [] for quality in dataset_qualities}

	shared_path = os.path.join("data", "Annotations", "Train")
	for quality in dataset_qualities:
		datasets[quality] = load_json_data(
			os.path.join(shared_path, f"{quality}_quality", "json_format", f"train_{quality}.json")
		)

		if quality == "silver":
			datasets[quality] = clean_incorrect_text_spans(
				data=datasets[quality],
				corrections=load_json_data(os.path.join("data", "metadata", "silver_incorrect_annotations.json"))[
					"clean"
				],
			)
		if quality == "bronze":
			datasets[quality] = clean_incorrect_text_spans(
				data=datasets[quality],
				corrections=load_json_data(os.path.join("data", "metadata", "bronze_incorrect_annotations.json"))[
					"clean"
				],
			)
			datasets[quality] = remove_incorrect_text_spans(
				data=datasets[quality],
				incorrect_annotations=load_json_data(
					os.path.join("data", "metadata", "bronze_incorrect_annotations.json")
				)["remove"],
			)

		if remove_html:
			datasets[quality] = remove_html_tags(datasets[quality])

	if model_name in ["sultan/BioM-ALBERT-xxlarge", "sultan/BioM-ALBERT-xxlarge-PMC"]:
		tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
	else:
		tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

	if model_type == "re":
		re_tokenizer = RelationTokenizer(
			datasets=list(datasets.values()),
			dataset_weights=dataset_weights,
			save_filename=os.path.join(experiment_name, dataset_dir_name, "training.pkl"),
			tokenizer=tokenizer,
			subtask=subtask,
		)
		re_tokenizer.process_files()
	else:
		bio_tokenizer = BIOTokenizer(
			datasets=list(datasets.values()),
			dataset_weights=dataset_weights,
			save_filename=os.path.join(experiment_name, dataset_dir_name, "training.pkl"),
			tokenizer=tokenizer,
		)
		bio_tokenizer.process_files()


def create_validation_dataset(
	experiment_name: str,
	model_name: str,
	model_type: str,
	dataset_dir_name: str,
	remove_html: bool,
	subtask: str = None,
):
	"""
	Create and preprocess the validation (dev) dataset, then tokenize and save it.

	Loads the validation dataset, optionally removes HTML, tokenizes the data using the
	appropriate tokenizer for the model, and saves the processed data as a pickle file.

	Args:
	    experiment_name (str): Name of the experiment for output directory structure.
	    model_name (str): Name of the pretrained model (used to select tokenizer).
	    dataset_dir_name (str): Name of the directory where the dataset will be saved.
	    remove_html (bool): Whether to remove HTML tags from the validation data.
	"""
	dev_data = load_json_data(os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))
	ner_predictions = load_json_data(os.path.join("data_inference_results", "NER_results", "ensemble_inference.json"))

	if remove_html:
		dev_data = remove_html_tags(data=dev_data)

	if model_name in ["sultan/BioM-ALBERT-xxlarge", "sultan/BioM-ALBERT-xxlarge-PMC"]:
		tokenizer = AlbertTokenizerFast.from_pretrained(model_name)
	else:
		tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

	if model_type == "re":
		for paper_id in ner_predictions:
			if paper_id not in dev_data:
				return ValueError("paper_id is not in dev data")
			dev_content = dev_data[paper_id]
			inference_content = ner_predictions[paper_id]
			merged = {"metadata": dev_content["metadata"], "entities": inference_content["entities"]}

		re_tokenizer = RelationTokenizer(
			datasets=[merged],
			save_filename=os.path.join(experiment_name, dataset_dir_name, "validation.pkl"),
			tokenizer=tokenizer,
			subtask=subtask,
		)
		re_tokenizer.process_files(is_ner_predictions=True)
	else:
		bio_tokenizer = BIOTokenizer(
			datasets=[dev_data],
			save_filename=os.path.join(experiment_name, dataset_dir_name, "validation.pkl"),
			tokenizer=tokenizer,
		)
		bio_tokenizer.process_files()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Clean and preprocess data and apply specified tokenizer")
	parser.add_argument("--config", type=str, required=True, help="Path to the training config YAML file")
	args = parser.parse_args()

	with open(args.config, "r") as file:
		config = yaml.safe_load(file)

	dataset_dir_name = make_dataset_dir_name(config)

	create_training_dataset(
		experiment_name=config["experiment_name"],
		model_name=config["model_name"],
		model_type=config["model_type"],
		dataset_qualities=config["dataset_qualities"],
		dataset_dir_name=dataset_dir_name,
		dataset_weights=config.get("dataset_weights"),
		remove_html=config["remove_html"],
		subtask=config["subtask"],
	)
	create_validation_dataset(
		experiment_name=config["experiment_name"],
		model_name=config["model_name"],
		model_type=config["model_type"],
		dataset_dir_name=dataset_dir_name,
		remove_html=config["remove_html"],
		subtask=config["subtask"],
	)
