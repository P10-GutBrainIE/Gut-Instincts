import argparse
import logging
import os
import yaml
from transformers import AutoTokenizer, AlbertTokenizerFast
from preprocessing.ner_tokenizer import BIOTokenizer
from preprocessing.re_tokenizer import RelationTokenizer
from preprocessing.remove_html import remove_html_tags
from preprocessing.data_cleanup import clean_incorrect_text_spans, remove_incorrect_text_spans
from utils.utils import load_json_data, make_dataset_dir_name, make_task_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_training_dataset(
	experiment_name: str,
	model_name: str,
	model_type: str,
	dataset_qualities: list[str],
	dataset_weights: list[float],
	dataset_dir_name: str,
	task_name: str,
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
	save_data_path = os.path.join(task_name, experiment_name, dataset_dir_name, "training.pkl")
	if os.path.exists(save_data_path):
		logger.info(f"Training dataset already exists at {save_data_path}. Skipping create_training_dataset().")
		return

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
			save_filename=save_data_path,
			tokenizer=tokenizer,
			subtask=subtask,
			negative_sample_multiplier=config["negative_sample_multiplier"],
		)
		re_tokenizer.process_files()
	else:
		bio_tokenizer = BIOTokenizer(
			datasets=list(datasets.values()),
			dataset_weights=dataset_weights,
			save_filename=save_data_path,
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
	task_name = make_task_name(config)
	create_training_dataset(
		experiment_name=config["experiment_name"],
		model_name=config["model_name"],
		model_type=config["model_type"],
		dataset_qualities=config["dataset_qualities"],
		dataset_weights=config.get("dataset_weights"),
		dataset_dir_name=dataset_dir_name,
		task_name=task_name,
		remove_html=config["remove_html"],
		subtask=config.get("subtask"),
	)
