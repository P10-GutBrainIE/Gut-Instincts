import argparse
import os
import yaml
from NER.inference import NERInference
from utils.utils import make_dataset_dir_name


def create_submission_dirs():
	os.makedirs(os.path.join("submissions"))


def ner_t61():
	pass


def re_t621():
	pass


def re_t622():
	pass


def re_t623():
	pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
	args = parser.parse_args()

	with open(args.config, "r") as file:
		config = yaml.safe_load(file)

	dataset_dir_name = make_dataset_dir_name(config)

	ner_inference = NERInference(
		test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		model_name_path=os.path.join("models", config["experiment_name"], dataset_dir_name),
		model_name=config["model_name"],
		model_type=config["model_type"],
		save_path=os.path.join("data_inference_results", config["experiment_name"], f"{dataset_dir_name}.json"),
		remove_html=config["remove_html"],
	)
	ner_inference.perform_inference()
