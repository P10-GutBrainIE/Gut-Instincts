import os
import yaml
from inference.ner_inference import NERInference
from utils.utils import make_dataset_dir_name

NER_CONFIG = os.path.join("training_configs", "weighted_training", "1,5-1,5-1-0,75.yaml")
RE_CONFIG = ""

# TODO: load test data and extract metadata with title and abstract
# to add to ner inference to use in relation extraction inference


def named_entity_recognition(config, dataset_dir_name):
	ner_inference = NERInference(
		test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		model_name_path=os.path.join("models", config["experiment_name"], dataset_dir_name),
		model_name=config["model_name"],
		model_type=config["model_type"],
		save_path=os.path.join("submissions", f"Gut-Instincts_T61_{config['experiment_name']}_{dataset_dir_name}.json"),
		remove_html=config["remove_html"],
	)
	ner_inference.perform_inference()


def binary_relation_extraction():
	pass


def ternary_tag_based_relation_extraction():
	pass


def ternary_mention_based_relation_extraction():
	pass


def pipeline():
	os.makedirs("submissions", exist_ok=True)()

	with open(NER_CONFIG, "r") as file:
		ner_config = yaml.safe_load(file)
	dataset_name = make_dataset_dir_name(ner_config)
	named_entity_recognition(ner_config, dataset_name)

	binary_relation_extraction()
	ternary_tag_based_relation_extraction()
	ternary_mention_based_relation_extraction()


if __name__ == "__main__":
	pipeline()
