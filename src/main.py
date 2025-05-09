import argparse
import os
from inference.re_inference import REInference
from utils.utils import make_dataset_dir_name, load_config, load_json_data, save_json_data


def load_and_combine_metadata_with_ner_results(ner_results_path, test_data_path):
	ner_results = load_json_data(ner_results_path)
	test_data = load_json_data(test_data_path)

	combined_data = {}
	for paper_id in test_data:
		entry = test_data[paper_id]
		if isinstance(entry, dict):
			meta = entry.get("metadata", entry)
			title = meta["title"]
			abstract = meta["abstract"]
			metadata = {"title": title, "abstract": abstract}
		combined_data[paper_id] = {"metadata": metadata, "entities": ner_results[paper_id]["entities"]}

	save_json_data(combined_data, os.path.join("combined_ner_and_test_data.json"))


def pipeline(
	ner_results_path,
	test_data_path,
	config,
):
	load_and_combine_metadata_with_ner_results(
		ner_results_path=ner_results_path,
		test_data_path=test_data_path,
	)

	dataset_dir_name = make_dataset_dir_name(config)
	re_inference = REInference(
		test_data_path=os.path.join("combined_ner_and_test_data.json"),
		model_name_path=os.path.join("models", dataset_dir_name),
		model_name=config["model_name"],
		model_type=config["model_type"],
		subtask=config["subtask"],
		save_path=os.path.join("data_inference_results", config["subtask"], f"{dataset_dir_name}.json"),
	)
	re_inference.perform_inference()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
	args = parser.parse_args()

	config = load_config(args.config)

	pipeline(
		ner_results_path=os.path.join(
			"data_inference_results_evaluated_on_test", "entity_ensemble", "9-entity-ensemble.json"
		),
		test_data_path=os.path.join("data", "Test_Data", "articles_test.json"),
		config=config,
	)
