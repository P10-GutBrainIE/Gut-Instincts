import os
from inference.re_inference import REInference
from utils.utils import make_dataset_dir_name, load_config, load_json_data, save_json_data


NER_RESULTS_PATH = os.path.join("data_inference_results_evaluated_on_dev", "entity_ensemble", "9-entity-ensemble.json")
TEST_DATA_PATH = os.path.join("data", "Articles","json_format", "articles_dev.json")


CONFIG_DIR = os.path.join("training_configs", "_re_623_top_3")


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

	save_json_data(combined_data, os.path.join("combined_data", "combined_ner_and_test_data_2.json"))


if __name__ == "__main__":
	for filename in os.listdir(CONFIG_DIR):
		file_path = os.path.join(CONFIG_DIR, filename)

		if os.path.isfile(file_path):
			config = load_config(file_path)

			dataset_dir_name = make_dataset_dir_name(config)
			re_inference = REInference(
				test_data_path=os.path.join("Annotations", "Dev","json_format", "dev.json"),
				model_name_path=os.path.join("models", dataset_dir_name),
				model_name=config["model_name"],
				model_type=config["model_type"],
				subtask=config["subtask"],
				save_path=os.path.join(
					"data_inference_results_re_evaluated_on_dev_ee9_3",
					f"{config['subtask']}",
					f"{dataset_dir_name}.json",
				),
			)
			re_inference.perform_inference()
		else:
			print(f"Provided path {file_path} is not a valid path.")
