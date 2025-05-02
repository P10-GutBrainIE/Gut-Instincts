import os
from inference.ner_inference import NERInference
from inference.re_inference import REInference
from utils.utils import make_dataset_dir_name, load_config, load_json_data, save_json_data
from training.calculate_metrics import compute_metrics


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

	save_json_data(combined_data, os.path.join("submissions", "combined_ner_and_test_data.json"))


def named_entity_recognition(config, dataset_dir_name, test_data_path, ner_save_path):
	ner_inference = NERInference(
		test_data_path=test_data_path,
		model_name_path=os.path.join("models", config["experiment_name"], dataset_dir_name),
		model_name=config["model_name"],
		model_type=config["model_type"],
		save_path=ner_save_path,
		remove_html=config["remove_html"],
	)
	ner_inference.perform_inference()


def binary_relation_extraction(config, dataset_dir_name):
	re_inference = REInference(
		test_data_path=os.path.join("submissions", "combined_ner_and_test_data.json"),
		model_name_path=os.path.join("models", config["experiment_name"], f"{config['subtask']}_{dataset_dir_name}"),
		model_name=config["model_name"],
		model_type=config["model_type"],
		subtask=config["subtask"],
		save_path=os.path.join("submissions", f"Gut-Instincts_T62_{config['experiment_name']}_{dataset_dir_name}.json"),
	)
	re_inference.perform_inference()


def ternary_tag_based_relation_extraction():
	pass


def ternary_mention_based_relation_extraction():
	pass


def pipeline(ner_config_path, re_config_path, test_data_path, run_evaluation):
	os.makedirs("submissions", exist_ok=True)

	ner_config = load_config(ner_config_path)
	ner_dataset_name = make_dataset_dir_name(ner_config)
	named_entity_recognition(
		config=ner_config,
		dataset_dir_name=ner_dataset_name,
		test_data_path=test_data_path,
		ner_save_path=os.path.join(
			"submissions", f"Gut-Instincts_T61_{ner_config['experiment_name']}_{ner_dataset_name}.json"
		),
	)

	load_and_combine_metadata_with_ner_results(
		ner_results_path=os.path.join(
			"submissions", f"Gut-Instincts_T61_{ner_config['experiment_name']}_{ner_dataset_name}.json"
		),
		test_data_path=test_data_path,
	)

	re_config = load_config(re_config_path)
	if re_config["subtask"] == "6.2.1":
		re_dataset_name = make_dataset_dir_name(re_config)
		binary_relation_extraction(re_config, re_dataset_name)
	elif re_config["subtask"] == "6.2.2":
		re_dataset_name = make_dataset_dir_name(re_config)
		ternary_tag_based_relation_extraction()
	elif re_config["subtask"] == "6.2.3":
		re_dataset_name = make_dataset_dir_name(re_config)
		ternary_mention_based_relation_extraction()

	if run_evaluation:
		compute_metrics(ner_config, test_data_path)
		compute_metrics(
			re_config,
			test_data_path=os.path.join(
				"submissions", f"Gut-Instincts_T62_{re_config['experiment_name']}_{re_dataset_name}.json"
			),
		)


if __name__ == "__main__":
	pipeline(
		ner_config_path=os.path.join(
			"training_configs", "weighted_training_with_html", "1,5-1,5-1-0,75_with_html.yaml"
		),
		re_config_path=os.path.join("training_configs", "relation_extraction", "621_biolinkbert-base.yaml"),
		test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		run_evaluation=True,
	)
