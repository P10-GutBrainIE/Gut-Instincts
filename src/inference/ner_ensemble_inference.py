import argparse
from collections import Counter, defaultdict
import os
import yaml
from inference.ner_inference import NERInference
from utils.utils import save_json_data, make_dataset_dir_name


def load_model_predictions(config):
	predictions_per_model = []
	for config_path in config["model_configs"]:
		with open(config_path, "r") as f:
			config = yaml.safe_load(f)

		dataset_dir_name = make_dataset_dir_name(config)
		ner_inference = NERInference(
			test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
			model_name_path=os.path.join("models", config["experiment_name"], dataset_dir_name),
			model_name=config["model_name"],
			model_type=config["model_type"],
			remove_html=config["remove_html"],
		)

		predictions_per_model.append(ner_inference.perform_inference())

	return predictions_per_model


def majority_vote(predictions):
	entity_votes = defaultdict(list)
	ensemble_results = defaultdict(lambda: {"entities": []})

	for model_predictions in predictions:
		for paper_id, content in model_predictions.items():
			for entity in content["entities"]:
				key = (paper_id, entity["start_idx"], entity["end_idx"], entity["location"])
				entity_votes[key].append((entity["label"], entity["text_span"]))

	for (paper_id, start_idx, end_idx, location), votes in entity_votes.items():
		if len(votes) < 2:
			continue
		labels, spans = zip(*votes)
		majority_span = Counter(spans).most_common(1)[0][0]
		majority_label = Counter(labels).most_common(1)[0][0]

		ensemble_results[paper_id]["entities"].append(
			{
				"start_idx": start_idx,
				"end_idx": end_idx,
				"location": location,
				"text_span": majority_span,
				"label": majority_label,
			}
		)

	return dict(ensemble_results)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
	args = parser.parse_args()

	with open(args.config, "r") as file:
		config = yaml.safe_load(file)
		os.makedirs("models", exist_ok=True)

	predictions = load_model_predictions(config)
	print("Model predictions loaded.")
	ensemble_predictions = majority_vote(predictions)
	print("Majority vote completed.")
	save_json_data(
		data=ensemble_predictions,
		output_path=os.path.join("data_inference_results", f"{config['experiment_name']}.json"),
	)
