import os
import json
import argparse
import yaml
from collections import Counter
from NER.inference import NERInference


def load_model_predictions(config):
	predictions_per_model = []
	for config_path in config["model_configs"]:
		with open(config_path, "r") as f:
			model_config = yaml.safe_load(f)

		ner_inference = NERInference(
			config["test_data_path"],
			model_name_path=os.path.join("models", f"{model_config['experiment_name']}"),
			save_path=os.path.join("data_inference_results", f"{model_config['experiment_name']}.json"),
		)

		predictions_per_model.append(ner_inference.perform_inference_return_data())

	return predictions_per_model


def majority_vote(predictions):
	entity_votes = {}
	ensemble_results = {}
	for model_predictions in predictions:
		for paper_id, content in model_predictions.items():
			for entity in content["entities"]:
				key = (paper_id, entity["start_idx"], entity["end_idx"], entity["location"])
				entity_votes[key].append(entity["label"], entity["text_span"])
				
	for (paper_id, start_idx, end_idx, location), votes in entity_votes.items():
		labels, spans = zip(*votes)
		majority_label, _ = Counter(labels).most_common(1)[0]
		majority_span = Counter(spans).most_common(1)[0]
		ensemble_results[paper_id].append({
			"start_idx": start_idx,
			"end_idx": end_idx,
			"location": location,
			"text_span": majority_span,
			"label": majority_label,
        })


def save_ensemble_results(predictions, save_path):
	with open(save_path, "w") as file:
		json.dump(predictions, file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
	args = parser.parse_args()

	with open(args.config, "r") as file:
		config = yaml.safe_load(file)
		os.makedirs("models", exist_ok=True)

	predictions = load_model_predictions(config)
	ensemble_predictions = majority_vote(predictions)
	# save_ensemble_results(predictions= ensemble_predictions, save_path = os.path.join("data_inference_results", f"{config['experiment_name']}.json"))
