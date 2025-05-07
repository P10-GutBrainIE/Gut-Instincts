import argparse
from collections import Counter, defaultdict
import os
import json
import numpy as np
from inference.ner_inference import NERInference
from utils.utils import make_dataset_dir_name, load_config, load_json_data


class EnsembleNERInference:
	def __init__(
		self, test_data_path, save_path=None, config_paths=None, token_ensemble_results_paths: list[str] = None
	):
		self.configs = [load_config(path) for path in config_paths] if config_paths else None
		self.test_data_path = test_data_path
		self.save_path = save_path
		self.predictions = (
			[load_json_data(path) for path in token_ensemble_results_paths]
			if token_ensemble_results_paths
			else self._load_model_predictions()
		)
		self.majority_threshold = (
			np.ceil(len(token_ensemble_results_paths) / 2)
			if token_ensemble_results_paths
			else np.ceil(len(config_paths) / 2)
		)

	def _load_model_predictions(self):
		predictions_per_model = []
		for config in self.configs:
			dataset_dir_name = make_dataset_dir_name(config)
			ner_inference = NERInference(
				test_data_path=self.test_data_path,
				model_name_path=os.path.join("models", dataset_dir_name),
				model_name=config["model_name"],
				model_type=config["model_type"],
				remove_html=config["remove_html"],
			)

			predictions_per_model.append(ner_inference.perform_inference())

		return predictions_per_model

	def perform_entity_level_inference(self):
		entity_votes = defaultdict(list)
		ensemble_results = defaultdict(lambda: {"entities": []})

		for model_predictions in self.predictions:
			for paper_id, content in model_predictions.items():
				for entity in content["entities"]:
					key = (paper_id, entity["start_idx"], entity["end_idx"], entity["location"])
					entity_votes[key].append((entity["label"], entity["text_span"]))

		for (paper_id, start_idx, end_idx, location), votes in entity_votes.items():
			if len(votes) < self.majority_threshold:
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

		if self.save_path:
			os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
			with open(self.save_path, "w") as f:
				json.dump(dict(ensemble_results), f, indent=4)
		else:
			return dict(ensemble_results)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
	args = parser.parse_args()

	config = load_config(args.config)

	ensemble_ner_inference = EnsembleNERInference(
		config_paths=config.get("model_configs"),
		test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		save_path=os.path.join("data_inference_results", "entity_ensemble", f"{config['experiment_name']}.json"),
		token_ensemble_results_paths=config.get("token_ensemble_results_paths"),
	)

	ensemble_ner_inference.perform_entity_level_inference()
