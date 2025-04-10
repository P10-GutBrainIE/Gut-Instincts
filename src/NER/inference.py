import os
import json
import logging
import argparse
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from utils.utils import load_json_data, load_bio_labels


class NERInference:
	def __init__(self, test_data_path: str, model_name_path: str, save_path: str):
		self.test_data = load_json_data(test_data_path)
		label_list, label2id, id2label = load_bio_labels()
		model = AutoModelForTokenClassification.from_pretrained(
			model_name_path, num_labels=len(label_list), id2label=id2label, label2id=label2id, use_safetensors=True
		)
		tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_fast=True)
		self.classifier = pipeline("ner", model=model, tokenizer=tokenizer)
		self.save_path = save_path

	def perform_inference(self):
		result = {}
		for paper_id, content in self.test_data.items():
			entity_predictions = []

			try:
				title_predictions = self.classifier(content["metadata"]["title"])
				entity_predictions.extend(self._merge_entities(title_predictions, "title"))

				abstract_predictions = self.classifier(content["metadata"]["abstract"])
				entity_predictions.extend(self._merge_entities(abstract_predictions, "abstract"))

				result[paper_id] = {"entities": entity_predictions}
			except Exception as e:
				logging.error(f"Error processing paper ID {paper_id}: {e}")

		os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
		with open(self.save_path, "w") as f:
			json.dump(result, f, indent=4)

	def _merge_entities(self, entities, location):
		merged = []
		current_entity = None

		for entity in entities:
			label = entity["entity"].split("-")[-1]
			word = entity["word"].replace("##", "")

			if entity["entity"].startswith("B-") or current_entity is None:
				if current_entity:
					merged.append(current_entity)
				current_entity = {
					"start_idx": entity["start"],
					"end_idx": entity["end"] - 1,
					"location": location,
					"text_span": word,
					"label": label,
				}
			else:
				if entity["start"] == current_entity["end_idx"] + 1:
					current_entity["text_span"] += word
				else:
					current_entity["text_span"] += " " + word
				current_entity["end_idx"] = entity["end"] - 1

		if current_entity:
			merged.append(current_entity)

		return merged


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
	args = parser.parse_args()

	with open(args.config, "r") as file:
		config = yaml.safe_load(file)
		os.makedirs("models", exist_ok=True)

	ner_inference = NERInference(
		os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		model_name_path=os.path.join("models", f"{config['experiment_name']}"),
		save_path=os.path.join("data_inference_results", f"{config['experiment_name']}.json"),
	)
	ner_inference.perform_inference()
