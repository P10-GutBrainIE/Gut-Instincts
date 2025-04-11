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
			model_name_path,
			num_labels=len(label_list),
			id2label=id2label,
			label2id=label2id,
			use_safetensors=True,
		)
		tokenizer = AutoTokenizer.from_pretrained(
			model_name_path, use_fast=True, max_length=512, truncation=True
		)
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

	def perform_inference_concatenated(self):
		result = {}
		for paper_id, content in self.test_data.items():
			text = content["metadata"]["title"] + ". " + content["metadata"]["abstract"]
			entity_predictions = []

			try:
				text_predictions = self.classifier(text)
				entity_predictions.extend(self._merge_entities_concatenated(text_predictions, len(content["metadata"]["title"])))

				result[paper_id] = {"entities": entity_predictions}
			except Exception as e:
				logging.error(f"Error processing paper ID {paper_id}: {e}")

		os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
		with open(self.save_path, "w") as f:
			json.dump(result, f, indent=4)

	def _merge_entities_concatenated(self, token_predictions, title_length):
		merged = []
		current_entity = None

		for token_prediction in token_predictions:
			prefix, label = token_prediction["entity"].split("-", 1)
			word = token_prediction["word"].replace("##", "")

			start_idx = token_prediction["start"]
			end_idx = token_prediction["end"] - 1
			location = "title" if end_idx < title_length else "abstract"

			# Adjust indices for abstract tokens
			if location == "abstract":
				start_idx -= title_length + 2
				end_idx -= title_length + 2

			if prefix == "B":
				if current_entity:
					merged.append(current_entity)
				current_entity = {
					"start_idx": start_idx,
					"end_idx": end_idx,
					"location": location,
					"text_span": word,
					"label": label,
				}
			elif prefix == "I":
				if current_entity is not None and current_entity["label"] == label:
					if start_idx == current_entity["end_idx"] + 1:
						current_entity["text_span"] += word
					else:
						current_entity["text_span"] += " " + word
					current_entity["end_idx"] = end_idx
				else:
					if current_entity:
						merged.append(current_entity)
					current_entity = {
						"start_idx": start_idx,
						"end_idx": end_idx,
						"location": location,
						"text_span": word,
						"label": label,
					}

		if current_entity:
			merged.append(current_entity)

		return merged

	def _merge_entities(self, token_predictions, location):
		merged = []
		current_entity = None

		for token_prediction in token_predictions:
			prefix, label = token_prediction["entity"].split("-", 1)
			word = token_prediction["word"].replace("##", "")

			if prefix == "B":
				if current_entity:
					merged.append(current_entity)
				current_entity = {
					"start_idx": token_prediction["start"],
					"end_idx": token_prediction["end"] - 1,
					"location": location,
					"text_span": word,
					"label": label,
				}
			elif prefix == "I":
				# Check if the current entity label is of the same as the previous entity's label
				# if not the same, then create a new entity
				if current_entity is not None and current_entity["label"] == label:
					if token_prediction["start"] == current_entity["end_idx"] + 1:
						current_entity["text_span"] += word
					else:
						current_entity["text_span"] += " " + word
					current_entity["end_idx"] = token_prediction["end"] - 1
				else:
					if current_entity is not None:
						merged.append(current_entity)
					current_entity = {
						"start_idx": token_prediction["start"],
						"end_idx": token_prediction["end"] - 1,
						"location": location,
						"text_span": word,
						"label": label,
					}

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
	ner_inference.perform_inference_concatenated()
