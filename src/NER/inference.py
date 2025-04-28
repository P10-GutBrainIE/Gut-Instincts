import os
import json
import argparse
import yaml
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, AlbertTokenizerFast, pipeline
from preprocessing.remove_html import remove_html_tags
from utils.utils import load_json_data, load_bio_labels, make_dataset_dir_name


class NERInference:
	def __init__(
		self,
		test_data_path: str,
		model_name: str,
		model_type: str,
		remove_html: bool,
		model_name_path: str = None,
		save_path: str = None,
		validation_model=None,
	):
		self.test_data = (
			remove_html_tags(load_json_data(test_data_path)) if remove_html else load_json_data(test_data_path)
		)
		label_list, label2id, self.id2label = load_bio_labels()
		if model_name in ["sultan/BioM-ALBERT-xxlarge", "sultan/BioM-ALBERT-xxlarge-PMC"]:
			self.tokenizer = AlbertTokenizerFast.from_pretrained(model_name, max_length=512, truncation=True)
		else:
			self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, max_length=512, truncation=True)
		self.model_type = model_type
		self.validation_model = validation_model
		self.model_name = model_name

		if model_type == "huggingface":
			if validation_model:
				self.model = validation_model
			else:
				model = AutoModelForTokenClassification.from_pretrained(
					model_name_path,
					num_labels=len(label_list),
					id2label=self.id2label,
					label2id=label2id,
					use_safetensors=True,
				)
				self.classifier = pipeline("ner", model=model, tokenizer=self.tokenizer)
		elif model_type == "bertlstmcrf":
			if validation_model:
				model = validation_model
			else:
				from NER.architectures.bert_lstm_crf import BertLSTMCRF

				model = BertLSTMCRF(
					model_name=model_name,
					num_labels=len(label_list),
				)
				state_dict = torch.load(os.path.join(model_name_path, "pytorch_model.bin"), map_location="cpu")
				model.load_state_dict(state_dict)
				model.eval()
			self.model = model
		else:
			raise ValueError("Unknown model_type")

		self.save_path = save_path

	def perform_inference(self):
		result = {}

		if self.model_type == "huggingface" and not self.validation_model:
			prediction_fn = self.classifier
		elif self.model_type == "bertlstmcrf" or self.validation_model:
			prediction_fn = self._ner_pipeline
		else:
			raise ValueError("Unknown model_type")

		for paper_id, content in tqdm(self.test_data.items(), total=len(self.test_data), desc="Inference"):
			entity_predictions = []
			for section in ["title", "abstract"]:
				text = content["metadata"][section]
				predictions = prediction_fn(text)
				merged = self._merge_entities(predictions, section)
				adjusted = self._adjust_casing(entity_predictions=merged, true_text=text)
				entity_predictions.extend(adjusted)

			result[paper_id] = {"entities": entity_predictions}

		if self.save_path:
			os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
			with open(self.save_path, "w") as f:
				json.dump(result, f, indent=4)
		else:
			return result

	def _ner_pipeline(self, text):
		result = self.tokenizer(
			text,
			return_tensors="pt",
			return_offsets_mapping=True,
			truncation=True,
			max_length=512,
		)
		device = next(self.model.parameters()).device
		input_ids = result["input_ids"].to(device)
		attention_mask = result["attention_mask"].to(device)

		if len(result["input_ids"][0]) > 512:
			print("Input longer than 512 tokens")

		tokens = self.tokenizer.convert_ids_to_tokens(result["input_ids"][0])[1:-1]
		offsets = result["offset_mapping"][0][1:-1]

		outputs = self.model.predict(input_ids, attention_mask)
		if isinstance(outputs[0], torch.Tensor):
			outputs = outputs[0].tolist()
		else:
			outputs = outputs[0]
		labels = [self.id2label[id] for id in outputs][1:-1]

		return [
			{"entity": label, "word": token, "start": int(start), "end": int(end)}
			for label, token, (start, end) in zip(labels, tokens, offsets)
			if label != "O"
		]

	def _process_lookahead(self, token_predictions, i, current_entity, model_name):
		lookahead = []
		prev_token_pred_end_idx = current_entity["end_idx"]
		skip = 0

		for token_pred in token_predictions:
			prefix, _ = token_pred["entity"].split("-", 1)
			if token_pred["start"] in [prev_token_pred_end_idx, prev_token_pred_end_idx + 1] and prefix == "I":
				lookahead.append(token_pred)
				prev_token_pred_end_idx = token_pred["end"]
			elif len(lookahead) >= 3 and len(set(tp["entity"].split("-")[1] for tp in lookahead)) > 1:
				_, label_last = lookahead[-1]["entity"].split("-", 1)
				if current_entity["label"] == label_last:
					for token_pred in lookahead:
						if model_name in ["sultan/BioM-ALBERT-xxlarge", "sultan/BioM-ALBERT-xxlarge-PMC"]:
							word = token_pred["word"].replace("▁", "")
						else:
							word = token_pred["word"].replace("##", "")
						if token_pred["start"] == current_entity["end_idx"] + 1:
							current_entity["text_span"] += word
						else:
							current_entity["text_span"] += " " + word
						current_entity["end_idx"] = token_pred["end"] - 1
					skip = len(lookahead)
					break
			else:
				break
		return current_entity, skip

	def _merge_entities(self, token_predictions, location):
		merged = []
		current_entity = None
		skip = 0

		for i, token_prediction in enumerate(token_predictions):
			if skip:
				skip -= 1
				continue

			prefix, label = token_prediction["entity"].split("-", 1)
			if self.model_name in ["sultan/BioM-ALBERT-xxlarge", "sultan/BioM-ALBERT-xxlarge-PMC"]:
				word = token_prediction["word"].replace("▁", "")
			else:
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

				current_entity, skip = self._process_lookahead(
					token_predictions[i + 1 : i + 10], i, current_entity, self.model_name
				)
				if skip:
					continue

			elif prefix == "I":
				if (
					current_entity
					and current_entity["label"] == label
					and token_prediction["start"] in [current_entity["end_idx"] + 1, current_entity["end_idx"] + 2]
				):
					if token_prediction["start"] == current_entity["end_idx"] + 1:
						current_entity["text_span"] += word
					elif token_prediction["start"] == current_entity["end_idx"] + 2:
						current_entity["text_span"] += " " + word
					current_entity["end_idx"] = token_prediction["end"] - 1
				else:
					if current_entity:
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

	def _adjust_casing(self, entity_predictions, true_text):
		for entity in entity_predictions:
			entity["text_span"] = true_text[entity["start_idx"] : entity["end_idx"] + 1]

		return entity_predictions


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
