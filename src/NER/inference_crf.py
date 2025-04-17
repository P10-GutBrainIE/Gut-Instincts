import os
import json
import logging
import argparse
import yaml
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, pipeline
from utils.utils import load_json_data, load_bio_labels

from NER.architectures.bert_lstm_crf import BertLSTMCRF
from NER.architectures.hf_token_classifier import HFTokenClassifier


class NERInference:
	def __init__(self, test_data_path: str, model_name_path: str, save_path: str, model_type: str = "huggingface"):
		self.test_data = load_json_data(test_data_path)
		label_list, label2id, id2label = load_bio_labels()
		self.id2label = id2label
		self.model_type = model_type.lower()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.save_path = save_path

		self.tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_fast=True, max_length=512, truncation=True)

		if self.model_type == "huggingface":
			model = HFTokenClassifier(
				model_name_path,
				num_labels=len(label_list),
				id2label=id2label,
				label2id=label2id,
				use_safetensors=True,
			)
			self.classifier = pipeline(
				"ner",
				model=model,
				tokenizer=self.tokenizer,
				aggregation_strategy="simple",
				device=0 if torch.cuda.is_available() else -1,
			)
		elif self.model_type == "bertlstmcrf":
			self.model = BertLSTMCRF(
				model_name_path,
				model_name=model_name_path,  # For loading the backbone
				num_labels=len(label_list),
				lstm_hidden_dim=256,  # Or load from config
				dropout_prob=0.3,
			).to(self.device)
			self.model.eval()
		else:
			raise ValueError(f"Unknown model_type: {self.model_type}")

	def perform_inference(self):
		result = {}
		for paper_id, content in tqdm(self.test_data.items(), total=len(self.test_data), desc="Performing inference"):
			entity_predictions = []
			try:
				if self.model_type == "huggingface":
					title_predictions = self.classifier(content["metadata"]["title"])
					entity_predictions.extend(self._merge_entities(title_predictions, "title"))

					abstract_predictions = self.classifier(content["metadata"]["abstract"])
					entity_predictions.extend(self._merge_entities(abstract_predictions, "abstract"))
				elif self.model_type == "bertlstmcrf":
					title_predictions = self.crf_inference(content["metadata"]["title"])
					entity_predictions.extend(
						self._merge_entities_crf(title_predictions, "title", content["metadata"]["title"])
					)

					abstract_predictions = self.crf_inference(content["metadata"]["abstract"])
					entity_predictions.extend(
						self._merge_entities_crf(abstract_predictions, "abstract", content["metadata"]["abstract"])
					)

				result[paper_id] = {"entities": entity_predictions}
			except Exception as e:
				logging.error(f"Error processing paper ID {paper_id}: {e}")

		os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
		with open(self.save_path, "w") as f:
			json.dump(result, f, indent=4)

	def crf_inference(self, text):
		encoding = self.tokenizer(
			text,
			return_tensors="pt",
			truncation=True,
			is_split_into_words=False,
		)
		input_ids = encoding["input_ids"].to(self.device)
		attention_mask = encoding["attention_mask"].to(self.device)
		with torch.no_grad():
			outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
			pred_ids = outputs["decoded_tags"][0]  # list of ints
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
		return list(
			zip(
				tokens,
				pred_ids,
				encoding["offset_mapping"][0].cpu().tolist() if "offset_mapping" in encoding else [None] * len(tokens),
			)
		)

	def _merge_entities_crf(self, token_predictions, location, text):
		merged = []
		current_entity = None
		for idx, (token, label_id, offset) in enumerate(token_predictions):
			label = self.id2label[label_id]
			if label == "O" or label == "[PAD]":
				if current_entity:
					merged.append(current_entity)
					current_entity = None
				continue
			if "-" in label:
				prefix, ent_type = label.split("-", 1)
			else:
				prefix, ent_type = "B", label
			if offset is not None and offset != [0, 0]:
				start_idx, end_idx = offset
			else:
				start_idx = idx
				end_idx = idx

			word = self.tokenizer.convert_tokens_to_string([token]).replace("##", "")

			if prefix == "B":
				if current_entity:
					merged.append(current_entity)
				current_entity = {
					"start_idx": start_idx,
					"end_idx": end_idx,
					"location": location,
					"text_span": word,
					"label": ent_type,
				}
			elif prefix == "I":
				if current_entity is not None and current_entity["label"] == ent_type:
					current_entity["end_idx"] = end_idx
					current_entity["text_span"] += word
				else:
					if current_entity:
						merged.append(current_entity)
					current_entity = {
						"start_idx": start_idx,
						"end_idx": end_idx,
						"location": location,
						"text_span": word,
						"label": ent_type,
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
	parser = argparse.ArgumentParser(description="NER Inference Script supporting both HF and CRF models")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
	parser.add_argument("--model_type", type=str, default="huggingface", help="Model type: huggingface or bertlstmcrf")
	args = parser.parse_args()

	with open(args.config, "r") as file:
		config = yaml.safe_load(file)

	ner_inference = NERInference(
		test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		model_name_path=os.path.join("models", f"{config['experiment_name']}"),
		save_path=os.path.join("data_inference_results", f"{config['experiment_name']}.json"),
		model_type=args.model_type,
	)
	ner_inference.perform_inference()
