import argparse
from collections import deque, Counter
import itertools
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AlbertTokenizerFast
from preprocessing.remove_html import remove_html_tags
from utils.utils import load_json_data, load_bio_labels, make_dataset_dir_name, load_config


class NERInference:
	def __init__(
		self,
		test_data_path: str,
		model_name: str | list[str],
		model_type: str | list[str],
		remove_html: bool,
		model_name_path: str | list[str],
		save_path: str = None,
		validation_model=None,
		ensemble_strategy: str = None,
	):
		self.test_data = (
			remove_html_tags(load_json_data(test_data_path)) if remove_html else load_json_data(test_data_path)
		)
		self.label_list, self.label2id, self.id2label = load_bio_labels()
		self.model_type = model_type
		self.model_name = model_name
		self.model_name_path = model_name_path
		self.validation_model = validation_model
		self.save_path = save_path
		self.ensemble_strategy = ensemble_strategy
		self.models = []
		self.tokenizers = []
		self._build_model()

	def _as_list(self, obj):
		"""Ensure the object is always a list for unified processing."""
		if isinstance(obj, list):
			return obj
		return [obj]

	def _build_model(self):
		model_types = self._as_list(self.model_type)
		model_names = self._as_list(self.model_name)
		model_name_paths = self._as_list(self.model_name_path)

		for m_type, m_name, m_path in zip(model_types, model_names, model_name_paths):
			if m_name in ["sultan/BioM-ALBERT-xxlarge", "sultan/BioM-ALBERT-xxlarge-PMC"]:
				self.tokenizers.append(AlbertTokenizerFast.from_pretrained(m_name, max_length=512, truncation=True))
			else:
				self.tokenizers.append(
					AutoTokenizer.from_pretrained(m_name, use_fast=True, max_length=512, truncation=True)
				)

			if m_type == "huggingface":
				if self.validation_model:
					model = self.validation_model
				else:
					from architectures.hf_token_classifier import HFTokenClassifier

					model = HFTokenClassifier(
						m_path,
						num_labels=len(self.label_list),
						id2label=self.id2label,
						label2id=self.label2id,
					)
			elif m_type == "bertlstmcrf":
				if self.validation_model:
					model = self.validation_model
				else:
					from architectures.bert_lstm_crf import BertLSTMCRF

					model = BertLSTMCRF(
						model_name=m_name,
						num_labels=len(self.label_list),
					)
					state_dict = torch.load(os.path.join(m_path, "pytorch_model.bin"), map_location="cpu")
					model.load_state_dict(state_dict)
					model.eval()
			elif m_type == "bertdensecrf":
				if self.validation_model:
					model = self.validation_model
				else:
					from architectures.bert_dense_crf import BertDenseCRF

					model = BertDenseCRF(
						model_name=m_name,
						num_labels=len(self.label_list),
					)
					state_dict = torch.load(os.path.join(m_path, "pytorch_model.bin"), map_location="cpu")
					model.load_state_dict(state_dict)
					model.eval()
			else:
				raise ValueError(f"Unknown model_type: {m_type}")
			self.models.append(model)

			if len(self.models) == 1 and self.ensemble_strategy is None:
				self.model = self.models[0]
				self.tokenizer = self.tokenizers[0]

	def perform_ensemble_inference(self):
		print(f"Performing ensemble inference with {self.ensemble_strategy} strategy")
		all_model_predictions = []
		for model_name, model, tokenizer in zip(self.model_name, self.models, self.tokenizers):
			self.model = model
			self.tokenizer = tokenizer
			model_predictions = {}
			for paper_id, content in tqdm(
				self.test_data.items(), total=len(self.test_data), desc=f"Inference with {model_name}"
			):
				model_predictions[paper_id] = {}
				for section in ["title", "abstract"]:
					text = content[section]
					predictions = self._ner_pipeline(text)
					model_predictions[paper_id][section] = {"tokens": predictions, "true_text": text}

			all_model_predictions.append(model_predictions)

		predictions = self._combine_ensemble_predictions(all_model_predictions)

	def _combine_ensemble_predictions(self, all_model_predictions):
		result = {}

		token_strategy, label_strategy = self.ensemble_strategy.split(".")

		for paper_id in tqdm(
			all_model_predictions[0].keys(),
			total=len(all_model_predictions[0].keys()),
			desc="Combining ensemble predictions",
		):
			entity_predictions = []
			for section in ["title", "abstract"]:
				combined_tokens = []
				token_queues = {
					model_number: deque(model_predictions[paper_id][section]["tokens"])
					for model_number, model_predictions in enumerate(all_model_predictions)
				}
				entity_type = "O"
				while True:
					tokens = {model_number: queue[0] for model_number, queue in token_queues.items() if queue}
					if not tokens:
						break

					min_start_value = min(token["start"] for _, token in tokens.items())
					tokens_lowest_start = [
						model_number for model_number, token in tokens.items() if token["start"] == min_start_value
					]

					if token_strategy == "union":
						entity_type = self._ensemble_entity_type(
							[tokens[i] for i in tokens_lowest_start], label_strategy
						)
					elif token_strategy == "majority":
						if len(tokens_lowest_start) >= int(np.ceil(len(self.model_name) / 2)):
							entity_type = self._ensemble_entity_type(
								[tokens[i] for i in tokens_lowest_start], label_strategy
							)
					elif token_strategy == "intersection":
						if len(tokens_lowest_start) == len(self.model_name):
							entity_type = self._ensemble_entity_type(
								[tokens[i] for i in tokens_lowest_start], label_strategy
							)

					for model_number in tokens_lowest_start:
						token_queues[model_number].popleft()

					if entity_type != "O":
						t = tokens[tokens_lowest_start[0]]
						combined_tokens.append(
							{
								"entity": entity_type,
								"word": t["word"],
								"start": t["start"],
								"end": t["end"],
							}
						)
						entity_type = "O"

				merged = self._merge_entities(combined_tokens, section)
				adjusted = self._adjust_casing(
					entity_predictions=merged, true_text=all_model_predictions[0][paper_id][section]["true_text"]
				)
				entity_predictions.extend(adjusted)

			result[paper_id] = {"entities": entity_predictions}

		if self.save_path:
			os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
			with open(self.save_path, "w") as f:
				json.dump(result, f, indent=4)
		else:
			return result

	def _ensemble_entity_type(self, tokens: list, strategy: str):
		if strategy == "majority":
			counter = Counter([token["entity"] for token in tokens])
			majority_vote = counter.most_common()
			if len(majority_vote) > 1 and majority_vote[0][1] == majority_vote[1][1]:
				total_softmax_sum = np.sum([token["softmax"] for token in tokens], axis=0)
				entity_type = self.id2label[np.argmax(total_softmax_sum)]
			else:
				entity_type = majority_vote[0][0]
		elif strategy == "softmax_sum":
			total_softmax_sum = np.sum([token["softmax"] for token in tokens], axis=0)
			entity_type = self.id2label[np.argmax(total_softmax_sum)]

		return entity_type

	def perform_inference(self):
		result = {}

		for paper_id, content in tqdm(self.test_data.items(), total=len(self.test_data), desc="Inference"):
			entity_predictions = []
			for section in ["title", "abstract"]:
				text = content[section]
				predictions = self._ner_pipeline(text)
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
		tokens = self.tokenizer.convert_ids_to_tokens(result["input_ids"][0])[1:-1]
		offsets = result["offset_mapping"][0][1:-1]

		if self.ensemble_strategy:
			outputs, probs = self.model.predict(input_ids, attention_mask, return_softmax=True)
			probs = probs[0][1:-1]
		else:
			outputs = self.model.predict(input_ids, attention_mask)

		outputs = outputs[0]
		labels = [self.id2label[id] for id in outputs][1:-1]

		if self.ensemble_strategy:
			return [
				{"entity": label, "softmax": prob, "word": token, "start": int(start), "end": int(end)}
				for label, prob, token, (start, end) in zip(labels, probs, tokens, offsets)
				if label != "O"
			]
		else:
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
		prev_entity_type = "O-nothing"
		prev_entity_end_idx = -100

		for i, token_prediction in enumerate(token_predictions):
			if skip:
				skip -= 1
				continue

			prefix, label = token_prediction["entity"].split("-", 1)
			if self.model_name in ["sultan/BioM-ALBERT-xxlarge", "sultan/BioM-ALBERT-xxlarge-PMC"]:
				word = token_prediction["word"].replace("▁", "")
			else:
				word = token_prediction["word"].replace("##", "")

			if (
				prefix == "B"
				and prev_entity_type != token_prediction["entity"]
				and not (
					prev_entity_type.split("-")[1] == token_prediction["entity"].split("-")[1]
					and token_prediction["start"] in [prev_entity_end_idx + 1, prev_entity_end_idx + 2]
				)
			):
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
				prev_entity_type = token_prediction["entity"]
				prev_entity_end_idx = token_prediction["end"]
				if skip:
					continue

			elif (
				prefix == "I"
				or prev_entity_type == token_prediction["entity"]
				or (
					prev_entity_type.split("-")[1] == token_prediction["entity"].split("-")[1]
					and token_prediction["start"] in [prev_entity_end_idx + 1, prev_entity_end_idx + 2]
				)
			):
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
				prev_entity_type = token_prediction["entity"]
				prev_entity_end_idx = token_prediction["end"]

		if current_entity:
			merged.append(current_entity)

		return merged

	def _adjust_casing(self, entity_predictions, true_text):
		for entity in entity_predictions:
			entity["text_span"] = true_text[entity["start_idx"] : entity["end_idx"] + 1]

			quote_indices = [index for index, char in enumerate(entity["text_span"]) if char == '"']
			if len(quote_indices) == 1:
				if quote_indices[0] == 0:
					entity["text_span"] = entity["text_span"][1:]
					entity["start_idx"] += 1

		return entity_predictions


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Load configuration from a YAML file or directory of YAML files.")
	parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file or directory")
	args = parser.parse_args()

	config_path = args.config

	if os.path.isfile(config_path):
		config = load_config(config_path)
		dataset_dir_name = make_dataset_dir_name(config)
		ner_inference = NERInference(
			test_data_path=os.path.join("data", "Test_Data", "articles_test.json"),
			model_name_path=os.path.join("models", dataset_dir_name),
			model_name=config["model_name"],
			model_type=config["model_type"],
			save_path=os.path.join("data_inference_results", "_ner_top_17_dev", f"{dataset_dir_name}.json"),
			remove_html=config["remove_html"],
		)
		ner_inference.perform_inference()

	elif os.path.isdir(config_path):
		model_name_path = []
		model_name = []
		model_type = []
		for config_name in os.listdir(config_path):
			full_path = os.path.join(config_path, config_name)
			if os.path.isfile(full_path) and config_name.endswith((".yml", ".yaml")):
				config = load_config(full_path)
				dataset_dir_name = make_dataset_dir_name(config)
				model_name_path.append(os.path.join("models", dataset_dir_name))
				model_name.append(config["model_name"])
				model_type.append(config["model_type"])

		# for token_strategy, entity_type_strategy in itertools.product(
		# 	["union", "majority", "intersection"], ["majority", "softmax_sum"]
		# ):
		for token_strategy, entity_type_strategy in itertools.product(["majority"], ["majority"]):
			strategy = f"{token_strategy}.{entity_type_strategy}"
			ner_inference = NERInference(
				test_data_path=os.path.join("data", "Articles", "json_format", "articles_dev.json"),
				model_name_path=model_name_path,
				model_name=model_name,
				model_type=model_type,
				save_path=os.path.join(
					"data_inference_results",
					"3t_3e",
					f"t_ensemble3_{strategy}.json",
				),
				remove_html=config["remove_html"],
				ensemble_strategy=strategy,
			)
			ner_inference.perform_ensemble_inference()
	else:
		raise ValueError(f"Provided config path '{config_path}' is neither a file nor a directory.")
