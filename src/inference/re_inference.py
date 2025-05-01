import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.utils import load_json_data, load_relation_labels


class REInference:
	def __init__(
		self,
		test_data_path: str,
		model_name: str,
		model_type: str,
		subtask: str,
		model_name_path: str = None,
		save_path: str = None,
		validation_model=None,
	):
		self.test_data = load_json_data(test_data_path)
		_, _, self.id2label = load_relation_labels()
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, max_length=512, truncation=True)
		self.model_type = model_type
		self.validation_model = validation_model
		self.subtask = subtask
		self.save_path = save_path

		self.tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
		self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
		self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

		if validation_model:
			self.model = validation_model
		else:
			from architectures.bert_with_entity_start import BertForREWithEntityStart

			self.model = BertForREWithEntityStart(model_name=model_name, subtask=self.subtask)
			state_dict = torch.load(os.path.join(model_name_path, "pytorch_model.bin"), map_location="cpu")
			self.model.load_state_dict(state_dict)
			self.model.eval()

	def _subtask_string(self, subtask):
		if subtask == "6.2.1":
			return "binary_tag_based_relations"
		elif subtask == "6.2.2":
			return "ternary_tag_based_relations"
		elif subtask == "6.2.3":
			return "ternary_mention_based_relations"
		else:
			raise ValueError(f"Unknown subtask type: {subtask}")

	def perform_inference(self):
		result = {}

		for paper_id, content in tqdm(
			self.test_data.items(), total=len(self.test_data), desc=f"RE inference ({self.subtask})"
		):
			predictions = self._re_pipeline(content)

			result[paper_id] = {f"{self._subtask_string(self.subtask)}": predictions}

		if self.save_path:
			os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
			with open(self.save_path, "w") as f:
				json.dump(result, f, indent=4)
		else:
			return result

	def _re_pipeline(self, content):
		predictions = set()

		offset = len(content["metadata"]["title"]) + 1
		full_text = f"{content['metadata']['title']} {content['metadata']['abstract']}"

		entity_combinations = [(a, b) for a in content["entities"] for b in content["entities"] if a != b]
		for subject, object in entity_combinations:
			input_ids, attention_mask = self._tokenize_with_entity_markers(
				full_text,
				subject={
					"start_idx": subject["start_idx"] + offset
					if subject["location"] == "abstract"
					else subject["start_idx"],
					"end_idx": subject["end_idx"] + offset + 1
					if subject["location"] == "abstract"
					else subject["end_idx"] + 1,
				},
				object={
					"start_idx": object["start_idx"] + offset
					if object["location"] == "abstract"
					else object["start_idx"],
					"end_idx": object["end_idx"] + offset + 1
					if object["location"] == "abstract"
					else object["end_idx"] + 1,
				},
			)

			prediction = self.model.predict(input_ids=input_ids, attention_mask=attention_mask)
			if self.subtask == "6.2.1":
				if prediction:
					predictions.add({"subject_label": subject["label"], "object_label": object["label"]})
			elif self.subtask in ["6.2.2", "6.2.3"]:
				predicate = self.id2label[prediction]
				if predicate != "no relation":
					if self.subtask == "6.2.2":
						predictions.add(
							{"subject_label": subject["label"], "predicate": predicate, "object_label": object["label"]}
						)
					elif self.subtask == "6.2.3":
						predictions.add(
							{
								"subject_text_span": subject["text_span"],
								"subject_label": subject["label"],
								"predicate": predicate,
								"object_text_span": object["text_span"],
								"object_label": object["label"],
							}
						)

		return list(predictions)

	def _tokenize_with_entity_markers(self, text, subject, object):
		if subject["start_idx"] < object["start_idx"]:
			spans = [
				(subject["start_idx"], subject["end_idx"], "[E1]", "[/E1]"),
				(object["start_idx"], object["end_idx"], "[E2]", "[/E2]"),
			]
		else:
			spans = [
				(object["start_idx"], object["end_idx"], "[E2]", "[/E2]"),
				(subject["start_idx"], subject["end_idx"], "[E1]", "[/E1]"),
			]

		marked_text = ""
		last_idx = 0
		for start, end, pre_tag, post_tag in spans:
			marked_text += text[last_idx:start]
			marked_text += f"{pre_tag}{text[start:end]}{post_tag}"
			last_idx = end
		marked_text += text[last_idx:]

		result = self.tokenizer(
			marked_text,
			return_attention_mask=True,
			truncation=True,
			padding="max_length",
			max_length=512,
		)

		device = next(self.model.parameters()).device
		input_ids = result["input_ids"].to(device)
		attention_mask = result["attention_mask"].to(device)

		return input_ids, attention_mask
