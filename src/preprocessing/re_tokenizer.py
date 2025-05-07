import os
import pickle
import random
import logging
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.utils import load_relation_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RelationTokenizer:
	def __init__(
		self,
		datasets: list[dict],
		tokenizer: AutoTokenizer,
		save_filename: str = None,
		dataset_weights: list = None,
		max_length: int = 512,
		subtask: str = None,
		negative_sample_multiplier: int = 1,
	):
		self.datasets = datasets
		self.dataset_weights = dataset_weights
		self.tokenizer = tokenizer
		self.tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
		self.save_filename = save_filename
		self.max_length = max_length
		self.subtask = subtask
		self.negative_sample_multiplier = negative_sample_multiplier
		_, self.relation2id, _ = load_relation_labels()

	def process_files(self):
		"""
		Load JSON files of papers and process each paper for relation extraction.
		"""
		logger.info("Starting to process files for relation extraction...")
		all_data = []

		if self.dataset_weights:
			for data, dataset_weight in zip(self.datasets, self.dataset_weights):
				for _, content in tqdm(data.items(), total=len(data.items()), desc="Preprocessing"):
					processed_data = self._process_paper(content, dataset_weight)
					all_data.extend(processed_data)
			logger.info(f"Datasets with weights: {self.dataset_weights} processed")
		else:
			for data in self.datasets:
				for _, content in tqdm(data.items(), total=len(data.items()), desc="Preprocessing"):
					processed_data = self._process_paper(content)
					all_data.extend(processed_data)
			logger.info("Datasets processed")

		if self.save_filename:
			self._save_to_pickle(all_data)
		else:
			return all_data

	def _process_paper(self, content, dataset_weight=None):
		samples = []
		positive_samples_lookup = []

		offset = len(content["metadata"]["title"]) + 1
		full_text = f"{content['metadata']['title']} {content['metadata']['abstract']}"

		for relation in content["relations"]:
			subject = {
				"start_idx": relation["subject_start_idx"] + offset
				if relation["subject_location"] == "abstract"
				else relation["subject_start_idx"],
				"end_idx": relation["subject_end_idx"] + offset + 1
				if relation["subject_location"] == "abstract"
				else relation["subject_end_idx"] + 1,
			}
			object = {
				"start_idx": relation["object_start_idx"] + offset
				if relation["object_location"] == "abstract"
				else relation["object_start_idx"],
				"end_idx": relation["object_end_idx"] + offset + 1
				if relation["object_location"] == "abstract"
				else relation["object_end_idx"] + 1,
			}
			positive_samples_lookup.append((subject, object))
			input_ids, attention_mask = self._tokenize_with_entity_markers(full_text, subject, object)
			if dataset_weight:
				samples.append(
					{
						"input_ids": input_ids,
						"attention_mask": attention_mask,
						"labels": self.relation2id[relation["predicate"]] if self.subtask in ["6.2.2", "6.2.3"] else 1,
						"weight": dataset_weight,
					}
				)
			else:
				samples.append(
					{
						"input_ids": input_ids,
						"attention_mask": attention_mask,
						"labels": self.relation2id[relation["predicate"]] if self.subtask in ["6.2.2", "6.2.3"] else 1,
					}
				)

		entity_combinations = [(a, b) for a in content["entities"] for b in content["entities"] if a != b]
		random.shuffle(entity_combinations)
		if len(samples) * self.negative_sample_multiplier > len(entity_combinations):
			number_negative_samples = len(entity_combinations)
		else:
			number_negative_samples = len(samples) * self.negative_sample_multiplier

		negative_samples_counter = 0
		for ent_a, ent_b in entity_combinations:
			if negative_samples_counter == number_negative_samples:
				break
			subject = {
				"start_idx": ent_a["start_idx"] + offset if ent_a["location"] == "abstract" else ent_a["start_idx"],
				"end_idx": ent_a["end_idx"] + offset + 1 if ent_a["location"] == "abstract" else ent_a["end_idx"] + 1,
			}
			object = {
				"start_idx": ent_b["start_idx"] + offset if ent_b["location"] == "abstract" else ent_b["start_idx"],
				"end_idx": ent_b["end_idx"] + offset + 1 if ent_b["location"] == "abstract" else ent_b["end_idx"] + 1,
			}
			if (subject, object) in positive_samples_lookup:
				continue
			else:
				input_ids, attention_mask = self._tokenize_with_entity_markers(full_text, subject, object)
				if dataset_weight:
					samples.append(
						{
							"input_ids": input_ids,
							"attention_mask": attention_mask,
							"labels": 0,
							"weight": dataset_weight,
						}
					)
				else:
					samples.append(
						{
							"input_ids": input_ids,
							"attention_mask": attention_mask,
							"labels": 0,
						}
					)
				negative_samples_counter += 1

		return samples

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

		encoding = self.tokenizer(
			marked_text,
			return_attention_mask=True,
			truncation=True,
			padding="max_length",
			max_length=self.max_length,
			return_tensors="pt",
		)

		return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

	def _save_to_pickle(self, data):
		os.makedirs(os.path.join("data_preprocessed", os.path.dirname(self.save_filename)), exist_ok=True)
		with open(os.path.join("data_preprocessed", self.save_filename), "wb") as f:
			pickle.dump(data, f)
			logger.info(f"Relation tokenized data saved to {self.save_filename}. Data size: {len(data)}")
