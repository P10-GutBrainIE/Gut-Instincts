import os
import pickle
import random
import logging
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.utils import load_bio_labels, load_relation_labels, load_json_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BIOTokenizer:
	def __init__(
		self,
		datasets: list[list[dict]],
		tokenizer,
		save_filename: str = None,
		dataset_weights: list = None,
		max_length: int = 512,
		concatenate_title_abstract: bool = True,
	):
		self.datasets = datasets
		self.dataset_weights = dataset_weights
		self.tokenizer = tokenizer
		self.save_filename = save_filename
		self.max_length = max_length
		self.concatenate_title_abstract = concatenate_title_abstract
		_, self.label2id, _ = load_bio_labels()

	def process_files(self):
		"""
		Load JSON files of papers and process each paper.

		This method reads the JSON files, processes each paper's content, and either returns or saves the processed data to a pickle file.
		"""
		logger.info("Starting to process files...")
		all_data = []

		if self.dataset_weights:
			for data, dataset_weight in zip(self.datasets, self.dataset_weights):
				for _, content in data.items():
					processed_data = self._process_paper(content, dataset_weight)
					all_data.extend(processed_data)
			logger.info(f"Datasets with weights: {self.dataset_weights} processed")
		else:
			for data in self.datasets:
				for _, content in data.items():
					processed_data = self._process_paper(content, self.dataset_weights)
					all_data.extend(processed_data)
			logger.info("Datasets processed")

		if self.save_filename:
			self._save_to_pickle(all_data)
		else:
			return all_data

	def _process_paper(self, content, dataset_weight):
		"""
		Process a single paper's content for both title and abstract.

		Args:
		    content (dict): Dictionary containing the paper's metadata and entities.

		Returns:
		    list[dict]: List of dictionaries with the processed data for title and abstract.
		"""
		processed = []

		text_lst, entities_lst = self._extract_entities(content)

		for text, entities in zip(text_lst, entities_lst):
			bio_tag_ids, input_ids, attention_mask = self._tokenize_with_bio(text, entities)
			if dataset_weight:
				processed.append(
					{
						"labels": bio_tag_ids,
						"input_ids": input_ids,
						"attention_mask": attention_mask,
						"weight": dataset_weight,
					}
				)
			else:
				processed.append(
					{
						"labels": bio_tag_ids,
						"input_ids": input_ids,
						"attention_mask": attention_mask,
					}
				)

		return processed

	def _extract_entities(self, content):
		text_lst = []
		entities_lst = []
		if self.concatenate_title_abstract:
			entities_lst.append(
				[
					{
						**entity,
						"start_idx": entity["start_idx"] + len(content["metadata"]["title"]) + 1,
						"end_idx": entity["end_idx"] + len(content["metadata"]["title"]) + 1,
					}
					if entity["location"] == "abstract"
					else entity
					for entity in content["entities"]
				]
			)
			text_lst.append(f"{content['metadata']['title']} {content['metadata']['abstract']}")
		else:
			for section in ["title", "abstract"]:
				entities_lst.append([entity for entity in content["entities"] if entity["location"] == section])
				text_lst.append(content["metadata"][section])

		return text_lst, entities_lst

	def _tokenize_with_bio(self, text, entities):
		"""
		Tokenize a given text using the fast tokenizer (with offset mapping) and assign BIO tags.

		Args:
		    text (str): The text to be tokenized.
		    entities (list[dict]): List of entities with their text spans and labels.
		    section (str): The section of the text (e.g., "title" or "abstract").

		Returns:
		    tuple: A tuple containing tokens, BIO tag IDs, input IDs, and attention mask.
		"""
		encoding = self.tokenizer(
			text, return_offsets_mapping=True, truncation=True, max_length=self.max_length, padding="max_length"
		)
		tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
		offsets = encoding["offset_mapping"]

		bio_tags = ["O"] * len(tokens)

		for entity in entities:
			first_token_assigned = False
			for i, (token_start, token_end) in enumerate(offsets):
				if i >= 1 and (token_start, token_end) == (0, 0):
					break

				if token_start >= entity["start_idx"] and token_end <= entity["end_idx"] + 1:
					if not first_token_assigned:
						bio_tags[i] = f"B-{entity['label']}"
						first_token_assigned = True
					else:
						bio_tags[i] = f"I-{entity['label']}"

		bio_tag_ids = []
		for offset, tag in zip(offsets, bio_tags):
			if offset == (0, 0):
				bio_tag_ids.append(-100)
			else:
				bio_tag_ids.append(self.label2id[tag])
		return bio_tag_ids, encoding["input_ids"], encoding["attention_mask"]

	def _save_to_pickle(self, data):
		"""
		Save the processed data to pickle files.

		Args:
		    Data (list[dict]): List of data.
		"""
		os.makedirs(os.path.join("data_preprocessed", os.path.dirname(self.save_filename)), exist_ok=True)

		with open(os.path.join("data_preprocessed", self.save_filename), "wb") as f:
			pickle.dump(data, f)
			logger.info(f"BIO tokenized data saved to {self.save_filename}. Data size: {len(data)}")


class RelationTokenizer:
	def __init__(
		self,
		datasets: list[dict],
		tokenizer: AutoTokenizer,
		save_filename: str = None,
		dataset_weights: list = None,
		max_length: int = 512,
		concatenate_title_abstract: bool = True,
		subtask: str = None,
		negative_sample_multiplier: int = 1,
	):
		self.datasets = datasets
		self.dataset_weights = dataset_weights
		self.tokenizer = tokenizer
		self.tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
		self.save_filename = save_filename
		self.max_length = max_length
		self.concatenate_title_abstract = concatenate_title_abstract
		self.subtask = subtask
		self.negative_sample_multiplier = negative_sample_multiplier
		_, self.relation2id, _ = load_relation_labels()

	def process_files(self):
		"""
		Load JSON files of papers and process each paper for relation classification.
		"""
		logger.info("Starting to process files for relation classification...")
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
					processed_data = self._process_paper(content, dataset_weight)
					all_data.extend(processed_data)
			logger.info("Datasets processed")

		if self.save_filename:
			self._save_to_pickle(all_data)
		else:
			return all_data

	def _process_paper(self, content, dataset_weight):
		samples = []

		offset = len(content["metadata"]["title"]) + 1 if self.concatenate_title_abstract else 0
		full_text = (
			f"{content['metadata']['title']} {content['metadata']['abstract']}"
			if self.concatenate_title_abstract
			else content["metadata"]["abstract"]
		)

		for relation in content["relations"]:
			input_ids, attention_mask = self._tokenize_with_entity_markers(
				full_text,
				subject={
					"start_idx": relation["subject_start_idx"] + offset
					if relation["subject_location"] == "abstract"
					else relation["subject_start_idx"],
					"end_idx": relation["subject_end_idx"] + offset + 1
					if relation["subject_location"] == "abstract"
					else relation["subject_end_idx"] + 1,
				},
				object={
					"start_idx": relation["object_start_idx"] + offset
					if relation["object_location"] == "abstract"
					else relation["object_start_idx"],
					"end_idx": relation["object_end_idx"] + offset + 1
					if relation["object_location"] == "abstract"
					else relation["object_end_idx"] + 1,
				},
			)
			if dataset_weight:
				samples.append(
					{
						"input_ids": input_ids,
						"attention_mask": attention_mask,
						"labels": relation["predicate"] if self.subtask in ["6.2.2", "6.2.3"] else 1,
						"subject_label": relation["subject_label"],
						"object_label": relation["object_label"],
						"subject_text_span": relation["subject_text_span"],
						"object_text_span": relation["object_text_span"],
						"weight": dataset_weight,
					}
				)
			else:
				samples.append(
					{
						"input_ids": input_ids,
						"attention_mask": attention_mask,
						"labels": relation["predicate"] if self.subtask in ["6.2.2", "6.2.3"] else 1,
						"subject_label": relation["subject_label"],
						"object_label": relation["object_label"],
						"subject_text_span": relation["subject_text_span"],
						"object_text_span": relation["object_text_span"],
					}
				)

		entity_combinations = [(a, b) for a in content["entities"] for b in content["entities"] if a != b]
		random.shuffle(entity_combinations)
		if len(samples) * self.negative_sample_multiplier > len(entity_combinations):
			number_negative_samples = len(entity_combinations)
		else:
			number_negative_samples = len(samples) * self.negative_sample_multiplier
		entity_combinations[:number_negative_samples]

		for ent_a, ent_b in entity_combinations:
			input_ids, attention_mask = self._tokenize_with_entity_markers(
				full_text,
				subject={
					"start_idx": ent_a["start_idx"] + offset if ent_a["location"] == "abstract" else ent_a["start_idx"],
					"end_idx": ent_a["end_idx"] + offset + 1
					if ent_a["location"] == "abstract"
					else ent_a["end_idx"] + 1,
				},
				object={
					"start_idx": ent_b["start_idx"] + offset if ent_b["location"] == "abstract" else ent_b["start_idx"],
					"end_idx": ent_b["end_idx"] + offset + 1
					if ent_b["location"] == "abstract"
					else ent_b["end_idx"] + 1,
				},
			)
			if dataset_weight:
				samples.append(
					{
						"input_ids": input_ids,
						"attention_mask": attention_mask,
						"labels": 0,
						"subject_label": ent_a["label"],
						"object_label": ent_b["label"],
						"subject_text_span": ent_a["text_span"],
						"object_text_span": ent_b["text_span"],
						"weight": dataset_weight,
					}
				)
			else:
				samples.append(
					{
						"input_ids": input_ids,
						"attention_mask": attention_mask,
						"labels": 0,
						"subject_label": ent_a["label"],
						"object_label": ent_b["label"],
						"subject_text_span": ent_a["text_span"],
						"object_text_span": ent_b["text_span"],
					}
				)

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

		return encoding["input_ids"], encoding["attention_mask"]

	def _save_to_pickle(self, data):
		os.makedirs(os.path.join("data_preprocessed", os.path.dirname(self.save_filename)), exist_ok=True)
		with open(os.path.join("data_preprocessed", self.save_filename), "wb") as f:
			pickle.dump(data, f)
			logger.info(f"Relation tokenized data saved to {self.save_filename}. Data size: {len(data)}")


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Dev")
	platinum_data = load_json_data(os.path.join(shared_path, "json_format", "dev.json"))

	tokenizer = AutoTokenizer.from_pretrained(
		"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", use_fast=True
	)
	re_tokenizer = RelationTokenizer(
		dataset_weights=[1], datasets=[platinum_data], tokenizer=tokenizer, subtask="6.2.1"
	)
	processed = re_tokenizer.process_files()
	for item in processed[:1]:
		print(item)
		print(tokenizer.convert_ids_to_tokens(item["input_ids"][0]))
