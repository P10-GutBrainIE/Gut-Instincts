import os
import pickle
import logging
from transformers import AutoTokenizer
from utils.utils import load_bio_labels, load_relation_labels, load_json_data
from preprocessing.remove_html import remove_html_tags
import random

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
		self.save_filename = save_filename
		self.max_length = max_length
		self.concatenate_title_abstract = concatenate_title_abstract
		self.subtask = subtask
		self.negative_sample_multiplier = negative_sample_multiplier
		_, self.relation2id, _ = load_relation_labels()

		# Register entity marker tokens and resize embeddings if applicable
		self.tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})

		# Resize model embeddings if the tokenizer has been attached to a model
		# if hasattr(self.tokenizer, "model_max_length") and hasattr(self.tokenizer, "model"):
		# 	try:
		# 		self.tokenizer.model.resize_token_embeddings(len(self.tokenizer))
		# 	except Exception:
		# 		pass  # You can log a warning here if you want

	def process_files(self):
		"""
		Load JSON files of papers and process each paper for relation classification.
		"""
		logger.info("Starting to process files for relation classification...")
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
		# processed = []
		positive_samples = []
		negative_samples = []

		title = content["metadata"]["title"]
		abstract = content["metadata"]["abstract"]
		offset = len(title) + 1 if self.concatenate_title_abstract else 0
		full_text = f"{title} {abstract}" if self.concatenate_title_abstract else abstract

		entities = content["entities"]
		gold_relations = {}
		for relation in content["relations"]:
			subject_key = (relation["subject_start_idx"], relation["subject_end_idx"], relation["subject_location"])
			object_key = (relation["object_start_idx"], relation["object_end_idx"], relation["object_location"])
			gold_relations[(subject_key, object_key)] = relation["predicate"]

		for i, subject_entity in enumerate(entities):
			for j, object_entity in enumerate(entities):
				if i == j:
					continue  # skip same entity

				subject_key = (subject_entity["start_idx"], subject_entity["end_idx"], subject_entity["location"])
				object_key = (object_entity["start_idx"], object_entity["end_idx"], object_entity["location"])

				if (subject_key, object_key) in gold_relations:
					if self.subtask == "6.2.1":
						label_id = 1
					else:
						relation_label = gold_relations[(subject_key, object_key)]
						if relation_label not in self.relation2id:
							self.relation2id[relation_label] = len(self.relation2id)
						label_id = self.relation2id[relation_label]
				else:
					if self.subtask == "6.2.1":
						label_id = 0
					else:
						label_id = self.relation2id["no relation"]

				# Adjust indices for title vs abstract
				subject_start = (
					subject_entity["start_idx"] + offset
					if subject_entity["location"] == "abstract"
					else subject_entity["start_idx"]
				)
				subject_end = (
					subject_entity["end_idx"] + offset + 1
					if subject_entity["location"] == "abstract"
					else subject_entity["end_idx"]
				)

				object_start = (
					object_entity["start_idx"] + offset
					if object_entity["location"] == "abstract"
					else object_entity["start_idx"]
				)
				object_end = (
					object_entity["end_idx"] + offset + 1
					if object_entity["location"] == "abstract"
					else object_entity["end_idx"]
				)

				input_ids, attention_mask = self._tokenize_with_entity_markers(
					full_text,
					subj={"start_idx": subject_start, "end_idx": subject_end},
					obj={"start_idx": object_start, "end_idx": object_end},
				)

				sample = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label_id}

				if dataset_weight:
					sample["weight"] = dataset_weight

				if (subject_key, object_key) in gold_relations:
					positive_samples.append(sample)
				else:
					negative_samples.append(sample)

		num_negatives_to_keep = min(len(negative_samples), self.negative_sample_multiplier * len(positive_samples))

		if num_negatives_to_keep < len(negative_samples):
			negative_samples = random.sample(negative_samples, num_negatives_to_keep)

		processed = positive_samples + negative_samples
		#random.shuffle(processed)

		return processed

	def _tokenize_with_entity_markers(self, text, subj, obj):
		# Ensure order is correct
		s_start, s_end = subj["start_idx"], subj["end_idx"]
		o_start, o_end = obj["start_idx"], obj["end_idx"]

		if s_start < o_start:
			spans = [(s_start, s_end, "[E1]", "[/E1]"), (o_start, o_end, "[E2]", "[/E2]")]
		else:
			spans = [(o_start, o_end, "[E2]", "[/E2]"), (s_start, s_end, "[E1]", "[/E1]")]

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


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Train")
	platinum_data = load_json_data(os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"))
	platinum_data = remove_html_tags(data=platinum_data)

	tokenizer = AutoTokenizer.from_pretrained(
		"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", use_fast=True
	)
	re_tokenizer = RelationTokenizer(
		datasets=[platinum_data],
		save_filename=os.path.join("test.pkl"),
		tokenizer=tokenizer,
		subtask="2.6.1"
	)
	re_tokenizer.process_files()
