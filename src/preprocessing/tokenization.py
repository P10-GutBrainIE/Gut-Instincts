import os
import pickle
import logging
from transformers import AutoTokenizer
from utils.utils import load_bio_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BIOTokenizer:
	def __init__(
		self,
		datasets: list[dict],
		dataset_weigths: list,
		tokenizer: AutoTokenizer,
		save_filename: str = None,
		max_length: int = 512,
		concatenate_title_abstract: bool = True,
	):
		self.datasets = datasets
		self.dataset_weigths = dataset_weigths
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
		for data, dataset_weight in zip(self.datasets, self.dataset_weigths):
			for _, content in data.items():
				processed_data = self._process_paper(content, dataset_weight)
				all_data.extend(processed_data)

		logger.info("Files processed")

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
			processed.append(
				{
					"labels": bio_tag_ids,
					"input_ids": input_ids,
					"attention_mask": attention_mask,
					"weight": dataset_weight,
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
		os.makedirs("data_preprocessed", exist_ok=True)

		with open(os.path.join("data_preprocessed", self.save_filename), "wb") as f:
			pickle.dump(data, f)
			logger.info(f"BIO tokenized data saved to {self.save_filename}. Data size: {len(data)}")


if __name__ == "__main__":
	from utils.utils import load_json_data, load_bio_labels

	data = load_json_data(os.path.join("tests", "test_data", "tokenization.json"))
	# data = load_json_data(os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))
	tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large", use_fast=True)
	bio_tokenizer = BIOTokenizer(datasets=[data], tokenizer=tokenizer, max_length=32, concatenate_title_abstract=True)
	processed_data = bio_tokenizer.process_files()
	print(processed_data)

	# print(processed_data[0])

	print(tokenizer.convert_ids_to_tokens(processed_data[0]["input_ids"]))

	# _, _, id2label = load_bio_labels()
	# print(id2label)
