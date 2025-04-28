import os
import pickle
import logging
from utils.utils import load_bio_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BIOTokenizer:
	"""
	Tokenizer for converting biomedical text and entities into tokenized data with BIO-tag annotations.

	This class supports processing multiple datasets, handling dataset weights, concatenating title and abstract,
	and saving the resulting tokenized data in pickle format.
	"""

	def __init__(
		self,
		datasets: list[list[dict]],
		tokenizer,
		save_filename: str = None,
		dataset_weights: list = None,
		max_length: int = 512,
		concatenate_title_abstract: bool = True,
	):
		"""
		Initialize the BIOTokenizer.

		Args:
		    datasets (list[list[dict]]): List of datasets to process.
		    tokenizer: Tokenizer object (e.g., from Hugging Face).
		    save_filename (str, optional): File name to save processed data as pickle.
		    dataset_weights (list, optional): Weights for each dataset (if any).
		    max_length (int, optional): Maximum sequence length for tokenization.
		    concatenate_title_abstract (bool, optional): Whether to concatenate title and abstract.
		"""
		self.datasets = datasets
		self.dataset_weights = dataset_weights
		self.tokenizer = tokenizer
		self.save_filename = save_filename
		self.max_length = max_length
		self.concatenate_title_abstract = concatenate_title_abstract
		_, self.label2id, _ = load_bio_labels()

	def process_files(self):
		"""
		Load and process datasets, converting papers to tokenized BIO-format data.

		Returns:
		    list[dict] or None: List of processed tokenized examples, or None if saved to file.
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
		Process a single paper's content (title and abstract) and tokenize with BIO tags.

		Args:
		    content (dict): Paper metadata and entities.
		    dataset_weight: Optional weight for the dataset.

		Returns:
		    list[dict]: List of processed data for title and/or abstract.
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
		"""
		Extracts text and associated entity annotations for tokenization.

		Args:
		    content (dict): Paper content with metadata and entities.

		Returns:
		    tuple: (list of texts, list of corresponding entity lists)
		"""
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
		Tokenize text and assign BIO tags to each token based on entity spans.

		Args:
		    text (str): The text to be tokenized.
		    entities (list[dict]): List of entity annotations.

		Returns:
		    tuple: BIO tag IDs, input IDs, and attention mask.
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
		Save processed data to a pickle file.

		Args:
		    data (list[dict]): List of processed data samples.
		"""
		os.makedirs(os.path.join("data_preprocessed", os.path.dirname(self.save_filename)), exist_ok=True)

		with open(os.path.join("data_preprocessed", self.save_filename), "wb") as f:
			pickle.dump(data, f)
			logger.info(f"BIO tokenized data saved to {self.save_filename}. Data size: {len(data)}")
