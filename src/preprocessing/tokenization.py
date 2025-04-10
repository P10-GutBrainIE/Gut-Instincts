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
		save_filename: str,
		tokenizer: AutoTokenizer,
		max_length: int = 512,
	):
		self.datasets = datasets
		self.save_filename = save_filename
		self.tokenizer = tokenizer
		self.max_length = max_length
		_, self.label2id, _ = load_bio_labels()

	def process_files(self):
		"""
		Load JSON files of papers and process each paper.

		This method reads the JSON files, processes each paper's content, and saves the processed data to a pickle file.
		"""
		logger.info("Starting to process files...")
		for data in self.datasets:
			all_data = []
			for _, content in data.items():
				processed_data = self._process_paper(content)
				all_data.extend(processed_data)

		logger.info("Files processed")

		self._save_to_pickle(all_data)

	def _process_paper(self, content):
		"""
		Process a single paper's content for both title and abstract.

		Args:
		    content (dict): Dictionary containing the paper's metadata and entities.

		Returns:
		    list[dict]: List of dictionaries with the processed data for title and abstract.
		"""
		processed = []
		metadata = content.get("metadata", {})
		title = metadata.get("title", "")
		abstract = metadata.get("abstract", "")
		entities = content.get("entities", [])

		for section, text in [("title", title), ("abstract", abstract)]:
			logger.debug(f"Processing {section} with length {len(text)}")
			bio_tag_ids, input_ids, attention_mask = self._tokenize_with_bio(text, entities, section)
			processed.append(
				{
					"labels": bio_tag_ids,
					"input_ids": input_ids,
					"attention_mask": attention_mask,
				}
			)
		return processed

	def _tokenize_with_bio(self, text, entities, section):
		"""
		Tokenize a given text using the fast tokenizer (with offset mapping) and assign BIO tags.

		Args:
		    text (str): The text to be tokenized.
		    entities (list[dict]): List of entities with their text spans and labels.
		    section (str): The section of the text (e.g., "title" or "abstract").

		Returns:
		    tuple: A tuple containing tokens, BIO tag IDs, input IDs, and attention mask.
		"""
		logger.debug(f"Tokenizing {section} text with {len(entities)} entities.")
		encoding = self.tokenizer(
			text, return_offsets_mapping=True, truncation=True, max_length=self.max_length, padding="max_length"
		)
		tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
		offsets = encoding["offset_mapping"]

		bio_tags = ["O"] * len(tokens)

		for entity in entities:
			if entity.get("location") != section:
				continue

			entity_text = entity["text_span"]
			entity_label = entity["label"]

			start_index = text.find(entity_text)
			if start_index == -1:
				continue
			end_index = start_index + len(entity_text)

			first_token_assigned = False
			for i, (token_start, token_end) in enumerate(offsets):
				if token_start is None or token_end is None or (token_start, token_end) == (0, 0):
					continue

				if token_end > start_index and token_start < end_index:
					if not first_token_assigned:
						bio_tags[i] = f"B-{entity_label}"
						first_token_assigned = True
					else:
						bio_tags[i] = f"I-{entity_label}"

		bio_tag_ids = []
		for offset, tag in zip(offsets, bio_tags):
			if offset is None or offset == (0, 0):
				bio_tag_ids.append(-100)
			else:
				bio_tag_ids.append(self.label2id.get(tag, 0))
		return bio_tag_ids, encoding["input_ids"], encoding["attention_mask"]

	def _save_to_pickle(self, data):
		"""
		Save the processed data to pickle files.

		Args:
		    training_data (list[dict]): List of training data.
		    validation_data (list[dict]): List of validation data.
		"""
		logger.info(f"Saving processed data to data_preprocessed/{self.save_filename}...")
		os.makedirs("data_preprocessed", exist_ok=True)

		with open(os.path.join("data_preprocessed", self.save_filename), "wb") as f:
			pickle.dump(data, f)
			logger.info(f"BIO tokenized data saved to {self.save_filename}. Data size: {len(data)}")
