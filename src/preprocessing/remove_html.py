import json
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def remove_html_tags(data: dict, replacement_char: str = "$", save_filename: str = None) -> None | dict:
	"""
	Remove HTML tags and adjust entity text span indices.

	This function processes one or more JSON files whose paths are specified in a list of dictionaries.
	Each dictionary should contain a single key-value pair where the key is an identifier (e.g., "gold")
	and the value is the file path to a JSON file containing text data.

	The processing steps are as follows:
	  1. For each JSON file:
	     - Load the file content.
	     - For each content item, replace every character in HTML tags within the "title" and "abstract" fields
	       of the "metadata" and within the "text_span" field of each entity with a replacement character (e.g., "$").
	  2. Adjust the entity indices:
	     - For each occurrence of the replacement character in the "title" and "abstract" fields,
	       update the `start_idx` and `end_idx` for entities that correspond to that text, compensating for the added
	       characters from the initial replacement.
	  3. Cleanup:
	     - Remove consecutive sequences of the replacement character from the text fields.

	Parameters:
	    file_paths_dict (list[dict]): A list of dictionaries, each mapping a file identifier to its JSON file path.
	                                  Example: [{"gold": "data/Annotations/Train/json_format/gold.json"}]
	    replacement_char (str): The character that replaces the HTML tags in the algorithm.
	    save_data (bool): If True, the processed JSON data is saved to a "data_preprocessed" directory.
	                      If False, the processed data is returned as a dictionary. Default is False.

	Returns:
	    dict[str, list] or None: If save_data is False, returns a dictionary where each key corresponds to the file identifier and
	                  the value is the processed JSON content. If save_data is True, the processed files are saved
	                  to disk and the function returns None.
	"""
	logger.info("Starting HTML removal...")
	try:
		for _, content in data.items():
			for text_type in ["title", "abstract"]:
				content["metadata"][text_type] = re.sub(
					r"</?[^>]+>", lambda m: replacement_char * len(m.group(0)), content["metadata"][text_type]
				)

			for type in ["entities", "relations", "ternary_mention_based_relations"]:
				for object in content[type]:
					if type == "entities":
						object["text_span"] = re.sub(
							r"</?[^>]+>", lambda m: replacement_char * len(m.group(0)), object["text_span"]
						)
					else:
						object["subject_text_span"] = re.sub(
							r"</?[^>]+>", lambda m: replacement_char * len(m.group(0)), object["subject_text_span"]
						)
						object["object_text_span"] = re.sub(
							r"</?[^>]+>", lambda m: replacement_char * len(m.group(0)), object["object_text_span"]
						)

		for _, content in data.items():
			for text_type in ["title", "abstract"]:
				replacement_char_counter = 0
				for i, c in enumerate(content["metadata"][text_type]):
					if c == "$":
						replacement_char_counter += 1
					for entity in content["entities"]:
						if i == entity["start_idx"] and entity["location"] == text_type:
							if c == "$":
								entity["start_idx"] -= replacement_char_counter - 1
							else:
								entity["start_idx"] -= replacement_char_counter
						if i == entity["end_idx"] and entity["location"] == text_type:
							entity["end_idx"] -= replacement_char_counter
					for relation in content["relations"]:
						for relation_type in ["subject", "object"]:
							if (
								i == relation[f"{relation_type}_start_idx"]
								and relation[f"{relation_type}_location"] == text_type
							):
								if c == "$":
									relation[f"{relation_type}_start_idx"] -= replacement_char_counter - 1
								else:
									relation[f"{relation_type}_start_idx"] -= replacement_char_counter
							if (
								i == relation[f"{relation_type}_end_idx"]
								and relation[f"{relation_type}_location"] == text_type
							):
								relation[f"{relation_type}_end_idx"] -= replacement_char_counter

		for _, content in data.items():
			for text_type in ["title", "abstract"]:
				content["metadata"][text_type] = re.sub(
					rf"{re.escape(replacement_char)}+", "", content["metadata"][text_type]
				)

			for type in ["entities", "relations", "ternary_mention_based_relations"]:
				for object in content[type]:
					if type == "entities":
						object["text_span"] = re.sub(rf"{re.escape(replacement_char)}+", "", object["text_span"])
					else:
						object["subject_text_span"] = re.sub(
							rf"{re.escape(replacement_char)}+", "", object["subject_text_span"]
						)
						object["object_text_span"] = re.sub(
							rf"{re.escape(replacement_char)}+", "", object["object_text_span"]
						)

	except Exception as e:
		logger.error(f"Error processing file: {e}")

	if save_filename:
		os.makedirs("data_preprocessed", exist_ok=True)
		output_file_path = os.path.join("data_preprocessed", f"{save_filename}")
		try:
			with open(output_file_path, "w", encoding="utf-8") as f:
				json.dump(data, f, indent=4)
			logger.info(f"Data saved to {output_file_path}")
		except Exception as e:
			logger.error(f"Error saving file {output_file_path}: {e}")
	else:
		logger.info("Returning processed data.")
		return data
