import json
import os
import re


def remove_html(file_paths_dict: list[dict], save_data: bool = False):
	"""
	Remove HTML tags from JSON content and adjust entity text span indices.

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
	    save_data (bool): If True, the processed JSON data is saved to a "data_preprocessed" directory.
	                      If False, the processed data is returned as a dictionary. Default is False.

	Returns:
	    dict or None: If save_data is False, returns a dictionary where each key corresponds to the file identifier and
	                  the value is the processed JSON content. If save_data is True, the processed files are saved
	                  to disk and the function returns None.

	Example:
	    >>> file_paths = [{"gold": "data/Annotations/Train/json_format/gold.json"}]
	    >>> processed_data = remove_html(file_paths, save_data=False)
	    >>> # processed_data now contains JSON data with HTML tags removed and entity indices adjusted.
	"""
	replacement_char = "$"
	all_file_data = {list(file_path.keys())[0]: {} for file_path in file_paths_dict}
	for file_path_element in file_paths_dict:
		key, file_path = file_path_element.popitem()
		with open(file_path, "r", encoding="utf-8") as f:
			file_data = json.load(f)

			for _, content in file_data.items():
				for text_type in ["title", "abstract"]:
					content["metadata"][text_type] = re.sub(
						r"</?[^>]+>", lambda m: replacement_char * len(m.group(0)), content["metadata"][text_type]
					)

				for entity in content["entities"]:
					entity["text_span"] = re.sub(
						r"</?[^>]+>", lambda m: replacement_char * len(m.group(0)), entity["text_span"]
					)

			for _, content in file_data.items():
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

			for _, content in file_data.items():
				for text_type in ["title", "abstract"]:
					content["metadata"][text_type] = re.sub(
						rf"{re.escape(replacement_char)}+", "", content["metadata"][text_type]
					)

				for entity in content["entities"]:
					entity["text_span"] = re.sub(rf"{re.escape(replacement_char)}+", "", entity["text_span"])

			all_file_data[key] = file_data

	if save_data:
		os.makedirs("data_preprocessed", exist_ok=True)
		for key, data in all_file_data.items():
			with open(os.path.join("data_preprocessed", f"{key}_html_removed.json"), "w", encoding="utf-8") as f:
				json.dump(data, f, indent=4)
	else:
		return all_file_data


if __name__ == "__main__":
	remove_html(
		file_paths_dict=[{"dev": os.path.join("data", "Annotations", "Dev", "json_format", "dev.json")}], save_data=True
	)
