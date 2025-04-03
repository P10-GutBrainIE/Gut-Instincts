import json
import os
from datetime import datetime
from utils.utils import load_json_data, save_json_data


def remove_documents_over_or_under_threshold(
	data: dict, threshold: int, type: str, remove_if: str = "under", save_data: bool = False
):
	"""
	Removes documents that are either under or over the threshold and saves the remaining ones
	in a new JSON file in data/preprocessed.

	Args:
	    data (dict): Dataset loaded as a dictionary
	    threshold (int): Threshold for the number of entities or relations
	    type (string): Should be either "entities" or "relations"
	    remove_if (string): "under" to remove documents below the threshold, "over" to remove above
		save_data (bool): If True, the processed JSON data is saved to a "data_preprocessed" directory.
	                      If False, the processed data is returned as a dictionary. Default is False.
	"""

	if remove_if == "under":
		filtered_data = {
			paper_id: content for paper_id, content in data.items() if len(content.get(type, [])) >= threshold
		}
		condition = "under"
	elif remove_if == "over":
		filtered_data = {
			paper_id: content for paper_id, content in data.items() if len(content.get(type, [])) <= threshold
		}
		condition = "over"
	else:
		raise ValueError('Invalid value for remove_if. Use "under" or "over".')

	if save_data:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_filename = f"filtered_{type}_{condition}_{threshold}_{timestamp}.json"

		output_path = os.path.join("data_preprocessed", output_filename)

		with open(output_path, "w", encoding="utf-8") as f:
			json.dump(filtered_data, f, indent=4, ensure_ascii=False)

		print(f"Filtered dataset saved to {output_path}")
	else:
		return filtered_data


def clean_incorrect_text_spans(
	data: dict, corrections: dict, check_ternary_relations: bool = False, save_data: bool = False
):
	"""
	Cleans incorrect entity annotations by replacing them with correct values and updating their start/end indices.

	This function loads a dataset and a set of corrections, then updates entities and relations in the dataset
	based on the provided corrections. If ternary relations are enabled, it also adjusts corresponding mentions.

	Args:
	    data (dict): Dataset loaded as a dictionary.
        corrections (dict): Dictionary containing corrections for incorrect text spans.
	    check_ternary_relations (bool, optional): Whether to update text spans in ternary mention-based relations. Defaults to False.
	    save_data (bool, optional): Whether to save the cleaned dataset to a file. If False, returns the updated dataset. Defaults to False.

	Returns:
	    dict or None: The cleaned dataset if `save_data` is False, otherwise None.
	"""

	missing_paper_ids = [paper_id for paper_id in corrections.keys() if paper_id not in data]
	if missing_paper_ids:
		raise ValueError(f"The following IDs were not found in data file: {', '.join(missing_paper_ids)}")

	for paper_id, corrections_list in corrections.items():
		if paper_id in data:
			for correction in corrections_list:
				for entity in data[paper_id]["entities"]:
					if (
						entity["start_idx"] == correction["start_idx"]
						and entity["end_idx"] == correction["end_idx"]
						and entity["location"] == correction["location"]
						and entity["text_span"] == correction["text_span"]
					):
						entity["text_span"] = correction["correct_text_span"]
						entity["start_idx"] = correction["correct_start_idx"]
						entity["end_idx"] = correction["correct_end_idx"]
				for relation in data[paper_id]["relations"]:
					if (
						relation["subject_start_idx"] == correction["start_idx"]
						and relation["subject_end_idx"] == correction["end_idx"]
						and relation["subject_location"] == correction["location"]
						and relation["subject_text_span"] == correction["text_span"]
					):
						relation["subject_text_span"] = correction["correct_text_span"]
						relation["subject_start_idx"] = correction["correct_start_idx"]
						relation["subject_end_idx"] = correction["correct_end_idx"]

					if (
						relation["object_start_idx"] == correction["start_idx"]
						and relation["object_end_idx"] == correction["end_idx"]
						and relation["object_location"] == correction["location"]
						and relation["object_text_span"] == correction["text_span"]
					):
						relation["object_text_span"] = correction["correct_text_span"]
						relation["object_start_idx"] = correction["correct_start_idx"]
						relation["object_end_idx"] = correction["correct_end_idx"]
				if check_ternary_relations:
					for relation in data[paper_id]["ternary_mention_based_relations"]:
						if (
							relation["subject_label"] == correction["label"]
							and relation["subject_text_span"] == correction["text_span"]
						):
							relation["subject_text_span"] = correction["correct_text_span"]
						if (
							relation["object_label"] == correction["label"]
							and relation["object_text_span"] == correction["text_span"]
						):
							relation["object_text_span"] = correction["correct_text_span"]

	if save_data:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_filename = f"cleaned_data_{timestamp}.json"
		output_path = os.path.join("data_preprocessed", output_filename)

		with open(output_path, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=4, ensure_ascii=False)

		print(f"Dataset with incorrect annotations cleaned saved to {output_path}")
	else:
		return data


def remove_incorrect_text_spans(
	data: dict, incorrect_annotations: dict, check_ternary_relations=False, save_data: bool = False
):
	"""
	Removes incorrect entity, relation, and ternary mention-based relation annotations from the dataset based on the corrections specified in corrections_file_path.
	Optionally, the cleaned dataset can be saved in a new JSON file in the 'data_preprocessed' directory.

	Args:
	    data (dict): Dataset loaded as a dictionary.
        incorrect_annotations (dict): Dictionary containing the annotations to be removed.
	    check_ternary_relations (bool, optional): If True, ternary mention-based relations will also be checked and cleaned. Defaults to False.
	    save_data (bool, optional): If True, the cleaned dataset will be saved in the 'data_preprocessed' directory. Defaults to False.

	Returns:
	    dict or None: Returns the cleaned dataset as a dictionary if 'save_data' is False, otherwise returns None.

	"""

	missing_paper_ids = [paper_id for paper_id in incorrect_annotations.keys() if paper_id not in data]
	if missing_paper_ids:
		raise ValueError(f"The following IDs were not found in data file: {', '.join(missing_paper_ids)}")

	for paper_id in list(data.keys()):
		if paper_id in incorrect_annotations:
			data[paper_id]["entities"] = [
				entity for entity in data[paper_id]["entities"] if entity not in incorrect_annotations[paper_id]
			]

			data[paper_id]["relations"] = [
				relation
				for relation in data[paper_id].get("relations", [])
				if not any(
					relation["subject_start_idx"] == incorrect_annotation["start_idx"]
					and relation["subject_end_idx"] == incorrect_annotation["end_idx"]
					and relation["subject_location"] == incorrect_annotation["location"]
					and relation["subject_text_span"] == incorrect_annotation["text_span"]
					or relation["object_start_idx"] == incorrect_annotation["start_idx"]
					and relation["object_end_idx"] == incorrect_annotation["end_idx"]
					and relation["object_location"] == incorrect_annotation["location"]
					and relation["object_text_span"] == incorrect_annotation["text_span"]
					for incorrect_annotation in incorrect_annotations[paper_id]
				)
			]
			if check_ternary_relations:
				data[paper_id]["ternary_mention_based_relations"] = [
					relation
					for relation in data[paper_id].get("ternary_mention_based_relations", [])
					if not any(
						(
							relation["subject_label"] == incorrect_annotation["label"]
							and relation["subject_text_span"] == incorrect_annotation["text_span"]
						)
						or (
							relation["object_label"] == incorrect_annotation["label"]
							and relation["object_text_span"] == incorrect_annotation["text_span"]
						)
						for incorrect_annotation in incorrect_annotations[paper_id]
					)
				]

	if save_data:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		output_filename = f"removed_data_{timestamp}.json"
		output_path = os.path.join("data_preprocessed", output_filename)

		with open(output_path, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=4, ensure_ascii=False)

		print(f"Dataset with incorrect annotations removed saved to {output_path}")
	else:
		return data


if __name__ == "__main__":
	bronze_path = os.path.join("data", "Annotations", "Train", "bronze_quality", "json_format", "train_bronze.json")
	silver_path = os.path.join("data", "Annotations", "Train", "silver_quality", "json_format", "train_silver.json")
	bronze_corrections_path = os.path.join("data", "metadata", "bronze_incorrect_annotations.json")
	silver_corrections_path = os.path.join("data", "metadata", "silver_incorrect_annotations.json")

	
	#Bronze cleanup pipeline example
	bronze_data = load_json_data(bronze_path)
	bronze_corrections = load_json_data(bronze_corrections_path)

	bronze_data = remove_incorrect_text_spans(bronze_data, bronze_corrections.get("remove", {}), check_ternary_relations=True)
	bronze_data = clean_incorrect_text_spans(bronze_data, bronze_corrections.get("clean", {}), check_ternary_relations=True)
	bronze_data = remove_documents_over_or_under_threshold(bronze_data, 1, "relations", "under")
	save_json_data(bronze_data, "cleaned_bronze_data.json")

	#Silver cleanup pipeline example
	silver_data = load_json_data(silver_path)
	silver_corrections = load_json_data(silver_corrections_path)

	silver_data = clean_incorrect_text_spans(silver_data, silver_corrections.get("clean", {}))
	silver_data = remove_documents_over_or_under_threshold(silver_data, 200, "relations", remove_if="over")
	save_json_data(silver_data, "cleaned_silver_data.json")


