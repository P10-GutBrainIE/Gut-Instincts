import json
from utils.utils import load_json_data
import os


def match_entity(inference_entity, dev_entities):
	for dev_entity in dev_entities:
		if dev_entity["location"] != inference_entity["location"]:
			continue

		if (
			inference_entity["start_idx"] == dev_entity["start_idx"]
			and inference_entity["end_idx"] == dev_entity["end_idx"]
			and inference_entity["text_span"] == dev_entity["text_span"]
		) or (
			inference_entity["start_idx"] >= dev_entity["start_idx"]
			and inference_entity["end_idx"] <= dev_entity["end_idx"]
			and inference_entity["text_span"] in dev_entity["text_span"]
		):
			return True
	return False


if __name__ == "__main__":
	dev_data = load_json_data(os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))
	inference_results = load_json_data(os.path.join("data_inference_results", "kkk.json"))
	save_path = "data_inference_results/kkk_matched_with_dev.json"

	results = {}
	for pmid, content in inference_results.items():
		results[pmid] = {"entities": []}
		inference_entities = content.get("entities", [])
		dev_entities = dev_data.get(pmid, {}).get("entities", [])

		for entity in inference_entities:
			matched = match_entity(entity, dev_entities)
			entity["matched_in_dev"] = matched
			results[pmid]["entities"].append(entity)

	with open(save_path, "w") as f:
		json.dump(results, f, indent=4)
