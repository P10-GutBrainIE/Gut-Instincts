import os
from utils.utils import load_json_data


def print_identical_label_and_text_span(datasets):
	for key, dataset in datasets.items():
		print(f"----- {key} -----")
		for _, content in dataset.items():
			for ent in content["entities"]:
				if ent["text_span"] == ent["label"]:
					print(ent)


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Train")
	platinum_data = load_json_data(os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"))
	gold_data = load_json_data(os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"))
	silver_data = load_json_data(os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"))
	dev_data = load_json_data(os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	print_identical_label_and_text_span(
		datasets={
			"platinum": platinum_data,
			"gold": gold_data,
			"silver": silver_data,
			"dev": dev_data,
		}
	)
