import os
import pandas as pd
import json

shared_path = os.path.join("data", "Annotations", "Train")

files = [
	os.path.join(shared_path, "bronze_quality", "json_format", "train_bronze.json"),
	os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
	os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
	os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
	os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
]


def extract_entities(file_path):
	entities = []
	with open(file_path, "r", encoding="utf-8") as f:
		file_path = json.load(f)
		for _, content in file_path.items():
			for entity in content["entities"]:
				entities.append({"Entity": entity["text_span"], "Category": entity["label"]})

	return pd.DataFrame(entities, columns=["Entity", "Category"])


def extract_all_entities(file_paths):
	entities = []
	for file_path in file_paths:
		with open(file_path, "r", encoding="utf-8") as f:
			data = json.load(f)
			for _, content in data.items():
				for entity in content["entities"]:
					entities.append({"Entity": entity["text_span"], "Category": entity["label"]})

	return pd.DataFrame(entities, columns=["Entity", "Category"])


bronze_entities = extract_entities(files[0])
bronze_entities.to_csv(os.path.join("entities", "bronze_entities.csv"))

silver_entities = extract_entities(files[1])
silver_entities.to_csv(os.path.join("entities", "silver_entities.csv"))

gold_entities = extract_entities(files[2])
gold_entities.to_csv(os.path.join("entities", "gold_entities.csv"))

platinum_entities = extract_entities(files[3])
platinum_entities.to_csv(os.path.join("entities", "platinum_entities.csv"))

all_entities = extract_all_entities(files)
all_entities.to_csv(os.path.join("entities", "all_entities.csv"))
