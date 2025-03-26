import json
import os
import numpy as np
import matplotlib.pyplot as plt


def entity_counter(file_path):
	category_dict = {
		"anatomical location": {},
		"animal": {},
		"biomedical technique": {},
		"bacteria": {},
		"chemical": {},
		"dietary supplement": {},
		"DDF": {},
		"drug": {},
		"food": {},
		"gene": {},
		"human": {},
		"microbiome": {},
		"statistical technique": {},
	}

	with open(file_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		for _, content in data.items():
			for entity in content["entities"]:
				if entity["label"] not in category_dict:
					continue  # Skip unexpected entity labels

				if entity["text_span"] in category_dict[entity["label"]]:
					category_dict[entity["label"]][entity["text_span"]] += 1
				else:
					category_dict[entity["label"]][entity["text_span"]] = 1

	return category_dict


def entity_counter_all(file_paths):
	common_entities_dict = {
		"anatomical location": {},
		"animal": {},
		"biomedical technique": {},
		"bacteria": {},
		"chemical": {},
		"dietary supplement": {},
		"DDF": {},
		"drug": {},
		"food": {},
		"gene": {},
		"human": {},
		"microbiome": {},
		"statistical technique": {},
	}

	for file_path in file_paths:
		file_entities = entity_counter(file_path)
		for category, entities in file_entities.items():
			for entity, count in entities.items():
				if entity in common_entities_dict[category]:
					common_entities_dict[category][entity] += count
				else:
					common_entities_dict[category][entity] = count
	return common_entities_dict


def visualize_entities_print(category_dict):
	for category, content in category_dict.items():
		sorted_value_index = np.argsort(list(content.values()))[::-1]
		sorted_dict = [f"{list(content.keys())[i]}: {list(content.values())[i]}" for i in sorted_value_index[:5]]
		print(f"{category}: {sorted_dict}")


def visualize_entities_plot(category_dict):
	categories = list(category_dict.keys())
	top_entities = {
		category: sorted(content.items(), key=lambda x: x[1], reverse=True)[:3]
		for category, content in category_dict.items()
	}

	x_labels = []
	y_values = []
	bar_colors = []
	legend_labels = []

	color_list = [
		"#1f77b4",
		"#ff7f0e",
		"#2ca02c",
		"#d62728",
		"#9467bd",
		"#8c564b",
		"#e377c2",
		"#7f7f7f",
		"#bcbd22",
		"#17becf",
		"#ff5733",
		"#8a2be2",
		"#3357ff",
	]  # A set of distinct, dark colors
	category_colors = {category: color_list[i % len(color_list)] for i, category in enumerate(categories)}

	for category in categories:
		for entity, count in top_entities.get(category, []):
			x_labels.append(entity)
			y_values.append(count)
			bar_colors.append(category_colors[category])
		legend_labels.append(category)

	plt.figure(figsize=(14, 8))
	plt.bar(x_labels, y_values, color=bar_colors)

	unique_colors = [category_colors[cat] for cat in categories]
	plt.legend(
		handles=[plt.Rectangle((0, 0), 1, 1, color=unique_colors[i]) for i in range(len(categories))],
		labels=categories,
		title="Categories",
	)

	plt.xticks(rotation=45, ha="right", fontsize=10)
	plt.xlabel("Entities")
	plt.ylabel("Count")
	plt.title("Top 3 Entities per Category")
	plt.show()


if __name__ == "__main__":
	platinum_path = os.path.join(
		"data",
		"Annotations",
		"Train",
		"platinum_quality",
		"json_format",
		"train_platinum.json",
	)
	gold_path = os.path.join(
		"data",
		"Annotations",
		"Train",
		"gold_quality",
		"json_format",
		"train_gold.json",
	)
	silver_path = os.path.join(
		"data",
		"Annotations",
		"Train",
		"silver_quality",
		"json_format",
		"train_silver.json",
	)
	bronze_path = os.path.join(
		"data",
		"Annotations",
		"Train",
		"bronze_quality",
		"json_format",
		"train_bronze.json",
	)
	dev_path = os.path.join(
		"data",
		"Annotations",
		"Dev",
		"json_format",
		"dev.json",
	)

	paths = [platinum_path, gold_path, silver_path, bronze_path, dev_path]
	common_entities = entity_counter_all(paths)
	visualize_entities_plot(common_entities)
