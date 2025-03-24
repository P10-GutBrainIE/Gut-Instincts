import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def analyze_text_length_distribution(file_paths):
	text_lengths = {quality: {"title": [], "abstract": []} for quality in file_paths.keys()}

	for quality, file_path in file_paths.items():
		with open(file_path, "r", encoding="utf-8") as f:
			file_data = json.load(f)

			for _, content in file_data.items():
				text_lengths[quality]["title"].append(len(content["metadata"]["title"].split(" ")))
				text_lengths[quality]["abstract"].append(len(content["metadata"]["abstract"].split(" ")))

	for quality, _ in text_lengths.items():
		mean_title_length = np.mean(text_lengths[quality]["title"])
		mean_abstract_length = np.mean(text_lengths[quality]["abstract"])
		max_title_length = np.max(text_lengths[quality]["title"])
		max_abstract_length = np.max(text_lengths[quality]["abstract"])
		min_title_length = np.min(text_lengths[quality]["title"])
		min_abstract_length = np.min(text_lengths[quality]["abstract"])
		print(f"{quality} title length: {min_title_length} - {mean_title_length} - {max_title_length}")
		print(f"{quality} abstract length: {min_abstract_length} - {mean_abstract_length} - {max_abstract_length}")

	data = []
	for quality, lengths in text_lengths.items():
		for text_type in ["title", "abstract"]:
			for length in lengths[text_type]:
				data.append({"Quality": quality.capitalize(), "Text Type": text_type, "Length": length})
		df = pd.DataFrame(data)

	sns.set_theme(style="whitegrid")
	_, axes = plt.subplots(2, 1, figsize=(16, 9))

	sns.violinplot(
		x="Quality",
		y="Length",
		data=df[df["Text Type"] == "title"],
		ax=axes[0],
		inner_kws=dict(box_width=10, whis_width=2),
		palette="Set3",
		hue="Quality",
	)
	axes[0].set_title("Title Length by Quality")
	axes[0].set_xlabel("")
	axes[0].set_ylabel("Word Count")

	sns.violinplot(
		x="Quality",
		y="Length",
		data=df[df["Text Type"] == "abstract"],
		ax=axes[1],
		inner_kws=dict(box_width=10, whis_width=2),
		palette="Set3",
		hue="Quality",
	)
	axes[1].set_title("Abstract Length by Quality")
	axes[1].set_xlabel("")
	axes[1].set_ylabel("Word Count")

	plt.tight_layout()

	os.makedirs("plots", exist_ok=True)
	plt.savefig(os.path.join("plots", "text_lengths.png"), dpi=300)


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Train")
	file_paths = {
		"platinum": os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
		"gold": os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
		"silver": os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
		"bronze": os.path.join(shared_path, "bronze_quality", "json_format", "train_bronze.json"),
	}

	analyze_text_length_distribution(file_paths)
