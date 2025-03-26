import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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

	data = []
	for quality, lengths in text_lengths.items():
		for text_type in ["title", "abstract"]:
			for length in lengths[text_type]:
				data.append({"quality": quality.capitalize(), "text type": text_type, "length": length})
		df = pd.DataFrame(data)

	df = df[df["length"] <= 900]

	sns.set_theme(style="ticks")
	_, axes = plt.subplots(2, 1, figsize=(14, 7))

	palette = sns.color_palette()

	sns.violinplot(
		x="quality",
		y="length",
		data=df[df["text type"] == "title"],
		ax=axes[0],
		inner="box",
		palette=palette,
		hue="quality",
	)
	axes[0].set_xlabel("")
	axes[0].set_xticklabels([])
	axes[0].set_ylabel("Word Count", fontsize=14)
	axes[0].set_title("Titles", fontsize=14)

	unique_qualities = df["quality"].unique()
	handles = [Patch(color=palette[i], label=quality) for i, quality in enumerate(unique_qualities)]
	handles.reverse()
	axes[0].legend(handles=handles, title="Quality", fontsize=12, title_fontsize=14, loc="upper right")

	sns.violinplot(
		x="quality",
		y="length",
		data=df[df["text type"] == "abstract"],
		ax=axes[1],
		inner="box",
		palette=palette,
		hue="quality",
	)
	axes[1].set_xlabel("")
	axes[1].set_xticklabels([])
	axes[1].set_ylabel("Word Count", fontsize=14)
	axes[1].set_title("Abstracts", fontsize=14)

	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	sns.despine()
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
