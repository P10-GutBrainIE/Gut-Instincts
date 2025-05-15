import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer


def text_length_histogram(
	file_paths: str, save_path: str = os.path.join("plots", "exploratory_analysis", "text_length_histogram.pdf")
):
	text_lengths = {quality: [] for quality in file_paths.keys()}

	tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base", use_fast=True)

	exclude_cnt = 0

	for quality, file_path in file_paths.items():
		with open(file_path, "r", encoding="utf-8") as f:
			file_data = json.load(f)

			for paper_id, content in file_data.items():
				length = sum(
					tokenizer(
						content["metadata"]["title"],
						truncation=False,
					)["attention_mask"]
				) + sum(
					tokenizer(
						content["metadata"]["abstract"],
						truncation=False,
					)["attention_mask"]
				)
				if length > 1000:
					print(f"Exluding {paper_id} from {quality} quality. Token count {length} > 1000.")
					exclude_cnt += 1
				else:
					text_lengths[quality].append(length)

	data = []
	for quality, lengths in text_lengths.items():
		for length in lengths:
			data.append({"quality": quality, "length": length})
	df = pd.DataFrame(data)

	print(f"{len(df[df['length'] >= 512]) + exclude_cnt} out of {len(df) + exclude_cnt} with length > 512")

	sns.set_theme(style="ticks")
	_, axes = plt.subplots(1, 4, figsize=(14, 5))

	palette = sns.color_palette("magma", n_colors=4, desat=1)[::-1]

	qualities = ["platinum", "gold", "silver", "bronze"]

	for i, quality in enumerate(qualities):
		sns.histplot(
			data=df[df["quality"] == quality],
			x="length",
			hue="quality",
			palette=palette[i : i + 1],
			multiple="stack",
			ax=axes[i],
			bins=25,
			legend=False,
			alpha=1,
			edgecolor="black",
			linewidth=0.5,
		)
		axes[i].set_xlabel("")
		axes[i].set_ylabel("")

		lengths_data = df["length"]
		x_min, x_max = lengths_data.min(), lengths_data.max()
		axes[i].set_xlim((x_min, x_max))
		axes[i].set_xticks(np.arange(0, x_max, step=200))
		if i == 0:
			axes[i].set_ylabel("Frequency", fontsize=14)

	unique_qualities = df["quality"].unique()
	handles = [
		Patch(facecolor=palette[i], label=quality.capitalize(), linewidth=0.5, edgecolor="black")
		for i, quality in enumerate(unique_qualities)
	]
	handles.reverse()
	plt.legend(title="Quality", handles=handles, fontsize=12, title_fontsize=14)

	plt.figtext(0.515, 0.0, "Token Count", ha="center", fontsize=14)
	sns.despine()
	plt.tight_layout(pad=1.3)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, format="pdf")


def text_length_violinplot(
	file_paths: str, save_path: str = os.path.join("plots", "exploratory_analysis", "text_length_violinplot.pdf")
):
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

	palette = sns.color_palette("magma", n_colors=4, desat=1)[::-1]

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
	handles = [
		Patch(facecolor=palette[i], label=quality, linewidth=0.5, edgecolor="black")
		for i, quality in enumerate(unique_qualities)
	]
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

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, format="pdf")


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Train")
	file_paths = {
		"platinum": os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
		"gold": os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
		"silver": os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
		"bronze": os.path.join(shared_path, "bronze_quality", "json_format", "train_bronze.json"),
	}

	text_length_histogram(file_paths)
	text_length_violinplot(file_paths)
