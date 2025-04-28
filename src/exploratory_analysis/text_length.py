import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import pandas as pd


def text_length_histogram(
	file_paths: str, save_path: str = os.path.join("plots", "exploratory_analysis", "text_length_histogram.pdf")
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
				data.append({"quality": quality, "text type": text_type, "length": length})
	df = pd.DataFrame(data)

	df = df[df["length"] <= 900]

	sns.set_theme(style="ticks")
	fig, axes = plt.subplots(2, 4, figsize=(14, 7))

	palette = sns.color_palette("magma", n_colors=4, desat=1)[::-1]

	qualities = ["platinum", "gold", "silver", "bronze"]

	title_data = df[df["text type"] == "title"]["length"]
	abstract_data = df[df["text type"] == "abstract"]["length"]

	title_xlim = (title_data.min(), title_data.max())
	abstract_xlim = (abstract_data.min(), abstract_data.max())

	title_data = df[df["text type"] == "title"]["length"]
	for i, quality in enumerate(qualities):
		sns.histplot(
			data=df[(df["text type"] == "title") & (df["quality"] == quality)],
			x="length",
			hue="quality",
			palette=palette[i : i + 1],
			multiple="stack",
			ax=axes[0, i],
			bins=25,
			legend=False,
			alpha=1,
			edgecolor="black",
			linewidth=0.5,
		)
		axes[0, i].set_xlabel("")
		axes[0, i].set_ylabel("")
		axes[0, i].set_xlim(title_xlim)
		axes[0, i].set_xticks(np.arange(0, title_data.max(), step=10))
	fig.text(0.46, 0.985, "Titles", ha="center", va="center", fontsize=16)

	unique_qualities = df["quality"].unique()
	handles = [
		Patch(facecolor=palette[i], label=quality.capitalize(), linewidth=0.5, edgecolor="black")
		for i, quality in enumerate(unique_qualities)
	]
	handles.reverse()
	axes[0, 3].legend(
		handles=handles, title="Quality", bbox_to_anchor=(1, 1), loc="upper left", fontsize=12, title_fontsize=14
	)

	title_data = df[df["text type"] == "abstract"]["length"]
	for i, quality in enumerate(qualities):
		sns.histplot(
			data=df[(df["text type"] == "abstract") & (df["quality"] == quality)],
			x="length",
			hue="quality",
			palette=palette[i : i + 1],
			multiple="stack",
			ax=axes[1, i],
			bins=20,
			legend=False,
			alpha=1,
			edgecolor="black",
			linewidth=0.5,
		)
		axes[1, i].set_xlabel("")
		axes[1, i].set_ylabel("")
		axes[1, i].set_xlim(abstract_xlim)
		axes[1, i].set_xticks(np.arange(0, title_data.max(), step=100))
	plt.subplots_adjust(hspace=5)
	fig.text(0.46, 0.485, "Abstracts", ha="center", va="center", fontsize=16)

	plt.figtext(0.46, 0.0, "Word Count", ha="center", fontsize=14)
	plt.figtext(0.006, 0.46, "Frequency", ha="center", rotation=90, fontsize=14)

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
