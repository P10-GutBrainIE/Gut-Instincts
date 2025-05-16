import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


def extract_data(file_paths: str) -> pd.DataFrame:
	data = []
	for quality, file_path in file_paths.items():
		with open(file_path, "r", encoding="utf-8") as f:
			file_data = json.load(f)

			for _, content in file_data.items():
				data.append(
					{
						"quality": quality.capitalize(),
						"entities": len(content["entities"]),
						"relations": len(content["relations"]),
						"paper_length": len(content["metadata"]["title"].split(" "))
						+ len(content["metadata"]["abstract"].split(" ")),
					}
				)

	df = pd.DataFrame(data)

	df = df[df["paper_length"] <= 900]
	return df


def entities_and_relations_vs_word_count(
	df,
	save_path_entities="plots/exploratory_analysis/entities_vs_word_count.pdf",
	save_path_relations="plots/exploratory_analysis/relations_vs_word_count.pdf",
):
	sns.set_theme(style="ticks")
	palette = sns.color_palette("magma", n_colors=len(df["quality"].unique()), desat=1)[::-1]

	plt.figure(figsize=(7, 7))
	ax1 = sns.scatterplot(
		x=df["paper_length"],
		y=df["entities"],
		hue=df["quality"],
		palette=palette,
		alpha=1,
		edgecolor="black",
		linewidth=0.5,
		legend=False,
	)
	ax1.set_xlabel("Word Count", fontsize=14)
	ax1.set_ylabel("Number of Entities", fontsize=14)
	sns.despine()
	plt.tight_layout(pad=1.3)
	os.makedirs(os.path.dirname(save_path_entities), exist_ok=True)
	plt.savefig(save_path_entities, format="pdf")
	plt.close()

	plt.figure(figsize=(7, 7))
	ax2 = sns.scatterplot(
		x=df["paper_length"],
		y=df["relations"],
		hue=df["quality"],
		palette=palette,
		alpha=1,
		edgecolor="black",
		linewidth=0.5,
		legend=False,
	)
	ax2.set_xlabel("Word Count", fontsize=14)
	ax2.set_ylabel("Number of Relations", fontsize=14)

	unique_qualities = df["quality"].unique()
	handles = [
		Patch(facecolor=palette[i], label=quality, edgecolor="black", linewidth=0.5)
		for i, quality in enumerate(unique_qualities)
	]
	handles.reverse()
	ax2.legend(handles=handles, title="Quality", loc="upper right", fontsize=12, title_fontsize=14)

	sns.despine()
	plt.tight_layout(pad=1.3)
	os.makedirs(os.path.dirname(save_path_relations), exist_ok=True)
	plt.savefig(save_path_relations, format="pdf")
	plt.close()


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Train")
	file_paths = {
		"platinum": os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
		"gold": os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
		"silver": os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
		"bronze": os.path.join(shared_path, "bronze_quality", "json_format", "train_bronze.json"),
	}

	data = extract_data(file_paths)
	entities_and_relations_vs_word_count(df=data)
