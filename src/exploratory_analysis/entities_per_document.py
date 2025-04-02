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

	# df = df[df["paper_length"] <= 900]
	# df = df[df["relations"] <= 60]
	return df


def entities_per_document(df: pd.DataFrame, save_path: str = os.path.join("plots", "entities_per_document.png")):
	sns.set_theme(style="ticks")
	_, axes = plt.subplots(2, 1, figsize=(14, 7))
	palette = sns.color_palette("magma", n_colors=len(df["quality"].unique()), desat=0.7)[::-1]

	sns.scatterplot(
		x=df["paper_length"],
		y=df["entities"],
		hue=df["quality"],
		palette=palette,
		alpha=1,
		edgecolor="black",
		linewidth=0.5,
		ax=axes[0],
		legend=False,
	)
	sns.regplot(
		data=df,
		x="paper_length",
		y="entities",
		scatter=False,
		color="blue",
		ax=axes[0],
		line_kws={"linewidth": 1, "alpha": 1},
	)
	axes[0].set_xlabel("Paper length (in words)", fontsize=14)
	axes[0].set_ylabel("Number of entities", fontsize=14)

	sns.scatterplot(
		x=df["paper_length"],
		y=df["relations"],
		hue=df["quality"],
		palette=palette,
		alpha=1,
		edgecolor="black",
		linewidth=0.5,
		ax=axes[1],
		legend=False,
	)
	sns.regplot(
		data=df,
		x="paper_length",
		y="relations",
		scatter=False,
		color="blue",
		ax=axes[1],
		line_kws={"linewidth": 1, "alpha": 1},
	)
	axes[1].set_xlabel("Paper length (in words)", fontsize=14)
	axes[1].set_ylabel("Number of relations", fontsize=14)

	unique_qualities = df["quality"].unique()
	handles = [Patch(color=palette[i], label=quality) for i, quality in enumerate(unique_qualities)]
	handles.reverse()
	axes[0].legend(handles=handles, title="Quality", loc="upper right", fontsize=12, title_fontsize=14)

	sns.despine()
	plt.tight_layout(pad=1.3)
	os.makedirs("plots", exist_ok=True)
	plt.savefig(save_path, dpi=300)
	plt.close()


def create_pairplot(df: pd.DataFrame, save_path: str = os.path.join("plots", "pairplot.png")):
	"""
	Create a pairplot for the variables 'entities', 'relations', and 'paper_length'.
	Colors the points by the 'quality' column.
	"""
	sns.set_theme(style="ticks")
	plt.figure(figsize=(14, 7))
	palette = sns.color_palette("magma", n_colors=len(df["quality"].unique()), desat=0.7)[::-1]

	sns.pairplot(
		df, hue="quality", palette=palette, vars=["entities", "relations", "paper_length"], markers=["o", "s", "D"]
	)

	os.makedirs("plots", exist_ok=True)
	plt.savefig(save_path, dpi=300)
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
	entities_per_document(df=data)
	create_pairplot(df=data)
