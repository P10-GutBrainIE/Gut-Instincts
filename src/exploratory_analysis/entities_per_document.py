import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
import pandas as pd


def entities_per_document(file_paths: str, save_path: str = os.path.join("plots", "entities_per_document.png")):
	densities_per_quality = {quality: [] for quality in file_paths.keys()}

	for quality, file_path in file_paths.items():
		with open(file_path, "r", encoding="utf-8") as f:
			file_data = json.load(f)

			highest_density = 0
			for paper_id, content in file_data.items():
				paper_length = len(content["metadata"]["title"].split(" ")) + len(
					content["metadata"]["abstract"].split(" ")
				)
				densities_per_quality[quality].append(len(content["relations"]) / paper_length)

				temp = len(content["relations"]) / paper_length
				if temp > highest_density:
					highest_density = temp
					paper = paper_id

			print(f"{quality}: {highest_density} ({paper})")

	data = []
	for quality, densities in densities_per_quality.items():
		for density in densities:
			data.append({"quality": quality, "density": density})
	df = pd.DataFrame(data)

	sns.set_theme(style="ticks")
	fig, axes = plt.subplots(1, 4, figsize=(14, 7))

	palette = sns.color_palette("magma", n_colors=4, desat=0.7)[::-1]

	densities_data = df["density"]
	x_limits = (densities_data.min(), densities_data.max())

	for i, quality in enumerate(list(densities_per_quality.keys())):
		outliers = df[(df["quality"] == quality)][(df["density"] < 0.005)]
		print(f"{quality}: {len(outliers)}")
		sns.histplot(
			data=df[df["quality"] == quality],
			x="density",
			hue="quality",
			palette=palette[i : i + 1],
			ax=axes[i],
			bins=25,
			legend=False,
			alpha=1,
			edgecolor="black",
			linewidth=0.5,
		)
		axes[i].set_xlabel("")
		axes[i].set_ylabel("")
		axes[i].set_xlim(x_limits)

	fig.legend(
		labels=[s.capitalize() for s in list(densities_per_quality.keys())],
		title="Quality",
		loc="upper right",
		fontsize=12,
		title_fontsize=14,
	)

	sns.despine()
	plt.tight_layout(pad=1.3)
	os.makedirs("plots", exist_ok=True)
	plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Train")
	file_paths = {
		"platinum": os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
		"gold": os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
		"silver": os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
		"bronze": os.path.join(shared_path, "bronze_quality", "json_format", "train_bronze.json"),
	}

	entities_per_document(file_paths)
