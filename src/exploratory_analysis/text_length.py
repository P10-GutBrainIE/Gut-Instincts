import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import shapiro
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer


def text_length_histogram(
	file_paths: dict, save_path: str = os.path.join("plots", "exploratory_analysis", "text_length_histogram.pdf")
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
	fig, axes = plt.subplots(2, 2, figsize=(14, 7))

	# palette = sns.color_palette("magma", n_colors=4, desat=1)
	palette = sns.color_palette("colorblind")[:8]
	palette = [palette[2]] + [palette[-1]] + [palette[1]] + [palette[0]]

	qualities = ["bronze", "silver", "gold", "platinum"]

	for i, quality in enumerate(qualities):
		lengths = df[df["quality"] == quality]["length"]

		# Apply Shapiro-Wilk test
		if len(lengths) >= 3 and len(lengths) <= 5000:  # Shapiro-Wilk limit is 5000
			stat, p_value = shapiro(lengths)
			print(f"[{quality.capitalize()}] Shapiro-Wilk: W={stat:.4f}, p={p_value:.4f}")
			if p_value < 0.05:
				print(f"❌ {quality.capitalize()} data is likely NOT normally distributed.")
			else:
				print(f"✅ {quality.capitalize()} data appears to be normally distributed.")
		else:
			print(f"[{quality.capitalize()}] Skipped Shapiro-Wilk (n={len(lengths)}): too many samples.")

		row, col = divmod(i, 2)
		ax = axes[row, col]
		sns.histplot(
			data=df[df["quality"] == quality],
			x="length",
			hue="quality",
			palette=palette[i : i + 1],
			multiple="stack",
			ax=ax,
			bins=30,
			legend=False,
			alpha=1,
			edgecolor="black",
			linewidth=0.5,
		)
		if row == 1:
			ax.set_xlabel("Token Sequence Length", fontsize=14)
		else:
			ax.set_xlabel("")
		if col == 0:
			ax.set_ylabel("Number of Articles", fontsize=14)
		else:
			ax.set_ylabel("")

		lengths_data = df["length"]
		x_min, x_max = lengths_data.min(), lengths_data.max()
		ax.set_xlim((x_min, x_max))
		ax.set_xticks(np.arange(0, x_max, step=200))

	handles = [
		Patch(facecolor=palette[i], label=quality.capitalize(), linewidth=0.5, edgecolor="black")
		for i, quality in enumerate(qualities)
	][::-1]
	handles.reverse()
	axes[0, 1].legend(title="Quality", handles=handles, fontsize=12, title_fontsize=14)

	sns.despine()
	plt.tight_layout(pad=1.3, rect=[0, 0.05, 1, 0.97])
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, format="pdf")
	plt.close()


if __name__ == "__main__":
	shared_path = os.path.join("data", "Annotations", "Train")
	file_paths = {
		"platinum": os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
		"gold": os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
		"silver": os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
		"bronze": os.path.join(shared_path, "bronze_quality", "json_format", "train_bronze.json"),
	}

	text_length_histogram(file_paths)
