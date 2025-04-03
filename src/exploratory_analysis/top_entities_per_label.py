import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def n_gram_per_label(file_paths: str, n: int = 1, top_k: int = 3) -> pd.DataFrame:
	"""
	Extract n-grams from the text spans of entities in the given JSON files and count their occurrences.

		Args:
			file_paths (str): Paths to the JSON files.
			n (int): The size of the n-grams to extract. Defaults to 1.
			top_k (int, optional): The number of top n-grams to return for each label. Defaults to 3.

		Returns:
			pd.DataFrame: A DataFrame containing the n-grams, their labels, and their counts.
	"""
	data = []
	for file_path in file_paths:
		with open(file_path, "r", encoding="utf-8") as f:
			file_data = json.load(f)
			for _, content in file_data.items():
				for entity in content["entities"]:
					words = entity["text_span"].split(" ")
					for i in range(len(words) - n + 1):
						ngram = " ".join(words[i : i + n])
						data.append(
							{
								"ngram": ngram,
								"label": entity["label"],
							}
						)

	df = pd.DataFrame(data)

	df["count"] = 1
	df = df.groupby(["ngram", "label"]).count().reset_index()
	df = df.sort_values(by=["label", "count"], ascending=[True, False])
	df = df.groupby("label").head(top_k).reset_index(drop=True)

	label_totals = df.groupby("label")["count"].sum().reset_index()
	label_totals = label_totals.rename(columns={"count": "total_count"})
	df = df.merge(label_totals, on="label")
	df = df.sort_values(by=["total_count", "label", "count"], ascending=[False, True, False]).reset_index(drop=True)
	df = df.drop(columns=["total_count"])

	return df


def plot_n_gram(df: pd.DataFrame, save_path: str = os.path.join("plots", "unigram_per_label.png")) -> None:
	"""
	Plot a bar chart of n-grams per label.

	Args:
	    df (pd.DataFrame): DataFrame containing n-grams, their labels, and their counts.
	    save_path (str): Path to save the plot. Defaults to "plots/unigram_per_label.png".
	"""
	sns.set_theme(style="ticks")
	plt.figure(figsize=(14, 7))

	palette = sns.color_palette("magma", n_colors=df["label"].nunique(), desat=0.7)
	sns.barplot(
		x="ngram",
		y="count",
		hue="label",
		data=df,
		palette=palette,
		dodge=False,
	)

	plt.xticks(rotation=45, ha="right", fontsize=12)
	plt.xlabel("Unigram per Label", fontsize=14)
	plt.ylabel("Count", fontsize=14)
	plt.legend(title="Label", loc="upper right", title_fontsize=14, fontsize=12)
	sns.despine()

	os.makedirs("plots", exist_ok=True)
	plt.tight_layout()
	plt.savefig(save_path, dpi=300)
	plt.close()


if __name__ == "__main__":
	shared_path = "data_preprocessed"
	file_paths = [
		os.path.join(shared_path, "platinum_html_removed.json"),
		os.path.join(shared_path, "gold_html_removed.json"),
		os.path.join(shared_path, "silver_html_removed.json"),
		os.path.join(shared_path, "bronze_html_removed.json"),
	]

	unigram = n_gram_per_label(file_paths=file_paths, n=1, top_k=3)
	plot_n_gram(df=unigram, save_path=os.path.join("plots", "unigram_per_label.png"))
