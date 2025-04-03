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
					words = [word.lower() for word in words]
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


def plot_n_gram(df: pd.DataFrame, save_path: str = os.path.join("plots", "unigram_per_label.pdf")) -> None:
	"""
	Plot a bar chart of n-grams per label.

	Args:
	    df (pd.DataFrame): DataFrame containing n-grams, their labels, and their counts.
	    save_path (str): Path to save the plot. Defaults to "plots/unigram_per_label.pdf".
	"""
	sns.set_theme(style="ticks")
	plt.figure(figsize=(14, 7))

	df["ngram_unique"] = df["ngram"] + "___" + df["label"]
	df["label"] = df["label"].apply(lambda x: x.title() if x != "DDF" else x)

	palette = sns.color_palette("magma", n_colors=df["label"].nunique(), desat=1)

	ax = sns.barplot(
		x="ngram_unique",
		y="count",
		hue="label",
		data=df,
		palette=palette,
		alpha=1,
		edgecolor="black",
		linewidth=0.5,
	)

	plt.xticks(ticks=df["ngram_unique"], labels=df["ngram"], rotation=45, ha="right", fontsize=12)
	plt.xlabel("")
	plt.ylabel("Count", fontsize=14)

	unique_labels = list(df["label"].unique())
	hatch_mapping = {label: "///" if i % 2 == 1 else "" for i, label in enumerate(unique_labels)}

	patch_idx = 0
	for label in unique_labels:
		n_bars = df[df["label"] == label].shape[0]
		hatch = hatch_mapping[label]
		for _ in range(n_bars):
			if patch_idx < len(ax.patches):
				ax.patches[patch_idx].set_hatch(hatch)
				patch_idx += 1

	leg = plt.legend(title="Label", loc="upper right", title_fontsize=14, fontsize=12)
	for handle, text in zip(leg.legend_handles, leg.get_texts()):
		label_text = text.get_text()
		hatch = hatch_mapping.get(label_text, "")
		handle.set_hatch(hatch)

	sns.despine()
	plt.tight_layout()

	os.makedirs("plots", exist_ok=True)
	plt.savefig(save_path, format="pdf")
	plt.close()


def plot_n_gram_sublpots(
	unigram: pd.DataFrame,
	bigram: pd.DataFrame,
	trigram: pd.DataFrame,
	save_path: str = os.path.join("plots", "n_gram_subplot.pdf"),
) -> None:
	"""
	Plot bar charts of n-grams per label (unigram, bigram, trigram) on separate subplots.

	Args:
	    unigram (pd.DataFrame): DataFrame containing unigram n-grams.
	    bigram (pd.DataFrame): DataFrame containing bigram n-grams.
	    trigram (pd.DataFrame): DataFrame containing trigram n-grams.
	    save_path (str): Path to save the plot. Defaults to "plots/n_gram_subplot.pdf".
	"""
	sns.set_theme(style="ticks")
	_, axes = plt.subplots(3, 1, figsize=(14, 21))

	for i, df in enumerate([unigram, bigram, trigram]):
		if i == 0:
			label_order = df["label"].tolist()
			label_order_dict = {label: idx for idx, label in enumerate(label_order)}
		else:
			df["order"] = df["label"].map(label_order_dict)
			df = df.sort_values("order").drop(columns="order")

		df["ngram_unique"] = df["ngram"] + "___" + df["label"]
		df["label"] = df["label"].apply(lambda x: x.title() if x != "DDF" else x)

		palette = sns.color_palette("magma", n_colors=df["label"].nunique(), desat=1)

		ax = axes[i]
		ax = sns.barplot(
			x="ngram_unique",
			y="count",
			hue="label",
			data=df,
			palette=palette,
			alpha=1,
			edgecolor="black",
			linewidth=0.5,
			ax=ax,
		)

		ax.set_xlabel("")
		ax.set_ylabel("Count", fontsize=14)
		ax.set_xticklabels(df["ngram"], rotation=45, ha="right", fontsize=12)
		ax.set_title(f"{['Unigram', 'Bigram', 'Trigram'][i]} per Label", fontsize=16)

		unique_labels = list(df["label"].unique())
		hatch_mapping = {label: "///" if i % 2 == 1 else "" for i, label in enumerate(unique_labels)}

		patch_idx = 0
		for label in unique_labels:
			n_bars = df[df["label"] == label].shape[0]
			hatch = hatch_mapping[label]
			for _ in range(n_bars):
				if patch_idx < len(ax.patches):
					ax.patches[patch_idx].set_hatch(hatch)
					patch_idx += 1

		ax.legend_.remove()

		if i == 0:
			leg = ax.legend(title="Label", loc="upper right", title_fontsize=14, fontsize=12)
			for handle, text in zip(leg.legend_handles, leg.get_texts()):
				label_text = text.get_text()
				hatch = hatch_mapping.get(label_text, "")
				handle.set_hatch(hatch)

	sns.despine()
	plt.tight_layout()

	os.makedirs("plots", exist_ok=True)
	plt.savefig(save_path, format="pdf")
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
	bigram = n_gram_per_label(file_paths=file_paths, n=2, top_k=3)
	trigram = n_gram_per_label(file_paths=file_paths, n=3, top_k=3)
	plot_n_gram(df=unigram, save_path=os.path.join("plots", "unigram_per_label.pdf"))
	plot_n_gram(df=bigram, save_path=os.path.join("plots", "bigram_per_label.pdf"))
	plot_n_gram(df=trigram, save_path=os.path.join("plots", "trigram_per_label.pdf"))
	plot_n_gram_sublpots(
		unigram=unigram, bigram=bigram, trigram=trigram, save_path=os.path.join("plots", "n_gram_subplot.pdf")
	)
