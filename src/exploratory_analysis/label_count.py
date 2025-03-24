import os
import matplotlib.pyplot as plt
import pandas as pd
from utils.utils import custom_colors, label_distribution


def label_distribution_stacked_bar_plot(label_distribution: dict):
	data = []
	for quality, labels in label_distribution.items():
		for label, count in labels.items():
			data.append({"Quality": quality, "Label": label, "Count": count})
	df = pd.DataFrame(data)

	df_pivot = df.pivot(index="Label", columns="Quality", values="Count").fillna(0)

	df_pivot["Total"] = df_pivot.sum(axis=1)
	df_pivot = df_pivot.sort_values(by="Total", ascending=False).drop(columns="Total")
	df_pivot = df_pivot[["Bronze", "Silver", "Gold", "Platinum"]]

	df_pivot.plot(
		kind="bar",
		stacked=True,
		figsize=(10, 8),
		color=[custom_colors()[col] for col in df_pivot.columns],
		linewidth=0.5,
		edgecolor="grey",
	)
	plt.xlabel("Label")
	plt.xticks(rotation=45, ha="right")
	plt.ylabel("Count")
	plt.legend(title="Quality")
	plt.tight_layout()
	plt.savefig(os.path.join("plots", "label_distribution_stacked_bar_plot.png"))


if __name__ == "__main__":
	label_distribution_stacked_bar_plot(label_distribution())
