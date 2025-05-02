import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix, classification_report
from utils.utils import load_json_data, load_entity_labels, make_dataset_dir_name


def compare_classes(test_data, inference_data):
	true_labels = []
	pred_labels = []

	for paper_id, inference_content in inference_data.items():
		test_entities = test_data[paper_id]["entities"]
		for entity in inference_content["entities"]:
			false_positive = False
			start_idx = entity["start_idx"]
			end_idx = entity["end_idx"]
			pred_label = entity["label"]

			for test_entity in test_entities:
				if test_entity["start_idx"] in range(start_idx - 2, start_idx + 2) and test_entity["end_idx"] in range(
					end_idx - 2, end_idx + 2
				):
					true_label = test_entity["label"]
					true_labels.append(true_label)
					pred_labels.append(pred_label)
					false_positive = False
					break
				else:
					false_positive = True

			if false_positive:
				true_labels.append("O")
				pred_labels.append(pred_label)

	return true_labels, pred_labels


def plot_confusion_matrix(
	y_test, y_pred, target_names, save_path=os.path.join("plots", "confusion_matrices", "confusion_matrix.pdf")
):
	cm = confusion_matrix(y_test, y_pred, labels=target_names)
	mask_diag = np.eye(cm.shape[0], dtype=bool)
	mask_off_diag = ~mask_diag

	fig, ax = plt.subplots(figsize=(12, 10))

	sns.heatmap(
		cm,
		mask=mask_diag,
		annot=True,
		fmt="d",
		cmap="Reds",
		cbar=False,
		ax=ax,
		xticklabels=target_names,
		yticklabels=target_names,
		linewidths=1,
		linecolor="gray",
		square=True,
	)
	sns.heatmap(
		cm,
		mask=mask_off_diag,
		annot=True,
		fmt="d",
		cmap="Blues",
		cbar=False,
		ax=ax,
		xticklabels=target_names,
		yticklabels=target_names,
		linewidths=1,
		linecolor="gray",
		square=True,
	)

	ax.set_xlabel("Predicted Label", fontsize=18)
	ax.set_ylabel("True Label", fontsize=18)
	ax.tick_params(axis="both", which="major", labelsize=14)
	plt.xticks(rotation=45, ha="right", fontsize=14)
	plt.yticks(rotation=0, fontsize=14)
	plt.tight_layout()

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, format="pdf")
	plt.close()


def plot_classification_report(
	y_test, y_pred, target_names, save_path=os.path.join("plots", "confusion_matrices", "classification_report.pdf")
):
	report_dict = classification_report(
		y_test, y_pred, labels=target_names, target_names=target_names, output_dict=True, zero_division=0
	)

	per_label = {k: v for k, v in report_dict.items() if k in target_names}
	df = pd.DataFrame(per_label).T[["precision", "recall", "f1-score", "support"]]

	df[["precision", "recall", "f1-score"]] = df[["precision", "recall", "f1-score"]].astype(float).round(4)
	df["support"] = df["support"].astype(int)

	_, ax = plt.subplots(figsize=(len(target_names) * 0.8 + 2, len(target_names) * 0.4 + 2))
	ax.axis("off")
	tbl = ax.table(
		cellText=df.values, rowLabels=df.index, colLabels=df.columns, cellLoc="center", rowLoc="center", loc="center"
	)
	tbl.auto_set_font_size(False)
	tbl.set_fontsize(12)
	tbl.scale(1.4, 1.2)
	plt.title("Per-label Classification Metrics", fontsize=16, pad=20)
	plt.tight_layout()

	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, format="pdf")
	plt.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Create confusion matrix")
	parser.add_argument("--config", type=str, required=True, help="Path to the training config YAML file")
	args = parser.parse_args()

	with open(args.config, "r") as file:
		config = yaml.safe_load(file)

	dataset_dir_name = make_dataset_dir_name(config)

	inference_results = load_json_data(
		os.path.join("data_inference_results", config["experiment_name"], f"{dataset_dir_name}.json")
	)
	test_data = load_json_data(os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	true_labels, pred_labels = compare_classes(
		test_data=test_data,
		inference_data=inference_results,
	)
	plot_confusion_matrix(
		y_test=true_labels,
		y_pred=pred_labels,
		target_names=load_entity_labels(),
		save_path=os.path.join("plots", "confusion_matrices", f"{config['experiment_name']}_{dataset_dir_name}.pdf"),
	)
	plot_classification_report(
		y_test=true_labels,
		y_pred=pred_labels,
		target_names=load_entity_labels(),
		save_path=os.path.join(
			"plots", "confusion_matrices", f"{config['experiment_name']}_{dataset_dir_name}_metrics.pdf"
		),
	)

	# inference_results = load_json_data(os.path.join("data_inference_results", "ensemble_inference.json"))
	# test_data = load_json_data(os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	# true_labels, pred_labels = compare_classes(
	# 	test_data=test_data,
	# 	inference_data=inference_results,
	# )
	# plot_confusion_matrix(
	# 	y_test=true_labels,
	# 	y_pred=pred_labels,
	# 	target_names=load_entity_labels(),
	# 	save_path=os.path.join("plots", "confusion_matrices", "ensemble.pdf"),
	# )
	# plot_classification_report(
	# 	y_test=true_labels,
	# 	y_pred=pred_labels,
	# 	target_names=load_entity_labels(),
	# 	save_path=os.path.join("plots", "confusion_matrices", "ensemble_metrics.pdf"),
	# )
