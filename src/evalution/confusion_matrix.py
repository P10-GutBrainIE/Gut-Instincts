import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.utils import load_json_data


def compare_classes(inference_data, test_data):
	"""
	Compare the entities between data and test_data based on start_idx and end_idx.

	Args:
	    data (dict): Dictionary containing the ground truth data.
	    test_data (dict): Dictionary containing the predicted data.

	Returns:
	    tuple: Two lists containing the true labels and predicted labels.
	"""
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


def plot_confusion_matrix(y_test, y_pred, target_names):
	"""
	Plot a confusion matrix with blue gradient for correct predictions (diagonal)
	and red gradient for wrong predictions (off-diagonal).

	Args:
	    y_test (list): List of true labels.
	    y_pred (list): List of predicted labels.
	    target_names (list): List of target names for the labels.
	"""
	cm = confusion_matrix(y_test, y_pred, labels=target_names)
	mask_diag = np.eye(cm.shape[0], dtype=bool)
	mask_off_diag = ~mask_diag

	fig, ax = plt.subplots(figsize=(10, 10))

	sns.heatmap(
		cm,
		mask=mask_diag,
		annot=False,
		fmt="d",
		cmap="Reds",
		cbar=False,
		ax=ax,
		xticklabels=target_names,
		yticklabels=target_names,
	)

	sns.heatmap(
		cm,
		mask=mask_off_diag,
		annot=False,
		fmt="d",
		cmap="Blues",
		cbar=True,
		ax=ax,
		xticklabels=target_names,
		yticklabels=target_names,
	)

	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j + 0.5, i + 0.5, format(cm[i, j], "d"), ha="center", va="center", color="black")

	ax.set_xlabel("Predicted Label")
	ax.set_ylabel("True Label")
	plt.tight_layout()

	os.makedirs("plots", exist_ok=True)
	plt.savefig(os.path.join("plots", "confusion_matrix.png"))
	plt.close()


if __name__ == "__main__":
	inference_results = load_json_data(os.path.join("data_inference_results", "ner.json"))
	test_data = load_json_data(os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"))

	true_labels, pred_labels = compare_classes(inference_results, test_data)
	target_names = [
		"O",
		"anatomical location",
		"animal",
		"biomedical technique",
		"bacteria",
		"chemical",
		"dietary supplement",
		"DDF",
		"drug",
		"food",
		"gene",
		"human",
		"microbiome",
		"statistical technique",
	]
	plot_confusion_matrix(true_labels, pred_labels, target_names)
