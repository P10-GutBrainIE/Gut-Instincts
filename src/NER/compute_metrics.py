import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(predictions, labels):
	predictions = np.argmax(predictions, axis=2)
	true_predictions = [p for pred, lbl in zip(predictions, labels) for p, la in zip(pred, lbl) if la != -100]
	true_labels = [la for pred, lbl in zip(predictions, labels) for _, la in zip(pred, lbl) if la != -100]

	precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
		true_labels, true_predictions, average="micro", zero_division=0
	)
	precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
		true_labels, true_predictions, average="macro", zero_division=0
	)
	accuracy = accuracy_score(true_labels, true_predictions)

	return {
		"accuracy": accuracy,
		"precision_micro": precision_micro,
		"recall_micro": recall_micro,
		"f1_micro": f1_micro,
		"precision_macro": precision_macro,
		"recall_macro": recall_macro,
		"f1_macro": f1_macro,
	}
