import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(predictions, labels):
	predictions = np.argmax(predictions, axis=2)

	true_predictions = [p for pred, lbl in zip(predictions, labels) for p, la in zip(pred, lbl) if la != -100]
	true_labels = [la for pred, lbl in zip(predictions, labels) for _, la in zip(pred, lbl) if la != -100]

	true_predictions_no_o = [
		p for pred, lbl in zip(predictions, labels) for p, la in zip(pred, lbl) if la not in [-100, 0]
	]
	true_labels_no_o = [la for pred, lbl in zip(predictions, labels) for _, la in zip(pred, lbl) if la not in [-100, 0]]

	metrics = {}
	log_metrics = {}
	for name, (preds, lbls) in {
		"all": (true_predictions, true_labels),
		"no_o": (true_predictions_no_o, true_labels_no_o),
	}.items():
		accuracy = accuracy_score(lbls, preds)
		precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
			lbls, preds, average="micro", zero_division=0
		)
		precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
			lbls, preds, average="macro", zero_division=0
		)
		total = accuracy + precision_micro + recall_micro + f1_micro + precision_macro + recall_macro + f1_macro

		metrics[name] = {
			"Accuracy": accuracy,
			"Precision_micro": precision_micro,
			"Recall_micro": recall_micro,
			"F1_micro": f1_micro,
			"Precision_macro": precision_macro,
			"Recall_macro": recall_macro,
			"F1_macro": f1_macro,
			"Total": total,
		}

		log_metrics.update(
			{
				f"{name}_accuracy": accuracy,
				f"{name}_precision_micro": precision_micro,
				f"{name}_recall_micro": recall_micro,
				f"{name}_f1_micro": f1_micro,
				f"{name}_precision_macro": precision_macro,
				f"{name}_recall_macro": recall_macro,
				f"{name}_f1_macro": f1_macro,
			}
		)

	return metrics, log_metrics
