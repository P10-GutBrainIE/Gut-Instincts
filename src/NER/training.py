import argparse
import yaml
import os
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
	AutoTokenizer,
	AutoModelForTokenClassification,
	DataCollatorForTokenClassification,
)
import torch
from utils.utils import load_bio_labels, load_pkl_data


def training(config):
	os.environ["MLFLOW_EXPERIMENT_NAME"] = config["experiment_name"]

	training_data = load_pkl_data(config["training_data_path"])
	validation_data = load_pkl_data(config["validation_data_path"])

	label_list, label2id, id2label = load_bio_labels()

	model = AutoModelForTokenClassification.from_pretrained(
		config["model_name"], num_labels=len(label_list), id2label=id2label, label2id=label2id
	)
	tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
	data_collator = DataCollatorForTokenClassification(tokenizer)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	train_loader = torch.utils.data.DataLoader(
		training_data,
		batch_size=config["hyperparameters"]["batch_size"],
		shuffle=True,
		collate_fn=data_collator,
	)
	val_loader = torch.utils.data.DataLoader(
		validation_data,
		batch_size=config["hyperparameters"]["batch_size"],
		shuffle=False,
		collate_fn=data_collator,
	)

	optimizer = torch.optim.AdamW(model.parameters(), lr=config["hyperparameters"]["learning_rate"])
	num_epochs = config["hyperparameters"]["num_epochs"]
	best_f1 = 0.0

	def _compute_metrics(predictions, labels):
		predictions = np.argmax(predictions, axis=2)
		true_predictions = [p for pred, lbl in zip(predictions, labels) for p, la in zip(pred, lbl) if la != -100]
		true_labels = [la for pred, lbl in zip(predictions, labels) for _, la in zip(pred, lbl) if la != -100]

		precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
			true_labels, true_predictions, average="micro"
		)
		precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
			true_labels, true_predictions, average="macro"
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

	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for batch in train_loader:
			batch = {k: v.to(device) for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss += loss.item()

		print(f"Epoch {epoch + 1}/{num_epochs} | Training loss: {total_loss / len(train_loader):.4f}")

		model.eval()
		all_preds = []
		all_labels = []
		with torch.no_grad():
			for batch in val_loader:
				labels = batch["labels"].cpu().numpy()
				batch = {k: v.to(device) for k, v in batch.items()}
				outputs = model(**batch)
				logits = outputs.logits.detach().cpu().numpy()
				all_preds.extend(logits)
				all_labels.extend(labels)

		metrics = _compute_metrics(all_preds, all_labels)
		print(f"Validation metrics (epoch {epoch + 1}): {metrics}")

		mlflow.log_metrics(metrics, step=epoch)

		if metrics["f1_micro"] > best_f1:
			best_f1 = metrics["f1_micro"]
			output_dir = os.path.join("models", config["experiment_name"])
			os.makedirs(output_dir, exist_ok=True)
			model.save_pretrained(output_dir)
			tokenizer.save_pretrained(output_dir)
			print(f"New best model saved with F1_micro: {best_f1:.4f}")

	mlflow.end_run()


if __name__ == "__main__":
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		print("Device count:", torch.cuda.device_count())
		print("CUDA is available. GPU:", torch.cuda.get_device_name(0))

		parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
		parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
		args = parser.parse_args()

		with open(args.config, "r") as file:
			config = yaml.safe_load(file)
			os.makedirs("models", exist_ok=True)
			training(config)

	else:
		print("CUDA is not available.")
		exit()
