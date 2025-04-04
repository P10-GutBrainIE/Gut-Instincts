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
	TrainingArguments,
	Trainer,
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

	def _compute_metrics(p):
		predictions, labels = p
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

	training_args = TrainingArguments(
		learning_rate=config["hyperparameters"]["learning_rate"],
		per_device_train_batch_size=config["hyperparameters"]["batch_size"],
		per_device_eval_batch_size=config["hyperparameters"]["batch_size"],
		num_train_epochs=config["hyperparameters"]["num_epochs"],
		output_dir=os.path.join("models", config["experiment_name"]),
		weight_decay=0.01,
		logging_strategy="epoch",
		eval_strategy="epoch",
		save_strategy="epoch",
		metric_for_best_model="f1_micro",
		push_to_hub=False,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=training_data,
		eval_dataset=validation_data,
		processing_class=tokenizer,
		data_collator=data_collator,
		compute_metrics=_compute_metrics,
	)

	trainer.train()

	mlflow.end_run()


if __name__ == "__main__":
	if torch.cuda.is_available():
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
