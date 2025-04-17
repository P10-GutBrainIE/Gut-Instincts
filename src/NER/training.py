import argparse
import os
import sys
import yaml
import mlflow
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.utils import load_bio_labels, load_pkl_data, print_metrics
from NER.compute_metrics import compute_metrics
from NER.dataset import Dataset
from NER.lr_scheduler import lr_scheduler

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def build_model(config, label_list, id2label, label2id):
	if config["model_type"] == "huggingface":
		from NER.architectures.hf_token_classifier import HFTokenClassifier

		return HFTokenClassifier(
			model_name=config["model_name"],
			num_labels=len(label_list),
			id2label=id2label,
			label2id=label2id,
		)
	elif config["model_type"] == "bertlstmcrf":
		from NER.architectures.bert_lstm_crf import BertLSTMCRF

		return BertLSTMCRF(
			model_name=config["model_name"],
			num_labels=len(label_list),
			lstm_hidden_dim=config.get("lstm_hidden_dim", 256),
			dropout_prob=config.get("dropout_prob", 0.3),
		)
	else:
		raise ValueError("Unknown model_type")


def training(config):
	output_dir = os.path.join("models", config["experiment_name"])
	os.makedirs(output_dir, exist_ok=True)

	mlflow.set_experiment(experiment_name=config["experiment_name"])
	mlflow.start_run()

	label_list, label2id, id2label = load_bio_labels()
	model = build_model(config, label_list, id2label, label2id)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
	tokenizer.save_pretrained(output_dir)

	training_data = load_pkl_data(config["training_data_path"])
	validation_data = load_pkl_data(config["validation_data_path"])
	training_dataset = Dataset(training_data, with_weights=False)
	validation_dataset = Dataset(validation_data, with_weights=False)
	train_loader = torch.utils.data.DataLoader(
		training_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True, pin_memory=True
	)
	val_loader = torch.utils.data.DataLoader(
		validation_dataset,
		batch_size=config["hyperparameters"]["batch_size"],
		shuffle=False,
	)

	current_lr = config["hyperparameters"]["lr_scheduler"]["learning_rate"]
	optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)
	scheduler = lr_scheduler(config["hyperparameters"]["lr_scheduler"], optimizer)

	best_f1_micro = 0.0
	num_epochs = config["hyperparameters"]["num_epochs"]
	for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
		model.train()
		total_loss = 0
		for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
			for k, v in batch.items():
				if isinstance(v, torch.Tensor):
					batch[k] = v.to(device)

			optimizer.zero_grad()
			outputs = model(
				batch["input_ids"],
				attention_mask=batch.get("attention_mask"),
				labels=batch.get("labels"),
			)
			loss = outputs["loss"]
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		current_lr = optimizer.param_groups[0]["lr"]
		avg_loss = total_loss / len(train_loader)
		mlflow.log_metrics({"lr": current_lr, "loss": avg_loss})
		print(
			f"Epoch {epoch + 1}/{num_epochs} | Training loss per batch: {avg_loss:.4f} | Learning rate: {current_lr:.8f}"
		)

		if config["hyperparameters"]["lr_scheduler"]["method"] == "custom":
			scheduler.step()
		else:
			scheduler.step(epoch)

		model.eval()
		all_preds = []
		all_labels = []
		with torch.no_grad():
			for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
				labels = batch["labels"].cpu().numpy()
				for k, v in batch.items():
					if isinstance(v, torch.Tensor):
						batch[k] = v.to(device)
				outputs = model(
					batch["input_ids"],
					attention_mask=batch.get("attention_mask"),
				)
				if config["model_type"] == "huggingface":
					preds = outputs["logits"].argmax(-1).cpu().tolist()
				else:
					preds = outputs.get("decoded_tags", outputs["logits"].argmax(-1).cpu().tolist())
				all_preds.extend(preds)
				all_labels.extend(labels)

		metrics, log_metrics = compute_metrics(all_preds, all_labels)
		print_metrics(metrics)
		mlflow.log_metrics(log_metrics, step=epoch)

		if metrics["all"]["F1_micro"] > best_f1_micro:
			best_f1_micro = metrics["all"]["F1_micro"]
			model.save(output_dir)
			print(f"New best model saved with All F1_micro: {best_f1_micro:.4f}")

		if epoch == num_epochs - 1:
			output_dir_last = os.path.join("models", f"{config['experiment_name']}_last_epoch")
			os.makedirs(output_dir_last, exist_ok=True)
			model.save(output_dir_last)
			print("Model at last epoch saved")

	mlflow.end_run()


if __name__ == "__main__":
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.manual_seed(17)
		torch.cuda.manual_seed_all(17)
		print("Device count:", torch.cuda.device_count())
		print("CUDA is available. GPU:", torch.cuda.get_device_name(0))

		parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
		parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
		args = parser.parse_args()

		with open(args.config, "r") as file:
			config = yaml.safe_load(file)
			os.makedirs("models", exist_ok=True)  # Top-level dir once
			training(config)
	else:
		print("CUDA is not available.")
		exit()
