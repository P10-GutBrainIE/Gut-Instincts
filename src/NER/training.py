import argparse
import yaml
import os
import mlflow
from transformers import (
	AutoTokenizer,
	AutoModelForTokenClassification,
)
import torch
from utils.utils import load_bio_labels, load_pkl_data, set_experiment_id
from NER.compute_metrics import compute_metrics
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


class CustomDataset(torch.utils.data.Dataset):
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		sample["input_ids"] = torch.tensor(sample["input_ids"], dtype=torch.long).clone().detach()
		sample["attention_mask"] = torch.tensor(sample["attention_mask"], dtype=torch.long).clone().detach()
		sample["labels"] = torch.tensor(sample["labels"], dtype=torch.long).clone().detach()
		return sample


def switch_freeze_state_model_parameters(model):
	# freezes embeddings and bottom 6 encoder layers
	for name, param in model.base_model.named_parameters():
		if any(layer in name for layer in [f"encoder.layer.{i}" for i in range(6)]) or "embeddings" in name:
			param.requires_grad = not param.requires_grad

	return model


def print_metrics(metrics):
	print("Validation metrics:\n")
	all_metrics = metrics.get("all", {})
	no_o_metrics = metrics.get("no_o", {})

	print(f"{'Metric':<25} {'All':>10} {'No_O':>10}")
	print("-" * 47)

	for key in sorted(all_metrics.keys()):
		all_value = all_metrics.get(key, 0.0)
		no_o_value = no_o_metrics.get(key, 0.0)
		print(f"{key:<25} {all_value:>10.4f} {no_o_value:>10.4f}")


def training(config):
	os.environ["MLFLOW_EXPERIMENT_NAME"] = config["experiment_name"]
	os.environ["MLFLOW_EXPERIMENT_ID"] = set_experiment_id(config["experiment_name"])

	training_data = load_pkl_data(config["training_data_path"])
	validation_data = load_pkl_data(config["validation_data_path"])

	label_list, label2id, id2label = load_bio_labels()

	model = AutoModelForTokenClassification.from_pretrained(
		config["model_name"], num_labels=len(label_list), id2label=id2label, label2id=label2id
	)
	tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

	freeze_epochs = config["hyperparameters"]["freeze_epochs"]
	if freeze_epochs > 0:
		print(f"Freezing model parameters for the first {freeze_epochs} epochs")
		model = switch_freeze_state_model_parameters(model)

	device = torch.device("cuda")
	model.to(device)

	training_dataset = CustomDataset(training_data)
	validation_dataset = CustomDataset(validation_data)

	train_loader = torch.utils.data.DataLoader(
		training_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True, pin_memory=True
	)
	val_loader = torch.utils.data.DataLoader(
		validation_dataset,
		batch_size=config["hyperparameters"]["batch_size"],
		shuffle=False,
	)

	current_lr = config["hyperparameters"]["learning_rate"]
	optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=config["hyperparameters"]["num_epochs"],
		eta_min=config["hyperparameters"]["min_learning_rate"],
	)

	num_epochs = config["hyperparameters"]["num_epochs"]
	best_f1 = 0.0

	for epoch in range(num_epochs):
		model.train()

		if freeze_epochs > 0 and epoch == freeze_epochs:
			print(f"\nUnfreezing model parameters after {epoch + 1} epochs")
			model = switch_freeze_state_model_parameters(model)

		total_loss = 0
		for batch in train_loader:
			for k, v in batch.items():
				if isinstance(v, torch.Tensor):
					batch[k] = v.to(device)
			outputs = model(**batch)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			total_loss += loss.item()

		current_lr = optimizer.param_groups[0]["lr"]
		avg_loss = total_loss / len(train_loader)
		mlflow.log_params(
			{
				"lr": current_lr,
				"loss": avg_loss,
			}
		)

		print(f"Epoch {epoch + 1}/{num_epochs} | Training loss: {avg_loss:.4f} | Learning rate: {current_lr}")

		scheduler.step(epoch)

		model.eval()
		all_preds = []
		all_labels = []
		with torch.no_grad():
			for batch in val_loader:
				labels = batch["labels"].cpu().numpy()
				for k, v in batch.items():
					if isinstance(v, torch.Tensor):
						batch[k] = v.to(device)
				outputs = model(**batch)
				logits = outputs.logits.detach().cpu().numpy()
				all_preds.extend(logits)
				all_labels.extend(labels)

		metrics = compute_metrics(all_preds, all_labels)
		print_metrics(metrics)

		mlflow.log_metrics(metrics, step=epoch)

		if metrics["no_o_f1_micro"] > best_f1:
			best_f1 = metrics["no_o_f1_micro"]
			output_dir = os.path.join("models", config["experiment_name"])
			os.makedirs(output_dir, exist_ok=True)
			model.save_pretrained(output_dir)
			tokenizer.save_pretrained(output_dir)
			print(f"New best model saved with F1_micro (ignoring O class): {best_f1:.4f}")

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
