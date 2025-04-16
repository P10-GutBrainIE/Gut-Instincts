import argparse
import yaml
import os
import mlflow
from transformers import (
	AutoTokenizer,
	AutoModelForTokenClassification,
)
import torch
from utils.utils import load_bio_labels, load_pkl_data, print_metrics
from NER.compute_metrics import compute_metrics
from NER.dataset import Dataset
from NER.lr_scheduler import lr_scheduler
import sys
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def switch_freeze_state_model_parameters(model):
	# freezes embeddings and bottom 6 encoder layers
	for name, param in model.base_model.named_parameters():
		if any(layer in name for layer in [f"encoder.layer.{i}" for i in range(6)]) or "embeddings" in name:
			param.requires_grad = not param.requires_grad

	return model


def training(config):
	mlflow.set_experiment(experiment_name=config["experiment_name"])
	mlflow.start_run()

	training_data = load_pkl_data(config["training_data_path"])
	validation_data = load_pkl_data(config["validation_data_path"])

	label_list, label2id, id2label = load_bio_labels()

	model = AutoModelForTokenClassification.from_pretrained(
		config["model_name"], num_labels=len(label_list), id2label=id2label, label2id=label2id
	)
	tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)

	output_dir = os.path.join("models", config["experiment_name"])
	os.makedirs(output_dir, exist_ok=True)
	tokenizer.save_pretrained(output_dir)

	freeze_epochs = config["hyperparameters"]["freeze_epochs"]
	if freeze_epochs > 0:
		print(f"Freezing model parameters for the first {freeze_epochs} epochs")
		model = switch_freeze_state_model_parameters(model)

	device = torch.device("cuda")
	model.to(device)

	training_dataset = Dataset(training_data, with_weights=config["weighted_training"])
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

	if config["weighted_training"]:
		loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

	best_f1 = 0.0

	num_epochs = config["hyperparameters"]["num_epochs"]
	for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
		model.train()

		if freeze_epochs > 0 and epoch == freeze_epochs:
			print(f"\nUnfreezing model parameters after {epoch + 1} epochs")
			model = switch_freeze_state_model_parameters(model)

		total_loss = 0
		for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
			for k, v in batch.items():
				if isinstance(v, torch.Tensor):
					batch[k] = v.to(device)

			if config["weighted_training"]:
				outputs = model(
					input_ids=batch["input_ids"],
					attention_mask=batch["attention_mask"],
					return_dict=True,
				)
				logits = outputs.logits
				labels = batch["labels"]
				loss_values = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
				loss_values = loss_values.view(labels.size(0), -1)
				mask = (labels != -100).float()
				mean_loss_per_instance = (loss_values * mask).sum(dim=1) / mask.sum(dim=1)

				weighted_loss_avg = (mean_loss_per_instance * batch["weight"].to(mean_loss_per_instance.device)).mean()
				weighted_loss_avg.backward()

				total_loss += (mean_loss_per_instance * batch["weight"].to(mean_loss_per_instance.device)).sum().item()
			else:
				outputs = model(**batch)
				loss = outputs.loss
				loss.backward()

				total_loss += loss.item()

			optimizer.step()
			optimizer.zero_grad()

		current_lr = optimizer.param_groups[0]["lr"]
		avg_loss = total_loss / len(train_loader)

		mlflow.log_metrics(
			{
				"lr": current_lr,
				"loss": avg_loss,
			},
			step=epoch,
		)

		print(
			f"Epoch {epoch + 1}/{num_epochs} | Training loss per batch: {avg_loss:.4f} | Learning rate: {current_lr:.8f}"
		)

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
				outputs = model(**batch)
				logits = outputs.logits.detach().cpu().numpy()
				all_preds.extend(logits)
				all_labels.extend(labels)

		metrics, log_metrics = compute_metrics(all_preds, all_labels)
		print_metrics(metrics)

		mlflow.log_metrics(log_metrics, step=epoch)

		if metrics["all"]["F1_micro"] > best_f1:
			best_f1 = metrics["all"]["F1_micro"]
			model.save_pretrained(output_dir)

			print(f"New best model saved with F1_micro (ignoring O class): {best_f1:.4f}")

		if epoch == num_epochs - 1:
			output_dir = os.path.join("models", f"{config['experiment_name']}_last_epoch")
			os.makedirs(output_dir, exist_ok=True)
			tokenizer.save_pretrained(output_dir)
			model.save_pretrained(output_dir)

			print("Model at last epoch saved")

	mlflow.end_run()


if __name__ == "__main__":
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.manual_seed(42)
		torch.cuda.manual_seed_all(42)
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
