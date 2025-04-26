import argparse
import os
import sys
import yaml
import mlflow
import torch
from tqdm import tqdm

from utils.utils import load_bio_labels, load_pkl_data, make_dataset_dir_name, print_metrics
from NER.compute_metrics import compute_metrics
from NER.dataset import Dataset
from NER.lr_scheduler import lr_scheduler

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def build_model(config, label_list, id2label, label2id):
	if config["model_type"] == "huggingface":
		from NER.architectures.hf_token_classifier import HFTokenClassifier

		if config["weighted_training"]:
			loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
			return HFTokenClassifier(
				model_name=config["model_name"],
				num_labels=len(label_list),
				id2label=id2label,
				label2id=label2id,
				loss_fn=loss_fn,
			)
		else:
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
		)
	else:
		raise ValueError("Unknown model_type")


def training(config):
	dataset_dir_name = make_dataset_dir_name(
		config["dataset_qualities"], config["weighted_training"], config.get("dataset_weights")
	)

	output_dir = os.path.join("models", config["experiment_name"], dataset_dir_name)
	os.makedirs(output_dir, exist_ok=True)

	mlflow.set_experiment(experiment_name=config["experiment_name"])
	mlflow.start_run()
	mlflow.log_params(params=config)

	label_list, label2id, id2label = load_bio_labels()
	model = build_model(config, label_list, id2label, label2id)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	training_data = load_pkl_data(
		os.path.join("data_preprocessed", config["experiment_name"], dataset_dir_name, "training.pkl")
	)
	training_dataset = Dataset(training_data, with_weights=config["weighted_training"])

	train_loader = torch.utils.data.DataLoader(
		training_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True, pin_memory=True
	)

	current_lr = config["hyperparameters"]["lr_scheduler"]["learning_rate"]
	optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)
	scheduler = lr_scheduler(
		lr_scheduler_dict=config["hyperparameters"]["lr_scheduler"],
		optimizer=optimizer,
		steps_per_epoch=len(train_loader),
	)

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
			if config["weighted_training"]:
				outputs = model(
					batch["input_ids"],
					attention_mask=batch.get("attention_mask"),
					labels=batch.get("labels"),
					weight=batch.get("weight"),
				)
			else:
				outputs = model(
					batch["input_ids"],
					attention_mask=batch.get("attention_mask"),
					labels=batch.get("labels"),
				)
			loss = outputs["loss"]
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

			if config["hyperparameters"]["lr_scheduler"]["method"] == "one cycle":
				scheduler.step()

		current_lr = optimizer.param_groups[0]["lr"]
		avg_loss = total_loss / len(train_loader)
		mlflow.log_metrics({"lr": current_lr, "loss": avg_loss}, step=epoch)
		print(
			f"Epoch {epoch + 1}/{num_epochs} | Avg. training loss per batch: {avg_loss:.4f} | Learning rate: {current_lr:.8f}"
		)

		if config["hyperparameters"]["lr_scheduler"]["method"] == "custom" or "cosine annealing":
			scheduler.step()
		elif config["hyperparameters"]["lr_scheduler"]["method"] == "reduce on plateau":
			scheduler.step(avg_loss)

		model.eval()
		if config["remove_html"]:
			data_path = os.path.join("data_preprocessed", config["experiment_name"], dataset_dir_name, "validation.pkl")
		else:
			data_path = os.path.join("data", "Annotations", "Dev", "json_format", "dev.json")
		metrics = compute_metrics(
			model=model,
			model_name=config["model_name"],
			model_type=config["model_type"],
			data_path=data_path,
		)
		model.to(device)
		print_metrics(metrics)
		mlflow.log_metrics(metrics, step=epoch)

		if metrics["F1_micro"] > best_f1_micro:
			best_f1_micro = metrics["F1_micro"]
			model.save(output_dir)
			print(f"New best model saved with F1_micro: {best_f1_micro:.4f}")

		if epoch == num_epochs - 1:
			mlflow.log_metric("Best F1_micro", best_f1_micro)

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
			os.makedirs("models", exist_ok=True)
			training(config)
	else:
		print("CUDA is not available.")
		exit()
