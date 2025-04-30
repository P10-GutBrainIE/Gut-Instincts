import argparse
import os
import sys
import yaml
import mlflow
import torch
from tqdm import tqdm

from utils.utils import (
	load_bio_labels,
	load_relation_labels,
	load_pkl_data,
	make_dataset_dir_name,
	print_evaluation_metrics,
)
from NER.compute_metrics import compute_evaluation_metrics
import torch
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
	elif config["model_type"] == "re":
		from NER.architectures.bert_with_entity_start import BertForREWithEntityStart

		return BertForREWithEntityStart(model_name=config["model_name"], num_labels=len(label_list))
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

	if config["model_type"] == "re":
		from NER.cached_re_dev_inputs import build_cached_inputs

		model_tag = config["model_name"].replace("/", "_")
		cached_input_path = os.path.join("data_preprocessed", f"dev_cached_inputs_{model_tag}.pkl")

		if not os.path.exists(cached_input_path):
			print("computing cached inference")
			build_cached_inputs(
				dev_json_path=os.path.join("data", "Annotation", "Dev", "json_format", "dev.json"),
				tokenizer_name=config["model_name"],
				output_pkl_path=cached_input_path
			)

		if config.get("subtask") == "6.2.1":
			label_list = ["relation"]
			label2id = {"relation": 1}
			id2label = {1: "relation"}
		else:
			label_list, label2id, id2label = load_relation_labels()
	else:
		label_list, label2id, id2label = load_bio_labels()

	model = build_model(config, label_list, id2label, label2id)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	training_data = load_pkl_data(
		os.path.join("data_preprocessed", config["experiment_name"], dataset_dir_name, "training.pkl")
	)
	validation_data = load_pkl_data(
		os.path.join("data_preprocessed", config["experiment_name"], dataset_dir_name, "validation.pkl")
	)
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

		current_lr = optimizer.param_groups[0]["lr"]
		avg_loss = total_loss / len(train_loader)
		mlflow.log_metrics({"lr": current_lr, "loss": avg_loss}, step=epoch)
		print(
			f"Epoch {epoch + 1}/{num_epochs} | Avg. training loss per batch: {avg_loss:.4f} | Learning rate: {current_lr:.8f}"
		)

		if config["hyperparameters"]["lr_scheduler"]["method"] == "custom":
			scheduler.step()
		elif config["hyperparameters"]["lr_scheduler"]["method"] == "reduce on plateau":
			scheduler.step(avg_loss)
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
				if config["model_type"] == "huggingface" or config["model_type"] == "re":
					preds = outputs["logits"].argmax(-1).cpu().tolist()
				else:
					preds = outputs.get("decoded_tags", outputs["logits"].argmax(-1).cpu().tolist())
				all_preds.extend(preds)
				all_labels.extend(labels)

		metrics = compute_evaluation_metrics(
			model=model, model_name=config["model_name"], model_type=config["model_type"], subtask=config["subtask"]
		)
		model.to(device)
		print_evaluation_metrics(metrics)
		mlflow.log_metrics(metrics, step=epoch)

		if metrics["F1_micro"] > best_f1_micro:
			best_f1_micro = metrics["F1_micro"]
			model.save(output_dir)
			print(f"New best model saved with F1_micro: {best_f1_micro:.4f}")

		if epoch == num_epochs - 1:
			mlflow.log_metric({"Best F1_micro": best_f1_micro})

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
