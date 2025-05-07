import argparse
import os
import sys
import yaml
import time
import matplotlib.pyplot as plt
import mlflow
import torch
from tqdm import tqdm
from utils.utils import load_bio_labels, load_pkl_data, make_dataset_dir_name, print_metrics
from training.calculate_metrics import compute_metrics
from training.dataset import Dataset
from training.freezing import freeze_bert, unfreeze_bert
from training.lr_scheduler import lr_scheduler
from training.seeding import seed_everything

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def build_model(config, label_list=None, id2label=None, label2id=None):
	if config["model_type"] == "huggingface":
		from architectures.hf_token_classifier import HFTokenClassifier

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
		from architectures.bert_lstm_crf import BertLSTMCRF

		return BertLSTMCRF(
			model_name=config["model_name"],
			num_labels=len(label_list),
		)
	elif config["model_type"] == "bertdensecrf":
		from architectures.bert_dense_crf import BertDenseCRF

		return BertDenseCRF(
			model_name=config["model_name"],
			num_labels=len(label_list),
		)
	elif config["model_type"] == "re":
		from architectures.bert_with_entity_start import BertForREWithEntityStart

		return BertForREWithEntityStart(model_name=config["model_name"], subtask=config["subtask"])
	else:
		raise ValueError("Unknown model_type")


def training(config):
	start_time = time.time()
	dataset_dir_name = make_dataset_dir_name(config)

	output_dir = os.path.join("models", dataset_dir_name)
	os.makedirs(output_dir, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if config.get("resume_checkpoint"):
		checkpoint = torch.load(os.path.join(output_dir, "checkpoint.pth"), map_location=device)
		start_epoch = checkpoint["epoch"] + 1
		best_f1_micro = checkpoint["best_f1_micro"]
		run_id = checkpoint["run_id"]
	else:
		start_epoch = 0
		best_f1_micro = 0.0
		run_id = None

	mlflow.set_experiment(experiment_name=config["experiment_name"])
	if run_id:
		mlflow.start_run(run_id=run_id)
	else:
		mlflow.start_run()
		mlflow.log_params(params=config)

	if config["model_type"] != "re":
		label_list, label2id, id2label = load_bio_labels()
		model = build_model(config, label_list, id2label, label2id)
	else:
		model = build_model(config)

	freeze_epochs = config["hyperparameters"].get("freeze_epochs", 0)
	if freeze_epochs > 0:
		print(f"Freezing BERT parameters for the first {freeze_epochs} epochs")
		freeze_bert(model)

	model.to(device)

	training_data = load_pkl_data(os.path.join("data_preprocessed", dataset_dir_name, "training.pkl"))
	training_dataset = Dataset(training_data, with_weights=config["weighted_training"])

	train_loader = torch.utils.data.DataLoader(
		training_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True, pin_memory=True
	)

	current_lr = config["hyperparameters"]["lr_scheduler"]["learning_rate"]
	optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr)
	scheduler = lr_scheduler(
		lr_scheduler_dict=config["hyperparameters"],
		optimizer=optimizer,
		steps_per_epoch=len(train_loader),
	)

	if config.get("resume_checkpoint"):
		model.load_state_dict(checkpoint["model_state_dict"])
		optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

	global_step = 0
	num_epochs = config["hyperparameters"]["num_epochs"]
	for epoch in tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch"):
		model.train()

		if freeze_epochs > 0 and epoch == freeze_epochs:
			print(f"\nUnfreezing BERT parameters after {epoch + 1} epochs")
			unfreeze_bert(model)

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
				current_lr = optimizer.param_groups[0]["lr"]
				mlflow.log_metric("batch_lr", current_lr, step=global_step)
				mlflow.log_metric("batch_loss", loss.item(), step=global_step)
				global_step += 1
				scheduler.step()

		current_lr = optimizer.param_groups[0]["lr"]
		avg_loss = total_loss / len(train_loader)
		mlflow.log_metrics({"epoch_lr": current_lr, "epoch_loss": avg_loss}, step=epoch)
		print(
			f"Epoch {epoch + 1}/{num_epochs} | Avg. training loss per batch: {avg_loss:.4f} | Learning rate: {current_lr:.8f}"
		)

		model.eval()
		metrics = compute_metrics(config=config, model=model)
		print_metrics(metrics)
		mlflow.log_metrics(metrics, step=epoch)

		if config["hyperparameters"]["lr_scheduler"]["method"] in ["custom", "cosine annealing"]:
			scheduler.step()
		elif config["hyperparameters"]["lr_scheduler"]["method"] == "reduce on plateau":
			scheduler.step(metrics["F1_micro"])

		if metrics["F1_micro"] > best_f1_micro:
			best_f1_micro = metrics["F1_micro"]
			model.save(output_dir)
			print(f"New best model saved with F1_micro: {best_f1_micro:.4f}")

		if epoch == num_epochs - 1:
			mlflow.log_metric("Best F1_micro", best_f1_micro)

		if time.time() - start_time > 32400:
			print("Training time exceeded 10 hours. Saving checkpoint and exiting.")
			checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
			torch.save(
				{
					"epoch": epoch,
					"model_state_dict": model.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"scheduler_state_dict": scheduler.state_dict(),
					"best_f1_micro": best_f1_micro,
					"run_id": mlflow.active_run().info.run_id,
				},
				checkpoint_path,
			)
			mlflow.end_run()
			return

	mlflow.end_run()


def find_optimal_lr(config, min_lr=1e-7, max_lr=1):
	print("Running learning rate finder instead of training")
	dataset_dir_name = make_dataset_dir_name(config)
	label_list, label2id, id2label = load_bio_labels()
	model = build_model(config, label_list, id2label, label2id)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	training_data = load_pkl_data(os.path.join("data_preprocessed", dataset_dir_name, "training.pkl"))
	training_dataset = Dataset(training_data, with_weights=config["weighted_training"])
	train_loader = torch.utils.data.DataLoader(
		training_dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True, pin_memory=True
	)

	optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr)
	lr_mult = (max_lr / min_lr) ** (1 / len(train_loader))
	lrs = []
	losses = []
	iter_loader = iter(train_loader)
	lr = min_lr
	best_loss = float("inf")
	avg_loss = 0.0
	beta = 0.98

	for batch_num in tqdm(range(len(train_loader)), desc="Finding best LR", unit="batch"):
		try:
			batch = next(iter_loader)
		except StopIteration:
			iter_loader = iter(train_loader)
			batch = next(iter_loader)
		for k, v in batch.items():
			if isinstance(v, torch.Tensor):
				batch[k] = v.to(device)
		optimizer.zero_grad()
		if config.get("weighted_training"):
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
		avg_loss = beta * avg_loss + (1 - beta) * loss.item()
		smoothed_loss = avg_loss / (1 - beta ** (batch_num + 1))
		if smoothed_loss < best_loss or batch_num == 0:
			best_loss = smoothed_loss
		lrs.append(lr)
		losses.append(smoothed_loss)
		loss.backward()
		optimizer.step()
		lr *= lr_mult
		for param_group in optimizer.param_groups:
			param_group["lr"] = lr
		if smoothed_loss > 4 * best_loss:
			break

	plt.figure()
	plt.plot(lrs, losses)
	plt.xscale("log")
	plt.xlabel("Learning Rate (log scale)")
	plt.ylabel("Smoothed Loss")
	plt.title("Learning Rate Finder")
	os.makedirs(os.path.join("plots", "lr_vs_loss"), exist_ok=True)
	plt.savefig(os.path.join("plots", "lr_vs_loss", f"{config['experiment_name']}.pdf"), format="pdf")

	print("Learning rate finder finished. Inspect the plot and select an LR just before the loss increases rapidly.")


if __name__ == "__main__":
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}. Device count: {torch.cuda.device_count()}.")

		parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
		parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
		args = parser.parse_args()

		with open(args.config, "r") as file:
			config = yaml.safe_load(file)
			os.makedirs("models", exist_ok=True)

			seed = config.get("seed", 666)
			print("Seed: ", seed)
			seed_everything(seed)

			if config.get("find_optimal_lr"):
				find_optimal_lr(config)
			else:
				training(config)
	else:
		print("CUDA is not available.")
		exit()
