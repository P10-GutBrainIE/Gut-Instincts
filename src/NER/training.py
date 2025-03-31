import os
import numpy as np
import evaluate
from transformers import (
	AutoTokenizer,
	AutoModelForTokenClassification,
	DataCollatorForTokenClassification,
	TrainingArguments,
	Trainer,
)
import mlflow
from utils.utils import load_bio_labels, load_pkl_data
# from dotenv import load_dotenv
# from huggingface_hub import login

# load_dotenv()
# HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
# login(HUGGING_FACE_TOKEN)

label_list, label2id, id2label = load_bio_labels()

training_data = load_pkl_data(os.path.join("data_preprocessed", "training.pkl"))
validation_data = load_pkl_data(os.path.join("data_preprocessed", "validation.pkl"))

MODEL_NAME = "michiyasunaga/BioLinkBERT-large"

model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=27, id2label=id2label, label2id=label2id)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

seqeval = evaluate.load("seqeval")


def compute_metrics(p):
	predictions, labels = p
	predictions = np.argmax(predictions, axis=2)

	true_predictions = [
		[label_list[p] for (p, la) in zip(prediction, label) if la != -100]
		for prediction, label in zip(predictions, labels)
	]
	true_labels = [
		[label_list[la] for (_, la) in zip(prediction, label) if la != -100]
		for prediction, label in zip(predictions, labels)
	]

	results = seqeval.compute(predictions=true_predictions, references=true_labels)
	return {
		"precision": results["overall_precision"],
		"recall": results["overall_recall"],
		"f1": results["overall_f1"],
		"accuracy": results["overall_accuracy"],
	}


with mlflow.start_run():
	mlflow.log_param("learning_rate", 2e-5)
	mlflow.log_param("batch_size", 16)
	mlflow.log_param("num_epochs", 5)
	mlflow.log_param("model_name", MODEL_NAME)

	training_args = TrainingArguments(
		output_dir="GutBrainIE_NER_v0",
		learning_rate=2e-5,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		num_train_epochs=5,
		weight_decay=0.01,
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		push_to_hub=False,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=training_data,
		eval_dataset=validation_data,
		processing_class=tokenizer,
		data_collator=data_collator,
		compute_metrics=compute_metrics,
	)

	trainer.train()

	eval_results = trainer.evaluate(validation_data)
	mlflow.log_metrics(
		{
			"eval_loss": eval_results["eval_loss"],
			"eval_accuracy": eval_results["eval_accuracy"],
			"eval_f1": eval_results["eval_f1"],
			"eval_precision": eval_results["eval_precision"],
			"eval_recall": eval_results["eval_recall"],
		}
	)

	mlflow.pytorch.log_model(model, "model")
