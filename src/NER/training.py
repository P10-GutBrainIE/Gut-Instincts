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
from utils.utils import load_bio_labels, load_data
from huggingface_hub import login

login("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")

label_list, label2id, id2label = load_bio_labels()

training_data = load_data(os.path.join("data", "preprocessed", "train.pkl"))
validation_data = load_data(os.path.join("data", "preprocessed", "val.pkl"))

model = AutoModelForTokenClassification.from_pretrained(
	"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", num_labels=27, id2label=id2label, label2id=label2id
)

tokenizer = AutoTokenizer.from_pretrained(
	"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", use_fast=True
)

data_collator = DataCollatorForTokenClassification(tokenizer)

seqeval = evaluate.load("seqeval")


def compute_metrics(p):
	predictions, labels = p
	predictions = np.argmax(predictions, axis=2)

	true_predictions = [
		[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]
	true_labels = [
		[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
		for prediction, label in zip(predictions, labels)
	]

	results = seqeval.compute(predictions=true_predictions, references=true_labels)
	return {
		"precision": results["overall_precision"],
		"recall": results["overall_recall"],
		"f1": results["overall_f1"],
		"accuracy": results["overall_accuracy"],
	}


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
	push_to_hub=True,
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
trainer.push_to_hub()
