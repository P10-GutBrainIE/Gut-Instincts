import os
import argparse
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from utils.utils import load_bio_labels, load_pkl_data
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
login(HUGGING_FACE_TOKEN)

# Parse model name from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()
model_name = args.model_name

label_list, label2id, id2label = load_bio_labels()
data_dir = os.path.join("data_preprocessed", model_name.split("/")[-1])
training_data = load_pkl_data(os.path.join(data_dir, "training.pkl"))
validation_data = load_pkl_data(os.path.join(data_dir, "validation.pkl"))

model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

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

output_dir = os.path.join("outputs", model_name.split("/")[-1] + "_NER")

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()
