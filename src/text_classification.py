from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import create_optimizer
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers.keras_callbacks import KerasMetricCallback
from transformers.keras_callbacks import PushToHubCallback
from huggingface_hub import login
from transformers import pipeline

login(token="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")


def preprocess_function(examples):
	return tokenizer(examples["text"], truncation=True)


imdb = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

tokenized_imdb = imdb.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
	predictions, labels = eval_pred
	predictions = np.argmax(predictions, axis=1)
	return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

batch_size = 8
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model = TFAutoModelForSequenceClassification.from_pretrained(
	"distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

tf_train_set = model.prepare_tf_dataset(
	tokenized_imdb["train"],
	shuffle=True,
	batch_size=8,
	collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
	tokenized_imdb["test"],
	shuffle=False,
	batch_size=8,
	collate_fn=data_collator,
)

model.compile(optimizer=optimizer)  # No loss argument!


metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

push_to_hub_callback = PushToHubCallback(
	output_dir="my_awesome_model",
	tokenizer=tokenizer,
)

callbacks = [metric_callback, push_to_hub_callback]

model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)

if __name__ == "__main__":
	print("testing")
