import os
import numpy as np
import evaluate
from transformers import (
	AutoConfig,
	AutoTokenizer,
	AutoModelForTokenClassification,
	DataCollatorForTokenClassification,
	TrainingArguments,
	Trainer,
)
from NER.model import BertWithPOS
from utils.utils import load_bio_labels, load_pkl_data
from dotenv import load_dotenv
from huggingface_hub import login

#config = AutoConfig.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
##config.num_pos_tags = 18
##config.num_labels = 27

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
login(HUGGING_FACE_TOKEN)

label_list, label2id, id2label = load_bio_labels()

training_data = load_pkl_data(os.path.join("data_preprocessed", "training.pkl"))
validation_data = load_pkl_data(os.path.join("data_preprocessed", "validation.pkl"))

print("training_data:", training_data[0]["labels"])
# print type of training_data
print("training_data type:", type(training_data))

model = BertWithPOS(
    model_name_or_path="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    num_pos_tags=18,
    num_labels=27
)
#model = BertModelWithPOSEmbeddings(config)

tokenizer = AutoTokenizer.from_pretrained(
	"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", use_fast=True
)

data_collator = DataCollatorForTokenClassification(tokenizer)
#data_collator = POSAwareDataCollator(tokenizer)

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
    #remove_unused_columns=False,
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

for batch in trainer.get_train_dataloader():
	print(batch.keys())  # should include 'pos_tag_ids'
	break



trainer.train()
trainer.push_to_hub()
