import os
import torch.nn as nn
from transformers import AutoModelForTokenClassification


class HFTokenClassifier(nn.Module):
	def __init__(self, model_name, num_labels, id2label=None, label2id=None):
		super().__init__()
		self.model = AutoModelForTokenClassification.from_pretrained(
			model_name,
			num_labels=num_labels,
			id2label=id2label,
			label2id=label2id,
		)

	def forward(self, input_ids, attention_mask=None, labels=None):
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			labels=labels,
			return_dict=True,
		)
		return {
			"loss": outputs.loss if labels is not None else None,
			"logits": outputs.logits,
		}

	def save(self, output_dir):
		os.makedirs(output_dir, exist_ok=True)
		self.model.save_pretrained(output_dir)

	@classmethod
	def load(cls, output_dir):
		model = AutoModelForTokenClassification.from_pretrained(output_dir)
		wrapper = cls.__new__(cls)
		wrapper.model = model
		return wrapper
