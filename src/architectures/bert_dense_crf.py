import torch
from transformers import AutoModel
from torchcrf import CRF
import os


class BertDenseCRF(torch.nn.Module):
	def __init__(self, model_name, num_labels):
		super().__init__()
		self.bert = AutoModel.from_pretrained(model_name)
		self.dropout = torch.nn.Dropout(p=0.3)
		self.dense = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
		self.dense_activation = torch.nn.GELU()
		self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
		self.crf = CRF(num_tags=num_labels, batch_first=True)

	def forward(self, input_ids, attention_mask=None, labels=None, weight=None):
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		sequence_output = self.dropout(outputs.last_hidden_state)
		dense_output = self.dense(sequence_output)
		dense_output = self.dense_activation(dense_output)
		logits = self.classifier(dense_output)

		output = {"logits": logits}

		if labels is not None:
			mask = (labels != -100) & attention_mask.bool()
			if not mask[:, 0].all():
				labels = labels.clone()
				labels[(labels[:, 0] == -100) & (attention_mask[:, 0] == 1), 0] = 0
				mask[:, 0] = True
			labels = labels.clone()
			labels[labels == -100] = 0

			crf_loss = -self.crf(logits, labels, mask=mask, reduction="none")
			if weight is not None:
				weighted_loss_avg = (crf_loss * weight.to(crf_loss.device)).mean()
				output["loss"] = weighted_loss_avg
			else:
				output["loss"] = crf_loss.mean()
		else:
			mask = attention_mask.bool() if attention_mask is not None else None
			output["decoded_tags"] = self.crf.decode(logits, mask=mask)

		return output

	def predict(self, input_ids, attention_mask=None):
		self.eval()
		with torch.no_grad():
			outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
			return outputs["decoded_tags"]

	def save(self, output_dir):
		os.makedirs(output_dir, exist_ok=True)
		torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

	@classmethod
	def load(cls, output_dir, model_name, num_labels):
		model = cls(model_name, num_labels)
		state_dict = torch.load(os.path.join(output_dir, "pytorch_model.bin"), map_location="cpu")
		model.load_state_dict(state_dict)
		return model
