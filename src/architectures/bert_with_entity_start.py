import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertForREWithEntityStart(nn.Module):
	def __init__(self, model_name: str, subtask: str):
		super().__init__()
		self.encoder = AutoModel.from_pretrained(model_name)
		self.hidden_size = self.encoder.config.hidden_size

		# Add and resize for special tokens
		special_tokens = {"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]}
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
		self.tokenizer.add_special_tokens(special_tokens)
		self.encoder.resize_token_embeddings(len(self.tokenizer))

		# Get correct token IDs
		self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
		self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

		# Classifier
		self.subtask = subtask
		self.classifier = nn.Linear(self.hidden_size * 2, 1 if subtask == "6.2.1" else 18)

	def forward(self, input_ids, attention_mask, labels=None):
		outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
		sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

		# Find positions of [E1] and [E2] tokens
		e1_positions = (input_ids == self.e1_token_id).float().argmax(dim=1)  # (batch_size,)
		e2_positions = (input_ids == self.e2_token_id).float().argmax(dim=1)

		# Extract corresponding hidden states
		batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
		e1_repr = sequence_output[batch_indices, e1_positions]  # (batch_size, hidden_size)
		e2_repr = sequence_output[batch_indices, e2_positions]

		# Concatenate and classify
		combined = torch.cat([e1_repr, e2_repr], dim=-1)
		logits = self.classifier(combined)

		# Compute loss if labels provided
		loss = None
		if labels is not None:
			if self.subtask == "6.2.1":
				loss_fct = nn.BCEWithLogitsLoss()
				loss = loss_fct(logits.squeeze(-1), labels.float())
			else:
				loss_fct = nn.CrossEntropyLoss()
				loss = loss_fct(logits, labels)

		return {"loss": loss, "logits": logits}

	def predict(self, input_ids, attention_mask):
		self.eval()
		with torch.no_grad():
			outputs = self.forward(
				input_ids=input_ids,
				attention_mask=attention_mask,
			)
			logits = outputs["logits"]
			if self.subtask == "6.2.1":
				probs = torch.sigmoid(logits).squeeze(-1)
				prediction = (probs >= 0.5).long()
			else:
				prediction = torch.argmax(logits, dim=-1).squeeze()

		return prediction

	def save(self, output_dir):
		os.makedirs(output_dir, exist_ok=True)
		torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))


# TODO: sometimes people use average spans between [E1] and [/E1] instead of just [E1] itself.
# If entity mentions are long, taking only the start token might not capture the full meaning.
