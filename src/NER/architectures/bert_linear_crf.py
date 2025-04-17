import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


class BertLSTMCRF(nn.Module):
	def __init__(self, model_name, num_labels, lstm_hidden_dim=256, bidirectional=True, dropout_prob=0.3):
		super().__init__()
		self.bert = AutoModel.from_pretrained(model_name)
		self.dropout = nn.Dropout(dropout_prob)

		self.lstm = nn.LSTM(
			input_size=self.bert.config.hidden_size,
			hidden_size=lstm_hidden_dim,
			num_layers=1,
			batch_first=True,
			bidirectional=bidirectional,
		)

		lstm_output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
		self.classifier = nn.Linear(lstm_output_dim, num_labels)

		self.crf = CRF(num_tags=num_labels, batch_first=True)

	def forward(self, input_ids, attention_mask=None):
		"""
		Standard forward pass. Returns emissions (logits) for loss computation outside.
		"""
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		sequence_output = self.dropout(outputs.last_hidden_state)

		lstm_output, _ = self.lstm(sequence_output)
		emissions = self.classifier(lstm_output)

		return emissions

	def compute_crf_loss(self, emissions, labels, attention_mask=None):
		"""
		Computes CRF negative log-likelihood loss.
		"""
		mask = attention_mask.bool() if attention_mask is not None else None
		log_likelihood = self.crf(emissions, labels, mask=mask, reduction="mean")
		return -log_likelihood

	def decode(self, emissions, attention_mask=None):
		"""
		Decodes the most probable sequence using CRF.
		"""
		mask = attention_mask.bool() if attention_mask is not None else None
		predictions = self.crf.decode(emissions, mask=mask)
		return predictions
