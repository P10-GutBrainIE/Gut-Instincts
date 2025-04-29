from transformers import AutoModel, PreTrainedModel
import torch.nn as nn
import torch
import os


# TODO: sometimes people use average spans between [E1] and [/E1] instead of just [E1] itself.
# If entity mentions are long, taking only the start token might not capture the full menaing.
class BertForREWithEntityStart(PreTrainedModel):
	def __init__(self, config, e1_token_id, e2_token_id):
		super().__init__(config)
		self.bert = AutoModel.from_config(config)

		type_vocab_size = getattr(config, "type_vocab_size", 2)
		self.bert.embeddings.token_type_embeddings = nn.Embedding(type_vocab_size, config.hidden_size)

		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

		self.e1_token_id = e1_token_id
		self.e2_token_id = e2_token_id

		self.init_weights()

	def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
		outputs = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=None
		)
		last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

		# Find [E1] and [E2] token positions for each sample in batch
		e1_mask = (input_ids == self.e1_token_id)
		e2_mask = (input_ids == self.e2_token_id)

		e1_hidden = self._entity_representation(last_hidden_state, e1_mask)
		e2_hidden = self._entity_representation(last_hidden_state, e2_mask)

		# Concatenate and classify
		combined = torch.cat([e1_hidden, e2_hidden], dim=-1)
		combined = self.dropout(combined)

		logits = self.classifier(combined)

		loss = None
		if labels is not None:
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(logits, labels)

		return {"loss": loss, "logits": logits}

	def _entity_representation(self, hidden_states, mask):
		mask = mask.unsqueeze(1).float()
		entity_hidden = torch.bmm(mask, hidden_states).squeeze(1)
		return entity_hidden

	def save(self, output_dir):
		os.makedirs(output_dir, exist_ok=True)
		torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
