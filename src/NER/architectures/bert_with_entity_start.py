import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class BertForREWithEntityStart(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
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
        self.classifier = nn.Linear(self.hidden_size * 2, num_labels)

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
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}
    
    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))




# TODO: sometimes people use average spans between [E1] and [/E1] instead of just [E1] itself.
# If entity mentions are long, taking only the start token might not capture the full meaning.
# class BertForREWithEntityStart(nn.Module):
# 	def __init__(self, config, e1_token_id, e2_token_id):
# 		super().__init__(config)
# 		self.bert = AutoModel.from_config(config)

# 		type_vocab_size = getattr(config, "type_vocab_size", 2)
# 		self.bert.embeddings.token_type_embeddings = nn.Embedding(type_vocab_size, config.hidden_size)

# 		self.dropout = nn.Dropout(config.hidden_dropout_prob)
# 		self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

# 		self.e1_token_id = e1_token_id
# 		self.e2_token_id = e2_token_id

# 		self.init_weights()

# 	def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
# 		outputs = self.bert(
# 			input_ids=input_ids,
# 			attention_mask=attention_mask,
# 			token_type_ids=None
# 		)
# 		last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

# 		# Find [E1] and [E2] token positions for each sample in batch
# 		e1_mask = (input_ids == self.e1_token_id)
# 		e2_mask = (input_ids == self.e2_token_id)

# 		e1_hidden = self._entity_representation(last_hidden_state, e1_mask)
# 		e2_hidden = self._entity_representation(last_hidden_state, e2_mask)

# 		# Concatenate and classify
# 		combined = torch.cat([e1_hidden, e2_hidden], dim=-1)
# 		combined = self.dropout(combined)

# 		logits = self.classifier(combined)

# 		loss = None
# 		if labels is not None:
# 			loss_fct = nn.CrossEntropyLoss()
# 			loss = loss_fct(logits, labels)

# 		return {"loss": loss, "logits": logits}

# 	def _entity_representation(self, hidden_states, mask):
# 		mask = mask.unsqueeze(1).float()
# 		entity_hidden = torch.bmm(mask, hidden_states).squeeze(1)
# 		return entity_hidden