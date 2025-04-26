from transformers import AutoModel, PreTrainedModel
import torch.nn as nn
import torch

class BertForREWithEntityStart(PreTrainedModel):
    def __init__(self, config, e1_token_id, e2_token_id):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

        self.e1_token_id = e1_token_id
        self.e2_token_id = e2_token_id

        self.init_weights()


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # Find [E1] and [E2] token positions for each sample in batch
        e1_mask = (input_ids == self.e1_token_id)
        e2_mask = (input_ids == self.e2_token_id)

        # Extract hidden states at [E1] and [E2] positions
        def extract_entity(hidden, mask):
            # mask: [batch, seq_len] -> [batch, 1, seq_len]
            mask = mask.unsqueeze(1).float()
            # masked selection (zero out non-[E1]/[E2]) then max-pool to reduce dim
            return (mask @ hidden).squeeze(1)  # [batch, hidden_size]

        e1_hidden = extract_entity(last_hidden, e1_mask)
        e2_hidden = extract_entity(last_hidden, e2_mask)
        #e1_hidden = hidden_states[e1_mask].view(-1, hidden_states.size(-1))
        #e2_hidden = hidden_states[e2_mask].view(-1, hidden_states.size(-1))

        # Concatenate and classify
        combined = torch.cat([e1_hidden, e2_hidden], dim=-1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits}