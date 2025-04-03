import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class BertWithPOS(nn.Module):
    def __init__(self, model_name_or_path, num_pos_tags, num_labels):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.hidden_size = self.bert.config.hidden_size

        # POS embedding layer
        self.pos_embeddings = nn.Embedding(num_pos_tags, self.hidden_size)

        # Classifier head
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Dropout
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask=None, pos_tag_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        if pos_tag_ids is not None:
            pos_embeds = self.pos_embeddings(pos_tag_ids)
            sequence_output = sequence_output + pos_embeds  # or torch.cat(...)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.shape[-1])[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
