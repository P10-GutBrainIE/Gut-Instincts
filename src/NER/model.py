from torch import nn
from transformers import AutoModel

class BertForTokenClassificationWithPOS(nn.Module):
    def __init__(self, model_name: str, num_labels: int, num_pos_tags: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        # POS tag embedding
        self.pos_embedding = nn.Embedding(num_pos_tags, self.bert.config.hidden_size)

        # Classification layer for NER
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, pos_tag_ids=None, labels=None):
        # Get outputs from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # Add POS embeddings
        if pos_tag_ids is not None:
            pos_embeddings = self.pos_embedding(pos_tag_ids)
            sequence_output = sequence_output + pos_embeddings

        # Dropout and classification
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
