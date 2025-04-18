import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF

class BertLinearCRF(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.linear(sequence_output)

        mask = attention_mask.bool()

        if labels is not None:
            # Set the padding labels to 0 as -100 is out of range for CRF
            labels = labels.clone()
            labels[~mask] = 0

            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            return {"loss": loss, "logits:": emissions}
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return {"logits": predictions}
