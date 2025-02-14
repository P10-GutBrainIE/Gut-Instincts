from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer



login(token="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")

wnut = load_dataset("wnut_17")

label_list = wnut["train"].features["ner_tags"].feature.names

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

example = wnut["train"][0]
tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)


