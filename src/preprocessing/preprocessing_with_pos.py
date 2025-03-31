import json
import os
import pickle
import random
import spacy
from transformers import AutoTokenizer
from utils.utils import load_bio_labels


class Preprocessor:
    def __init__(
        self,
        file_paths: list[str],
        save_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        train_val_split: float = 0.9,
    ):
        self.file_paths = file_paths
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_val_split = train_val_split
        _, self.label2id, _ = load_bio_labels()
        self.nlp = spacy.load("en_core_web_sm")
        self.pos2id = self._build_pos2id()

    def _build_pos2id(self):
        universal_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ',
                          'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                          'SCONJ', 'SYM', 'VERB', 'X']
        return {tag: i + 1 for i, tag in enumerate(universal_tags)}

    def process_files(self):
        processed_papers = []
        for file_path in self.file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)

                all_data = []
                for _, content in file_data.items():
                    processed_data = self._process_paper(content)
                    all_data.extend(processed_data)

            processed_papers.append(all_data)

        training_data, validation_data = self._train_val_split(processed_papers)
        self._save_to_pickle(training_data, validation_data)

    def _process_paper(self, content):
        processed = []
        metadata = content.get("metadata", {})
        title = metadata.get("title", "")
        abstract = metadata.get("abstract", "")
        entities = content.get("entities", [])

        for section, text in [("title", title), ("abstract", abstract)]:
            tokens, bio_tag_ids, input_ids, attention_mask, pos_tag_ids = self._tokenize_with_bio(text, entities, section)
            processed.append({
                "words": tokens,
                "labels": bio_tag_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pos_tag_ids": pos_tag_ids,
            })
        return processed

    def _tokenize_with_bio(self, text, entities, section):
        encoding = self.tokenizer(
            text, return_offsets_mapping=True, truncation=True,
            max_length=self.max_length, padding="max_length"
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        offsets = encoding["offset_mapping"]

        bio_tags = ["O"] * len(tokens)
        pos_tag_ids = [0] * len(tokens)

        doc = self.nlp(text)
        word_spans = [(token.idx, token.idx + len(token.text), token.pos_) for token in doc]

		# Assign POS tags
        for i, (start, end) in enumerate(offsets):
            if start is None or end is None or (start, end) == (0, 0):
                continue
			
            for spacy_start, spacy_end, spacy_pos in word_spans:
                if start >= spacy_start and end <= spacy_end:
                    pos_tag_ids[i] = self.pos2id.get(spacy_pos, 0)
                    break
        
        #print(tokens[0:33])
        #print(pos_tag_ids[0:33])
        #print(self.pos2id)
        #exit()

		# Assign BIO tags
        for entity in entities:
            if entity.get("location") != section:
                continue

            entity_text = entity["text_span"]
            entity_label = entity["label"]
            start_index = text.find(entity_text)
            if start_index == -1:
                continue
            end_index = start_index + len(entity_text)

            first_token_assigned = False
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start is None or token_end is None or (token_start, token_end) == (0, 0):
                    continue

                if token_end > start_index and token_start < end_index:
                    if not first_token_assigned:
                        bio_tags[i] = f"B-{entity_label}"
                        first_token_assigned = True
                    else:
                        bio_tags[i] = f"I-{entity_label}"

        bio_tag_ids = []
        for offset, tag in zip(offsets, bio_tags):
            if offset is None or offset == (0, 0):
                bio_tag_ids.append(-100)
            else:
                bio_tag_ids.append(self.label2id.get(tag, 0))

        return tokens, bio_tag_ids, encoding["input_ids"], encoding["attention_mask"], pos_tag_ids

    def _train_val_split(self, processed_papers):
        training_data = []
        validation_data = []

        for data in processed_papers:
            random.shuffle(data)
            split_index = int(len(data) * self.train_val_split)
            training_data.extend(data[:split_index])
            validation_data.extend(data[split_index:])

        return training_data, validation_data

    def _save_to_pickle(self, training_data, validation_data):
        os.makedirs(self.save_path, exist_ok=True)

        with open(os.path.join(self.save_path, "training.pkl"), "wb") as f:
            pickle.dump(training_data, f)
        with open(os.path.join(self.save_path, "validation.pkl"), "wb") as f:
            pickle.dump(validation_data, f)


if __name__ == "__main__":
    shared_path = os.path.join("data", "Annotations", "Train")
    file_paths = [
        os.path.join(shared_path, "platinum_quality", "json_format", "train_platinum.json"),
        os.path.join(shared_path, "gold_quality", "json_format", "train_gold.json"),
        os.path.join(shared_path, "silver_quality", "json_format", "train_silver.json"),
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", use_fast=True
    )

    preprocessor = Preprocessor(
        file_paths, os.path.join("data_preprocessed"), tokenizer, max_length=512, train_val_split=0.9
    )
    preprocessor.process_files()
