import os
import json
import torch
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.nn.functional import softmax
from collections import defaultdict
from utils.utils import load_json_data, load_bio_labels
from dotenv import load_dotenv
from huggingface_hub import login

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)

load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
login(HUGGING_FACE_TOKEN)


class LogitEnsembleNER:
	def __init__(
		self,
		test_data_path: str,
		model_names: list[str],
		save_path: str,
		device="cuda" if torch.cuda.is_available() else "cpu",
	):
		self.test_data = load_json_data(test_data_path)
		self.save_path = save_path
		self.device = device
		self.label_list, self.label2id, self.id2label = load_bio_labels()

		self.models = []
		self.tokenizers = []

		logging.info(f"Using device: {self.device}")
		logging.info("Loading models...")

		for model_name in model_names:
			logging.info(f"Loading model: {model_name}")
			model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device).eval()
			tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=512)
			self.models.append(model)
			self.tokenizers.append(tokenizer)

	def perform_inference(self):
		result = {}
		logging.info(f"Starting inference on {len(self.test_data)} papers...")

		for idx, (paper_id, content) in enumerate(self.test_data.items(), start=1):
			try:
				logging.info(f"[{idx}] Processing paper ID: {paper_id}")
				title_entities = self._ensemble_predict(content["metadata"]["title"], "title")
				abstract_entities = self._ensemble_predict(content["metadata"]["abstract"], "abstract")
				result[paper_id] = {"entities": title_entities + abstract_entities}
			except Exception as e:
				logging.error(f"Error processing paper ID {paper_id}: {e}")

		os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
		with open(self.save_path, "w") as f:
			json.dump(result, f, indent=4)

		logging.info(f"Inference complete. Results saved to: {self.save_path}")

	def _ensemble_predict(self, text, location):
		token_span_scores = defaultdict(lambda: defaultdict(list))

		for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
			flag = False
			encoding = tokenizer(
				text,
				return_offsets_mapping=True,
				return_tensors="pt",
				truncation=True,
				padding="max_length",
				max_length=512,
			)
			input_ids = encoding["input_ids"].to(self.device)
			attention_mask = encoding["attention_mask"].to(self.device)
			offset_mapping = encoding["offset_mapping"][0].tolist()

			with torch.no_grad():
				logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(0).cpu().numpy()
				for i, (start, end) in enumerate(offset_mapping):
					if start == end or (start == 0 and end == 0):
						continue
					for label_id, logit in enumerate(logits[i]):
						label = self.id2label[label_id]
						token_span_scores[(start, end)][label].append(logit)
				#probs = softmax(logits, dim=-1).cpu().numpy()

		# Average scores across models
		final_labels = {}
		#print("\n\nFinal labels:")
		#print("token_span_scores:", token_span_scores.keys())
		
		for span, label_scores in token_span_scores.items():
			avg_logits = np.array([np.mean(label_scores[label]) for label in self.label_list])
			probs = softmax(torch.tensor(avg_logits), dim=-1).numpy()
			best_label_id = int(np.argmax(probs))
			best_label = self.id2label[best_label_id]
			final_labels[span] = best_label
			
			#print (f"\nspan: {span}, \n  label_scores: {label_scores}")

			#avg_scores = {label: np.mean(scores) for label, scores in label_scores.items()}
			#print(f"\nspan: {span}, \n  avg_scores: {avg_scores}")
			#best_label = max(avg_scores.items(), key=lambda x: x[1])[0]
			#final_labels[span] = best_label       
		return self._merge_entities(final_labels, text, location)

	def _merge_entities(self, final_labels, text, location):
		entities = []
		current_entity = None

		for start, end in sorted(final_labels.keys()):
			label = final_labels[(start, end)]

			if label == "O":
				if current_entity:
					entities.append(current_entity)
					current_entity = None
				continue

			tag_type, entity_label = label.split("-")
			word = text[start:end].lower()

			if tag_type == "B" or current_entity is None:
				if current_entity:
					entities.append(current_entity)
				current_entity = {
					"start_idx": start,
					"end_idx": end - 1,
					"location": location,
					"text_span": word,
					"label": entity_label,
				}
			else:
				if start == current_entity["end_idx"] + 1:
					current_entity["text_span"] += word
				else:
					current_entity["text_span"] += " " + word
				current_entity["end_idx"] = end - 1

		if current_entity:
			entities.append(current_entity)

		return entities


if __name__ == "__main__":
	model_names = [
		"ihlen/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext_NER",
		"ihlen/scibert_scivocab_uncased_NER",
		"ihlen/BioLinkBERT-large_NER",
	]

	inference = LogitEnsembleNER(
		test_data_path=os.path.join("data", "Annotations", "Dev", "json_format", "dev.json"),
		model_names=model_names,
		save_path="data_inference_results/ensemble_logits.json",
	)
	inference.perform_inference()
