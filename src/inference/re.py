import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from utils.utils import load_json_data, load_relation_labels


class REInference:
	def __init__(
		self,
		test_data_path: str,
		ner_predictions_path: str,
		model_name: str,
		model_type: str,
		experiment_name: str,
		subtask: str,
		model_name_path: str = None,
		save_path: str = None,
		validation_model=None,
	):
		self.test_data = load_json_data(test_data_path)
		self.ner_predictions = load_json_data(ner_predictions_path)
		_, _, self.id2label = load_relation_labels()
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, max_length=512, truncation=True)
		self.model_type = model_type
		self.experiment_name = experiment_name
		self.validation_model = validation_model
		self.subtask = subtask
		self.save_path = save_path

		self.tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
		self.e1_token_id = self.tokenizer.convert_tokens_to_ids("[E1]")
		self.e2_token_id = self.tokenizer.convert_tokens_to_ids("[E2]")

		if validation_model:
			self.model = validation_model
		else:
			from architectures.bert_with_entity_start import BertForREWithEntityStart

			config = AutoConfig.from_pretrained(model_name_path)
			print("config number of labels: ", config.num_labels)
			self.model = BertForREWithEntityStart(
				config=config, e1_token_id=self.e1_token_id, e2_token_id=self.e2_token_id
			)
			state_dict = torch.load(os.path.join(model_name_path, "pytorch_model.bin"), map_location="cpu")
			self.model.load_state_dict(state_dict)
			self.model.eval()

	def perform_inference(self):
		if not self.validation_model:
			raise ValueError("Validation model is required for inference.")

		result = {}

		for sample in tqdm(validation_data, desc=f"Performing RE inference ({self.subtask})"):
			input_ids = sample["input_ids"].unsqueeze(0)
			attention_mask = sample["attention_mask"].unsqueeze(0)
			paper_id = sample.get("paper_id", "unknown")

			with torch.no_grad():
				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				logits = outputs["logits"]
				pred_label = logits.argmax(-1).item()

			if self.subtask == "6.2.1":
				if paper_id not in result:
					result[paper_id] = {"binary_tag_based_relations": []}
				if pred_label == 1:
					result[paper_id]["binary_tag_based_relations"].append(
						{
							"subject_label": sample["subj_label"],
							"object_label": sample["obj_label"],
						}
					)
			elif self.subtask == "6.2.2":
				pred_relation = self.id2label[pred_label]
				if paper_id not in result:
					result[paper_id] = {"ternary_tag_based_relations": []}
				if pred_relation != "no relation":
					result[paper_id]["ternary_tag_based_relations"].append(
						{
							"subject_label": sample["subj_label"],
							"predicate": pred_relation,
							"object_label": sample["obj_label"],
						}
					)
			elif self.subtask == "6.2.3":
				pred_relation = self.id2label[pred_label]
				if paper_id not in result:
					result[paper_id] = {"ternary_mention_based_relations": []}
				if pred_relation != "no relation":
					result[paper_id]["ternary_mention_based_relations"].append(
						{
							"subject_text_span": sample["subj"]["text_span"],
							"subject_label": sample["subj"],
							"predicate": pred_relation,
							"object_label": sample["obj"],
							"object_text_span": sample["obj"]["text_span"],
						}
					)
			else:
				raise ValueError(f"Unsupported subtask: {self.subtask}")

		if self.save_path:
			os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
			with open(self.save_path, "w") as f:
				json.dump(result, f, indent=4)
		else:
			return result
